"""Versioned reads and cooperating, atomic context writes.

The digest covers raw bytes. The lock serializes AFS writers, including
unconditional writes and appends. Editors and other non-cooperating writers
must use this API to participate in the compare-and-write guarantee.
"""

from __future__ import annotations

import hashlib
import io
import os
import re
import stat
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from .atomic_io import atomic_write_bytes
from .path_safety import assert_no_linklike_components


class FileWriteConflict(ValueError):
    """The destination no longer has the version supplied by the caller."""

    def __init__(self, path: Path, expected: str, actual: str) -> None:
        self.expected = expected
        self.actual = actual
        super().__init__(f"write conflict: {path}: expected {expected}, found {actual}")


def read_text_snapshot(
    path: Path, *, encoding: str = "utf-8", errors: str = "replace", preserve_newlines: bool = False
) -> tuple[str, str]:
    """Return text and a digest from the same read, including original newlines."""
    data = path.read_bytes()
    with io.TextIOWrapper(
        io.BytesIO(data), encoding=encoding, errors=errors,
        newline="" if preserve_newlines else None,
    ) as reader:
        return reader.read(), hashlib.sha256(data).hexdigest()


@contextmanager
def _write_lock(path: Path) -> Iterator[int | None]:
    # POSIX directory locks survive atomic replacement of the destination and
    # avoid putting lock files into indexed context mounts.
    if os.name == "posix":
        import fcntl

        flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW
        parent = path.absolute().parent
        descriptor = os.open(parent.anchor, flags)
        try:
            for component in parent.parts[1:]:
                child = os.open(component, flags, dir_fd=descriptor)
                os.close(descriptor)
                descriptor = child
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            yield descriptor
        finally:
            os.close(descriptor)
    elif sys.platform == "win32":  # pragma: no cover - Windows CI
        import msvcrt

        lock_path = path.with_name(f".{path.name}.afs-write.lock")
        assert_no_linklike_components(lock_path)
        descriptor = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            if os.fstat(descriptor).st_nlink != 1:
                raise ValueError("write lock must not have hard links")
            msvcrt.locking(descriptor, msvcrt.LK_LOCK, 1)
            yield None
        finally:
            os.close(descriptor)
    else:  # pragma: no cover - fail closed on unsupported runtimes
        raise OSError("atomic context writes require POSIX or Windows file locking")


def write_text_checked(
    path: Path,
    content: str,
    *,
    encoding: str = "utf-8",
    append: bool = False,
    if_match: str | None = None,
) -> str:
    """Check ``if_match`` (SHA-256 or ``missing``), then publish under one lock."""
    if if_match is not None:
        if not isinstance(if_match, str) or not re.fullmatch(
            r"(?:[0-9a-fA-F]{64}|missing)", if_match
        ):
            raise ValueError("if_match must be a SHA-256 digest or 'missing'")
        if_match = if_match.lower()
    # Callers resolve allowed mount links before reaching this boundary.
    # Reject links introduced after that resolution, including dangling links.
    assert_no_linklike_components(path)
    with _write_lock(path) as directory_fd:
        assert_no_linklike_components(path)
        target_name = str(path) if directory_fd is None else path.name
        needs_read = append or if_match is not None
        try:
            # Refuse special files before opening them for write (a FIFO can
            # fail or block on open). Recheck the opened descriptor below.
            target_stat = os.stat(target_name, dir_fd=directory_fd, follow_symlinks=False)
            if not stat.S_ISREG(target_stat.st_mode):
                raise ValueError("context write destination must be a regular file")
            # Enforce effective write permission, including ACLs, without
            # truncating the original. Ordinary overwrites need no read access.
            descriptor = os.open(
                target_name,
                (os.O_RDWR if needs_read else os.O_WRONLY)
                | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0),
                dir_fd=directory_fd,
            )
            with os.fdopen(descriptor, "r+b" if needs_read else "wb") as handle:
                file_stat = os.fstat(handle.fileno())
                if not stat.S_ISREG(file_stat.st_mode):
                    raise ValueError("context write destination must be a regular file")
                mode = stat.S_IMODE(file_stat.st_mode) & 0o777
                previous = handle.read() if needs_read else b""
            actual = hashlib.sha256(previous).hexdigest()
        except FileNotFoundError:
            previous, actual, mode = b"", "missing", 0o600
        if if_match is not None and if_match != actual:
            raise FileWriteConflict(path, if_match, actual)
        buffer = io.BytesIO(previous if append else b"")
        buffer.seek(0, io.SEEK_END)
        with io.TextIOWrapper(buffer, encoding=encoding, newline="") as writer:
            writer.write(content)
            writer.flush()
            data = buffer.getvalue()
        if directory_fd is not None:
            opened = os.fstat(directory_fd)
            linked = path.parent.stat(follow_symlinks=False)
            if (opened.st_dev, opened.st_ino) != (linked.st_dev, linked.st_ino):
                raise ValueError("write destination directory changed during the operation")
        atomic_write_bytes(path, data, mode=mode, directory_fd=directory_fd)
        return hashlib.sha256(data).hexdigest()
