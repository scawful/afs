"""Shared atomic filesystem primitives for AFS state files.

Standard write path for durable state: publish whole files atomically so
concurrent readers never observe partial content, apply restrictive
permissions before a file becomes visible at its final path, and never
silently overwrite artifacts that must be immutable.

See docs/ENGINEERING_PRACTICES.md for when to use which primitive.
"""

from __future__ import annotations

import contextlib
import ctypes
import errno
import os
import stat
import sys
import uuid
from pathlib import Path

__all__ = [
    "atomic_create_text",
    "atomic_write_text",
    "exclusive_create_text",
    "fsync_directory",
    "secure_mkdir",
    "strict_fsync_directory",
]

_AT_FDCWD = -100
_RENAME_NOREPLACE = 0x00000001
_RENAME_EXCL = 0x00000004


def _rename_noreplace(source: Path, destination: Path) -> None:
    """Atomically rename without replacing an existing destination."""

    library = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin" and getattr(library, "renamex_np", None) is not None:
        renamex_np = library.renamex_np
        renamex_np.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint)
        renamex_np.restype = ctypes.c_int
        result = renamex_np(os.fsencode(source), os.fsencode(destination), _RENAME_EXCL)
    elif sys.platform.startswith("linux") and getattr(library, "renameat2", None) is not None:
        renameat2 = library.renameat2
        renameat2.argtypes = (
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        )
        renameat2.restype = ctypes.c_int
        result = renameat2(
            _AT_FDCWD,
            os.fsencode(source),
            _AT_FDCWD,
            os.fsencode(destination),
            _RENAME_NOREPLACE,
        )
    else:
        raise OSError(errno.ENOTSUP, "atomic no-replace rename is unavailable")
    if result != 0:
        error_number = ctypes.get_errno()
        raise OSError(error_number, os.strerror(error_number), destination)


def atomic_create_text(
    path: Path,
    text: str,
    *,
    encoding: str = "utf-8",
    mode: int = 0o600,
    durable: bool = False,
) -> None:
    """Atomically publish a new immutable file without replacing a target.

    Content is completed in a private sibling temporary file, then renamed to
    the final path with create-or-fail semantics. A crash can leave a complete
    temporary file, but never a partial final receipt.
    """

    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(temporary, flags, mode)
    try:
        with os.fdopen(descriptor, "w", encoding=encoding) as handle:
            handle.write(text)
            if hasattr(os, "fchmod"):
                os.fchmod(handle.fileno(), mode)
            if durable:
                handle.flush()
                os.fsync(handle.fileno())
        _rename_noreplace(temporary, path)
        if durable:
            strict_fsync_directory(path.parent)
    finally:
        with contextlib.suppress(OSError):
            temporary.unlink(missing_ok=True)


def fsync_directory(directory: Path) -> None:
    """Best-effort fsync of a directory entry.

    Directory file descriptors cannot be opened on some platforms
    (notably Windows); those failures are tolerated because the caller's
    os.replace() is still atomic — the directory fsync only strengthens
    crash durability where the platform supports it.
    """
    try:
        fd = os.open(directory, os.O_RDONLY)
    except OSError:
        return
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def strict_fsync_directory(directory: Path) -> None:
    """Durably sync a directory or raise when the platform cannot do so.

    Namespace-changing transactions such as context activation cannot accept
    the best-effort semantics of :func:`fsync_directory`.
    """

    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_CLOEXEC", 0)
    descriptor = os.open(directory, flags)
    try:
        directory_stat = os.fstat(descriptor)
        if not stat.S_ISDIR(directory_stat.st_mode):
            raise NotADirectoryError(directory)
        if directory_stat.st_nlink < 1:
            raise OSError(f"directory has no durable link: {directory}")
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def atomic_write_text(
    path: Path,
    text: str,
    *,
    encoding: str = "utf-8",
    mode: int | None = None,
    durable: bool = False,
) -> None:
    """Atomically publish ``text`` at ``path`` via exclusive temp + rename.

    A concurrent reader sees either the old file or the new file, never a
    partial write. When ``mode`` is given it is applied to the temp file
    before the rename, so the final path never exists with looser
    permissions. When ``durable`` is true the content is fsynced before
    the rename and the directory entry is fsynced after it.

    On failure the temp file is removed and the original error re-raised;
    the destination is left untouched.
    """
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(temporary, "x", encoding=encoding) as handle:
            handle.write(text)
            if durable:
                handle.flush()
                os.fsync(handle.fileno())
        if mode is not None:
            os.chmod(temporary, mode)
        os.replace(temporary, path)
        if durable:
            fsync_directory(path.parent)
    except OSError:
        with contextlib.suppress(OSError):
            temporary.unlink(missing_ok=True)
        raise


def exclusive_create_text(
    path: Path,
    text: str,
    *,
    encoding: str = "utf-8",
    mode: int = 0o600,
) -> None:
    """Create ``path`` with ``text``, failing if anything already exists there.

    O_CREAT|O_EXCL guarantees create-or-fail semantics: an existing file,
    directory, or symlink at ``path`` (dangling or not) raises
    FileExistsError and nothing is written. O_NOFOLLOW additionally
    refuses to write through a symlink where the platform supports it.
    Use this for artifacts that must never be overwritten (immutable
    revisions, one-shot claims).
    """
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    fd = os.open(path, flags, mode)
    with os.fdopen(fd, "w", encoding=encoding) as handle:
        handle.write(text)
    # os.open's mode argument is masked by the process umask; re-apply so
    # the declared permissions hold regardless of the caller's umask.
    os.chmod(path, mode)


def secure_mkdir(path: Path, *, mode: int = 0o700) -> Path:
    """``mkdir -p`` that applies ``mode`` to every directory it creates.

    ``Path.mkdir(mode=..., parents=True)`` applies the mode only to the
    leaf; intermediate directories get umask defaults. This helper chmods
    each directory this call actually created, leaving pre-existing
    ancestors untouched.
    """
    missing: list[Path] = []
    current = path
    while not current.exists():
        missing.append(current)
        current = current.parent
    path.mkdir(parents=True, exist_ok=True)
    for directory in missing:
        os.chmod(directory, mode)
    return path
