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
from collections.abc import Collection
from pathlib import Path

from .path_safety import is_linklike

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

    On failure before the rename the temp file is removed and the destination
    is left untouched. A durable directory-sync failure is reported after the
    rename, so the new destination may already be visible even though crash
    durability could not be established.
    """
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        with open(temporary, "x", encoding=encoding) as handle:
            handle.write(text)
            if mode is not None:
                if hasattr(os, "fchmod"):
                    os.fchmod(handle.fileno(), mode)
                else:
                    os.chmod(temporary, mode)
            if durable:
                handle.flush()
                os.fsync(handle.fileno())
        os.replace(temporary, path)
        if durable:
            strict_fsync_directory(path.parent)
    finally:
        with contextlib.suppress(OSError):
            temporary.unlink(missing_ok=True)


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


def _supports_anchored_mkdir() -> bool:
    supports_dir_fd: Collection[object] = getattr(
        os,
        "supports_dir_fd",
        frozenset(),
    )
    supports_follow_symlinks: Collection[object] = getattr(
        os,
        "supports_follow_symlinks",
        frozenset(),
    )
    return (
        os.open in supports_dir_fd
        and os.mkdir in supports_dir_fd
        and os.stat in supports_dir_fd
        and os.stat in supports_follow_symlinks
        and hasattr(os, "O_DIRECTORY")
        and hasattr(os, "O_NOFOLLOW")
        and hasattr(os, "fchmod")
    )


def _anchored_directory_flags() -> int:
    return os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | getattr(os, "O_CLOEXEC", 0)


def _component_identity(path_stat: os.stat_result) -> tuple[int, int]:
    return path_stat.st_dev, path_stat.st_ino


def _secure_mkdir_anchored(path: Path, *, mode: int, durable: bool) -> None:
    target = Path(os.path.abspath(path))
    components = target.parts[1:]
    parent_fd = os.open(target.anchor, _anchored_directory_flags())
    try:
        for component in components:
            created = False
            created_identity: tuple[int, int] | None = None
            try:
                child_fd = os.open(
                    component,
                    _anchored_directory_flags(),
                    dir_fd=parent_fd,
                )
            except FileNotFoundError:
                try:
                    os.mkdir(component, mode=mode, dir_fd=parent_fd)
                    created = True
                except FileExistsError:
                    # A concurrent creator won. Open and validate its entry,
                    # but do not change permissions on a directory we did not
                    # create.
                    pass
                if created:
                    created_stat = os.stat(
                        component,
                        dir_fd=parent_fd,
                        follow_symlinks=False,
                    )
                    if not stat.S_ISDIR(created_stat.st_mode):
                        raise OSError(
                            errno.ESTALE,
                            "new directory component was replaced",
                            component,
                        ) from None
                    created_identity = _component_identity(created_stat)
                child_fd = os.open(
                    component,
                    _anchored_directory_flags(),
                    dir_fd=parent_fd,
                )
            except OSError as exc:
                if exc.errno == errno.ENOTDIR:
                    raise FileExistsError(
                        errno.EEXIST,
                        "path component is not a real directory",
                        component,
                    ) from exc
                raise

            advance = False
            try:
                child_stat = os.fstat(child_fd)
                child_identity = _component_identity(child_stat)
                if created_identity is not None and child_identity != created_identity:
                    raise OSError(
                        errno.ESTALE,
                        "new directory component changed while opening",
                        component,
                    )
                if created:
                    os.fchmod(child_fd, mode)
                    if durable:
                        os.fsync(child_fd)
                        os.fsync(parent_fd)
                linked_stat = os.stat(
                    component,
                    dir_fd=parent_fd,
                    follow_symlinks=False,
                )
                if _component_identity(linked_stat) != child_identity:
                    raise OSError(
                        errno.ESTALE,
                        "directory component changed during traversal",
                        component,
                    )
                advance = True
            finally:
                if not advance:
                    os.close(child_fd)
            os.close(parent_fd)
            parent_fd = child_fd
    finally:
        os.close(parent_fd)


def _secure_mkdir_portable(path: Path, *, mode: int, durable: bool) -> None:
    if durable:
        raise OSError(
            errno.ENOTSUP,
            "durable secure directory creation requires dir_fd and O_NOFOLLOW",
            path,
        )
    target = Path(os.path.abspath(path))
    current = Path(target.anchor)
    for component in target.parts[1:]:
        current /= component
        created = False
        try:
            current.mkdir(mode=mode)
            created = True
        except FileExistsError:
            pass
        path_stat = os.lstat(current)
        if is_linklike(path_stat) or not stat.S_ISDIR(path_stat.st_mode):
            raise FileExistsError(
                errno.EEXIST,
                "path component is not a real directory",
                current,
            )
        if created:
            os.chmod(current, mode)


def secure_mkdir(
    path: Path,
    *,
    mode: int = 0o700,
    durable: bool = False,
) -> Path:
    """Create a no-follow directory path and chmod only new components.

    Supported POSIX platforms walk from the filesystem anchor using directory
    file descriptors, ``mkdirat``/``openat`` semantics, and ``O_NOFOLLOW``.
    Durable creation fsyncs each new directory and its pinned parent before
    continuing. Platforms without those primitives retain non-durable
    ``mkdir -p`` behavior but reject durable creation rather than weakening it.
    """

    if _supports_anchored_mkdir():
        _secure_mkdir_anchored(path, mode=mode, durable=durable)
    else:
        _secure_mkdir_portable(path, mode=mode, durable=durable)
    return path
