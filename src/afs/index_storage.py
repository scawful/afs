"""Portable locking and durable publication helpers for rebuildable indexes."""

from __future__ import annotations

import os
import re
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import BinaryIO

try:  # POSIX
    import fcntl  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]

try:  # Windows
    import msvcrt  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - POSIX
    msvcrt = None  # type: ignore[assignment]

_GENERATION_ID = re.compile(r"^[a-f0-9]{32}$")


class IndexLockTimeout(TimeoutError):
    """Raised when an index publication lock cannot be acquired in time."""


@contextmanager
def index_file_lock(
    path: Path,
    *,
    shared: bool = False,
    timeout: float = 30.0,
) -> Iterator[None]:
    """Acquire a cross-process index lock.

    POSIX uses shared/exclusive ``flock``. Windows uses an exclusive byte-range
    lock for both modes because ``msvcrt`` has no shared-lock primitive.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+b")
    try:
        _acquire(handle, shared=shared, timeout=timeout, path=path)
        yield
    finally:
        _release(handle)
        handle.close()


def atomic_write_text(path: Path, text: str) -> None:
    """Write, fsync, atomically replace, and fsync the containing directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{time.time_ns()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def fsync_directory(path: Path) -> None:
    """Best-effort directory durability (unsupported on some platforms)."""
    flags = getattr(os, "O_DIRECTORY", 0) | os.O_RDONLY
    try:
        descriptor = os.open(path, flags)
    except OSError:
        return
    try:
        os.fsync(descriptor)
    except OSError:
        pass
    finally:
        os.close(descriptor)


def read_generation_id(current_path: Path) -> str | None:
    """Read and validate a CURRENT pointer; absence means legacy layout."""
    try:
        generation_id = current_path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    if not _GENERATION_ID.fullmatch(generation_id):
        raise ValueError(f"Invalid index generation pointer: {current_path}")
    return generation_id


def _acquire(handle: BinaryIO, *, shared: bool, timeout: float, path: Path) -> None:
    deadline = time.monotonic() + max(0.0, timeout)
    while True:
        try:
            if fcntl is not None:
                operation = fcntl.LOCK_SH if shared else fcntl.LOCK_EX
                fcntl.flock(handle.fileno(), operation | fcntl.LOCK_NB)
                return
            if msvcrt is not None:  # pragma: no cover - Windows
                handle.seek(0)
                if handle.read(1) == b"":
                    handle.write(b"0")
                    handle.flush()
                handle.seek(0)
                msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
                return
            return  # pragma: no cover - unsupported interpreter
        except (BlockingIOError, OSError):
            if time.monotonic() >= deadline:
                raise IndexLockTimeout(f"Timed out acquiring index lock: {path}") from None
            time.sleep(0.01)


def _release(handle: BinaryIO) -> None:
    try:
        if fcntl is not None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        elif msvcrt is not None:  # pragma: no cover - Windows
            handle.seek(0)
            msvcrt.locking(handle.fileno(), msvcrt.LK_UNLCK, 1)
    except OSError:
        pass
