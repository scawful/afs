"""Conflict and publication guarantees shared by CLI and MCP writes."""

import hashlib
import stat
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from afs.file_io import FileWriteConflict, read_text_snapshot, write_text_checked


def test_stale_write_does_not_change_bytes(tmp_path: Path) -> None:
    path = tmp_path / "note"
    digest = write_text_checked(path, "original", if_match="missing")
    write_text_checked(path, "winner", if_match=digest)
    with pytest.raises(FileWriteConflict) as conflict:
        write_text_checked(path, "stale", if_match=digest)
    assert conflict.value.actual == hashlib.sha256(b"winner").hexdigest()
    assert path.read_bytes() == b"winner"
    with pytest.raises(FileWriteConflict):
        write_text_checked(path, "new", if_match="missing")


def test_competing_writers_have_exactly_one_winner(tmp_path: Path) -> None:
    path = tmp_path / "note"
    digest = write_text_checked(path, "original")

    def update(i: int) -> bool:
        try:
            write_text_checked(path, str(i), if_match=digest)
        except FileWriteConflict:
            return False
        return True

    with ThreadPoolExecutor(max_workers=12) as executor:
        assert sum(executor.map(update, range(24))) == 1


@pytest.mark.parametrize("encoding", ["utf-8", "utf-16"])
def test_snapshot_hashes_bytes_and_append_preserves_them(
    tmp_path: Path, encoding: str
) -> None:
    path = tmp_path / "note"
    original = "before\r\n".encode(encoding)
    path.write_bytes(original)
    text, digest = read_text_snapshot(path, encoding=encoding)
    assert text == "before\n"
    assert digest == hashlib.sha256(original).hexdigest()
    result = write_text_checked(path, "after", encoding=encoding, append=True, if_match=digest)
    assert path.read_bytes().startswith(original)
    assert path.read_bytes().decode(encoding) == "before\r\nafter"
    assert result == hashlib.sha256(path.read_bytes()).hexdigest()


def test_failed_publication_preserves_old_file(monkeypatch, tmp_path: Path) -> None:
    path = tmp_path / "note"
    path.write_bytes(b"old")

    def fail(*_args: object, **_kwargs: object) -> None:
        raise OSError("injected replace failure")

    monkeypatch.setattr("afs.atomic_io.os.replace", fail)
    with pytest.raises(OSError, match="injected"):
        write_text_checked(path, "replacement")
    assert path.read_bytes() == b"old"
    assert list(tmp_path.iterdir()) == [path]


def test_rejects_symlinks_and_invalid_hashes(tmp_path: Path) -> None:
    path = tmp_path / "note"
    target = tmp_path / "outside"
    target.write_bytes(b"outside")
    path.symlink_to(target)
    with pytest.raises(ValueError, match="symbolic link"):
        write_text_checked(path, "changed")
    assert target.read_bytes() == b"outside"
    with pytest.raises(ValueError, match="SHA-256"):
        write_text_checked(target, "changed", if_match="typo")
    assert target.read_bytes() == b"outside"


def test_preserves_existing_permissions(tmp_path: Path) -> None:
    path = tmp_path / "note"
    path.write_bytes(b"old")
    path.chmod(0o640)
    write_text_checked(path, "new")
    assert stat.S_IMODE(path.stat().st_mode) == 0o640


def test_parallel_appends_do_not_lose_updates(tmp_path: Path) -> None:
    path = tmp_path / "log"
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(lambda i: write_text_checked(path, f"{i}\n", append=True), range(24)))
    assert sorted(map(int, path.read_text().splitlines())) == list(range(24))


def test_rejects_non_regular_destination_without_blocking(tmp_path: Path) -> None:
    import os

    if not hasattr(os, "mkfifo"):
        pytest.skip("POSIX FIFO boundary")
    path = tmp_path / "pipe"
    os.mkfifo(path)
    with pytest.raises(ValueError, match="regular file"):
        write_text_checked(path, "text")
