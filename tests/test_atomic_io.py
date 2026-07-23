"""Tests for the shared atomic filesystem primitives."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

import afs.atomic_io as atomic_io
from afs.atomic_io import (
    atomic_create_text,
    atomic_write_text,
    exclusive_create_text,
    secure_mkdir,
    strict_fsync_directory,
)


def _mode_of(path: Path) -> int:
    return stat.S_IMODE(path.stat().st_mode)


def test_atomic_write_publishes_content_without_temp_residue(tmp_path: Path) -> None:
    target = tmp_path / "state.json"
    atomic_write_text(target, '{"ok": true}')
    assert target.read_text(encoding="utf-8") == '{"ok": true}'
    leftovers = [p for p in tmp_path.iterdir() if p != target]
    assert leftovers == []


def test_atomic_write_replaces_existing_content(tmp_path: Path) -> None:
    target = tmp_path / "state.json"
    target.write_text("old", encoding="utf-8")
    atomic_write_text(target, "new")
    assert target.read_text(encoding="utf-8") == "new"


def test_atomic_write_applies_mode_regardless_of_umask(
    tmp_path: Path,
) -> None:
    target = tmp_path / "private.txt"
    previous_umask = os.umask(0o000)
    try:
        atomic_write_text(target, "secret", mode=0o600)
    finally:
        os.umask(previous_umask)
    assert _mode_of(target) == 0o600


def test_atomic_write_durable_smoke(tmp_path: Path) -> None:
    target = tmp_path / "durable.txt"
    atomic_write_text(target, "flushed", durable=True)
    assert target.read_text(encoding="utf-8") == "flushed"


def test_atomic_write_durable_fails_when_parent_directory_open_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "durable.txt"

    def fail_directory_open(*_args: object, **_kwargs: object) -> int:
        raise OSError("simulated directory open failure")

    monkeypatch.setattr(atomic_io.os, "open", fail_directory_open)
    with pytest.raises(OSError, match="simulated directory open failure"):
        atomic_write_text(target, "published", durable=True)

    assert target.read_text(encoding="utf-8") == "published"
    assert list(tmp_path.iterdir()) == [target]


def test_atomic_write_durable_fails_when_parent_directory_fsync_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "durable.txt"
    real_fsync = atomic_io.os.fsync

    def fail_directory_fsync(descriptor: int) -> None:
        if stat.S_ISDIR(os.fstat(descriptor).st_mode):
            raise OSError("simulated directory fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(atomic_io.os, "fsync", fail_directory_fsync)
    with pytest.raises(OSError, match="simulated directory fsync failure"):
        atomic_write_text(target, "published", durable=True)

    assert target.read_text(encoding="utf-8") == "published"
    assert list(tmp_path.iterdir()) == [target]


def test_atomic_write_durable_applies_mode_before_sync_and_publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "durable.txt"
    events: list[str] = []
    real_fchmod = atomic_io.os.fchmod
    real_fsync = atomic_io.os.fsync
    real_replace = atomic_io.os.replace

    def record_fchmod(descriptor: int, mode: int) -> None:
        events.append("fchmod")
        real_fchmod(descriptor, mode)

    def record_fsync(descriptor: int) -> None:
        kind = "directory_fsync" if stat.S_ISDIR(os.fstat(descriptor).st_mode) else "file_fsync"
        events.append(kind)
        real_fsync(descriptor)

    def record_replace(source: Path, destination: Path) -> None:
        events.append("replace")
        real_replace(source, destination)

    monkeypatch.setattr(atomic_io.os, "fchmod", record_fchmod)
    monkeypatch.setattr(atomic_io.os, "fsync", record_fsync)
    monkeypatch.setattr(atomic_io.os, "replace", record_replace)
    atomic_write_text(target, "published", mode=0o600, durable=True)

    assert events == ["fchmod", "file_fsync", "replace", "directory_fsync"]
    assert _mode_of(target) == 0o600


def test_atomic_write_non_durable_skips_strict_directory_sync(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "state.txt"

    def fail_strict_sync(_directory: Path) -> None:
        raise AssertionError("non-durable write requested a strict directory sync")

    monkeypatch.setattr(atomic_io, "strict_fsync_directory", fail_strict_sync)
    atomic_write_text(target, "best effort")

    assert target.read_text(encoding="utf-8") == "best effort"


def test_atomic_write_encoding_failure_cleans_temp_and_preserves_target(
    tmp_path: Path,
) -> None:
    target = tmp_path / "state.txt"
    target.write_text("original", encoding="utf-8")

    with pytest.raises(UnicodeEncodeError):
        atomic_write_text(target, "snowman: \N{SNOWMAN}", encoding="ascii")

    assert target.read_text(encoding="utf-8") == "original"
    assert list(tmp_path.iterdir()) == [target]


def test_atomic_write_failure_cleans_temp_and_preserves_target(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    target = tmp_path / "state.json"
    target.write_text("original", encoding="utf-8")

    def boom(src: object, dst: object) -> None:
        raise OSError("simulated rename failure")

    monkeypatch.setattr(os, "replace", boom)
    with pytest.raises(OSError, match="simulated rename failure"):
        atomic_write_text(target, "new")
    monkeypatch.undo()

    assert target.read_text(encoding="utf-8") == "original"
    assert list(tmp_path.iterdir()) == [target]


def test_exclusive_create_writes_with_declared_mode(tmp_path: Path) -> None:
    target = tmp_path / "artifact.md"
    previous_umask = os.umask(0o000)
    try:
        exclusive_create_text(target, "immutable", mode=0o600)
    finally:
        os.umask(previous_umask)
    assert target.read_text(encoding="utf-8") == "immutable"
    assert _mode_of(target) == 0o600


def test_exclusive_create_durable_smoke(tmp_path: Path) -> None:
    target = tmp_path / "durable-receipt.json"
    atomic_create_text(target, '{"status":"complete"}\n', durable=True)
    assert target.read_text(encoding="utf-8") == '{"status":"complete"}\n'
    assert _mode_of(target) == 0o600


def test_atomic_create_never_exposes_final_path_when_publish_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "receipt.json"
    monkeypatch.setattr(
        atomic_io,
        "_rename_noreplace",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(OSError("publish failed")),
    )
    with pytest.raises(OSError, match="publish failed"):
        atomic_create_text(target, "complete", durable=True)
    assert not target.exists()
    assert list(tmp_path.iterdir()) == []


def test_atomic_create_refuses_to_replace_existing_target(tmp_path: Path) -> None:
    target = tmp_path / "receipt.json"
    target.write_text("original", encoding="utf-8")
    with pytest.raises(FileExistsError):
        atomic_create_text(target, "replacement", durable=True)
    assert target.read_text(encoding="utf-8") == "original"
    assert list(tmp_path.iterdir()) == [target]


def test_strict_fsync_directory_rejects_regular_file(tmp_path: Path) -> None:
    target = tmp_path / "not-a-directory"
    target.write_text("x", encoding="utf-8")
    with pytest.raises((NotADirectoryError, OSError)):
        strict_fsync_directory(target)


def test_exclusive_create_refuses_existing_file(tmp_path: Path) -> None:
    target = tmp_path / "artifact.md"
    target.write_text("first", encoding="utf-8")
    with pytest.raises(FileExistsError):
        exclusive_create_text(target, "second")
    assert target.read_text(encoding="utf-8") == "first"


def test_exclusive_create_refuses_symlink_at_path(tmp_path: Path) -> None:
    escape_target = tmp_path / "outside.txt"
    link = tmp_path / "artifact.md"
    link.symlink_to(escape_target)
    with pytest.raises(OSError):
        exclusive_create_text(link, "smuggled")
    assert not escape_target.exists()


def test_secure_mkdir_applies_mode_to_all_created_directories(
    tmp_path: Path,
) -> None:
    pre_existing = tmp_path / "base"
    pre_existing.mkdir(mode=0o755)
    pre_existing_mode = _mode_of(pre_existing)

    previous_umask = os.umask(0o022)
    try:
        leaf = secure_mkdir(pre_existing / "a" / "b" / "c", mode=0o700)
    finally:
        os.umask(previous_umask)

    assert leaf == pre_existing / "a" / "b" / "c"
    for created in (
        pre_existing / "a",
        pre_existing / "a" / "b",
        pre_existing / "a" / "b" / "c",
    ):
        assert _mode_of(created) == 0o700
    assert _mode_of(pre_existing) == pre_existing_mode


def test_secure_mkdir_is_idempotent(tmp_path: Path) -> None:
    target = tmp_path / "x" / "y"
    secure_mkdir(target)
    secure_mkdir(target)
    assert target.is_dir()
