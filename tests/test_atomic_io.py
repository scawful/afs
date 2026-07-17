"""Tests for the shared atomic filesystem primitives."""

from __future__ import annotations

import os
import stat
from pathlib import Path

import pytest

from afs.atomic_io import (
    atomic_write_text,
    exclusive_create_text,
    secure_mkdir,
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
