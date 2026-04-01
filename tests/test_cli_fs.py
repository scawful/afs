"""Tests for the afs fs CLI command surface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from afs.cli.fs import register_parsers
from afs.context_index import ContextSQLiteIndex
from afs.manager import AFSManager
from afs.models import MountType
from afs.schema import AFSConfig, GeneralConfig


def _make_manager(tmp_path: Path) -> tuple[AFSManager, Path]:
    context_root = tmp_path / "context"
    general = GeneralConfig(context_root=context_root)
    manager = AFSManager(config=AFSConfig(general=general))
    project_path = tmp_path / "project"
    project_path.mkdir()
    manager.ensure(path=project_path, context_root=context_root)
    return manager, project_path


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    register_parsers(subparsers)
    return parser


def test_fs_registers_delete_and_move_parsers() -> None:
    parser = _make_parser()

    delete_args = parser.parse_args(["fs", "delete", "scratchpad", "notes.md"])
    assert delete_args.command == "fs"
    assert delete_args.fs_command == "delete"
    assert hasattr(delete_args, "func")

    move_args = parser.parse_args(
        ["fs", "move", "scratchpad", "before.md", "knowledge", "after.md"]
    )
    assert move_args.command == "fs"
    assert move_args.fs_command == "move"
    assert hasattr(move_args, "func")


def test_fs_delete_removes_file_and_index_entry(capsys, monkeypatch, tmp_path: Path) -> None:
    manager, project_path = _make_manager(tmp_path)
    context_root = manager.config.general.context_root
    target = context_root / "scratchpad" / "delete-me.md"
    target.write_text("delete marker", encoding="utf-8")

    index = ContextSQLiteIndex(manager, context_root)
    index.rebuild(mount_types=[MountType.SCRATCHPAD])

    monkeypatch.setattr("afs.cli.fs.load_manager", lambda _config_path: manager)
    parser = _make_parser()
    args = parser.parse_args(
        [
            "fs",
            "delete",
            "scratchpad",
            "delete-me.md",
            "--path",
            str(project_path),
            "--context-root",
            str(context_root),
            "--json",
        ]
    )
    status = args.func(args)
    assert status == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["deleted"] is True
    assert payload["rows_deleted"] == 1
    assert not target.exists()

    entries = index.query(query="delete marker", mount_types=[MountType.SCRATCHPAD])
    assert entries == []


def test_fs_move_supports_cross_mount_moves_and_rebuilds_index(
    capsys, monkeypatch, tmp_path: Path
) -> None:
    manager, project_path = _make_manager(tmp_path)
    context_root = manager.config.general.context_root
    source = context_root / "scratchpad" / "before.md"
    source.write_text("move marker", encoding="utf-8")

    index = ContextSQLiteIndex(manager, context_root)
    index.rebuild(mount_types=[MountType.SCRATCHPAD, MountType.KNOWLEDGE])

    monkeypatch.setattr("afs.cli.fs.load_manager", lambda _config_path: manager)
    parser = _make_parser()
    args = parser.parse_args(
        [
            "fs",
            "move",
            "scratchpad",
            "before.md",
            "knowledge",
            "docs/after.md",
            "--path",
            str(project_path),
            "--context-root",
            str(context_root),
            "--mkdirs",
            "--json",
        ]
    )
    status = args.func(args)
    assert status == 0

    payload = json.loads(capsys.readouterr().out)
    destination = context_root / "knowledge" / "docs" / "after.md"
    assert payload["source_mount_type"] == "scratchpad"
    assert payload["destination_mount_type"] == "knowledge"
    assert destination.exists()
    assert not source.exists()

    scratchpad_entries = index.query(
        query="move marker",
        mount_types=[MountType.SCRATCHPAD],
    )
    knowledge_entries = index.query(
        query="move marker",
        mount_types=[MountType.KNOWLEDGE],
    )
    assert scratchpad_entries == []
    assert len(knowledge_entries) == 1
    assert knowledge_entries[0]["relative_path"] == "docs/after.md"
