"""Tests for the afs fs CLI command surface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from afs.cli.fs import register_parsers
from afs.context_index import ContextSQLiteIndex
from afs.context_layout import scaffold_v2
from afs.manager import AFSManager
from afs.models import MountType
from afs.project_registry import ProjectRecord, ProjectRegistry
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


def _make_v2_manager(
    tmp_path: Path,
) -> tuple[AFSManager, Path, Path, ProjectRecord, ProjectRecord]:
    context_root = tmp_path / "central-context"
    scaffold_v2(context_root)
    manager = AFSManager(
        config=AFSConfig(general=GeneralConfig(context_root=context_root))
    )
    alpha_path = tmp_path / "alpha"
    beta_path = tmp_path / "beta"
    alpha_path.mkdir()
    beta_path.mkdir()
    registry = ProjectRegistry(context_root)
    alpha = registry.register(alpha_path, name="alpha")
    beta = registry.register(beta_path, name="beta")
    return manager, alpha_path, beta_path, alpha, beta


def _run(parser: argparse.ArgumentParser, argv: list[str]) -> int:
    args = parser.parse_args(argv)
    return args.func(args)


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


def test_v2_files_scope_all_operations_and_isolates_projects(
    capsys, monkeypatch, tmp_path: Path
) -> None:
    manager, alpha_path, _beta_path, alpha, beta = _make_v2_manager(tmp_path)
    context_root = manager.config.general.context_root
    alpha_root = context_root / "scratchpad" / "projects" / alpha.project_id
    beta_root = context_root / "scratchpad" / "projects" / beta.project_id
    alpha_root.mkdir(parents=True)
    beta_root.mkdir(parents=True)
    (alpha_root / "same.md").write_text("alpha", encoding="utf-8")
    (beta_root / "same.md").write_text("beta", encoding="utf-8")
    (beta_root / "private.md").write_text("private", encoding="utf-8")

    monkeypatch.setattr("afs.cli.fs.load_manager", lambda _config_path: manager)
    parser = _make_parser()
    base = ["--path", str(alpha_path), "--context-root", str(context_root)]

    assert _run(parser, ["files", "read", "scratchpad", "same.md", *base]) == 0
    assert capsys.readouterr().out == "alpha"

    assert (
        _run(
            parser,
            [
                "files",
                "read",
                "scratchpad",
                f"../{beta.project_id}/private.md",
                *base,
            ],
        )
        == 1
    )
    assert "escapes mount root" in capsys.readouterr().out

    assert (
        _run(
            parser,
            [
                "files",
                "write",
                "scratchpad",
                "drafts/new.md",
                "--content",
                "new content",
                "--mkdirs",
                "--json",
                *base,
            ],
        )
        == 0
    )
    write_payload = json.loads(capsys.readouterr().out)
    assert write_payload["scope_id"] == alpha.scope_id
    assert write_payload["path"] == "drafts/new.md"
    assert (alpha_root / "drafts" / "new.md").read_text() == "new content"
    index = ContextSQLiteIndex(manager, context_root)
    scratchpad_entries = index.query(
        query="new content", mount_types=[MountType.SCRATCHPAD]
    )
    assert [entry["relative_path"] for entry in scratchpad_entries] == [
        f"projects/{alpha.project_id}/drafts/new.md"
    ]

    assert (
        _run(
            parser,
            [
                "files",
                "info",
                "scratchpad",
                "drafts/new.md",
                "--json",
                *base,
            ],
        )
        == 0
    )
    info_payload = json.loads(capsys.readouterr().out)
    assert info_payload["path"] == "drafts/new.md"
    assert info_payload["relative_path"] == "drafts/new.md"
    assert info_payload["scope_id"] == alpha.scope_id
    assert alpha.project_id not in info_payload["path"]

    assert (
        _run(
            parser,
            [
                "files",
                "list",
                "scratchpad",
                "--files-only",
                "--max-depth",
                "3",
                "--json",
                *base,
            ],
        )
        == 0
    )
    list_payload = json.loads(capsys.readouterr().out)
    assert list_payload["scope_id"] == alpha.scope_id
    assert {entry["relative_path"] for entry in list_payload["entries"]} == {
        "same.md",
        "drafts/new.md",
    }
    assert all(entry["scope_id"] == alpha.scope_id for entry in list_payload["entries"])

    assert (
        _run(
            parser,
            [
                "files",
                "move",
                "scratchpad",
                "drafts/new.md",
                "knowledge",
                "moved.md",
                "--json",
                *base,
            ],
        )
        == 0
    )
    move_payload = json.loads(capsys.readouterr().out)
    destination = context_root / "knowledge" / "projects" / alpha.project_id / "moved.md"
    assert move_payload["source"] == "drafts/new.md"
    assert move_payload["destination"] == "moved.md"
    assert destination.read_text() == "new content"

    knowledge_entries = index.query(
        query="new content", mount_types=[MountType.KNOWLEDGE]
    )
    assert [entry["relative_path"] for entry in knowledge_entries] == [
        f"projects/{alpha.project_id}/moved.md"
    ]

    assert (
        _run(
            parser,
            [
                "files",
                "delete",
                "knowledge",
                "moved.md",
                "--json",
                *base,
            ],
        )
        == 0
    )
    delete_payload = json.loads(capsys.readouterr().out)
    assert delete_payload["relative_path"] == "moved.md"
    assert delete_payload["rows_deleted"] == 1
    assert not destination.exists()
    assert (beta_root / "private.md").read_text() == "private"


def test_v2_files_empty_scope_list_does_not_create_directories(
    capsys, monkeypatch, tmp_path: Path
) -> None:
    manager, alpha_path, _beta_path, alpha, _beta = _make_v2_manager(tmp_path)
    context_root = manager.config.general.context_root
    scoped_root = context_root / "human" / "projects" / alpha.project_id
    assert not scoped_root.exists()

    monkeypatch.setattr("afs.cli.fs.load_manager", lambda _config_path: manager)
    parser = _make_parser()
    status = _run(
        parser,
        [
            "files",
            "list",
            "human",
            "--path",
            str(alpha_path),
            "--context-root",
            str(context_root),
            "--json",
        ],
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["scope_id"] == alpha.scope_id
    assert payload["entries"] == []
    assert not scoped_root.exists()

    scoped_root.mkdir(parents=True)
    (scoped_root / "intent.md").write_text("human intent", encoding="utf-8")
    status = _run(
        parser,
        [
            "files",
            "read",
            "human",
            "intent.md",
            "--path",
            str(alpha_path),
            "--context-root",
            str(context_root),
            "--json",
        ],
    )
    assert status == 0
    read_payload = json.loads(capsys.readouterr().out)
    assert read_payload["content"] == "human intent"
    assert read_payload["path"] == "intent.md"


def test_v2_files_common_scope_is_explicit(
    capsys, monkeypatch, tmp_path: Path
) -> None:
    manager, _alpha_path, _beta_path, alpha, _beta = _make_v2_manager(tmp_path)
    context_root = manager.config.general.context_root
    unregistered_path = tmp_path / "unregistered"
    unregistered_path.mkdir()
    project_root = context_root / "scratchpad" / "projects" / alpha.project_id
    common_root = context_root / "scratchpad" / "common"
    project_root.mkdir(parents=True)
    common_root.mkdir(parents=True)
    (project_root / "scope.md").write_text("project", encoding="utf-8")
    (common_root / "scope.md").write_text("common", encoding="utf-8")

    monkeypatch.setattr("afs.cli.fs.load_manager", lambda _config_path: manager)
    parser = _make_parser()
    status = _run(
        parser,
        [
            "files",
            "read",
            "scratchpad",
            "scope.md",
            "--path",
            str(unregistered_path),
            "--common",
            "--json",
        ],
    )

    assert status == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["content"] == "common"
    assert payload["scope_id"] == "common"
    assert payload["path"] == "scope.md"


def test_v2_files_rejects_internal_mounts(
    capsys, monkeypatch, tmp_path: Path
) -> None:
    manager, alpha_path, _beta_path, _alpha, _beta = _make_v2_manager(tmp_path)
    context_root = manager.config.general.context_root
    monkeypatch.setattr("afs.cli.fs.load_manager", lambda _config_path: manager)
    parser = _make_parser()

    status = _run(
        parser,
        [
            "files",
            "list",
            "hivemind",
            "--path",
            str(alpha_path),
            "--context-root",
            str(context_root),
        ],
    )

    assert status == 1
    assert "internal v2 mount" in capsys.readouterr().out


def test_v1_files_preserves_legacy_mount_paths(
    capsys, monkeypatch, tmp_path: Path
) -> None:
    manager, project_path = _make_manager(tmp_path)
    context_root = manager.config.general.context_root
    monkeypatch.setattr("afs.cli.fs.load_manager", lambda _config_path: manager)
    parser = _make_parser()

    status = _run(
        parser,
        [
            "files",
            "write",
            "scratchpad",
            "legacy.md",
            "--content",
            "legacy",
            "--path",
            str(project_path),
            "--context-root",
            str(context_root),
        ],
    )

    assert status == 0
    target = context_root / "scratchpad" / "legacy.md"
    assert target.read_text() == "legacy"
    assert capsys.readouterr().out.strip() == f"wrote: {target}"
    assert not (context_root / "scratchpad" / "common").exists()
