"""Tests for the afs context/index CLI command surface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from afs.cli.context import register_parsers
from afs.manager import AFSManager
from afs.schema import AFSConfig, GeneralConfig


def _make_manager(tmp_path: Path) -> tuple[AFSManager, Path, Path]:
    context_root = tmp_path / "context"
    general = GeneralConfig(context_root=context_root)
    manager = AFSManager(config=AFSConfig(general=general))
    project_path = tmp_path / "project"
    project_path.mkdir()
    manager.ensure(path=project_path, context_root=context_root)
    return manager, project_path, context_root


def _make_local_context_manager(tmp_path: Path) -> tuple[AFSManager, Path, Path]:
    general = GeneralConfig(context_root=tmp_path / "shared-context")
    manager = AFSManager(config=AFSConfig(general=general))
    project_path = tmp_path / "project"
    project_path.mkdir()
    manager.ensure(path=project_path)
    return manager, project_path, project_path / ".context"


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    register_parsers(subparsers)
    return parser


def test_context_and_index_query_parsers_register() -> None:
    parser = _make_parser()

    context_args = parser.parse_args(["context", "query", "needle"])
    assert context_args.command == "context"
    assert context_args.context_command == "query"
    assert hasattr(context_args, "func")

    shortcut_args = parser.parse_args(["query", "needle"])
    assert shortcut_args.command == "query"
    assert hasattr(shortcut_args, "func")

    index_args = parser.parse_args(["index", "query", "needle"])
    assert index_args.command == "index"
    assert index_args.index_command == "query"
    assert hasattr(index_args, "func")

    rebuild_args = parser.parse_args(["index", "rebuild", "--mount", "scratchpad"])
    assert rebuild_args.command == "index"
    assert rebuild_args.index_command == "rebuild"
    assert hasattr(rebuild_args, "func")

    discover_args = parser.parse_args(["context", "discover", "--include-nested"])
    assert discover_args.command == "context"
    assert discover_args.context_command == "discover"
    assert discover_args.include_nested is True


def test_context_query_auto_indexes_and_returns_entries(
    capsys, monkeypatch, tmp_path: Path
) -> None:
    manager, project_path, context_root = _make_manager(tmp_path)
    note_path = context_root / "scratchpad" / "notes.md"
    note_path.write_text("context query marker", encoding="utf-8")

    monkeypatch.setattr("afs.cli.context.load_manager", lambda _config_path: manager)
    parser = _make_parser()
    args = parser.parse_args(
        [
            "context",
            "query",
            "query marker",
            "--mount",
            "scratchpad",
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
    assert payload["count"] == 1
    assert payload["entries"][0]["relative_path"] == "notes.md"
    assert "index_rebuild" in payload


def test_context_query_resolves_parent_context_for_nested_path(
    capsys, monkeypatch, tmp_path: Path
) -> None:
    manager, project_path, context_root = _make_local_context_manager(tmp_path)
    nested_path = project_path / "docs" / "dev"
    nested_path.mkdir(parents=True)
    (nested_path / "afs.toml").write_text("[project]\nname = 'docs-dev'\n", encoding="utf-8")
    note_path = context_root / "scratchpad" / "notes.md"
    note_path.write_text("parent route marker", encoding="utf-8")

    monkeypatch.setattr("afs.cli.context.load_manager", lambda _config_path: manager)
    parser = _make_parser()
    args = parser.parse_args(
        [
            "context",
            "query",
            "route marker",
            "--mount",
            "scratchpad",
            "--path",
            str(nested_path),
            "--json",
        ]
    )
    status = args.func(args)
    assert status == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["context_path"] == str(context_root)
    assert payload["count"] == 1
    assert payload["entries"][0]["relative_path"] == "notes.md"


def test_context_ensure_still_creates_nested_child_when_requested(
    capsys, monkeypatch, tmp_path: Path
) -> None:
    manager, project_path, _context_root = _make_local_context_manager(tmp_path)
    nested_path = project_path / "docs" / "dev"
    nested_path.mkdir(parents=True)

    monkeypatch.setattr("afs.cli.context.load_manager", lambda _config_path: manager)
    parser = _make_parser()
    args = parser.parse_args(
        [
            "context",
            "ensure",
            "--path",
            str(nested_path),
        ]
    )
    status = args.func(args)
    assert status == 0

    output = capsys.readouterr().out
    nested_context = nested_path / ".context"
    assert nested_context.is_dir()
    assert f"context_path: {nested_context}" in output


def test_context_query_raises_when_no_existing_context(
    monkeypatch, tmp_path: Path
) -> None:
    manager = AFSManager(config=AFSConfig(general=GeneralConfig(context_root=tmp_path / "shared")))
    project_path = tmp_path / "orphan-project"
    project_path.mkdir()

    monkeypatch.setattr("afs.cli.context.load_manager", lambda _config_path: manager)
    parser = _make_parser()
    args = parser.parse_args(
        [
            "context",
            "query",
            "missing",
            "--path",
            str(project_path),
            "--json",
        ]
    )

    try:
        args.func(args)
    except FileNotFoundError as exc:
        assert str(project_path) in str(exc)
        assert "afs context ensure" in str(exc)
    else:
        raise AssertionError("expected FileNotFoundError")


def test_index_rebuild_alias_outputs_summary(capsys, monkeypatch, tmp_path: Path) -> None:
    manager, project_path, context_root = _make_manager(tmp_path)
    note_path = context_root / "scratchpad" / "daily.md"
    note_path.write_text("rebuild marker", encoding="utf-8")

    monkeypatch.setattr("afs.cli.context.load_manager", lambda _config_path: manager)
    parser = _make_parser()
    args = parser.parse_args(
        [
            "index",
            "rebuild",
            "--mount",
            "scratchpad",
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
    assert payload["rows_written"] >= 1
    assert payload["by_mount_type"]["scratchpad"] >= 1
