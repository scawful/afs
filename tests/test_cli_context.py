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

    overview_args = parser.parse_args(["context", "overview"])
    assert overview_args.command == "context"
    assert overview_args.context_command == "overview"
    assert hasattr(overview_args, "func")


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


def test_context_overview_outputs_codebase_summary(
    capsys, monkeypatch, tmp_path: Path
) -> None:
    manager, project_path, _context_path = _make_local_context_manager(tmp_path)
    (project_path / "README.md").write_text("# Demo\n", encoding="utf-8")
    (project_path / "AGENTS.md").write_text("# Agents\n", encoding="utf-8")
    (project_path / "pyproject.toml").write_text("[project]\nname = 'demo'\n", encoding="utf-8")
    (project_path / "src").mkdir()
    (project_path / "src" / "demo.py").write_text("def demo() -> int:\n    return 1\n", encoding="utf-8")
    (project_path / "tests").mkdir()
    (project_path / "tests" / "test_demo.py").write_text("def test_demo():\n    assert True\n", encoding="utf-8")
    (project_path / "docs").mkdir()
    (project_path / "docs" / "guide.md").write_text("# Guide\n", encoding="utf-8")

    monkeypatch.setattr("afs.cli.context.load_manager", lambda _config_path: manager)
    parser = _make_parser()
    args = parser.parse_args(
        [
            "context",
            "overview",
            "--path",
            str(project_path),
            "--json",
        ]
    )
    status = args.func(args)
    assert status == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["project_name"] == "project"
    assert payload["codebase"]["project_root"] == str(project_path.resolve())
    assert "pyproject.toml" in payload["codebase"]["manifests"]
    assert "src" in payload["codebase"]["source_roots"]
    assert "tests" in payload["codebase"]["test_roots"]
    assert "docs" in payload["codebase"]["docs_roots"]
    assert payload["codebase"]["language_hints"]["python"] >= 2


def test_context_overview_supports_raw_project_without_context(
    capsys, monkeypatch, tmp_path: Path
) -> None:
    manager = AFSManager(config=AFSConfig(general=GeneralConfig(context_root=tmp_path / "shared-context")))
    project_path = tmp_path / "scawfulbot"
    project_path.mkdir()
    (project_path / "config").mkdir()
    (project_path / "config" / "registry.json").write_text('{"models": []}\n', encoding="utf-8")
    (project_path / "config" / "system_prompt.md").write_text("# Prompt\n", encoding="utf-8")
    (project_path / "data").mkdir()
    (project_path / "eval").mkdir()
    (project_path / "eval" / "smoke.py").write_text("def smoke() -> bool:\n    return True\n", encoding="utf-8")
    (project_path / "models").mkdir()
    (project_path / "scripts").mkdir()
    (project_path / "scripts" / "train.py").write_text("def train() -> None:\n    pass\n", encoding="utf-8")
    (project_path / "training").mkdir()
    (project_path / "training" / "dataset.py").write_text("def load_dataset() -> list[str]:\n    return []\n", encoding="utf-8")

    monkeypatch.setattr("afs.cli.context.load_manager", lambda _config_path: manager)
    parser = _make_parser()
    args = parser.parse_args(
        [
            "context",
            "overview",
            "--path",
            str(project_path),
            "--json",
        ]
    )
    status = args.func(args)
    assert status == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["context_available"] is False
    assert payload["context_path"] is None
    assert payload["project_name"] == "scawfulbot"
    assert "config" in payload["codebase"]["workflow_roots"]
    assert "eval" in payload["codebase"]["workflow_roots"]
    assert "models" in payload["codebase"]["workflow_roots"]
    assert "training" in payload["codebase"]["workflow_roots"]
    assert "scripts" in payload["codebase"]["script_roots"]
    assert "scripts/train.py" in payload["codebase"]["sample_paths"]


def test_context_overview_prefers_requested_project_codebase_over_ancestor_context(
    capsys, monkeypatch, tmp_path: Path
) -> None:
    manager = AFSManager(config=AFSConfig(general=GeneralConfig(context_root=tmp_path / "shared-context")))
    lab_path = tmp_path / "lab"
    lab_path.mkdir()
    manager.ensure(path=lab_path)

    project_path = lab_path / "scawfulbot"
    project_path.mkdir()
    (project_path / "config").mkdir()
    (project_path / "config" / "registry.json").write_text('{"models": []}\n', encoding="utf-8")
    (project_path / "scripts").mkdir()
    (project_path / "scripts" / "train.py").write_text("def train() -> None:\n    pass\n", encoding="utf-8")
    (project_path / "training").mkdir()
    (project_path / "training" / "dataset.py").write_text("def load_dataset() -> list[str]:\n    return []\n", encoding="utf-8")

    monkeypatch.setattr("afs.cli.context.load_manager", lambda _config_path: manager)
    parser = _make_parser()
    args = parser.parse_args(
        [
            "context",
            "overview",
            "--path",
            str(project_path),
            "--json",
        ]
    )
    status = args.func(args)
    assert status == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["context_available"] is True
    assert payload["context_path"] == str(lab_path / ".context")
    assert payload["project_name"] == "scawfulbot"
    assert payload["context_project_name"] == "lab"
    assert payload["codebase"]["project_root"] == str(project_path)
    assert "training" in payload["codebase"]["workflow_roots"]
    assert "scripts/train.py" in payload["codebase"]["sample_paths"]


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
