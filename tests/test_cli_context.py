"""Tests for the afs context/index CLI command surface."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from afs.cli.context import register_parsers
from afs.context_index import ContextSQLiteIndex
from afs.context_layout import scaffold_v2
from afs.manager import AFSManager
from afs.models import MountType
from afs.project_registry import ProjectRegistry
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


def _make_v2_manager(
    tmp_path: Path,
) -> tuple[AFSManager, Path, Path, str, str]:
    context_root = tmp_path / "central-context"
    alpha = tmp_path / "alpha"
    beta = tmp_path / "beta"
    alpha.mkdir()
    beta.mkdir()
    scaffold_v2(context_root)
    registry = ProjectRegistry(context_root)
    alpha_record = registry.register(alpha)
    beta_record = registry.register(beta)
    manager = AFSManager(
        config=AFSConfig(general=GeneralConfig(context_root=context_root))
    )
    return manager, alpha, context_root, alpha_record.project_id, beta_record.project_id


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


def test_v2_context_query_auto_indexes_only_current_and_common_scope(
    capsys, monkeypatch, tmp_path: Path
) -> None:
    manager, alpha, context_root, alpha_id, beta_id = _make_v2_manager(tmp_path)
    knowledge = context_root / "knowledge"
    alpha_root = knowledge / "projects" / alpha_id
    beta_root = knowledge / "projects" / beta_id
    common_root = knowledge / "common"
    for root, marker in (
        (alpha_root, "alpha-search-marker"),
        (beta_root, "beta-private-marker"),
        (common_root, "common-search-marker"),
    ):
        root.mkdir(parents=True)
        (root / "note.md").write_text(marker, encoding="utf-8")

    index = ContextSQLiteIndex(manager, context_root)
    index.rebuild(mount_types=[MountType.KNOWLEDGE], include_content=True)
    index.delete_relative_prefix(MountType.KNOWLEDGE, f"projects/{alpha_id}")
    index.delete_relative_prefix(MountType.KNOWLEDGE, "common")

    real_iterdir = Path.iterdir

    def guarded_iterdir(path: Path):
        if path == beta_root:
            raise AssertionError("beta scope was traversed")
        return real_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", guarded_iterdir)
    monkeypatch.setattr("afs.cli.context.load_manager", lambda _config_path: manager)
    parser = _make_parser()
    args = parser.parse_args(
        [
            "context",
            "query",
            "search-marker",
            "--mount",
            "knowledge",
            "--path",
            str(alpha),
            "--context-root",
            str(context_root),
            "--json",
        ]
    )

    assert args.func(args) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["scope_id"] == f"project:{alpha_id}"
    assert {entry["relative_path"] for entry in payload["entries"]} == {
        f"projects/{alpha_id}/note.md",
        "common/note.md",
    }
    assert index.count_entries(
        mount_types=[MountType.KNOWLEDGE],
        relative_prefixes=[f"projects/{beta_id}/"],
    ) >= 1


def test_v2_context_index_rebuild_requires_all_projects_for_other_scopes(
    capsys, monkeypatch, tmp_path: Path
) -> None:
    manager, alpha, context_root, alpha_id, beta_id = _make_v2_manager(tmp_path)
    for project_id, marker in (
        (alpha_id, "alpha-index"),
        (beta_id, "beta-index"),
    ):
        path = context_root / "knowledge" / "projects" / project_id / "note.md"
        path.parent.mkdir(parents=True)
        path.write_text(marker, encoding="utf-8")

    monkeypatch.setattr("afs.cli.context.load_manager", lambda _config_path: manager)
    parser = _make_parser()
    base = [
        "index",
        "rebuild",
        "--mount",
        "knowledge",
        "--path",
        str(alpha),
        "--context-root",
        str(context_root),
        "--json",
    ]
    scoped_args = parser.parse_args(base)
    assert scoped_args.func(scoped_args) == 0
    scoped_payload = json.loads(capsys.readouterr().out)
    assert scoped_payload["scope_id"] == f"project:{alpha_id}"
    index = ContextSQLiteIndex(manager, context_root)
    assert index.count_entries(
        mount_types=[MountType.KNOWLEDGE],
        relative_prefixes=[f"projects/{beta_id}/"],
    ) == 0

    all_args = parser.parse_args([*base[:-1], "--all-projects", "--json"])
    assert all_args.func(all_args) == 0
    all_payload = json.loads(capsys.readouterr().out)
    assert all_payload["scope_id"] == "all-projects"
    assert index.count_entries(
        mount_types=[MountType.KNOWLEDGE],
        relative_prefixes=[f"projects/{beta_id}/"],
    ) >= 1


def test_v2_context_freshness_does_not_traverse_other_projects(
    capsys, monkeypatch, tmp_path: Path
) -> None:
    manager, alpha, context_root, alpha_id, beta_id = _make_v2_manager(tmp_path)
    alpha_root = context_root / "knowledge" / "projects" / alpha_id
    beta_root = context_root / "knowledge" / "projects" / beta_id
    for root, marker in ((alpha_root, "alpha"), (beta_root, "beta")):
        root.mkdir(parents=True)
        (root / "note.md").write_text(marker, encoding="utf-8")
    ContextSQLiteIndex(manager, context_root).rebuild(
        mount_types=[MountType.KNOWLEDGE],
        include_content=True,
    )

    real_iterdir = Path.iterdir

    def guarded_iterdir(path: Path):
        if path == beta_root:
            raise AssertionError("beta scope was traversed")
        return real_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", guarded_iterdir)
    monkeypatch.setattr("afs.cli.context.load_manager", lambda _config_path: manager)
    parser = _make_parser()
    args = parser.parse_args(
        [
            "context",
            "freshness",
            "--mount",
            "knowledge",
            "--path",
            str(alpha),
            "--context-root",
            str(context_root),
            "--json",
        ]
    )

    assert args.func(args) == 0
    payload = json.loads(capsys.readouterr().out)
    paths = {
        entry["relative_path"] for entry in payload["files"]["knowledge"]
    }
    assert f"projects/{alpha_id}/note.md" in paths
    assert f"projects/{beta_id}/note.md" not in paths


def test_v2_central_cli_common_scope_is_explicit_and_usable(
    capsys, monkeypatch, tmp_path: Path
) -> None:
    manager, _alpha, context_root, alpha_id, _beta_id = _make_v2_manager(tmp_path)
    common_note = context_root / "knowledge" / "common" / "shared.md"
    alpha_note = context_root / "knowledge" / "projects" / alpha_id / "private.md"
    common_note.parent.mkdir(parents=True)
    alpha_note.parent.mkdir(parents=True)
    common_note.write_text("common-only-marker", encoding="utf-8")
    alpha_note.write_text("project-private-marker", encoding="utf-8")

    monkeypatch.setattr("afs.cli.context.load_manager", lambda _config_path: manager)
    parser = _make_parser()
    base = [
        "--path",
        str(context_root),
        "--context-root",
        str(context_root),
        "--common",
        "--json",
    ]
    rebuild = parser.parse_args(
        ["index", "rebuild", "--mount", "knowledge", *base]
    )
    assert rebuild.func(rebuild) == 0
    assert json.loads(capsys.readouterr().out)["scope_id"] == "common"

    query = parser.parse_args(
        ["context", "query", "marker", "--mount", "knowledge", *base]
    )
    assert query.func(query) == 0
    query_payload = json.loads(capsys.readouterr().out)
    assert {entry["relative_path"] for entry in query_payload["entries"]} == {
        "common/shared.md"
    }

    freshness = parser.parse_args(
        ["context", "freshness", "--mount", "knowledge", *base]
    )
    assert freshness.func(freshness) == 0
    freshness_payload = json.loads(capsys.readouterr().out)
    assert {
        entry["relative_path"]
        for entry in freshness_payload["files"]["knowledge"]
    } == {"common/shared.md"}


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


def test_v2_context_overview_prunes_nested_project_and_visible_context(
    capsys,
    monkeypatch,
    tmp_path: Path,
) -> None:
    alpha = tmp_path / "workspace"
    beta = alpha / "nested-beta"
    context_root = alpha / "central-context"
    beta.mkdir(parents=True)
    scaffold_v2(context_root)
    registry = ProjectRegistry(context_root)
    registry.register(alpha)
    registry.register(beta)
    (alpha / "alpha_safe.py").write_text("ALPHA_SAFE = True\n", encoding="utf-8")
    (beta / "beta_confidential_canary.py").write_text(
        "BETA_PRIVATE = True\n",
        encoding="utf-8",
    )
    manager = AFSManager(
        config=AFSConfig(general=GeneralConfig(context_root=context_root))
    )
    monkeypatch.setattr("afs.cli.context.load_manager", lambda _path: manager)
    args = _make_parser().parse_args(
        [
            "context",
            "overview",
            "--path",
            str(alpha),
            "--context-root",
            str(context_root),
            "--json",
        ]
    )

    assert args.func(args) == 0
    payload = json.loads(capsys.readouterr().out)
    rendered = json.dumps(payload["codebase"])
    assert "alpha_safe.py" in rendered
    assert "nested-beta" not in rendered
    assert "beta_confidential_canary" not in rendered
    assert "central-context" not in rendered


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
