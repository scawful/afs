from __future__ import annotations

import argparse
import json
from pathlib import Path

from afs.cli import build_parser
from afs.cli.friendly import (
    handoff_create_command,
    handoff_revise_command,
    messages_clean_command,
    messages_list_command,
    messages_send_command,
    notes_archive_command,
    notes_create_command,
    notes_draft_command,
    notes_list_command,
    notes_promote_command,
    projects_current_command,
    projects_import_command,
    projects_list_command,
)
from afs.context_layout import scaffold_v2
from afs.project_registry import ProjectRegistry


def _central(tmp_path: Path, monkeypatch) -> tuple[Path, Path, Path]:
    root = tmp_path / ".context"
    project = tmp_path / "project"
    other = tmp_path / "other"
    project.mkdir()
    other.mkdir()
    scaffold_v2(root)
    registry = ProjectRegistry(root)
    registry.register(project)
    registry.register(other)
    config = tmp_path / "afs.toml"
    config.write_text(f'[general]\ncontext_root = "{root}"\n', encoding="utf-8")
    monkeypatch.setenv("AFS_CONFIG_PATH", str(config))
    return root, project, other


def test_projects_current_and_list_use_central_registry(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    root, project, _other = _central(tmp_path, monkeypatch)
    current = argparse.Namespace(
        config=None,
        path=str(project),
        context_root=None,
        context_dir=None,
        json=True,
    )
    assert projects_current_command(current) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["context_root"] == str(root.resolve())
    assert payload["scope_id"].startswith("project:prj_")

    listing = argparse.Namespace(config=None, context_root=None, json=True)
    assert projects_list_command(listing) == 0
    assert len(json.loads(capsys.readouterr().out)) == 2


def test_projects_import_is_dry_run_until_apply(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    root, _project, _other = _central(tmp_path, monkeypatch)
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (tmp_path / "WORKSPACE.toml").write_text(
        '[[item]]\npath = "candidate"\ndescription = "new project"\n',
        encoding="utf-8",
    )

    args = argparse.Namespace(
        workspace_root=str(tmp_path),
        config=None,
        context_root=str(root),
        no_local=False,
        apply=False,
        json=True,
    )
    assert projects_import_command(args) == 0
    preview = json.loads(capsys.readouterr().out)
    assert preview["applied"] is False
    assert preview["candidates"] == [str(candidate.resolve())]
    assert ProjectRegistry(root).resolve(candidate) is None

    args.apply = True
    assert projects_import_command(args) == 0
    applied = json.loads(capsys.readouterr().out)
    assert applied["applied"] is True
    assert len(applied["registered"]) == 1
    assert ProjectRegistry(root).resolve(candidate) is not None


def test_messages_commands_enforce_current_scope(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    _root, project, other = _central(tmp_path, monkeypatch)

    def args(path: Path, **values):
        defaults = {
            "config": None,
            "path": str(path),
            "context_root": None,
            "context_dir": None,
            "all_projects": False,
            "include_legacy": False,
            "json": True,
        }
        defaults.update(values)
        return argparse.Namespace(**defaults)

    assert messages_send_command(
        args(
            project,
            from_agent="alpha",
            type="status",
            payload='{"ok": true}',
            to=None,
            topic=None,
            ttl_hours=None,
            scope=None,
        )
    ) == 0
    capsys.readouterr()

    assert messages_list_command(
        args(project, agent=None, type=None, topic=None, limit=10)
    ) == 0
    assert len(json.loads(capsys.readouterr().out)) == 1

    assert messages_list_command(
        args(other, agent=None, type=None, topic=None, limit=10)
    ) == 0
    assert json.loads(capsys.readouterr().out) == []


def test_friendly_top_level_parsers_are_discoverable() -> None:
    parser = build_parser(["start", "--help"])
    choices = next(
        action.choices
        for action in parser._actions
        if isinstance(action, argparse._SubParsersAction)
    )

    assert {"start", "projects", "notes", "handoff", "messages"}.issubset(choices)


def test_message_cleanup_requires_explicit_all_projects() -> None:
    args = argparse.Namespace(all_projects=False)

    try:
        messages_clean_command(args)
    except PermissionError as exc:
        assert "--all-projects" in str(exc)
    else:  # pragma: no cover - assertion helper without pytest dependency
        raise AssertionError("cleanup unexpectedly crossed the scope boundary")


def _artifact_args(project: Path, **values) -> argparse.Namespace:
    defaults = {
        "config": None,
        "path": str(project),
        "context_root": None,
        "context_dir": None,
        "common": False,
        "json": True,
    }
    defaults.update(values)
    return argparse.Namespace(**defaults)


def test_notes_drafts_promote_and_archive_within_current_scope(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    _root, project, other = _central(tmp_path, monkeypatch)
    content = {
        "title": "Readable planning note",
        "body": "Keep the decision boundary explicit.",
        "body_file": None,
        "task_id": None,
        "agent_name": "codex",
        "author_kind": "agent",
        "sensitivity": "internal",
    }
    assert notes_create_command(_artifact_args(project, **content)) == 0
    durable = json.loads(capsys.readouterr().out)

    assert notes_list_command(_artifact_args(other, limit=10)) == 0
    assert json.loads(capsys.readouterr().out) == []

    assert notes_draft_command(_artifact_args(project, **content)) == 0
    draft = json.loads(capsys.readouterr().out)
    draft_id = draft["metadata"]["artifact_id"]
    assert "readable-planning-note" in Path(draft["path"]).name

    assert notes_promote_command(_artifact_args(project, identifier=draft_id)) == 0
    promoted = json.loads(capsys.readouterr().out)
    assert promoted["metadata"]["provenance"]["source_artifact_id"] == draft_id
    assert promoted["path"] != durable["path"]

    assert notes_archive_command(_artifact_args(project, identifier=draft_id)) == 0
    archived = json.loads(capsys.readouterr().out)
    assert Path(archived["path"]).parent.name == "archive"


def test_handoff_create_and_revise_require_readable_titles(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    _root, project, _other = _central(tmp_path, monkeypatch)
    base = {
        "title": "Context v2 implementation",
        "agent_name": "codex",
        "accomplished": ["layout complete"],
        "blocked": [],
        "next_steps": ["review search"],
        "target_agent": "reviewer",
        "priority": "high",
    }
    assert handoff_create_command(_artifact_args(project, **base)) == 0
    first = json.loads(capsys.readouterr().out)

    revised = dict(base)
    revised["title"] = "Context v2 review follow-up"
    assert handoff_revise_command(
        _artifact_args(project, revision_id=first["revision_id"], **revised)
    ) == 0
    second = json.loads(capsys.readouterr().out)

    assert second["stream_id"] == first["stream_id"]
    assert second["supersedes"] == [first["revision_id"]]
    assert "context-v2-review-follow-up" in Path(second["artifact_path"]).name
