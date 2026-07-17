from __future__ import annotations

import argparse
import json
from pathlib import Path

from afs.cli import build_parser
from afs.cli.friendly import (
    messages_clean_command,
    messages_list_command,
    messages_send_command,
    projects_current_command,
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

    assert {"start", "projects", "messages"}.issubset(choices)


def test_message_cleanup_requires_explicit_all_projects() -> None:
    args = argparse.Namespace(all_projects=False)

    try:
        messages_clean_command(args)
    except PermissionError as exc:
        assert "--all-projects" in str(exc)
    else:  # pragma: no cover - assertion helper without pytest dependency
        raise AssertionError("cleanup unexpectedly crossed the scope boundary")
