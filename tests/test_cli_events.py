from __future__ import annotations

import argparse
import json
from argparse import Namespace
from pathlib import Path

import afs.cli.events as events_cli
from afs.cli.events import (
    events_analytics_command,
    events_replay_command,
    events_tail_command,
    register_parsers,
)
from afs.history import append_history_event
from afs.manager import AFSManager
from afs.schema import AFSConfig, GeneralConfig


def test_events_tail_uses_resolved_project_context(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    project_path = tmp_path / "project"
    project_path.mkdir()
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    context_root = tmp_path / "context"
    manager = AFSManager(
        config=AFSConfig(
            general=GeneralConfig(
                context_root=context_root,
                agent_workspaces_dir=context_root / "workspaces",
            )
        )
    )
    manager.ensure(path=project_path, context_root=context_root)
    history_root = context_root / "history"
    history_root.mkdir(parents=True, exist_ok=True)
    append_history_event(history_root, "session", "afs.session", op="bootstrap")

    monkeypatch.setattr(events_cli, "load_manager", lambda _config_path=None: manager)
    monkeypatch.chdir(elsewhere)

    exit_code = events_tail_command(
        Namespace(
            config=None,
            path=str(project_path),
            context_root=str(context_root),
            context_dir=None,
            limit=5,
            json=True,
        )
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert any(event["source"] == "afs.session" for event in payload)


def test_events_analytics_reports_mcp_metrics(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    project_path = tmp_path / "project"
    project_path.mkdir()
    context_root = tmp_path / "context"
    manager = AFSManager(
        config=AFSConfig(
            general=GeneralConfig(
                context_root=context_root,
                agent_workspaces_dir=context_root / "workspaces",
            )
        )
    )
    manager.ensure(path=project_path, context_root=context_root)
    history_root = context_root / "history"
    history_root.mkdir(parents=True, exist_ok=True)
    append_history_event(
        history_root,
        "mcp_tool",
        "afs.mcp",
        op="call",
        metadata={"tool_name": "context.status", "duration_ms": 25, "ok": True},
    )

    monkeypatch.setattr(events_cli, "load_manager", lambda _config_path=None: manager)

    exit_code = events_analytics_command(
        Namespace(
            config=None,
            path=str(project_path),
            context_root=str(context_root),
            context_dir=None,
            hours=24,
            type=None,
            json=True,
        )
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["mcp_tools"]["context.status"]["count"] == 1


def test_events_replay_filters_by_session_id(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    project_path = tmp_path / "project"
    project_path.mkdir()
    context_root = tmp_path / "context"
    manager = AFSManager(
        config=AFSConfig(
            general=GeneralConfig(
                context_root=context_root,
                agent_workspaces_dir=context_root / "workspaces",
            )
        )
    )
    manager.ensure(path=project_path, context_root=context_root)
    history_root = context_root / "history"
    history_root.mkdir(parents=True, exist_ok=True)
    append_history_event(
        history_root,
        "session",
        "afs.session",
        op="bootstrap",
        metadata={"session_id": "session-a"},
    )
    append_history_event(
        history_root,
        "session",
        "afs.session",
        op="bootstrap",
        metadata={"session_id": "session-b"},
    )

    monkeypatch.setattr(events_cli, "load_manager", lambda _config_path=None: manager)

    exit_code = events_replay_command(
        Namespace(
            config=None,
            path=str(project_path),
            context_root=str(context_root),
            context_dir=None,
            session_id="session-a",
            limit=50,
            include_payloads=False,
            json=True,
        )
    )

    assert exit_code == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["session_id"] == "session-a"
    assert payload["count"] == 1


def test_events_parser_accepts_context_flags_after_subcommand() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    register_parsers(subparsers)

    args = parser.parse_args(
        [
            "events",
            "analytics",
            "--path",
            "/tmp/project",
            "--hours",
            "12",
        ]
    )

    assert args.command == "events"
    assert args.events_command == "analytics"
    assert args.path == "/tmp/project"
    assert args.hours == 12
