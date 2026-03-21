from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import afs.cli.events as events_cli
from afs.cli.events import events_tail_command
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
