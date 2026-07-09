from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import afs.cli.missions as missions_cli
from afs.cli.missions import (
    mission_create_command,
    mission_list_command,
    mission_show_command,
    mission_update_command,
)
from afs.manager import AFSManager
from afs.missions import MissionStore
from afs.schema import AFSConfig, GeneralConfig


def _wire(tmp_path: Path, monkeypatch):
    context_root = tmp_path / ".context"
    config = AFSConfig(general=GeneralConfig(context_root=context_root))
    manager = AFSManager(config=config)
    project_path = tmp_path / "project"
    project_path.mkdir()
    manager.ensure(path=project_path, context_root=context_root)
    monkeypatch.setattr(missions_cli, "load_manager", lambda _c: manager)
    monkeypatch.setattr(
        missions_cli,
        "resolve_context_paths",
        lambda _a, _m: (project_path, context_root, None, None),
    )
    return manager, context_root


def _args(**kwargs) -> Namespace:
    values = {
        "config": None,
        "path": None,
        "context_root": None,
        "context_dir": None,
        "json": False,
    }
    values.update(kwargs)
    return Namespace(**values)


def test_cli_create_and_list(tmp_path, monkeypatch, capsys) -> None:
    _wire(tmp_path, monkeypatch)
    rc = mission_create_command(
        _args(title="Triage 4821", summary="incident", owner="gemini", next_step=["pull logs"], tag=["incident"], json=True)
    )
    assert rc == 0
    created = json.loads(capsys.readouterr().out)
    assert created["status"] == "active"
    mid = created["mission_id"]

    assert mission_list_command(_args(status=None, limit=50)) == 0
    out = capsys.readouterr().out
    assert mid in out
    assert "Triage 4821" in out


def test_cli_update_and_show(tmp_path, monkeypatch, capsys) -> None:
    _, context_root = _wire(tmp_path, monkeypatch)
    mission = MissionStore(context_root).create(title="Ship fusion")

    rc = mission_update_command(
        _args(mission_id=mission.mission_id, status="blocked", blocker=["waiting"], note="pinged", actor="claude")
    )
    assert rc == 0
    capsys.readouterr()  # discard the update's human-readable output before show's JSON

    assert mission_show_command(_args(mission_id=mission.mission_id)) == 0
    shown = json.loads(capsys.readouterr().out)
    assert shown["status"] == "blocked"
    assert shown["blockers"] == ["waiting"]
    assert shown["log"][0]["note"] == "pinged"


def test_cli_update_invalid_status_returns_2(tmp_path, monkeypatch, capsys) -> None:
    _, context_root = _wire(tmp_path, monkeypatch)
    mission = MissionStore(context_root).create(title="X")
    assert mission_update_command(_args(mission_id=mission.mission_id, status="bogus")) == 2


def test_cli_show_missing_returns_1(tmp_path, monkeypatch, capsys) -> None:
    _wire(tmp_path, monkeypatch)
    assert mission_show_command(_args(mission_id="mission_missing")) == 1
