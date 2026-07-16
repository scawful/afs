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


def test_cli_create_with_acceptance(tmp_path, monkeypatch, capsys) -> None:
    _wire(tmp_path, monkeypatch)
    # The --acceptance flag requires a typed terminal confirmation.
    monkeypatch.setattr(missions_cli, "_TTY_READER", lambda prompt: "human")
    rc = mission_create_command(
        _args(title="Ship it", acceptance="all five steps land with tests", json=True)
    )
    assert rc == 0
    created = json.loads(capsys.readouterr().out)
    assert created["acceptance"] == "all five steps land with tests"
    assert created["acceptance_set_by"]  # OS user recorded as provenance
    assert created["acceptance_set_at"]
    assert created["acceptance_human_confirmed"] is True
    assert created["acceptance_set_via"] == "controlling_terminal"


def test_cli_create_acceptance_refused_headless(tmp_path, monkeypatch, capsys) -> None:
    """A headless agent passing --acceptance is refused: the anchor is human-authored."""
    _wire(tmp_path, monkeypatch)
    monkeypatch.setattr(missions_cli, "_TTY_READER", lambda prompt: None)
    rc = mission_create_command(
        _args(title="Ship it", acceptance="fabricated by an agent", json=True)
    )
    assert rc == 2
    captured = capsys.readouterr()
    assert "human-authored" in captured.err
    assert captured.out == ""  # refusal never contaminates stdout


def test_cli_create_without_tty_skips_prompt_and_nudges(tmp_path, monkeypatch, capsys) -> None:
    _wire(tmp_path, monkeypatch)
    # Simulate a headless caller: no prompt may block, a nudge is printed.
    monkeypatch.setattr(missions_cli, "_TTY_READER", lambda prompt: None)
    rc = mission_create_command(_args(title="Background job", acceptance=None))
    assert rc == 0
    out = capsys.readouterr().out
    assert "acceptance not set" in out
    assert "--acceptance" in out


def test_cli_create_json_never_prompts(tmp_path, monkeypatch, capsys) -> None:
    """--json output must stay machine-readable even with a terminal attached."""
    _wire(tmp_path, monkeypatch)

    def _fail_reader(prompt):
        raise AssertionError("prompted during --json output")

    monkeypatch.setattr(missions_cli, "_TTY_READER", _fail_reader)
    rc = mission_create_command(_args(title="Ship it", acceptance=None, json=True))
    assert rc == 0
    created = json.loads(capsys.readouterr().out)
    assert created["acceptance"] == ""


def test_cli_create_interactive_prompt_records_provenance(
    tmp_path, monkeypatch, capsys
) -> None:
    _wire(tmp_path, monkeypatch)
    monkeypatch.setattr(
        missions_cli, "_TTY_READER", lambda prompt: "demo runs end to end"
    )
    rc = mission_create_command(_args(title="Ship it", acceptance=None))
    assert rc == 0
    store = MissionStore(tmp_path / ".context")
    mission = store.list(limit=1)[0]
    assert mission.acceptance == "demo runs end to end"
    assert mission.acceptance_set_by
    assert mission.acceptance_human_confirmed is True


def _update_args(mid: str, **overrides):
    base = {
        "mission_id": mid,
        "status": None,
        "summary": None,
        "owner": None,
        "acceptance": None,
        "next_step": None,
        "blocker": None,
        "link_session": None,
        "link_handoff": None,
        "tag": None,
        "note": None,
        "actor": None,
        "json": True,
    }
    base.update(overrides)
    return _args(**base)


def test_cli_update_acceptance(tmp_path, monkeypatch, capsys) -> None:
    _wire(tmp_path, monkeypatch)
    monkeypatch.setattr(missions_cli, "_TTY_READER", lambda prompt: "human")
    mission_create_command(_args(title="Ship it", acceptance="v1", json=True))
    mid = json.loads(capsys.readouterr().out)["mission_id"]
    rc = mission_update_command(_update_args(mid, acceptance="v2 with docs"))
    assert rc == 0
    updated = json.loads(capsys.readouterr().out)
    assert updated["acceptance"] == "v2 with docs"
    assert updated["acceptance_set_by"]
    assert updated["acceptance_human_confirmed"] is True


def test_cli_update_acceptance_refused_headless(tmp_path, monkeypatch, capsys) -> None:
    """An agent cannot rewrite or clear the human's acceptance after the fact."""
    _wire(tmp_path, monkeypatch)
    monkeypatch.setattr(missions_cli, "_TTY_READER", lambda prompt: "human")
    mission_create_command(_args(title="Ship it", acceptance="v1", json=True))
    mid = json.loads(capsys.readouterr().out)["mission_id"]

    monkeypatch.setattr(missions_cli, "_TTY_READER", lambda prompt: None)
    for attempted in ("agent rewrite", ""):
        rc = mission_update_command(_update_args(mid, acceptance=attempted))
        assert rc == 2
    store = MissionStore(tmp_path / ".context")
    assert store.get(mid).acceptance == "v1"
