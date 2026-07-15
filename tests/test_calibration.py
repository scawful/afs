"""Tests for the calibration trail (decision resurfacing + outcome scoring)."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from afs.calibration import (
    collect_decisions,
    load_outcomes,
    load_predictions,
    record_outcome,
    record_prediction,
)
from afs.missions import MissionStore
from afs.work_assistant import WorkAssistantStore


def _context(tmp_path: Path) -> Path:
    context_root = tmp_path / ".context"
    context_root.mkdir()
    return context_root


# ---------------------------------------------------------------------------
# trail primitives
# ---------------------------------------------------------------------------


def test_prediction_and_outcome_round_trip(tmp_path: Path) -> None:
    context = _context(tmp_path)
    entry = record_prediction(
        context,
        kind="bootstrap_top_priority",
        predicted="fix the reactor",
        actual="fix the reactor",
        match=True,
        session_id="sess-1",
    )
    assert entry["id"].startswith("pred_")

    loaded = load_predictions(context)
    assert len(loaded) == 1
    assert loaded[0]["match"] is True

    score = record_outcome(context, ref=entry["id"], outcome="hit", note="good call")
    assert score["kind"] == "prediction"
    assert load_outcomes(context)[0]["ref"] == entry["id"]


def test_outcome_validation(tmp_path: Path) -> None:
    context = _context(tmp_path)
    with pytest.raises(ValueError):
        record_outcome(context, ref="mission_x", outcome="great")
    with pytest.raises(ValueError):
        record_outcome(context, ref="   ", outcome="hit")


def test_ref_kind_inference(tmp_path: Path) -> None:
    context = _context(tmp_path)
    assert record_outcome(context, ref="mission_abc", outcome="hit")["kind"] == "mission"
    assert record_outcome(context, ref="pred_abc", outcome="miss")["kind"] == "prediction"
    assert record_outcome(context, ref="apr_abc", outcome="unclear")["kind"] == "approval"


# ---------------------------------------------------------------------------
# collect_decisions
# ---------------------------------------------------------------------------


def _isolate_gate(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AFS_AGENT_APPROVALS_PATH", str(tmp_path / "gate.json"))


def test_collect_decisions_surfaces_rationales_and_acceptance(
    tmp_path: Path, monkeypatch
) -> None:
    _isolate_gate(monkeypatch, tmp_path)
    context = _context(tmp_path)

    store = WorkAssistantStore(context)
    approval_id = store.create_approval(
        target_system="local",
        target_id="note",
        action="internal_note",
        summary="Record a note",
    )
    store.approve(approval_id, approved_by="human", rationale="note content verified")

    missions = MissionStore(context)
    mission = missions.create(title="Ship it", acceptance="lands with tests")
    missions.update(mission.mission_id, status="done")

    record_prediction(
        context, kind="bootstrap_top_priority", predicted="a", actual="b", match=False
    )

    report = collect_decisions(context, days=7)
    assert report["window_days"] == 7

    approvals = {entry["ref"]: entry for entry in report["approvals"]}
    assert approvals[approval_id]["rationale"] == "note content verified"
    assert approvals[approval_id]["status"] == "approved"

    assert len(report["missions"]) == 1
    assert report["missions"][0]["ref"] == mission.mission_id
    assert report["missions"][0]["acceptance"] == "lands with tests"

    assert len(report["predictions"]) == 1
    assert report["predictions"][0]["match"] is False

    # Scoring shows up in the scored map on the next collect.
    record_outcome(context, ref=mission.mission_id, outcome="hit")
    rescored = collect_decisions(context, days=7)
    assert rescored["scored"][mission.mission_id]["outcome"] == "hit"


def test_collect_decisions_ignores_open_and_out_of_window(
    tmp_path: Path, monkeypatch
) -> None:
    _isolate_gate(monkeypatch, tmp_path)
    context = _context(tmp_path)

    missions = MissionStore(context)
    missions.create(title="Still open", acceptance="never closes")

    store = WorkAssistantStore(context)
    store.create_approval(
        target_system="local",
        target_id="note",
        action="internal_note",
        summary="Still pending",
    )

    report = collect_decisions(context, days=7)
    assert report["approvals"] == []
    assert report["missions"] == []


def test_collect_decisions_includes_gate_decisions(tmp_path: Path, monkeypatch) -> None:
    _isolate_gate(monkeypatch, tmp_path)
    context = _context(tmp_path)

    from datetime import datetime, timezone

    from afs.agents.guardrails import ApprovalGate, ApprovalRequest

    gate = ApprovalGate()
    gate._pending.append(
        ApprovalRequest(
            agent="scout",
            action="git_push",
            detail="push to origin",
            timestamp=datetime.now(timezone.utc).isoformat(),
        )
    )
    gate._save()
    gate.approve("scout", "git_push", rationale="diff reviewed")

    report = collect_decisions(context, days=7)
    gate_entries = [e for e in report["approvals"] if e["source"] == "gate"]
    assert len(gate_entries) == 1
    assert gate_entries[0]["rationale"] == "diff reviewed"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _cli_args(context_root: Path, **kwargs) -> Namespace:
    values = {
        "config": None,
        "path": None,
        "context_root": str(context_root),
        "context_dir": None,
        "json": False,
    }
    values.update(kwargs)
    return Namespace(**values)


def _wire_cli(tmp_path: Path, monkeypatch) -> Path:
    import afs.cli.calibration as calibration_cli
    from afs.manager import AFSManager
    from afs.schema import AFSConfig, GeneralConfig

    context_root = tmp_path / ".context"
    config = AFSConfig(general=GeneralConfig(context_root=context_root))
    manager = AFSManager(config=config)
    project_path = tmp_path / "project"
    project_path.mkdir()
    manager.ensure(path=project_path, context_root=context_root)
    monkeypatch.setattr(calibration_cli, "load_manager", lambda _c: manager)
    monkeypatch.setattr(
        calibration_cli,
        "resolve_context_paths",
        lambda _a, _m: (project_path, context_root, None, None),
    )
    return context_root


def test_cli_review_and_score(tmp_path: Path, monkeypatch, capsys) -> None:
    _isolate_gate(monkeypatch, tmp_path)
    from afs.cli.calibration import calibration_review_command, calibration_score_command

    context_root = _wire_cli(tmp_path, monkeypatch)

    missions = MissionStore(context_root)
    mission = missions.create(title="Ship it", acceptance="lands with tests")
    missions.update(mission.mission_id, status="done")

    assert calibration_review_command(_cli_args(context_root, days=7, markdown=False)) == 0
    out = capsys.readouterr().out
    assert "Ship it" in out
    assert "acceptance was: lands with tests" in out
    assert f"afs calibration score {mission.mission_id}" in out

    assert (
        calibration_score_command(
            _cli_args(context_root, ref=mission.mission_id, outcome="hit", note="shipped")
        )
        == 0
    )
    assert "scored mission" in capsys.readouterr().out

    # After scoring, the review reflects the recorded outcome.
    assert calibration_review_command(_cli_args(context_root, days=7, markdown=False)) == 0
    assert "scored: hit" in capsys.readouterr().out


def test_cli_review_markdown_digest(tmp_path: Path, monkeypatch, capsys) -> None:
    _isolate_gate(monkeypatch, tmp_path)
    from afs.cli.calibration import calibration_review_command

    context_root = _wire_cli(tmp_path, monkeypatch)
    missions = MissionStore(context_root)
    missions.update(
        missions.create(title="Ship it", acceptance="lands with tests").mission_id,
        status="done",
    )
    record_prediction(
        context_root, kind="bootstrap_top_priority", predicted="a", actual="a", match=True
    )

    assert calibration_review_command(_cli_args(context_root, days=7, markdown=True)) == 0
    out = capsys.readouterr().out
    assert out.startswith("## Calibration")
    assert "### Closed missions" in out
    assert "1/1" in out


def test_cli_score_rejects_bad_outcome(tmp_path: Path, monkeypatch, capsys) -> None:
    from afs.cli.calibration import calibration_score_command

    context_root = _wire_cli(tmp_path, monkeypatch)
    args = _cli_args(context_root, ref="mission_x", outcome="great", note=None)
    assert calibration_score_command(args) == 2


# ---------------------------------------------------------------------------
# bootstrap --engage helpers
# ---------------------------------------------------------------------------


def test_top_priority_item_precedence() -> None:
    from afs.session_bootstrap import top_priority_item

    summary = {
        "handoff": {"next_steps": ["  finish the reactor  "]},
        "missions": {"active": [{"title": "Mission title"}]},
        "tasks": {"items": [{"title": "Task title"}]},
    }
    assert top_priority_item(summary) == "finish the reactor"
    summary["handoff"] = {"next_steps": []}
    assert top_priority_item(summary) == "Mission title"
    summary["missions"] = {"active": []}
    assert top_priority_item(summary) == "Task title"
    summary["tasks"] = {}
    assert top_priority_item(summary) == ""


def test_run_engage_prediction_records_trail(tmp_path: Path, capsys) -> None:
    from afs.session_bootstrap import run_engage_prediction

    context = _context(tmp_path)
    summary = {"handoff": {"next_steps": ["Fix the reactor cursor race"]}}
    entry = run_engage_prediction(
        context,
        summary,
        input_fn=lambda prompt: "fix the reactor",
        interactive=True,
    )
    assert entry is not None
    assert entry["match"] is True
    assert "matched" in capsys.readouterr().out

    trail = load_predictions(context)
    assert len(trail) == 1
    assert trail[0]["kind"] == "bootstrap_top_priority"


def test_run_engage_prediction_skips_when_headless_or_empty(
    tmp_path: Path, capsys
) -> None:
    from afs.session_bootstrap import run_engage_prediction

    context = _context(tmp_path)
    assert run_engage_prediction(context, {}, interactive=False) is None
    assert "requires an interactive terminal" in capsys.readouterr().out

    assert (
        run_engage_prediction(
            context, {}, input_fn=lambda prompt: "   ", interactive=True
        )
        is None
    )
    assert load_predictions(context) == []
