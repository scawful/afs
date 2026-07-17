"""Tests for the calibration trail (decision resurfacing + outcome scoring)."""

from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

import pytest

from afs.calibration import (
    UnknownDecisionRefError,
    calibration_root,
    collect_decisions,
    human_outcome_scope,
    human_prediction_scope,
    load_outcomes,
    load_predictions,
    record_human_outcome,
    record_human_prediction,
    record_outcome,
    record_prediction,
)
from afs.context_layout import scaffold_v2
from afs.missions import MissionStore
from afs.work_assistant import WorkAssistantStore


def _human_authorization(scope: str):
    from afs.human_provenance import _broker_for_reader

    authorization = _broker_for_reader(lambda _prompt: "confirm").confirm_token(
        "confirm",
        "prompt",
        scope=scope,
    )
    assert authorization is not None
    return authorization


@pytest.fixture(autouse=True)
def _isolated_gate_outcomes(monkeypatch, tmp_path: Path):
    """Every test gets an isolated global gate-outcome trail.

    load_outcomes merges the global trail, so without this a test could read
    (or a gate-ref score could write) the developer's real state file.
    """
    monkeypatch.setenv(
        "AFS_AGENT_APPROVAL_OUTCOMES_PATH", str(tmp_path / "gate_outcomes.jsonl")
    )


def _context(tmp_path: Path) -> Path:
    context_root = tmp_path / ".context"
    context_root.mkdir(parents=True)
    return context_root


def _symlink_or_skip(link: Path, target: Path, *, directory: bool = False) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    try:
        link.symlink_to(target, target_is_directory=directory)
    except OSError as exc:  # pragma: no cover - Windows without symlink privilege
        pytest.skip(f"symlinks unavailable: {exc}")


# ---------------------------------------------------------------------------
# trail primitives
# ---------------------------------------------------------------------------


def test_v1_calibration_preserves_external_remapped_scratchpad(tmp_path: Path) -> None:
    context = _context(tmp_path)
    external = tmp_path / "external-scratchpad"
    external.mkdir()
    (context / "metadata.json").write_text(
        json.dumps({"directories": {"scratchpad": str(external)}}),
        encoding="utf-8",
    )

    created = record_prediction(
        context,
        kind="bootstrap_top_priority",
        predicted="keep v1 remap",
        actual="keep v1 remap",
        match=True,
    )

    expected = external / "calibration" / "predictions.jsonl"
    assert calibration_root(context) == external / "calibration"
    assert expected.is_file()
    assert load_predictions(context)[0]["id"] == created["id"]


def test_v2_calibration_reads_pre_fix_trail_and_writes_only_common(
    tmp_path: Path,
) -> None:
    context = tmp_path / ".context"
    scaffold_v2(context)
    canonical = context / "scratchpad" / "common" / "calibration"
    legacy = context / "scratchpad" / "calibration"

    assert calibration_root(context) == canonical
    assert load_predictions(context) == []
    assert not canonical.exists()
    assert not legacy.exists()

    legacy.mkdir(parents=True)
    old_entry = {
        "id": "pred_pre_fix",
        "timestamp": "2026-07-16T00:00:00+00:00",
        "kind": "bootstrap_top_priority",
        "predicted": "old path",
        "actual": "old path",
        "match": True,
    }
    legacy_path = legacy / "predictions.jsonl"
    legacy_path.write_text(json.dumps(old_entry) + "\n", encoding="utf-8")
    old_outcome = {
        "ref": "mission_pre_fix",
        "kind": "mission",
        "outcome": "hit",
        "timestamp": "2026-07-16T00:00:00+00:00",
    }
    legacy_outcomes = legacy / "outcomes.jsonl"
    legacy_outcomes.write_text(json.dumps(old_outcome) + "\n", encoding="utf-8")

    assert [entry["id"] for entry in load_predictions(context)] == ["pred_pre_fix"]
    assert [entry["ref"] for entry in load_outcomes(context)] == ["mission_pre_fix"]
    assert not (legacy / "predictions.jsonl.lock").exists()
    before = legacy_path.read_bytes()
    outcomes_before = legacy_outcomes.read_bytes()

    created = record_prediction(
        context,
        kind="bootstrap_top_priority",
        predicted="new common path",
        actual="new common path",
        match=True,
    )
    score = record_outcome(context, ref="pred_pre_fix", outcome="hit")

    assert (canonical / "predictions.jsonl").is_file()
    assert (canonical / "outcomes.jsonl").is_file()
    assert legacy_path.read_bytes() == before
    assert legacy_outcomes.read_bytes() == outcomes_before
    assert not (legacy / "predictions.jsonl.lock").exists()
    assert not (legacy / "outcomes.jsonl.lock").exists()
    assert [entry["id"] for entry in load_predictions(context)] == [
        "pred_pre_fix",
        created["id"],
    ]
    assert [entry["ref"] for entry in load_outcomes(context)] == [
        "mission_pre_fix",
        score["ref"],
    ]


def test_v2_calibration_deduplicates_copy_migrated_records(tmp_path: Path) -> None:
    context = tmp_path / ".context"
    scaffold_v2(context)
    legacy = context / "scratchpad" / "calibration"
    canonical = context / "scratchpad" / "common" / "calibration"
    legacy.mkdir(parents=True)
    canonical.mkdir(parents=True)
    prediction = {
        "id": "pred_copied",
        "timestamp": "2026-07-16T00:00:00+00:00",
        "kind": "bootstrap_top_priority",
        "predicted": "copied",
        "actual": "copied",
        "match": True,
    }
    outcome = {
        "ref": "mission_copied",
        "kind": "mission",
        "outcome": "hit",
        "scored_by": "reviewer",
        "timestamp": "2026-07-16T01:00:00+00:00",
    }
    for root in (legacy, canonical):
        (root / "predictions.jsonl").write_text(
            json.dumps(prediction) + "\n",
            encoding="utf-8",
        )
        (root / "outcomes.jsonl").write_text(
            json.dumps(outcome) + "\n",
            encoding="utf-8",
        )

    assert [entry["id"] for entry in load_predictions(context)] == ["pred_copied"]
    assert [entry["ref"] for entry in load_outcomes(context)] == ["mission_copied"]


@pytest.mark.parametrize("root_kind", ["canonical", "legacy"])
def test_v2_calibration_rejects_linked_roots(
    tmp_path: Path,
    root_kind: str,
) -> None:
    context = tmp_path / ".context"
    scaffold_v2(context)
    outside = tmp_path / f"outside-{root_kind}"
    outside.mkdir()
    outside_file = outside / "predictions.jsonl"
    outside_file.write_text(
        json.dumps({"id": "pred_outside", "timestamp": "2026-07-17T00:00:00+00:00"})
        + "\n",
        encoding="utf-8",
    )
    root = (
        context / "scratchpad" / "common" / "calibration"
        if root_kind == "canonical"
        else context / "scratchpad" / "calibration"
    )
    _symlink_or_skip(root, outside, directory=True)
    before = outside_file.read_bytes()

    with pytest.raises(ValueError, match="symbolic link or reparse point"):
        load_predictions(context)
    if root_kind == "canonical":
        with pytest.raises(ValueError, match="symbolic link or reparse point"):
            record_prediction(
                context,
                kind="bootstrap_top_priority",
                predicted="must not escape",
                actual="must not escape",
            )
    else:
        record_prediction(
            context,
            kind="bootstrap_top_priority",
            predicted="safe canonical write",
            actual="safe canonical write",
        )
    assert outside_file.read_bytes() == before


@pytest.mark.parametrize("leaf_kind", ["jsonl", "lock"])
@pytest.mark.parametrize("root_kind", ["canonical", "legacy"])
def test_v2_calibration_rejects_linked_leaves(
    tmp_path: Path,
    leaf_kind: str,
    root_kind: str,
) -> None:
    context = tmp_path / ".context"
    scaffold_v2(context)
    root = (
        context / "scratchpad" / "common" / "calibration"
        if root_kind == "canonical"
        else context / "scratchpad" / "calibration"
    )
    root.mkdir(parents=True)
    jsonl = root / "predictions.jsonl"
    outside = tmp_path / f"outside-{leaf_kind}"
    outside.write_text(
        json.dumps({"id": "pred_outside", "timestamp": "2026-07-17T00:00:00+00:00"})
        + "\n",
        encoding="utf-8",
    )
    if leaf_kind == "jsonl":
        _symlink_or_skip(jsonl, outside)
    else:
        jsonl.write_text(
            json.dumps({"id": "pred_local", "timestamp": "2026-07-17T00:00:00+00:00"})
            + "\n",
            encoding="utf-8",
        )
        _symlink_or_skip(jsonl.with_suffix(".jsonl.lock"), outside)
    before = outside.read_bytes()

    with pytest.raises(ValueError, match="symbolic link or reparse point"):
        load_predictions(context)
    if root_kind == "canonical":
        with pytest.raises(ValueError, match="symbolic link or reparse point"):
            record_prediction(
                context,
                kind="bootstrap_top_priority",
                predicted="must not follow leaf",
                actual="must not follow leaf",
            )
    else:
        record_prediction(
            context,
            kind="bootstrap_top_priority",
            predicted="safe canonical write",
            actual="safe canonical write",
        )
    assert outside.read_bytes() == before


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
    assert entry["human_confirmed"] is False
    assert entry["predicted_via"] == "programmatic"

    loaded = load_predictions(context)
    assert len(loaded) == 1
    assert loaded[0]["match"] is True
    assert collect_decisions(context, days=7)["predictions"] == []

    score = record_outcome(context, ref=entry["id"], outcome="hit", note="good call")
    assert score["kind"] == "prediction"
    assert load_outcomes(context)[0]["ref"] == entry["id"]


def test_prediction_append_repairs_torn_jsonl_tail(tmp_path: Path) -> None:
    context = _context(tmp_path)
    path = calibration_root(context) / "predictions.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = {
        "id": "pred_existing",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "kind": "bootstrap_top_priority",
    }
    path.write_bytes(
        (json.dumps(existing) + "\n" + '{"id":"pred_torn"').encode("utf-8")
    )

    created = record_prediction(
        context,
        kind="bootstrap_top_priority",
        predicted="repair the trail",
        actual="repair the trail",
        match=True,
    )

    assert [entry["id"] for entry in load_predictions(context)] == [
        "pred_existing",
        created["id"],
    ]
    assert path.read_bytes().endswith(b"\n")


def test_prediction_append_preserves_complete_tail_without_newline(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    path = calibration_root(context) / "predictions.jsonl"
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = {
        "id": "pred_existing",
        "timestamp": "2026-01-01T00:00:00+00:00",
        "kind": "bootstrap_top_priority",
    }
    path.write_text(json.dumps(existing), encoding="utf-8")

    created = record_prediction(
        context,
        kind="bootstrap_top_priority",
        predicted="keep the complete record",
        actual="keep the complete record",
        match=True,
    )

    assert [entry["id"] for entry in load_predictions(context)] == [
        "pred_existing",
        created["id"],
    ]


def test_outcome_validation(tmp_path: Path) -> None:
    context = _context(tmp_path)
    with pytest.raises(ValueError):
        record_outcome(context, ref="mission_x", outcome="great")
    with pytest.raises(ValueError):
        record_outcome(context, ref="   ", outcome="hit")


def test_ref_kind_inference(tmp_path: Path) -> None:
    context = _context(tmp_path)
    # force=True: these synthetic refs exist in no store; only the kind
    # inference is under test here.
    assert (
        record_outcome(context, ref="mission_abc", outcome="hit", force=True)["kind"]
        == "mission"
    )
    assert (
        record_outcome(context, ref="pred_abc", outcome="miss", force=True)["kind"]
        == "prediction"
    )
    assert (
        record_outcome(context, ref="apr_abc", outcome="unclear", force=True)["kind"]
        == "approval"
    )


def test_unknown_ref_is_rejected(tmp_path: Path, monkeypatch) -> None:
    """A typo'd or fabricated ref must not silently poison the trail."""
    monkeypatch.setenv("AFS_AGENT_APPROVALS_PATH", str(tmp_path / "gate.json"))
    context = _context(tmp_path)
    with pytest.raises(UnknownDecisionRefError):
        record_outcome(context, ref="mission_nonexistent", outcome="hit")
    with pytest.raises(UnknownDecisionRefError):
        record_outcome(context, ref="pred_nonexistent", outcome="hit")
    with pytest.raises(UnknownDecisionRefError):
        record_outcome(context, ref="gate_nonexistent", outcome="hit")
    assert load_outcomes(context) == []


def test_known_refs_are_accepted(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("AFS_AGENT_APPROVALS_PATH", str(tmp_path / "gate.json"))
    context = _context(tmp_path)
    mission = MissionStore(context).create(title="Real mission")
    prediction = record_prediction(
        context, kind="bootstrap_top_priority", predicted="a", actual="b", match=False
    )
    assert (
        record_outcome(context, ref=mission.mission_id, outcome="hit")["ref"]
        == mission.mission_id
    )
    assert (
        record_outcome(context, ref=prediction["id"], outcome="miss")["ref"]
        == prediction["id"]
    )


def test_programmatic_outcome_cannot_forge_human_score(tmp_path: Path) -> None:
    context = _context(tmp_path)
    mission = MissionStore(context).create(title="Real mission")
    MissionStore(context).update(mission.mission_id, status="done")

    entry = record_outcome(
        context,
        ref=mission.mission_id,
        outcome="hit",
        scored_by="human",
        scored_via="controlling_terminal",
    )
    assert entry["human_confirmed"] is False
    assert entry["scored_by"] == "unauthenticated"
    assert collect_decisions(context, days=7)["scored"] == {}


def test_human_prediction_capability_binds_content_and_is_single_use(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    scope = human_prediction_scope(
        context,
        kind="bootstrap_top_priority",
        predicted="repair archive",
        actual="repair archive",
        match=True,
    )
    authorization = _human_authorization(scope)

    with pytest.raises(ValueError, match="authorization"):
        record_human_prediction(
            context,
            kind="bootstrap_top_priority",
            predicted="different claim",
            actual="repair archive",
            match=True,
            authorization=authorization,
        )
    record_human_prediction(
        context,
        kind="bootstrap_top_priority",
        predicted="repair archive",
        actual="repair archive",
        match=True,
        authorization=authorization,
    )
    with pytest.raises(ValueError, match="authorization"):
        record_human_prediction(
            context,
            kind="bootstrap_top_priority",
            predicted="repair archive",
            actual="repair archive",
            match=True,
            authorization=authorization,
        )


def test_human_outcome_capability_binds_note_and_is_single_use(
    tmp_path: Path,
) -> None:
    context = _context(tmp_path)
    missions = MissionStore(context)
    mission = missions.create(title="Real mission")
    missions.update(mission.mission_id, status="done")
    scope = human_outcome_scope(
        context,
        ref=mission.mission_id,
        outcome="hit",
        note="verified in production",
    )
    authorization = _human_authorization(scope)

    with pytest.raises(ValueError, match="authorization"):
        record_human_outcome(
            context,
            ref=mission.mission_id,
            outcome="hit",
            note="caller changed the note",
            authorization=authorization,
        )
    record_human_outcome(
        context,
        ref=mission.mission_id,
        outcome="hit",
        note="verified in production",
        authorization=authorization,
    )
    with pytest.raises(ValueError, match="authorization"):
        record_human_outcome(
            context,
            ref=mission.mission_id,
            outcome="hit",
            note="verified in production",
            authorization=authorization,
        )


# ---------------------------------------------------------------------------
# collect_decisions
# ---------------------------------------------------------------------------


def _isolate_gate(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("AFS_AGENT_APPROVALS_PATH", str(tmp_path / "gate.json"))
    monkeypatch.setenv(
        "AFS_AGENT_APPROVAL_OUTCOMES_PATH", str(tmp_path / "gate_outcomes.jsonl")
    )


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
    store.approve_human(
        approval_id,
        rationale="note content verified",
        authorization=_human_authorization(
            store.human_authorization_scope(
                "approve", approval_id, "note content verified"
            )
        ),
    )

    missions = MissionStore(context)
    mission = missions.create(
        title="Ship it",
        acceptance="lands with tests",
        acceptance_authorization=_human_authorization(
            missions.human_acceptance_scope(
                "create", "Ship it", "lands with tests"
            )
        ),
    )
    missions.update(mission.mission_id, status="done")

    prediction_scope = human_prediction_scope(
        context,
        kind="bootstrap_top_priority",
        predicted="a",
        actual="b",
        match=False,
    )
    record_human_prediction(
        context,
        kind="bootstrap_top_priority",
        predicted="a",
        actual="b",
        match=False,
        authorization=_human_authorization(prediction_scope),
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
    record_human_outcome(
        context,
        ref=mission.mission_id,
        outcome="hit",
        authorization=_human_authorization(
            human_outcome_scope(
                context, ref=mission.mission_id, outcome="hit"
            )
        ),
    )
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

    from afs.agents.guardrails import ApprovalGate

    gate = ApprovalGate()
    gate._queue("scout", "git_push", "push to origin")
    request = gate.find_pending("scout", "git_push")
    assert request is not None
    gate.approve_human(
        "scout",
        "git_push",
        rationale="diff reviewed",
        authorization=_human_authorization(
            gate.human_authorization_scope(
                "approve", request.request_id, "diff reviewed"
            )
        ),
    )

    report = collect_decisions(context, days=7)
    gate_entries = [e for e in report["approvals"] if e["source"] == "gate"]
    assert len(gate_entries) == 1
    assert gate_entries[0]["rationale"] == "diff reviewed"
    # Refs are per-request ids, never the agent:action pair (which repeats
    # across requests and contexts), and the global store path is visible.
    assert gate_entries[0]["ref"].startswith("gate_")
    assert gate_entries[0]["store"]


def test_gate_refs_are_unique_per_request(tmp_path: Path, monkeypatch) -> None:
    _isolate_gate(monkeypatch, tmp_path)
    context = _context(tmp_path)

    from afs.agents.guardrails import ApprovalGate

    gate = ApprovalGate()
    gate._queue("scout", "git_push", "push to origin")
    gate._queue("scout", "mass_delete", "clean old artifacts")
    approve_request = gate.find_pending("scout", "git_push")
    reject_request = gate.find_pending("scout", "mass_delete")
    assert approve_request is not None and reject_request is not None
    gate.approve_human(
        "scout",
        "git_push",
        rationale="diff reviewed",
        authorization=_human_authorization(
            gate.human_authorization_scope(
                "approve", approve_request.request_id, "diff reviewed"
            )
        ),
    )
    gate.reject_human(
        "scout",
        "mass_delete",
        rationale="too broad",
        authorization=_human_authorization(
            gate.human_authorization_scope(
                "reject", reject_request.request_id, "too broad"
            )
        ),
    )

    report = collect_decisions(context, days=7)
    refs = [e["ref"] for e in report["approvals"] if e["source"] == "gate"]
    assert len(refs) == 2
    assert len(set(refs)) == 2
    assert all(ref.startswith("gate_") for ref in refs)


def test_cleared_gate_decision_remains_in_calibration_history(
    tmp_path: Path, monkeypatch
) -> None:
    _isolate_gate(monkeypatch, tmp_path)
    context = _context(tmp_path)

    from afs.agents.guardrails import ApprovalGate

    gate = ApprovalGate()
    gate._queue("scout", "git_push", "push to origin")
    request = gate.find_pending("scout", "git_push")
    assert request is not None
    gate.approve_human(
        "scout",
        "git_push",
        rationale="diff reviewed",
        authorization=_human_authorization(
            gate.human_authorization_scope(
                "approve", request.request_id, "diff reviewed"
            )
        ),
    )
    ref = gate.all_requests()[0].request_id
    removed, remaining = gate.clear_completed()
    assert (removed, remaining) == (1, 0)

    report = collect_decisions(context, days=7)
    assert ref in {entry["ref"] for entry in report["approvals"]}


def test_collect_decisions_includes_applied_work_approvals(
    tmp_path: Path, monkeypatch
) -> None:
    """Approved-then-executed approvals are decisions worth scoring."""
    _isolate_gate(monkeypatch, tmp_path)
    context = _context(tmp_path)

    store = WorkAssistantStore(context)
    approval_id = store.create_approval(
        target_system="local",
        target_id="note",
        action="internal_note",
        summary="Record a note",
    )
    store.approve_human(
        approval_id,
        rationale="content verified",
        authorization=_human_authorization(
            store.human_authorization_scope(
                "approve", approval_id, "content verified"
            )
        ),
    )
    store.record_approval_result(
        approval_id, result={"ok": True}, status="applied"
    )

    report = collect_decisions(context, days=7)
    work_entries = {e["ref"]: e for e in report["approvals"] if e["source"] == "work"}
    assert approval_id in work_entries
    assert work_entries[approval_id]["status"] == "applied"


def test_work_decisions_pass_config_through(tmp_path: Path, monkeypatch) -> None:
    """Custom mount layouts must reach the work store, not just defaults."""
    _isolate_gate(monkeypatch, tmp_path)
    context = _context(tmp_path)
    captured: dict = {}

    import afs.work_assistant as work_assistant_module

    real_store = work_assistant_module.WorkAssistantStore

    def _capturing_store(context_root, **kwargs):
        captured.update(kwargs)
        return real_store(context_root, **kwargs)

    monkeypatch.setattr(work_assistant_module, "WorkAssistantStore", _capturing_store)
    from afs.schema import AFSConfig

    config = AFSConfig()
    collect_decisions(context, days=7, config=config)
    assert captured.get("config") is config


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
    import afs.cli.calibration as calibration_cli
    from afs.cli.calibration import calibration_review_command, calibration_score_command

    context_root = _wire_cli(tmp_path, monkeypatch)

    missions = MissionStore(context_root)
    mission = missions.create(
        title="Ship it",
        acceptance="lands with tests",
        acceptance_authorization=_human_authorization(
            missions.human_acceptance_scope(
                "create", "Ship it", "lands with tests"
            )
        ),
    )
    missions.update(mission.mission_id, status="done")

    assert calibration_review_command(_cli_args(context_root, days=7, markdown=False)) == 0
    out = capsys.readouterr().out
    assert "Ship it" in out
    assert "acceptance was: lands with tests" in out
    assert f"afs calibration score {mission.mission_id}" in out

    # Scoring is a human judgment: it requires the tty confirmation.
    monkeypatch.setattr(calibration_cli, "_TTY_READER", lambda prompt: "hit")
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
        missions.create(
            title="Ship it",
            acceptance="lands with tests",
            acceptance_authorization=_human_authorization(
                missions.human_acceptance_scope(
                    "create", "Ship it", "lands with tests"
                )
            ),
        ).mission_id,
        status="done",
    )
    record_human_prediction(
        context_root,
        kind="bootstrap_top_priority",
        predicted="a",
        actual="a",
        match=True,
        authorization=_human_authorization(
            human_prediction_scope(
                context_root,
                kind="bootstrap_top_priority",
                predicted="a",
                actual="a",
                match=True,
            )
        ),
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


def test_run_engage_prediction_records_trail(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    import afs.session_bootstrap as session_bootstrap
    from afs.session_bootstrap import run_engage_prediction

    context = _context(tmp_path)
    summary = {"handoff": {"next_steps": ["Fix the reactor cursor race"]}}
    monkeypatch.setattr(
        session_bootstrap, "_ENGAGE_READER", lambda prompt: "fix the reactor"
    )
    entry = run_engage_prediction(context, summary)
    assert entry is not None
    assert entry["match"] is True
    assert "matched" in capsys.readouterr().out

    trail = load_predictions(context)
    assert len(trail) == 1
    assert trail[0]["kind"] == "bootstrap_top_priority"
    assert trail[0]["human_confirmed"] is True


def test_run_engage_prediction_skips_when_headless_or_empty(
    tmp_path: Path, monkeypatch, capsys
) -> None:
    import afs.session_bootstrap as session_bootstrap
    from afs.session_bootstrap import run_engage_prediction

    context = _context(tmp_path)
    monkeypatch.setattr(session_bootstrap, "_ENGAGE_READER", lambda prompt: None)
    assert run_engage_prediction(context, {}) is None
    assert "requires an interactive controlling terminal" in capsys.readouterr().out

    monkeypatch.setattr(session_bootstrap, "_ENGAGE_READER", lambda prompt: "   ")
    assert run_engage_prediction(context, {}) is None
    assert load_predictions(context) == []


def test_cli_score_refused_headless(tmp_path: Path, monkeypatch, capsys) -> None:
    """An agent cannot record an outcome score: that would be grading its
    own work — the exact offload the calibration trail exists to counter."""
    _isolate_gate(monkeypatch, tmp_path)
    import afs.cli.calibration as calibration_cli
    from afs.cli.calibration import calibration_score_command

    context_root = _wire_cli(tmp_path, monkeypatch)
    mission = MissionStore(context_root).create(title="Ship it")
    monkeypatch.setattr(calibration_cli, "_TTY_READER", lambda prompt: None)

    exit_code = calibration_score_command(
        _cli_args(context_root, ref=mission.mission_id, outcome="hit", note="")
    )
    assert exit_code == 2
    assert "interactive human confirmation" in capsys.readouterr().err
    assert load_outcomes(context_root) == []


def test_cli_score_wrong_token_refused(tmp_path: Path, monkeypatch, capsys) -> None:
    _isolate_gate(monkeypatch, tmp_path)
    import afs.cli.calibration as calibration_cli
    from afs.cli.calibration import calibration_score_command

    context_root = _wire_cli(tmp_path, monkeypatch)
    mission = MissionStore(context_root).create(title="Ship it")
    monkeypatch.setattr(calibration_cli, "_TTY_READER", lambda prompt: "yes")

    exit_code = calibration_score_command(
        _cli_args(context_root, ref=mission.mission_id, outcome="hit", note="")
    )
    assert exit_code == 2
    assert load_outcomes(context_root) == []


def test_cli_score_records_provenance(tmp_path: Path, monkeypatch, capsys) -> None:
    _isolate_gate(monkeypatch, tmp_path)
    import afs.cli.calibration as calibration_cli
    from afs.cli.calibration import calibration_score_command

    context_root = _wire_cli(tmp_path, monkeypatch)
    mission = MissionStore(context_root).create(title="Ship it")
    monkeypatch.setattr(calibration_cli, "_TTY_READER", lambda prompt: "miss")

    exit_code = calibration_score_command(
        _cli_args(context_root, ref=mission.mission_id, outcome="miss", note="slipped")
    )
    assert exit_code == 0
    (entry,) = load_outcomes(context_root)
    assert entry["scored_via"] == "controlling_terminal"
    assert entry["scored_by"]
    assert entry["human_confirmed"] is True


def test_gate_outcomes_are_global_across_contexts(tmp_path: Path, monkeypatch) -> None:
    """Gate decisions live in a global store, so their scores must too: a
    decision scored while reviewing one context must not resurface as
    unscored (and scorable again) in another."""
    _isolate_gate(monkeypatch, tmp_path)
    from afs.agents.guardrails import ApprovalGate

    context_a = _context(tmp_path / "a")
    context_b = _context(tmp_path / "b")

    gate = ApprovalGate()
    gate._queue("scout", "git_push", "push to origin")
    request = gate.find_pending("scout", "git_push")
    assert request is not None
    gate.approve_human(
        "scout",
        "git_push",
        rationale="diff reviewed",
        authorization=_human_authorization(
            gate.human_authorization_scope(
                "approve", request.request_id, "diff reviewed"
            )
        ),
    )
    ref = ApprovalGate()._pending[0].request_id

    entry = record_human_outcome(
        context_a,
        ref=ref,
        outcome="hit",
        authorization=_human_authorization(
            human_outcome_scope(context_a, ref=ref, outcome="hit")
        ),
    )
    assert entry["kind"] == "gate"
    # The score landed in the global trail, not context A's outcomes file.
    assert (tmp_path / "gate_outcomes.jsonl").exists()
    context_a_file = calibration_root(context_a) / "outcomes.jsonl"
    assert not context_a_file.exists()

    # Both contexts see the decision as scored.
    for context in (context_a, context_b):
        report = collect_decisions(context, days=7)
        assert ref in report["scored"]
        assert report["scored"][ref]["outcome"] == "hit"
