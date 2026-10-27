"""Tests for agent guardrails: quota tracking, model fallback, approvals, iteration caps."""

from __future__ import annotations

import json
import multiprocessing
import os
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import pytest

import afs.agents.approvals as approvals_module
import afs.agents.guardrails as guardrails
from afs.agents.guardrails import (
    ALWAYS_APPROVE,
    AUTO_APPROVE,
    ApprovalGate,
    ApprovalRequest,
    ApprovalStateError,
    GuardrailConfig,
    GuardrailedAgent,
    QuotaTracker,
    _file_lock,
    resolve_model,
)
from afs.claude_defaults import DEFAULT_CLAUDE_GENERATION_MODEL
from afs.gemini_defaults import DEFAULT_GEMINI_GENERATION_MODEL, DEFAULT_GEMINI_SUBTASK_MODEL


def _require_xml_expat() -> None:
    pytest.importorskip(
        "pyexpat",
        reason="XML/plist parsing requires a working pyexpat module",
        exc_type=ImportError,
    )


# ---------------------------------------------------------------------------
# QuotaTracker
# ---------------------------------------------------------------------------


class TestQuotaTracker:
    def test_fresh_tracker_allows_all_providers(self, tmp_path: Path) -> None:
        tracker = QuotaTracker(path=tmp_path / "quota.json")
        assert tracker.can_call("claude") is True
        assert tracker.can_call("gemini") is True
        assert tracker.can_call("codex") is True
        assert tracker.can_call("local") is True

    def test_local_always_allowed(self, tmp_path: Path) -> None:
        tracker = QuotaTracker(path=tmp_path / "quota.json")
        # Record many calls — local should still be allowed
        for _ in range(1000):
            tracker.record_call("local")
        assert tracker.can_call("local") is True

    def test_hourly_limit_blocks(self, tmp_path: Path) -> None:
        tracker = QuotaTracker(
            path=tmp_path / "quota.json",
            quotas={"claude": {"hourly": 3, "daily": 100, "cost_ceiling_usd": 10.0}},
        )
        for _ in range(3):
            assert tracker.can_call("claude") is True
            tracker.record_call("claude")
        assert tracker.can_call("claude") is False

    def test_daily_limit_blocks(self, tmp_path: Path) -> None:
        tracker = QuotaTracker(
            path=tmp_path / "quota.json",
            quotas={"gemini": {"hourly": 100, "daily": 2, "cost_ceiling_usd": 10.0}},
        )
        tracker.record_call("gemini")
        tracker.record_call("gemini")
        assert tracker.can_call("gemini") is False

    def test_cost_ceiling_blocks(self, tmp_path: Path) -> None:
        tracker = QuotaTracker(
            path=tmp_path / "quota.json",
            quotas={"claude": {"hourly": 100, "daily": 100, "cost_ceiling_usd": 1.0}},
        )
        tracker.record_call("claude", cost_usd=0.6)
        assert tracker.can_call("claude") is True
        tracker.record_call("claude", cost_usd=0.5)
        assert tracker.can_call("claude") is False

    def test_persistence_across_instances(self, tmp_path: Path) -> None:
        path = tmp_path / "quota.json"
        quotas = {"claude": {"hourly": 5, "daily": 100, "cost_ceiling_usd": 10.0}}

        tracker1 = QuotaTracker(path=path, quotas=quotas)
        tracker1.record_call("claude")
        tracker1.record_call("claude")

        # New instance should read persisted state
        tracker2 = QuotaTracker(path=path, quotas=quotas)
        assert tracker2.can_call("claude") is True
        tracker2.record_call("claude")
        tracker2.record_call("claude")
        tracker2.record_call("claude")
        assert tracker2.can_call("claude") is False

    def test_window_rolling_resets_hourly(self, tmp_path: Path) -> None:
        path = tmp_path / "quota.json"
        quotas = {"claude": {"hourly": 2, "daily": 100, "cost_ceiling_usd": 10.0}}

        tracker = QuotaTracker(path=path, quotas=quotas)
        tracker.record_call("claude")
        tracker.record_call("claude")
        assert tracker.can_call("claude") is False

        # Manually set the hour window to a past hour
        entry = tracker._get_entry("claude")
        entry.hour_window = "2020-01-01T00"
        tracker._save()

        # Re-load — window should roll and reset
        tracker2 = QuotaTracker(path=path, quotas=quotas)
        assert tracker2.can_call("claude") is True

    def test_window_rolling_resets_daily(self, tmp_path: Path) -> None:
        path = tmp_path / "quota.json"
        quotas = {"claude": {"hourly": 100, "daily": 1, "cost_ceiling_usd": 10.0}}

        tracker = QuotaTracker(path=path, quotas=quotas)
        tracker.record_call("claude")
        assert tracker.can_call("claude") is False

        # Set day window to past date
        entry = tracker._get_entry("claude")
        entry.day_window = "2020-01-01"
        tracker._save()

        tracker2 = QuotaTracker(path=path, quotas=quotas)
        assert tracker2.can_call("claude") is True

    def test_usage_summary(self, tmp_path: Path) -> None:
        tracker = QuotaTracker(path=tmp_path / "quota.json")
        tracker.record_call("claude", cost_usd=0.05)
        tracker.record_call("gemini")
        summary = tracker.usage_summary()
        assert summary["claude"]["calls_this_hour"] == 1
        assert summary["claude"]["cost_today_usd"] == 0.05
        assert summary["gemini"]["calls_this_hour"] == 1

    def test_unknown_provider_allowed(self, tmp_path: Path) -> None:
        tracker = QuotaTracker(path=tmp_path / "quota.json")
        assert tracker.can_call("unknown_provider") is True

    def test_corrupt_quota_file_handled(self, tmp_path: Path) -> None:
        path = tmp_path / "quota.json"
        path.write_text("not json", encoding="utf-8")
        tracker = QuotaTracker(path=path)
        assert tracker.can_call("claude") is True

    def test_zero_limits_mean_unlimited(self, tmp_path: Path) -> None:
        tracker = QuotaTracker(
            path=tmp_path / "quota.json",
            quotas={"claude": {"hourly": 0, "daily": 0, "cost_ceiling_usd": 0.0}},
        )
        for _ in range(50):
            tracker.record_call("claude", cost_usd=1.0)
        assert tracker.can_call("claude") is True


# ---------------------------------------------------------------------------
# resolve_model
# ---------------------------------------------------------------------------


class TestResolveModel:
    def test_standard_tier_prefers_claude(self, tmp_path: Path) -> None:
        tracker = QuotaTracker(path=tmp_path / "quota.json")
        route = resolve_model(quota_tracker=tracker, task_tier="standard")
        assert route.provider == "claude"
        assert route.model_id == DEFAULT_CLAUDE_GENERATION_MODEL

    def test_subtask_tier_prefers_stable_gemini_flash(self, tmp_path: Path) -> None:
        tracker = QuotaTracker(path=tmp_path / "quota.json")
        route = resolve_model(quota_tracker=tracker, task_tier="subtask")
        assert route.provider == "gemini"
        assert route.model_id == DEFAULT_GEMINI_SUBTASK_MODEL

    def test_standard_tier_honors_explicit_preference(self, tmp_path: Path) -> None:
        tracker = QuotaTracker(path=tmp_path / "quota.json")
        route = resolve_model(
            preferred="gemini",
            quota_tracker=tracker,
            task_tier="standard",
        )
        assert route.provider == "gemini"

    def test_critical_tier_does_not_fall_back_to_local(self, tmp_path: Path) -> None:
        tracker = QuotaTracker(
            path=tmp_path / "quota.json",
            quotas={
                "claude": {"hourly": 1, "daily": 1, "cost_ceiling_usd": 0.01},
                "gemini": {"hourly": 1, "daily": 1, "cost_ceiling_usd": 0.01},
                "codex": {"hourly": 0, "daily": 0, "cost_ceiling_usd": 0.0},
                "local": {"hourly": 0, "daily": 0, "cost_ceiling_usd": 0.0},
            },
        )
        tracker.record_call("claude", cost_usd=0.02)
        tracker.record_call("gemini", cost_usd=0.02)

        with pytest.raises(RuntimeError, match="critical task"):
            resolve_model(quota_tracker=tracker, task_tier="critical")

    def test_background_tier_prefers_local(self, tmp_path: Path) -> None:
        tracker = QuotaTracker(path=tmp_path / "quota.json")
        route = resolve_model(quota_tracker=tracker, task_tier="background")
        assert route.provider == "local"
        assert route.model_id == "qwen2.5-coder:14b"

    def test_fallback_on_quota_exhaustion(self, tmp_path: Path) -> None:
        tracker = QuotaTracker(
            path=tmp_path / "quota.json",
            quotas={
                "claude": {"hourly": 0, "daily": 0, "cost_ceiling_usd": 0.01},
                "gemini": {"hourly": 100, "daily": 100, "cost_ceiling_usd": 10.0},
                "codex": {"hourly": 100, "daily": 100, "cost_ceiling_usd": 10.0},
                "local": {"hourly": 0, "daily": 0, "cost_ceiling_usd": 0.0},
            },
        )
        # Exhaust Claude quota
        tracker.record_call("claude", cost_usd=0.02)
        route = resolve_model(quota_tracker=tracker, task_tier="standard")
        assert route.provider == "gemini"
        assert route.model_id == DEFAULT_GEMINI_GENERATION_MODEL


def test_ensure_worktree_refuses_ambiguous_existing_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    root = tmp_path / "worktrees"
    ambiguous = root / repo.name / "feature"
    ambiguous.mkdir(parents=True)
    monkeypatch.setenv("AFS_WORKTREES_ROOT", str(root))

    class Result:
        returncode = 128
        stdout = ""
        stderr = "not a worktree"

    monkeypatch.setattr(guardrails.subprocess, "run", lambda *args, **kwargs: Result())

    with pytest.raises(RuntimeError, match="Refusing to replace"):
        guardrails.ensure_worktree(repo, "feature")

    def test_critical_tier_raises_when_all_exhausted(self, tmp_path: Path) -> None:
        tracker = QuotaTracker(
            path=tmp_path / "quota.json",
            quotas={
                "claude": {"hourly": 1, "daily": 1, "cost_ceiling_usd": 1.0},
                "gemini": {"hourly": 1, "daily": 1, "cost_ceiling_usd": 1.0},
                "codex": {"hourly": 1, "daily": 1, "cost_ceiling_usd": 1.0},
                "local": {"hourly": 1, "daily": 1, "cost_ceiling_usd": 1.0},
            },
        )
        for provider in ["claude", "gemini", "codex", "local"]:
            tracker.record_call(provider)
        with pytest.raises(RuntimeError, match="exhausted"):
            resolve_model(quota_tracker=tracker, task_tier="critical")

    def test_standard_falls_to_local_when_all_exhausted(self, tmp_path: Path) -> None:
        tracker = QuotaTracker(
            path=tmp_path / "quota.json",
            quotas={
                "claude": {"hourly": 1, "daily": 1, "cost_ceiling_usd": 1.0},
                "gemini": {"hourly": 1, "daily": 1, "cost_ceiling_usd": 1.0},
                "codex": {"hourly": 1, "daily": 1, "cost_ceiling_usd": 1.0},
                "local": {"hourly": 1, "daily": 1, "cost_ceiling_usd": 1.0},
            },
        )
        for provider in ["claude", "gemini", "codex", "local"]:
            tracker.record_call(provider)
        route = resolve_model(quota_tracker=tracker, task_tier="standard")
        assert route.provider == "local"
        assert "exhausted" in route.reason

    def test_custom_fallback_chain(self, tmp_path: Path) -> None:
        tracker = QuotaTracker(path=tmp_path / "quota.json")
        route = resolve_model(
            fallback_chain=["local", "gemini"],
            quota_tracker=tracker,
            task_tier="standard",
        )
        assert route.provider == "local"


# ---------------------------------------------------------------------------
# ApprovalGate
# ---------------------------------------------------------------------------


def _hours_ago(hours: float) -> str:
    return (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()


def _shift_clock(
    monkeypatch: pytest.MonkeyPatch, *, hours: float = 0.0, days: float = 0.0
) -> None:
    """Freeze the gate clock ``hours``/``days`` past its current reading."""
    shifted = approvals_module._now() + timedelta(hours=hours, days=days)
    monkeypatch.setattr(approvals_module, "_now", lambda: shifted)


def _human_approve(gate: ApprovalGate, request_id: str, rationale: str = "reviewed") -> None:
    from afs.human_provenance import _broker_for_reader

    authorization = _broker_for_reader(lambda _prompt: "confirm").confirm_token(
        "confirm",
        "prompt",
        scope=gate.human_authorization_scope("approve", request_id, rationale),
    )
    assert gate.approve_human(
        request_id=request_id, rationale=rationale, authorization=authorization
    )


def _concurrent_checker(path: str, agent: str, count: int, barrier: Any) -> None:
    """Worker for the multi-process approval test."""
    gate = ApprovalGate(path=Path(path))
    barrier.wait(timeout=10)
    for n in range(count):
        gate.check(agent, "git_push", f"target {n}")


class TestApprovalGate:
    def test_auto_approve_read_actions(self, tmp_path: Path) -> None:
        gate = ApprovalGate(path=tmp_path / "approvals.json")
        assert gate.check("agent1", "file_read") is True
        assert gate.check("agent1", "context_read") is True
        assert gate.check("agent1", "run_tests") is True

    def test_blocks_dangerous_actions(self, tmp_path: Path) -> None:
        gate = ApprovalGate(path=tmp_path / "approvals.json")
        assert gate.check("agent1", "git_push") is False
        assert gate.check("agent1", "file_delete") is False
        assert gate.check("agent1", "rom_edit") is False

    def test_queues_pending_request(self, tmp_path: Path) -> None:
        gate = ApprovalGate(path=tmp_path / "approvals.json")
        gate.check("agent1", "git_push", "push to main")
        pending = gate.pending_requests()
        assert len(pending) == 1
        assert pending[0].agent == "agent1"
        assert pending[0].action == "git_push"
        assert pending[0].detail == "push to main"

    def test_no_duplicate_pending(self, tmp_path: Path) -> None:
        gate = ApprovalGate(path=tmp_path / "approvals.json")
        gate.check("agent1", "git_push")
        gate.check("agent1", "git_push")
        assert len(gate.pending_requests()) == 1

    def test_long_lived_gate_observes_another_process_approval(self, tmp_path: Path) -> None:
        from afs.human_provenance import _broker_for_reader

        path = tmp_path / "approvals.json"
        long_lived = ApprovalGate(path=path)
        assert long_lived.check("agent1", "git_push", "origin/main") is False

        reviewer = ApprovalGate(path=path)
        request = reviewer.find_pending("agent1", "git_push")
        assert request is not None
        authorization = _broker_for_reader(lambda _prompt: "confirm").confirm_token(
            "confirm",
            "prompt",
            scope=reviewer.human_authorization_scope("approve", request.request_id, "reviewed"),
        )
        assert reviewer.approve_human(
            "agent1",
            "git_push",
            rationale="reviewed",
            authorization=authorization,
        )

        assert long_lived.check("agent1", "git_push", "origin/main") is True
        assert long_lived.pending_requests() == []

    def test_approval_is_bound_to_exact_action_detail(self, tmp_path: Path) -> None:
        from afs.human_provenance import _broker_for_reader

        gate = ApprovalGate(path=tmp_path / "approvals.json")
        assert gate.check("agent1", "git_push", "origin/main") is False
        request = gate.find_pending("agent1", "git_push")
        assert request is not None
        authorization = _broker_for_reader(lambda _prompt: "confirm").confirm_token(
            "confirm",
            "prompt",
            scope=gate.human_authorization_scope("approve", request.request_id, "reviewed"),
        )
        assert gate.approve_human(
            "agent1",
            "git_push",
            rationale="reviewed",
            authorization=authorization,
        )

        assert gate.check("agent1", "git_push", "origin/main") is True
        assert gate.check("agent1", "git_push", "origin/release") is False
        assert [request.detail for request in gate.pending_requests()] == ["origin/release"]

    def test_torn_legacy_file_is_skipped_and_never_overwritten(self, tmp_path: Path) -> None:
        path = tmp_path / "approvals.json"
        path.write_text('{"torn":', encoding="utf-8")
        corrupted = path.read_bytes()

        gate = ApprovalGate(path=path)
        assert gate.check("agent1", "git_push", "origin/main") is False

        assert path.read_bytes() == corrupted
        assert gate.health()["legacy_skipped"] == 1
        assert [r.detail for r in gate.pending_requests()] == ["origin/main"]

    def test_programmatic_approve_does_not_authorize_action(self, tmp_path: Path) -> None:
        gate = ApprovalGate(path=tmp_path / "approvals.json")
        gate.check("agent1", "git_push")
        assert gate.approve("agent1", "git_push", reviewer="human", reviewed_via="tty") is True
        decided = gate.all_requests()[0]
        assert decided.human_confirmed is False
        assert decided.reviewed_via == "programmatic"
        assert gate.check("agent1", "git_push") is False

    def test_broker_authorized_approve_allows_action(self, tmp_path: Path) -> None:
        from afs.human_provenance import _broker_for_reader

        gate = ApprovalGate(path=tmp_path / "approvals.json")
        gate.check("agent1", "git_push")
        request = gate.find_pending("agent1", "git_push")
        assert request is not None
        authorization = _broker_for_reader(lambda _prompt: "confirm").confirm_token(
            "confirm",
            "prompt",
            scope=gate.human_authorization_scope("approve", request.request_id, "reviewed"),
        )
        assert authorization is not None
        with pytest.raises(ValueError, match="authorization"):
            gate.approve_human(
                "agent1",
                "git_push",
                rationale="caller changed the rationale",
                authorization=authorization,
            )
        assert gate.approve_human(
            "agent1",
            "git_push",
            rationale="reviewed",
            authorization=authorization,
        )
        assert gate.check("agent1", "git_push") is True

    def test_reject_keeps_blocked(self, tmp_path: Path) -> None:
        gate = ApprovalGate(path=tmp_path / "approvals.json")
        gate.check("agent1", "deploy")
        assert gate.reject("agent1", "deploy") is True
        # After rejection, a new check should queue a new request
        assert gate.check("agent1", "deploy") is False

    def test_requests_get_stable_unique_ids(self, tmp_path: Path) -> None:
        path = tmp_path / "approvals.json"
        gate = ApprovalGate(path=path)
        gate.check("agent1", "git_push")
        gate.check("agent1", "deploy")
        ids = [r.request_id for r in gate.pending_requests()]
        assert all(rid.startswith("gate_") for rid in ids)
        assert len(set(ids)) == 2
        # Ids survive a reload — they are calibration refs.
        reloaded = ApprovalGate(path=path)
        assert [r.request_id for r in reloaded.pending_requests()] == ids

    def test_legacy_records_are_backfilled_with_stable_ids(self, tmp_path: Path) -> None:
        path = tmp_path / "approvals.json"
        path.write_text(
            json.dumps(
                [
                    {
                        "agent": "old",
                        "action": "git_push",
                        "detail": "",
                        "timestamp": _hours_ago(1),
                        "status": "pending",
                        "reviewed_by": "",
                        "reviewed_at": "",
                    }
                ]
            ),
            encoding="utf-8",
        )
        first = ApprovalGate(path=path).pending_requests()[0].request_id
        assert first.startswith("gate_")
        # The id is derived from the record, so a fresh import sees the same one.
        path.with_suffix(".sqlite3").unlink()
        second = ApprovalGate(path=path).pending_requests()[0].request_id
        assert second == first

    def test_legacy_pending_past_ttl_expires_on_import(self, tmp_path: Path) -> None:
        path = tmp_path / "approvals.json"
        path.write_text(
            json.dumps(
                [
                    {"agent": "old", "action": "git_push", "timestamp": _hours_ago(100)},
                    {"agent": "new", "action": "git_push", "timestamp": _hours_ago(1)},
                ]
            ),
            encoding="utf-8",
        )
        gate = ApprovalGate(path=path)
        by_agent = {r.agent: r for r in gate.all_requests()}
        assert by_agent["old"].status == "expired"
        assert by_agent["old"].human_confirmed is False
        assert by_agent["new"].status == "pending"

    def test_decision_does_not_clobber_concurrent_queue(self, tmp_path: Path) -> None:
        """Each operation is its own transaction, so a request queued by
        another gate between two calls is never lost."""
        path = tmp_path / "approvals.json"
        gate_a = ApprovalGate(path=path)
        gate_a.check("agent1", "git_push")

        gate_b = ApprovalGate(path=path)
        gate_b.check("agent2", "deploy")

        assert gate_a.approve("agent1", "git_push", rationale="reviewed") is True
        statuses = {(r.agent, r.status) for r in ApprovalGate(path=path).all_requests()}
        assert statuses == {("agent1", "approved"), ("agent2", "pending")}

    def test_clear_prunes_old_system_rows_and_keeps_decisions(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "approvals.json"
        gate = ApprovalGate(path=path)
        gate.check("agent1", "git_push")
        gate.approve("agent1", "git_push", rationale="reviewed")
        gate.check("agent2", "deploy", source="mission:a")
        gate.withdraw("agent2", source="mission:a", reason="phase removed")
        gate.check("agent3", "deploy")

        assert gate.clear_completed() == (0, 1)
        _shift_clock(monkeypatch, days=31)
        assert gate.clear_completed() == (1, 0)

        remaining = {(r.agent, r.status) for r in gate.all_requests()}
        # agent3 expired on the way (72h TTL); expiry is recent, so it is kept.
        assert remaining == {("agent1", "approved"), ("agent3", "expired")}

    def test_torn_history_line_is_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "approvals.json"
        good = {
            "agent": "agent1",
            "action": "git_push",
            "timestamp": _hours_ago(200),
            "status": "rejected",
            "rationale": "unsafe target",
            "request_id": "gate_old",
        }
        (tmp_path / "approvals.history.jsonl").write_text(
            json.dumps(good) + '\n{"request_id":"torn"', encoding="utf-8"
        )
        gate = ApprovalGate(path=path)
        history = gate.all_requests()
        assert [(r.request_id, r.rationale) for r in history] == [("gate_old", "unsafe target")]
        assert gate.health()["legacy_skipped"] == 1

    def test_failed_decision_rolls_back(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "approvals.json"
        gate = ApprovalGate(path=path)
        gate.check("agent1", "git_push", "preserve this request")
        original_mark = ApprovalGate._mark_decided

        def mark_then_crash(*args: Any, **kwargs: Any) -> bool:
            assert original_mark(*args, **kwargs)
            raise OSError("simulated crash before commit")

        monkeypatch.setattr(ApprovalGate, "_mark_decided", staticmethod(mark_then_crash))
        with pytest.raises(OSError, match="simulated crash before commit"):
            gate.approve("agent1", "git_push", rationale="reviewed")

        reloaded = ApprovalGate(path=path).all_requests()
        assert [(r.status, r.rationale, r.detail) for r in reloaded] == [
            ("pending", "", "preserve this request")
        ]

    def test_one_malformed_record_is_skipped_without_dropping_rest(self, tmp_path: Path) -> None:
        path = tmp_path / "approvals.json"
        good = {
            "agent": "agent1",
            "action": "git_push",
            "detail": "",
            "timestamp": _hours_ago(1),
            "status": "pending",
        }
        bad = {"agent": "agent2", "unexpected_field": True}
        original = json.dumps([bad, good]).encode()
        path.write_bytes(original)

        gate = ApprovalGate(path=path)
        assert [r.agent for r in gate.pending_requests()] == ["agent1"]
        assert gate.health()["legacy_skipped"] == 1
        assert path.read_bytes() == original

    def test_persistence_across_instances(self, tmp_path: Path) -> None:
        path = tmp_path / "approvals.json"
        gate1 = ApprovalGate(path=path)
        gate1.check("agent1", "git_push")

        gate2 = ApprovalGate(path=path)
        assert len(gate2.pending_requests()) == 1

    def test_corrupt_database_fails_closed(self, tmp_path: Path) -> None:
        path = tmp_path / "approvals.sqlite3"
        path.write_bytes(b"not a database, just bytes " * 64)
        corrupted = path.read_bytes()
        gate = ApprovalGate(path=path)
        with pytest.raises(ApprovalStateError, match="unreadable"):
            gate.check("agent1", "git_push")
        assert path.read_bytes() == corrupted

    def test_unknown_action_queued(self, tmp_path: Path) -> None:
        gate = ApprovalGate(path=tmp_path / "approvals.json")
        assert gate.check("agent1", "custom_unknown_action") is False
        assert len(gate.pending_requests()) == 1

    def test_human_approved_unknown_action_is_honored(self, tmp_path: Path) -> None:
        from afs.human_provenance import _broker_for_reader

        gate = ApprovalGate(path=tmp_path / "approvals.json")
        assert gate.check("agent1", "custom_unknown_action", "exact target") is False
        request = gate.find_pending("agent1", "custom_unknown_action")
        assert request is not None
        authorization = _broker_for_reader(lambda _prompt: "confirm").confirm_token(
            "confirm",
            "prompt",
            scope=gate.human_authorization_scope(
                "approve", request.request_id, "reviewed unknown action"
            ),
        )
        assert gate.approve_human(
            "agent1",
            "custom_unknown_action",
            rationale="reviewed unknown action",
            authorization=authorization,
        )
        assert gate.check("agent1", "custom_unknown_action", "exact target") is True

    def test_string_human_confirmation_fails_closed(self, tmp_path: Path) -> None:
        path = tmp_path / "approvals.json"
        request = ApprovalRequest(
            agent="agent1",
            action="git_push",
            detail="origin/main",
            timestamp="2026-01-01T00:00:00+00:00",
            status="approved",
            request_id="gate_untrusted",
            human_confirmed=True,
        ).to_dict()
        request["human_confirmed"] = "false"
        request["identity_authenticated"] = "true"
        path.write_text(json.dumps([request]), encoding="utf-8")

        gate = ApprovalGate(path=path)
        assert gate.check("agent1", "git_push", "origin/main") is False
        assert any(item.status == "pending" for item in gate.pending_requests())
        loaded = gate.get("gate_untrusted")
        assert loaded is not None
        assert loaded.human_confirmed is False
        assert loaded.identity_authenticated is False

    # -- lifecycle ---------------------------------------------------------

    def test_pending_request_expires_after_ttl(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        gate = ApprovalGate(path=tmp_path / "approvals.json", ttl_hours=2)
        gate.check("agent1", "git_push")
        _shift_clock(monkeypatch, hours=1)
        assert len(gate.pending_requests()) == 1

        _shift_clock(monkeypatch, hours=2)
        assert gate.pending_requests() == []
        expired = gate.all_requests()[0]
        assert (expired.status, expired.reviewed_via) == ("expired", "ttl")
        assert expired.human_confirmed is False
        # An expired request cannot be approved; the next check queues anew.
        assert gate.approve("agent1", "git_push") is False
        assert gate.check("agent1", "git_push") is False
        assert len(gate.pending_requests()) == 1

    def test_new_run_supersedes_older_request_from_same_source(self, tmp_path: Path) -> None:
        gate = ApprovalGate(path=tmp_path / "approvals.json")
        assert gate.check("runner", "deploy", "v1", source="mission:a", run_id="run_1") is False
        # The same run asking again does not requeue.
        assert gate.check("runner", "deploy", "v1", source="mission:a", run_id="run_1") is False
        assert len(gate.pending_requests()) == 1

        gate.check("runner", "deploy", "v2", source="mission:a", run_id="run_2")
        gate.check("runner", "deploy", "v1", source="mission:b", run_id="run_3")
        by_run = {r.run_id: r for r in gate.all_requests()}
        assert by_run["run_1"].status == "superseded"
        assert by_run["run_1"].superseded_by == by_run["run_2"].request_id
        assert by_run["run_2"].status == "pending"
        assert by_run["run_3"].status == "pending"

    def test_once_approval_is_consumed_by_one_run(self, tmp_path: Path) -> None:
        gate = ApprovalGate(path=tmp_path / "approvals.json")
        key = ("runner", "mission_phase_act", "phase act")
        assert gate.check(*key, source="mission:a", run_id="run_1", scope="once") is False
        _human_approve(gate, gate.pending_requests()[0].request_id)

        assert gate.check(*key, source="mission:a", run_id="run_2", scope="once") is True
        assert gate.check(*key, source="mission:a", run_id="run_2", scope="once") is True
        assert gate.check(*key, source="mission:a", run_id="run_3", scope="once") is False
        consumed = [r for r in gate.all_requests() if r.status == "approved"][0]
        assert consumed.consumed_by_run == "run_2"
        assert [r.run_id for r in gate.pending_requests()] == ["run_3"]

    def test_withdraw_closes_pending_requests_for_source(self, tmp_path: Path) -> None:
        gate = ApprovalGate(path=tmp_path / "approvals.json")
        gate.check("runner", "mission_phase_a", source="mission:a")
        gate.check("runner", "mission_phase_b", source="mission:a")
        gate.check("runner", "mission_phase_a", source="mission:b")

        assert gate.withdraw("runner", action="mission_phase_a", source="mission:a") == 1
        assert gate.withdraw("runner", source="mission:a", reason="mission retired") == 1
        assert [r.source for r in gate.pending_requests()] == ["mission:b"]
        with pytest.raises(ValueError, match="action or a source"):
            gate.withdraw("runner")

    def test_revoke_stops_authorization_and_keeps_decision(self, tmp_path: Path) -> None:
        gate = ApprovalGate(path=tmp_path / "approvals.json")
        gate.check("agent1", "git_push", "origin/main")
        request_id = gate.pending_requests()[0].request_id
        _human_approve(gate, request_id)
        assert gate.check("agent1", "git_push", "origin/main") is True

        with pytest.raises(ValueError, match="reason"):
            gate.revoke(request_id, reason=" ")
        assert gate.revoke(request_id, reason="branch frozen") is True
        assert gate.revoke(request_id, reason="again") is False
        assert gate.check("agent1", "git_push", "origin/main") is False

        kept = gate.get(request_id)
        assert kept is not None
        assert (kept.status, kept.human_confirmed) == ("approved", True)
        assert kept.revoked_reason == "branch frozen"

    def test_rejected_run_stays_blocked_without_requeue(self, tmp_path: Path) -> None:
        from afs.human_provenance import _broker_for_reader

        gate = ApprovalGate(path=tmp_path / "approvals.json")
        gate.check("runner", "deploy", source="mission:a", run_id="run_1")
        request_id = gate.pending_requests()[0].request_id
        authorization = _broker_for_reader(lambda _prompt: "confirm").confirm_token(
            "confirm",
            "prompt",
            scope=gate.human_authorization_scope("reject", request_id, "not now"),
        )
        assert gate.reject_human(
            request_id=request_id, rationale="not now", authorization=authorization
        )
        assert gate.check("runner", "deploy", source="mission:a", run_id="run_1") is False
        assert gate.pending_requests() == []
        assert gate.check("runner", "deploy", source="mission:a", run_id="run_2") is False
        assert [r.run_id for r in gate.pending_requests()] == ["run_2"]

    def test_batch_reject_is_bound_to_the_exact_id_set(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from afs.human_provenance import _broker_for_reader

        gate = ApprovalGate(path=tmp_path / "approvals.json")
        gate.check("old-a", "deploy")
        gate.check("old-b", "deploy")
        _shift_clock(monkeypatch, hours=30)
        gate.check("fresh", "deploy")

        stale = sorted(r.request_id for r in gate.stale_pending(24))
        assert len(stale) == 2
        broker = _broker_for_reader(lambda _prompt: "reject 2")
        wrong_set = broker.confirm_token(
            "reject 2",
            "prompt",
            scope=gate.batch_authorization_scope("reject", stale[:1], "stale"),
        )
        with pytest.raises(ValueError, match="authorization"):
            gate.reject_human_batch(stale, rationale="stale", authorization=wrong_set)

        authorization = broker.confirm_token(
            "reject 2",
            "prompt",
            scope=gate.batch_authorization_scope("reject", stale, "stale"),
        )
        assert gate.reject_human_batch(stale, rationale="stale", authorization=authorization) == stale
        statuses = {r.agent: (r.status, r.human_confirmed) for r in gate.all_requests()}
        assert statuses == {
            "old-a": ("rejected", True),
            "old-b": ("rejected", True),
            "fresh": ("pending", False),
        }

    def test_concurrent_checks_from_many_processes(self, tmp_path: Path) -> None:
        path = tmp_path / "approvals.sqlite3"
        ctx = multiprocessing.get_context("spawn")
        barrier = ctx.Barrier(4)
        workers = [
            ctx.Process(target=_concurrent_checker, args=(str(path), f"agent{i}", 5, barrier))
            for i in range(4)
        ]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(timeout=60)
            assert worker.exitcode == 0

        pending = ApprovalGate(path=path).pending_requests()
        assert sorted((r.agent, r.detail) for r in pending) == sorted(
            (f"agent{i}", f"target {n}") for i in range(4) for n in range(5)
        )

    def test_auto_approve_constants(self) -> None:
        assert "file_read" in AUTO_APPROVE
        assert "context_read" in AUTO_APPROVE
        assert "run_tests" in AUTO_APPROVE
        assert "git_push" in ALWAYS_APPROVE
        assert "rom_edit" in ALWAYS_APPROVE


# ---------------------------------------------------------------------------
# GuardrailedAgent
# ---------------------------------------------------------------------------


class TestGuardrailedAgent:
    def test_iteration_cap(self) -> None:
        config = GuardrailConfig(max_iterations=3)
        guard = GuardrailedAgent("test-agent", config=config)
        assert guard.should_continue() is True  # 1
        assert guard.should_continue() is True  # 2
        assert guard.should_continue() is True  # 3
        assert guard.should_continue() is False  # 4 — exceeded

    def test_checkpoint_interval(self) -> None:
        config = GuardrailConfig(checkpoint_interval=3)
        guard = GuardrailedAgent("test-agent", config=config)
        guard.should_continue()  # iteration 1
        assert guard.should_checkpoint() is False
        guard.should_continue()  # iteration 2
        assert guard.should_checkpoint() is False
        guard.should_continue()  # iteration 3
        assert guard.should_checkpoint() is True

    def test_dry_run_blocks_all_actions(self) -> None:
        config = GuardrailConfig(dry_run=True)
        guard = GuardrailedAgent("test-agent", config=config)
        assert guard.can_do("git_push") is False
        assert guard.can_do("file_read") is False  # even reads blocked in dry run

    def test_disabled_approvals_allows_all(self) -> None:
        config = GuardrailConfig(enable_approvals=False)
        guard = GuardrailedAgent("test-agent", config=config)
        assert guard.can_do("git_push") is True
        assert guard.can_do("file_delete") is True

    def test_resolve_model_with_background_tier(self, tmp_path: Path) -> None:
        config = GuardrailConfig(task_tier="background")
        guard = GuardrailedAgent("test-agent", config=config)
        # Override tracker with isolated path
        guard._tracker = QuotaTracker(path=tmp_path / "quota.json")
        model = guard.resolve_model()
        assert model.provider == "local"

    def test_record_call_tracks_usage(self, tmp_path: Path) -> None:
        config = GuardrailConfig()
        guard = GuardrailedAgent("test-agent", config=config)
        # Use isolated quota and approval paths
        guard._tracker = QuotaTracker(path=tmp_path / "quota.json")
        guard.record_call("claude", cost_usd=0.05)
        usage = guard.usage_summary()
        assert usage["claude"]["calls_this_hour"] == 1
        assert usage["claude"]["cost_today_usd"] == 0.05

    def test_pending_approvals_scoped_to_agent(self, tmp_path: Path) -> None:
        config = GuardrailConfig()
        guard = GuardrailedAgent("agent-a", config=config)
        guard._gate = ApprovalGate(path=tmp_path / "approvals.json")
        guard.can_do("git_push")
        assert len(guard.pending_approvals()) == 1
        assert guard.pending_approvals()[0].agent == "agent-a"

    def test_usage_summary_empty_when_disabled(self) -> None:
        config = GuardrailConfig(enable_quota=False)
        guard = GuardrailedAgent("test-agent", config=config)
        assert guard.usage_summary() == {}

    def test_iteration_property(self) -> None:
        guard = GuardrailedAgent("test-agent")
        assert guard.iteration == 0
        guard.should_continue()
        assert guard.iteration == 1


# ---------------------------------------------------------------------------
# File locking
# ---------------------------------------------------------------------------


def _concurrent_writer(path: str, provider: str, count: int, barrier: Any) -> None:
    """Worker function for concurrent locking test."""
    tracker = QuotaTracker(path=Path(path))
    barrier.wait(timeout=10)
    for _ in range(count):
        tracker.record_call(provider, cost_usd=0.01)


class TestFileLocking:
    def test_lock_basic(self, tmp_path: Path) -> None:
        target = tmp_path / "test.json"
        target.write_text("{}", encoding="utf-8")
        with _file_lock(target):
            # Should be able to read/write while holding lock
            target.write_text('{"locked": true}', encoding="utf-8")
        data = json.loads(target.read_text(encoding="utf-8"))
        assert data["locked"] is True

    def test_lock_creates_parent_dirs(self, tmp_path: Path) -> None:
        target = tmp_path / "deep" / "nested" / "test.json"
        with _file_lock(target):
            pass
        assert (tmp_path / "deep" / "nested" / "test.json.lock").exists()

    def test_lock_cleanup(self, tmp_path: Path) -> None:
        target = tmp_path / "test.json"
        target.write_text("{}", encoding="utf-8")
        with _file_lock(target):
            pass
        # Lock file should exist (not deleted, just unlocked)
        assert (tmp_path / "test.json.lock").exists()

    def test_lock_uses_windows_backend_when_fcntl_is_unavailable(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[tuple[int, int]] = []

        class FakeMsvcrt:
            LK_NBLCK = 1
            LK_UNLCK = 2

            @staticmethod
            def locking(_fd: int, mode: int, size: int) -> None:
                calls.append((mode, size))

        monkeypatch.setattr(guardrails, "fcntl", None)
        monkeypatch.setattr(guardrails, "msvcrt", FakeMsvcrt)

        with _file_lock(tmp_path / "test.json"):
            pass

        assert calls == [(FakeMsvcrt.LK_NBLCK, 1), (FakeMsvcrt.LK_UNLCK, 1)]

    def test_lock_timeout_does_not_enter_protected_body(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        entered = False

        def always_contended(_fd: int) -> None:
            raise BlockingIOError("lock held")

        monkeypatch.setattr(guardrails, "_try_lock", always_contended)

        with pytest.raises(TimeoutError, match="Lock timeout"):
            with _file_lock(tmp_path / "test.json", timeout=0):
                entered = True

        assert entered is False

    def test_concurrent_quota_writes(self, tmp_path: Path) -> None:
        """Verify synchronized writers preserve every quota and cost increment."""
        path = tmp_path / "quota.json"
        calls_per_process = 10
        num_processes = 4
        context = multiprocessing.get_context("spawn")
        barrier = context.Barrier(num_processes)

        procs = []
        for _i in range(num_processes):
            p = context.Process(
                target=_concurrent_writer,
                args=(str(path), "claude", calls_per_process, barrier),
            )
            procs.append(p)
            p.start()

        for p in procs:
            p.join(timeout=30)
            assert p.exitcode == 0

        expected_calls = calls_per_process * num_processes
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert data["claude"]["calls_this_hour"] == expected_calls
        assert data["claude"]["calls_today"] == expected_calls
        assert data["claude"]["cost_today_usd"] == pytest.approx(0.4)

    def test_quota_tracker_accumulates_across_instances(self, tmp_path: Path) -> None:
        """Verify each record_call transaction starts from current disk state."""
        path = tmp_path / "quota.json"
        quotas = {"claude": {"hourly": 100, "daily": 1000, "cost_ceiling_usd": 50.0}}

        # Writer A records 3 calls
        tracker_a = QuotaTracker(path=path, quotas=quotas)
        for _ in range(3):
            tracker_a.record_call("claude")

        # Writer B starts fresh, records 5 calls
        tracker_b = QuotaTracker(path=path, quotas=quotas)
        for _ in range(5):
            tracker_b.record_call("claude")

        final = QuotaTracker(path=path, quotas=quotas)
        entry = final._get_entry("claude")
        assert entry.calls_this_hour == 8
        assert entry.calls_today == 8


# ---------------------------------------------------------------------------
# LaunchAgent plist
# ---------------------------------------------------------------------------


class TestLaunchAgentPlist:
    def test_plist_is_valid_xml(self) -> None:
        _require_xml_expat()
        import xml.etree.ElementTree as ET

        plist_path = Path(__file__).parent.parent / "scripts" / "com.afs.supervisor.plist"
        if not plist_path.exists():
            pytest.skip("plist not found")
        tree = ET.parse(str(plist_path))
        root = tree.getroot()
        assert root.tag == "plist"

    def test_plist_has_required_keys(self) -> None:
        _require_xml_expat()
        import plistlib

        plist_path = Path(__file__).parent.parent / "scripts" / "com.afs.supervisor.plist"
        if not plist_path.exists():
            pytest.skip("plist not found")
        with open(plist_path, "rb") as f:
            data = plistlib.load(f)
        assert data["Label"] == "com.afs.supervisor"
        assert "ProgramArguments" in data
        assert "StartInterval" in data
        assert data["StartInterval"] == 900
        assert "afs.agents.supervisor" in data["ProgramArguments"]

    def test_ctl_script_is_executable(self) -> None:
        ctl_path = Path(__file__).parent.parent / "scripts" / "afs-supervisor-ctl"
        if not ctl_path.exists():
            pytest.skip("ctl script not found")
        assert os.access(str(ctl_path), os.X_OK)
