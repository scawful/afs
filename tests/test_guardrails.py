"""Tests for agent guardrails: quota tracking, model fallback, approvals, iteration caps."""

from __future__ import annotations

import json
import multiprocessing
import os
from pathlib import Path
from typing import Any

import pytest

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
        assert route.model_id == "claude-3-5-sonnet"

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

    def test_long_lived_gate_observes_another_process_approval(
        self, tmp_path: Path
    ) -> None:
        from afs.human_provenance import _broker_for_reader

        path = tmp_path / "approvals.json"
        long_lived = ApprovalGate(path=path)
        assert long_lived.check("agent1", "git_push", "origin/main") is False

        reviewer = ApprovalGate(path=path)
        request = reviewer.find_pending("agent1", "git_push")
        assert request is not None
        authorization = _broker_for_reader(
            lambda _prompt: "confirm"
        ).confirm_token(
            "confirm",
            "prompt",
            scope=reviewer.human_authorization_scope(
                "approve", request.request_id, "reviewed"
            ),
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
        authorization = _broker_for_reader(
            lambda _prompt: "confirm"
        ).confirm_token(
            "confirm",
            "prompt",
            scope=gate.human_authorization_scope(
                "approve", request.request_id, "reviewed"
            ),
        )
        assert gate.approve_human(
            "agent1",
            "git_push",
            rationale="reviewed",
            authorization=authorization,
        )

        assert gate.check("agent1", "git_push", "origin/main") is True
        assert gate.check("agent1", "git_push", "origin/release") is False
        assert [request.detail for request in gate.pending_requests()] == [
            "origin/release"
        ]

    def test_corrupt_active_state_is_never_overwritten(self, tmp_path: Path) -> None:
        path = tmp_path / "approvals.json"
        gate = ApprovalGate(path=path)
        path.write_text('{"torn":', encoding="utf-8")
        corrupted = path.read_bytes()

        with pytest.raises(ApprovalStateError, match="refusing to overwrite"):
            gate.check("agent1", "git_push", "origin/main")

        assert path.read_bytes() == corrupted
        with pytest.raises(ApprovalStateError, match="refusing to overwrite"):
            ApprovalGate(path=path)

    def test_programmatic_approve_does_not_authorize_action(self, tmp_path: Path) -> None:
        gate = ApprovalGate(path=tmp_path / "approvals.json")
        gate.check("agent1", "git_push")
        assert gate.approve(
            "agent1", "git_push", reviewer="human", reviewed_via="tty"
        ) is True
        assert gate._pending[0].human_confirmed is False
        assert gate._pending[0].reviewed_via == "programmatic"
        assert gate.check("agent1", "git_push") is False

    def test_broker_authorized_approve_allows_action(self, tmp_path: Path) -> None:
        from afs.human_provenance import _broker_for_reader

        gate = ApprovalGate(path=tmp_path / "approvals.json")
        gate.check("agent1", "git_push")
        request = gate.find_pending("agent1", "git_push")
        assert request is not None
        authorization = _broker_for_reader(
            lambda _prompt: "confirm"
        ).confirm_token(
            "confirm",
            "prompt",
            scope=gate.human_authorization_scope(
                "approve", request.request_id, "reviewed"
            ),
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
        import json as json_module

        path = tmp_path / "approvals.json"
        path.write_text(
            json_module.dumps(
                [
                    {
                        "agent": "old",
                        "action": "git_push",
                        "detail": "",
                        "timestamp": "2026-01-01T00:00:00+00:00",
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
        # The backfilled id was persisted, so a second load sees the same one.
        second = ApprovalGate(path=path).pending_requests()[0].request_id
        assert second == first

    def test_decision_does_not_clobber_concurrent_queue(self, tmp_path: Path) -> None:
        """A decision re-reads the store under the lock, so requests queued by
        another process between our load and save are preserved."""
        path = tmp_path / "approvals.json"
        gate_a = ApprovalGate(path=path)
        gate_a.check("agent1", "git_push")

        # Another process queues a second request after gate_a loaded.
        gate_b = ApprovalGate(path=path)
        gate_b.check("agent2", "deploy")

        assert gate_a.approve("agent1", "git_push", rationale="reviewed") is True
        final = ApprovalGate(path=path)
        assert len(final._pending) == 2
        statuses = {(r.agent, r.status) for r in final._pending}
        assert ("agent1", "approved") in statuses
        assert ("agent2", "pending") in statuses

    def test_clear_does_not_clobber_concurrent_queue(self, tmp_path: Path) -> None:
        """clear_completed re-reads under the lock, so a request queued by
        another process after this gate loaded survives the clear."""
        path = tmp_path / "approvals.json"
        gate_a = ApprovalGate(path=path)
        gate_a.check("agent1", "git_push")
        gate_a.approve("agent1", "git_push", rationale="reviewed")

        # Another process queues a new request after gate_a's last read.
        gate_b = ApprovalGate(path=path)
        gate_b.check("agent2", "deploy")

        removed, remaining = gate_a.clear_completed()
        assert removed == 1
        assert remaining == 1
        final = ApprovalGate(path=path)
        assert [(r.agent, r.status) for r in final._pending] == [("agent2", "pending")]
        history = final.all_requests()
        archived = [request for request in history if request.agent == "agent1"]
        assert len(archived) == 1
        assert archived[0].rationale == "reviewed"

    def test_clear_repairs_torn_archive_tail_before_compacting(
        self, tmp_path: Path
    ) -> None:
        path = tmp_path / "approvals.json"
        gate = ApprovalGate(path=path)
        gate.check("agent1", "git_push")
        request = gate.find_pending("agent1", "git_push")
        assert request is not None
        assert gate.reject("agent1", "git_push", rationale="unsafe target")

        # Simulate a process dying partway through a prior JSONL append.
        gate._archive_path.write_bytes(b'{"request_id":"torn"')

        assert gate.clear_completed() == (1, 0)
        history = gate.all_requests()
        assert [item.request_id for item in history] == [request.request_id]
        assert history[0].rationale == "unsafe target"
        assert gate._archive_path.read_bytes().endswith(b"\n")

    def test_resolve_replace_failure_preserves_active_approvals(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "approvals.json"
        gate = ApprovalGate(path=path)
        gate.check("agent1", "git_push", "preserve this request")
        original = path.read_bytes()

        def fail_replace(source: Path, destination: Path) -> None:
            source_path = Path(source)
            assert source_path.parent == path.parent
            assert Path(destination) == path
            assert json.loads(source_path.read_text(encoding="utf-8"))[0][
                "rationale"
            ] == "reviewed"
            raise OSError("simulated crash before replace")

        monkeypatch.setattr(guardrails.os, "replace", fail_replace)

        with pytest.raises(OSError, match="simulated crash before replace"):
            gate.approve("agent1", "git_push", rationale="reviewed")

        assert path.read_bytes() == original
        assert [(item.status, item.rationale) for item in gate._pending] == [
            ("pending", "")
        ]
        assert not list(path.parent.glob(f".{path.name}.*.tmp"))
        reloaded = ApprovalGate(path=path).pending_requests()
        assert [(item.status, item.detail) for item in reloaded] == [
            ("pending", "preserve this request")
        ]

    def test_resolve_fsync_failure_preserves_active_approvals(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        path = tmp_path / "approvals.json"
        gate = ApprovalGate(path=path)
        gate.check("agent1", "deploy", "preserve this request")
        original = path.read_bytes()

        def fail_fsync(_fd: int) -> None:
            raise OSError("simulated full disk during flush")

        monkeypatch.setattr(guardrails.os, "fsync", fail_fsync)

        with pytest.raises(OSError, match="simulated full disk during flush"):
            gate.reject("agent1", "deploy", rationale="unsafe")

        assert path.read_bytes() == original
        assert [(item.status, item.rationale) for item in gate._pending] == [
            ("pending", "")
        ]
        assert not list(path.parent.glob(f".{path.name}.*.tmp"))
        reloaded = ApprovalGate(path=path).pending_requests()
        assert [(item.status, item.detail) for item in reloaded] == [
            ("pending", "preserve this request")
        ]

    def test_one_malformed_record_fails_closed_without_dropping_rest(
        self, tmp_path: Path
    ) -> None:
        import json as json_module

        path = tmp_path / "approvals.json"
        good = {
            "agent": "agent1",
            "action": "git_push",
            "detail": "",
            "timestamp": "2026-01-01T00:00:00+00:00",
            "status": "pending",
        }
        bad = {"agent": "agent2", "unexpected_field": True}
        original = json_module.dumps([bad, good]).encode()
        path.write_bytes(original)

        with pytest.raises(ApprovalStateError, match="record 0 is malformed"):
            ApprovalGate(path=path)

        assert path.read_bytes() == original

    def test_persistence_across_instances(self, tmp_path: Path) -> None:
        path = tmp_path / "approvals.json"
        gate1 = ApprovalGate(path=path)
        gate1.check("agent1", "git_push")

        gate2 = ApprovalGate(path=path)
        assert len(gate2.pending_requests()) == 1

    def test_corrupt_file_fails_closed(self, tmp_path: Path) -> None:
        path = tmp_path / "approvals.json"
        path.write_text("not json", encoding="utf-8")
        with pytest.raises(ApprovalStateError, match="refusing to overwrite"):
            ApprovalGate(path=path)

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
        loaded = [item for item in gate._pending if item.request_id == "gate_untrusted"][0]
        assert loaded.human_confirmed is False
        assert loaded.identity_authenticated is False

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
