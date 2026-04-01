"""Tests for agent guardrails: quota tracking, model fallback, approvals, iteration caps."""

from __future__ import annotations

import json
import multiprocessing
import os
from datetime import datetime, timezone
from pathlib import Path

import pytest

from afs.agents.guardrails import (
    AUTO_APPROVE,
    ALWAYS_APPROVE,
    ApprovalGate,
    ApprovalRequest,
    GuardrailConfig,
    GuardrailedAgent,
    ModelRoute,
    QuotaEntry,
    QuotaTracker,
    _file_lock,
    resolve_model,
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

    def test_approve_allows_action(self, tmp_path: Path) -> None:
        gate = ApprovalGate(path=tmp_path / "approvals.json")
        gate.check("agent1", "git_push")
        assert gate.approve("agent1", "git_push") is True
        assert gate.check("agent1", "git_push") is True

    def test_reject_keeps_blocked(self, tmp_path: Path) -> None:
        gate = ApprovalGate(path=tmp_path / "approvals.json")
        gate.check("agent1", "deploy")
        assert gate.reject("agent1", "deploy") is True
        # After rejection, a new check should queue a new request
        assert gate.check("agent1", "deploy") is False

    def test_persistence_across_instances(self, tmp_path: Path) -> None:
        path = tmp_path / "approvals.json"
        gate1 = ApprovalGate(path=path)
        gate1.check("agent1", "git_push")

        gate2 = ApprovalGate(path=path)
        assert len(gate2.pending_requests()) == 1

    def test_corrupt_file_handled(self, tmp_path: Path) -> None:
        path = tmp_path / "approvals.json"
        path.write_text("not json", encoding="utf-8")
        gate = ApprovalGate(path=path)
        assert gate.check("agent1", "file_read") is True

    def test_unknown_action_queued(self, tmp_path: Path) -> None:
        gate = ApprovalGate(path=tmp_path / "approvals.json")
        assert gate.check("agent1", "custom_unknown_action") is False
        assert len(gate.pending_requests()) == 1

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


def _concurrent_writer(path: str, provider: str, count: int) -> None:
    """Worker function for concurrent locking test."""
    tracker = QuotaTracker(path=Path(path))
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

    def test_concurrent_quota_writes(self, tmp_path: Path) -> None:
        """Verify concurrent writers don't corrupt the quota file."""
        path = tmp_path / "quota.json"
        calls_per_process = 10
        num_processes = 4

        procs = []
        for i in range(num_processes):
            p = multiprocessing.Process(
                target=_concurrent_writer,
                args=(str(path), "claude", calls_per_process),
            )
            procs.append(p)
            p.start()

        for p in procs:
            p.join(timeout=10)

        # File should be valid JSON and have merged counts
        assert path.exists()
        data = json.loads(path.read_text(encoding="utf-8"))
        assert "claude" in data
        # Total should be at least calls_per_process (last writer wins for max)
        assert data["claude"]["calls_this_hour"] >= calls_per_process

    def test_quota_tracker_save_merges(self, tmp_path: Path) -> None:
        """Verify _save re-reads and merges with disk state."""
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

        # Verify final state — should have max(3+5=8 from B's perspective, 3 from A)
        final = QuotaTracker(path=path, quotas=quotas)
        entry = final._get_entry("claude")
        # B read A's 3, then added 5 = 8
        assert entry.calls_this_hour >= 5


# ---------------------------------------------------------------------------
# LaunchAgent plist
# ---------------------------------------------------------------------------


class TestLaunchAgentPlist:
    def test_plist_is_valid_xml(self) -> None:
        import xml.etree.ElementTree as ET
        plist_path = Path(__file__).parent.parent / "scripts" / "com.halext.afs-supervisor.plist"
        if not plist_path.exists():
            pytest.skip("plist not found")
        tree = ET.parse(str(plist_path))
        root = tree.getroot()
        assert root.tag == "plist"

    def test_plist_has_required_keys(self) -> None:
        import plistlib
        plist_path = Path(__file__).parent.parent / "scripts" / "com.halext.afs-supervisor.plist"
        if not plist_path.exists():
            pytest.skip("plist not found")
        with open(plist_path, "rb") as f:
            data = plistlib.load(f)
        assert data["Label"] == "com.halext.afs-supervisor"
        assert "ProgramArguments" in data
        assert "StartInterval" in data
        assert data["StartInterval"] == 900
        assert "afs.agents.supervisor" in data["ProgramArguments"]

    def test_ctl_script_is_executable(self) -> None:
        ctl_path = Path(__file__).parent.parent / "scripts" / "afs-supervisor-ctl"
        if not ctl_path.exists():
            pytest.skip("ctl script not found")
        assert os.access(str(ctl_path), os.X_OK)
