"""Tests for mission runner agent — TOML loading, OODA phases, guardrails integration."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from afs.agents.mission_runner import (
    AGENT_NAME,
    Mission,
    MissionGuardrails,
    MissionPhase,
    _discover_missions,
    _execute_phase,
    _load_mission,
    _run_mission,
    _run_phase_tools,
    main,
)
from afs.agents.guardrails import GuardrailConfig, GuardrailedAgent


@pytest.fixture(autouse=True)
def _isolated_agent_registry(tmp_path: Path, monkeypatch):
    monkeypatch.setenv("AFS_AGENT_REGISTRY_PATH", str(tmp_path / "agent_registry.json"))
    monkeypatch.setenv("HOME", str(tmp_path))


@pytest.fixture
def context_root(tmp_path: Path) -> Path:
    root = tmp_path / ".context"
    missions_dir = root / "scratchpad" / "missions"
    missions_dir.mkdir(parents=True)
    return root


@pytest.fixture
def sample_mission_toml() -> str:
    return """\
[mission]
name = "test-mission"
description = "A test mission"
tier = "background"
owner = "test-agent"
status = "pending"

[mission.guardrails]
require_worktree = false
require_approval_for = ["git_push"]
max_iterations = 10
dry_run = false

[[mission.phases]]
name = "observe"
description = "Gather data"
tools = ["context.read"]
outputs = ["observations.json"]

[[mission.phases]]
name = "orient"
description = "Analyze findings"
tools = ["context.query"]
outputs = ["analysis.json"]

[[mission.phases]]
name = "decide"
description = "Determine actions"
tools = []
outputs = ["decisions.json"]
requires_approval = true

[[mission.phases]]
name = "act"
description = "Execute plan"
tools = ["context.write"]
outputs = ["results.json"]
"""


# ---------------------------------------------------------------------------
# Mission loading
# ---------------------------------------------------------------------------


class TestLoadMission:
    def test_load_valid_mission(self, tmp_path: Path, sample_mission_toml: str) -> None:
        path = tmp_path / "test.toml"
        path.write_text(sample_mission_toml, encoding="utf-8")
        mission = _load_mission(path)
        assert mission is not None
        assert mission.name == "test-mission"
        assert mission.description == "A test mission"
        assert mission.tier == "background"
        assert mission.status == "pending"
        assert len(mission.phases) == 4

    def test_load_mission_phases(self, tmp_path: Path, sample_mission_toml: str) -> None:
        path = tmp_path / "test.toml"
        path.write_text(sample_mission_toml, encoding="utf-8")
        mission = _load_mission(path)
        assert mission is not None
        assert mission.phases[0].name == "observe"
        assert mission.phases[0].tools == ["context.read"]
        assert mission.phases[0].outputs == ["observations.json"]
        assert mission.phases[2].requires_approval is True
        assert mission.phases[3].requires_approval is False

    def test_load_mission_guardrails(self, tmp_path: Path, sample_mission_toml: str) -> None:
        path = tmp_path / "test.toml"
        path.write_text(sample_mission_toml, encoding="utf-8")
        mission = _load_mission(path)
        assert mission is not None
        assert mission.guardrails.require_worktree is False
        assert mission.guardrails.max_iterations == 10
        assert "git_push" in mission.guardrails.require_approval_for

    def test_load_invalid_toml(self, tmp_path: Path) -> None:
        path = tmp_path / "bad.toml"
        path.write_text("not { valid toml !!!", encoding="utf-8")
        mission = _load_mission(path)
        assert mission is None

    def test_load_missing_fields_uses_defaults(self, tmp_path: Path) -> None:
        path = tmp_path / "minimal.toml"
        path.write_text(
            '[mission]\nname = "minimal"\n\n[[mission.phases]]\nname = "observe"\n',
            encoding="utf-8",
        )
        mission = _load_mission(path)
        assert mission is not None
        assert mission.name == "minimal"
        assert mission.tier == "background"
        assert mission.guardrails.max_iterations == 50
        assert len(mission.phases) == 1

    def test_load_uses_filename_as_fallback_name(self, tmp_path: Path) -> None:
        path = tmp_path / "my-mission.toml"
        path.write_text("[mission]\n", encoding="utf-8")
        mission = _load_mission(path)
        assert mission is not None
        assert mission.name == "my-mission"


# ---------------------------------------------------------------------------
# Mission discovery
# ---------------------------------------------------------------------------


class TestDiscoverMissions:
    def test_discovers_pending_missions(self, context_root: Path, sample_mission_toml: str) -> None:
        missions_dir = context_root / "scratchpad" / "missions"
        (missions_dir / "m1.toml").write_text(sample_mission_toml, encoding="utf-8")
        missions = _discover_missions(context_root)
        assert len(missions) == 1
        assert missions[0][1].name == "test-mission"

    def test_skips_non_pending(self, context_root: Path) -> None:
        missions_dir = context_root / "scratchpad" / "missions"
        (missions_dir / "done.toml").write_text(
            '[mission]\nname = "done"\nstatus = "completed"\n',
            encoding="utf-8",
        )
        missions = _discover_missions(context_root)
        assert len(missions) == 0

    def test_skips_invalid_files(self, context_root: Path) -> None:
        missions_dir = context_root / "scratchpad" / "missions"
        (missions_dir / "bad.toml").write_text("not valid toml!!!", encoding="utf-8")
        (missions_dir / "readme.md").write_text("# not a toml", encoding="utf-8")
        missions = _discover_missions(context_root)
        assert len(missions) == 0

    def test_empty_directory(self, context_root: Path) -> None:
        missions = _discover_missions(context_root)
        assert len(missions) == 0

    def test_missing_directory(self, tmp_path: Path) -> None:
        missions = _discover_missions(tmp_path / "nonexistent")
        assert len(missions) == 0


# ---------------------------------------------------------------------------
# Phase execution
# ---------------------------------------------------------------------------


class TestPhaseExecution:
    def test_observe_phase_writes_output(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "output"
        phase = MissionPhase(
            name="observe",
            description="Gather data",
            tools=["context.read"],
            outputs=["observations.json"],
        )
        mission = Mission(name="test", tier="background")
        guard = GuardrailedAgent("test", GuardrailConfig(
            task_tier="background", enable_approvals=False,
        ))
        result = _execute_phase(phase, mission, guard, output_dir)
        assert result["status"] == "ok"
        assert result["phase"] == "observe"
        assert (output_dir / "observations.json").exists()

    def test_phase_requiring_approval_blocks(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "output"
        phase = MissionPhase(
            name="decide",
            description="Decide",
            requires_approval=True,
        )
        mission = Mission(name="test", tier="standard")
        guard = GuardrailedAgent("test", GuardrailConfig(enable_approvals=True))
        result = _execute_phase(phase, mission, guard, output_dir)
        assert result["status"] == "awaiting_approval"

    def test_phase_tracks_model(self, tmp_path: Path) -> None:
        output_dir = tmp_path / "output"
        phase = MissionPhase(name="observe", outputs=["out.json"])
        mission = Mission(name="test", tier="background")
        guard = GuardrailedAgent("test", GuardrailConfig(
            task_tier="background", enable_approvals=False,
        ))
        result = _execute_phase(phase, mission, guard, output_dir)
        assert "model" in result
        assert result["model"]["provider"] in ("claude", "gemini", "codex", "local")

    def test_observe_with_context_root(self, context_root: Path) -> None:
        """Observe phase should attempt AFS calls with a real context root."""
        output_dir = context_root / "scratchpad" / "missions" / "test-obs"
        phase = MissionPhase(name="observe", outputs=["observations.json"])
        mission = Mission(name="test", tier="background")
        guard = GuardrailedAgent("test", GuardrailConfig(
            task_tier="background", enable_approvals=False,
        ))
        result = _execute_phase(phase, mission, guard, output_dir, context_root)
        assert result["status"] == "ok"
        # Should have notes from the observe phase tools
        assert any("Context health" in n or "context_root" in n for n in result["notes"])

    def test_phase_tools_no_context_root(self, tmp_path: Path) -> None:
        """Phase tools should gracefully handle missing context root."""
        phase = MissionPhase(name="observe")
        mission = Mission(name="test")
        guard = GuardrailedAgent("test", GuardrailConfig(enable_approvals=False))
        result = _run_phase_tools(phase, mission, guard, tmp_path, context_root=None)
        assert "no context_root" in result["notes"][0]

    def test_orient_reads_observe_output(self, context_root: Path) -> None:
        """Orient phase should read observations from observe output."""
        output_dir = context_root / "scratchpad" / "missions" / "test-orient"
        output_dir.mkdir(parents=True, exist_ok=True)
        # Write fake observe output
        (output_dir / "observations.json").write_text(
            json.dumps({"data": {"index_diff": {"added": 5, "modified": 3, "deleted": 0}}}),
            encoding="utf-8",
        )
        phase = MissionPhase(name="orient", outputs=["analysis.json"])
        mission = Mission(name="test")
        guard = GuardrailedAgent("test", GuardrailConfig(enable_approvals=False))
        result = _run_phase_tools(phase, mission, guard, output_dir, context_root)
        # Should detect the index drift from observations
        findings = result.get("data", {}).get("findings", [])
        drift = [f for f in findings if f.get("type") == "index_drift"]
        assert len(drift) == 1
        assert drift[0]["detail"] == "8 files changed since last index"

    def test_decide_produces_actions(self, context_root: Path) -> None:
        """Decide phase should produce actions from orient findings."""
        output_dir = context_root / "scratchpad" / "missions" / "test-decide"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "analysis.json").write_text(json.dumps({
            "data": {
                "findings": [
                    {"type": "index_drift", "severity": "warning", "detail": "10 files changed"},
                    {"type": "missing_dirs", "severity": "warning", "detail": "Missing: tools"},
                ],
            },
        }), encoding="utf-8")
        phase = MissionPhase(name="decide", outputs=["decisions.json"])
        mission = Mission(name="test")
        guard = GuardrailedAgent("test", GuardrailConfig(enable_approvals=False))
        result = _run_phase_tools(phase, mission, guard, output_dir, context_root)
        actions = result.get("data", {}).get("actions", [])
        deferred = result.get("data", {}).get("deferred", [])
        assert len(actions) == 1
        assert actions[0]["action"] == "embedding_update"
        assert len(deferred) == 1
        assert deferred[0]["action"] == "context_repair"

    def test_act_skips_unapproved(self, context_root: Path) -> None:
        """Act phase should skip actions that need approval."""
        output_dir = context_root / "scratchpad" / "missions" / "test-act"
        output_dir.mkdir(parents=True, exist_ok=True)
        (output_dir / "decisions.json").write_text(json.dumps({
            "data": {"actions": [{"action": "git_push", "reason": "push changes"}]},
        }), encoding="utf-8")
        phase = MissionPhase(name="act", outputs=["results.json"])
        mission = Mission(name="test")
        guard = GuardrailedAgent("test", GuardrailConfig(enable_approvals=True))
        result = _run_phase_tools(phase, mission, guard, output_dir, context_root)
        skipped = result.get("data", {}).get("skipped", [])
        assert len(skipped) == 1
        assert skipped[0]["reason"] == "approval required"

    def test_unknown_phase_type_handled(self, context_root: Path) -> None:
        """Unknown phase types should not crash."""
        phase = MissionPhase(name="custom_phase")
        mission = Mission(name="test")
        guard = GuardrailedAgent("test", GuardrailConfig(enable_approvals=False))
        result = _run_phase_tools(phase, mission, guard, context_root, context_root)
        assert "Unknown phase type" in result["notes"][0]


# ---------------------------------------------------------------------------
# Full mission execution
# ---------------------------------------------------------------------------


class TestRunMission:
    def test_full_mission_no_approval_gates(
        self, context_root: Path, tmp_path: Path,
    ) -> None:
        mission_path = tmp_path / "test.toml"
        mission = Mission(
            name="simple-test",
            description="Simple test mission",
            tier="background",
            guardrails=MissionGuardrails(max_iterations=20),
            phases=[
                MissionPhase(name="observe", outputs=["obs.json"]),
                MissionPhase(name="orient", outputs=["orient.json"]),
                MissionPhase(name="act", outputs=["act.json"]),
            ],
        )
        result = _run_mission(mission_path, mission, context_root)
        assert result["status"] == "ok"
        assert len(result["phases"]) == 3
        assert all(p["status"] == "ok" for p in result["phases"])

    def test_mission_stops_on_approval_gate(
        self, context_root: Path, tmp_path: Path,
    ) -> None:
        mission_path = tmp_path / "test.toml"
        mission = Mission(
            name="gated-test",
            tier="standard",
            guardrails=MissionGuardrails(max_iterations=20),
            phases=[
                MissionPhase(name="observe", outputs=["obs.json"]),
                MissionPhase(name="decide", requires_approval=True, outputs=["decide.json"]),
                MissionPhase(name="act", outputs=["act.json"]),
            ],
        )
        result = _run_mission(mission_path, mission, context_root)
        assert result["status"] == "awaiting_approval"
        assert len(result["phases"]) == 2  # stopped after decide
        assert result["phases"][1]["status"] == "awaiting_approval"

    def test_mission_writes_result_file(
        self, context_root: Path, tmp_path: Path,
    ) -> None:
        mission_path = tmp_path / "test.toml"
        mission = Mission(
            name="output-test",
            tier="background",
            guardrails=MissionGuardrails(max_iterations=20),
            phases=[MissionPhase(name="observe", outputs=["obs.json"])],
        )
        result = _run_mission(mission_path, mission, context_root)
        result_file = context_root / "scratchpad" / "missions" / "output-test" / "result.json"
        assert result_file.exists()
        data = json.loads(result_file.read_text(encoding="utf-8"))
        assert data["mission"] == "output-test"

    def test_mission_includes_quota_usage(
        self, context_root: Path, tmp_path: Path,
    ) -> None:
        mission_path = tmp_path / "test.toml"
        mission = Mission(
            name="quota-test",
            tier="background",
            phases=[MissionPhase(name="observe", outputs=["obs.json"])],
        )
        result = _run_mission(mission_path, mission, context_root)
        assert "quota_usage" in result

    def test_mission_respects_iteration_cap(
        self, context_root: Path, tmp_path: Path,
    ) -> None:
        mission_path = tmp_path / "test.toml"
        # Create a mission with many phases but low iteration cap
        phases = [MissionPhase(name=f"phase_{i}", outputs=[f"out_{i}.json"]) for i in range(10)]
        mission = Mission(
            name="cap-test",
            tier="background",
            guardrails=MissionGuardrails(max_iterations=2),
            phases=phases,
        )
        result = _run_mission(mission_path, mission, context_root)
        assert result["status"] == "iteration_cap"
        # Should have executed 2 phases and skipped the rest
        executed = [p for p in result["phases"] if p.get("status") == "ok"]
        skipped = [p for p in result["phases"] if p.get("status") == "skipped"]
        assert len(executed) == 2
        assert len(skipped) > 0


# ---------------------------------------------------------------------------
# CLI main
# ---------------------------------------------------------------------------


class TestMissionRunnerCLI:
    def test_no_missions_exits_ok(self, context_root: Path) -> None:
        rc = main(["--context-root", str(context_root), "--stdout", "--quiet"])
        assert rc == 0

    def test_specific_mission(self, context_root: Path, sample_mission_toml: str) -> None:
        missions_dir = context_root / "scratchpad" / "missions"
        (missions_dir / "target.toml").write_text(sample_mission_toml, encoding="utf-8")
        rc = main([
            "--context-root", str(context_root),
            "--mission", "test-mission",
            "--stdout", "--quiet",
        ])
        assert rc == 0

    def test_nonexistent_mission_exits_ok(self, context_root: Path) -> None:
        rc = main([
            "--context-root", str(context_root),
            "--mission", "does-not-exist",
            "--stdout", "--quiet",
        ])
        assert rc == 0
