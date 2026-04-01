"""Tests for the dashboard-export agent."""

from __future__ import annotations

import json
from pathlib import Path

from afs.agents import dashboard_export as agent


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")


def _populate_sources(tmp_path: Path) -> tuple[Path, Path, Path]:
    """Create realistic source data files and return (agent_root, scratchpad, missions)."""
    agent_root = tmp_path / "agents"
    scratchpad = tmp_path / "scratchpad"
    missions = tmp_path / "missions"
    agent_root.mkdir()
    scratchpad.mkdir()
    missions.mkdir()

    # Supervisor audit via agent_supervisor.json
    _write_json(scratchpad / "agent_supervisor.json", {
        "payload": {
            "audit": {
                "counts": {
                    "running": 6,
                    "failed": 1,
                    "stopped": 1,
                    "configured": 8,
                },
                "stale_pid_files": ["context-audit"],
                "agents": [],
            }
        }
    })

    # Workspace health
    _write_json(scratchpad / "workspace_health.json", {
        "summary": {
            "total_repos": 49,
            "dirty": 30,
            "stale": 15,
            "ok": 4,
            "total_uncommitted": 260,
        }
    })

    # Quota
    _write_json(agent_root / "quota.json", {
        "claude": {
            "provider": "claude",
            "calls_today": 50,
            "limit": 100,
            "used": 50,
        },
        "local": {
            "provider": "local",
            "calls_today": 88,
        },
    })

    # Approvals
    _write_json(agent_root / "approvals.json", [
        {"agent": "mission-runner", "status": "pending", "action": "deploy"},
        {"agent": "context-warm", "status": "approved", "action": "repair"},
    ])

    # Missions (TOML files)
    (missions / "workspace-health.toml").write_text(
        '[mission]\nname = "workspace-health"\nstatus = "pending"\n',
        encoding="utf-8",
    )
    (missions / "zelda-research.toml").write_text(
        '[mission]\nname = "zelda-research"\nstatus = "completed"\n',
        encoding="utf-8",
    )
    (missions / "broken-build.toml").write_text(
        '[mission]\nname = "broken-build"\nstatus = "blocked"\n',
        encoding="utf-8",
    )

    return agent_root, scratchpad, missions


# ---------------------------------------------------------------------------
# JSON export format
# ---------------------------------------------------------------------------

def test_dashboard_json_format(tmp_path: Path) -> None:
    """Verify the dashboard JSON contains all required top-level keys."""
    agent_root, scratchpad, missions = _populate_sources(tmp_path)
    dashboard = agent.build_dashboard(
        agent_output_root=agent_root,
        scratchpad_root=scratchpad,
        missions_root=missions,
    )

    assert "timestamp" in dashboard
    assert isinstance(dashboard["timestamp"], str)

    # Agents section
    agents = dashboard["agents"]
    assert agents["running"] == 6
    assert agents["failed"] == 1
    assert agents["total"] == 8

    # Workspace section
    ws = dashboard["workspace"]
    assert ws["repos"] == 49
    assert ws["dirty"] == 30
    assert ws["stale"] == 15
    assert ws["uncommitted_files"] == 260

    # Quota section
    assert "claude" in dashboard["quota"]
    assert dashboard["quota"]["claude"]["used"] == 50
    assert dashboard["quota"]["claude"]["limit"] == 100

    # Approvals
    assert dashboard["approvals_pending"] == 1  # only 1 is "pending"

    # Missions
    assert dashboard["missions"]["active"] == 1
    assert dashboard["missions"]["completed"] == 1
    assert dashboard["missions"]["blocked"] == 1

    # Alerts
    assert isinstance(dashboard["alerts"], list)
    # With 1 failed agent, 30 dirty repos, 260 uncommitted files, 1 approval
    assert any("failed" in a for a in dashboard["alerts"])
    assert any("uncommitted" in a for a in dashboard["alerts"])
    assert any("approval" in a for a in dashboard["alerts"])


# ---------------------------------------------------------------------------
# Status line format
# ---------------------------------------------------------------------------

def test_status_line_format(tmp_path: Path) -> None:
    """Verify the one-line status string matches expected barista format."""
    agent_root, scratchpad, missions = _populate_sources(tmp_path)
    dashboard = agent.build_dashboard(
        agent_output_root=agent_root,
        scratchpad_root=scratchpad,
        missions_root=missions,
    )

    status = agent.format_status_line(dashboard)

    assert "agents:6/8" in status
    assert "repos:30dirty" in status
    assert "quota:ok" in status
    assert "approvals:1" in status
    assert " | " in status


def test_status_line_quota_warn() -> None:
    """Verify quota warning surfaces in the status line."""
    dashboard = {
        "agents": {"running": 3, "total": 3, "failed": 0},
        "workspace": {"repos": 10, "dirty": 2, "stale": 1, "uncommitted_files": 5},
        "quota": {"claude": {"used": 90, "limit": 100}},
        "approvals_pending": 0,
        "missions": {"active": 0, "completed": 0, "blocked": 0},
        "alerts": [],
    }
    status = agent.format_status_line(dashboard)
    assert "quota:warn" in status


def test_status_line_quota_over() -> None:
    """Verify quota over surfaces in the status line."""
    dashboard = {
        "agents": {"running": 3, "total": 3, "failed": 0},
        "workspace": {"repos": 10, "dirty": 2, "stale": 1, "uncommitted_files": 5},
        "quota": {"claude": {"used": 100, "limit": 100}},
        "approvals_pending": 0,
        "missions": {"active": 0, "completed": 0, "blocked": 0},
        "alerts": [],
    }
    status = agent.format_status_line(dashboard)
    assert "quota:over" in status


# ---------------------------------------------------------------------------
# Graceful defaults with missing data files
# ---------------------------------------------------------------------------

def test_missing_data_graceful_defaults(tmp_path: Path) -> None:
    """When source data files are missing, the dashboard should use safe defaults."""
    empty_agents = tmp_path / "agents"
    empty_scratchpad = tmp_path / "scratchpad"
    empty_missions = tmp_path / "missions"
    empty_agents.mkdir()
    empty_scratchpad.mkdir()
    # missions directory intentionally not created

    dashboard = agent.build_dashboard(
        agent_output_root=empty_agents,
        scratchpad_root=empty_scratchpad,
        missions_root=empty_missions,
    )

    assert dashboard["agents"]["running"] == 0
    assert dashboard["agents"]["failed"] == 0
    assert dashboard["agents"]["total"] == 0
    assert dashboard["workspace"]["repos"] == 0
    assert dashboard["workspace"]["dirty"] == 0
    assert dashboard["workspace"]["uncommitted_files"] == 0
    assert dashboard["quota"] == {}
    assert dashboard["approvals_pending"] == 0
    assert dashboard["missions"]["active"] == 0
    assert dashboard["alerts"] == []


def test_missing_all_directories(tmp_path: Path) -> None:
    """Even when directories do not exist at all, no exception is raised."""
    nonexistent = tmp_path / "nope"
    dashboard = agent.build_dashboard(
        agent_output_root=nonexistent / "agents",
        scratchpad_root=nonexistent / "scratchpad",
        missions_root=nonexistent / "missions",
    )

    assert isinstance(dashboard["timestamp"], str)
    assert dashboard["agents"]["total"] == 0
    assert dashboard["alerts"] == []


# ---------------------------------------------------------------------------
# File export
# ---------------------------------------------------------------------------

def test_export_writes_files(tmp_path: Path) -> None:
    """Verify export_dashboard writes both JSON and status files."""
    agent_root, scratchpad, missions = _populate_sources(tmp_path)
    dashboard = agent.build_dashboard(
        agent_output_root=agent_root,
        scratchpad_root=scratchpad,
        missions_root=missions,
    )

    dp = tmp_path / "output" / "dashboard.json"
    sp = tmp_path / "output" / "status.txt"

    agent.export_dashboard(dashboard, dashboard_path=dp, status_path=sp)

    assert dp.exists()
    assert sp.exists()

    loaded = json.loads(dp.read_text(encoding="utf-8"))
    assert loaded["agents"]["running"] == 6

    status_text = sp.read_text(encoding="utf-8").strip()
    assert "agents:6/8" in status_text


# ---------------------------------------------------------------------------
# CLI main() exits 0
# ---------------------------------------------------------------------------

def test_main_exits_zero(tmp_path: Path) -> None:
    """Verify main() runs to completion and exits 0."""
    agent_root, scratchpad, missions = _populate_sources(tmp_path)
    dp = tmp_path / "out" / "dashboard.json"
    sp = tmp_path / "out" / "status.txt"

    rc = agent.main([
        "--quiet",
        "--agent-output-root", str(agent_root),
        "--scratchpad-root", str(scratchpad),
        "--missions-root", str(missions),
        "--dashboard-path", str(dp),
        "--status-path", str(sp),
    ])

    assert rc == 0
    assert dp.exists()
    assert sp.exists()

    loaded = json.loads(dp.read_text(encoding="utf-8"))
    assert "timestamp" in loaded
    assert loaded["agents"]["total"] == 8


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------

def test_malformed_json_files(tmp_path: Path) -> None:
    """Malformed JSON files should not crash the dashboard builder."""
    agent_root = tmp_path / "agents"
    scratchpad = tmp_path / "scratchpad"
    agent_root.mkdir()
    scratchpad.mkdir()

    (agent_root / "quota.json").write_text("{bad json", encoding="utf-8")
    (agent_root / "approvals.json").write_text("not json at all", encoding="utf-8")
    (scratchpad / "workspace_health.json").write_text("", encoding="utf-8")

    dashboard = agent.build_dashboard(
        agent_output_root=agent_root,
        scratchpad_root=scratchpad,
        missions_root=tmp_path / "no_missions",
    )

    assert dashboard["quota"] == {}
    assert dashboard["approvals_pending"] == 0
    assert dashboard["workspace"]["repos"] == 0


def test_alerts_dirty_repos_threshold(tmp_path: Path) -> None:
    """Dirty repo alert should only fire when count exceeds threshold (>5)."""
    agent_root = tmp_path / "agents"
    scratchpad = tmp_path / "scratchpad"
    agent_root.mkdir()
    scratchpad.mkdir()

    # 3 dirty repos => no alert
    _write_json(scratchpad / "workspace_health.json", {
        "summary": {"total_repos": 10, "dirty": 3, "stale": 0, "total_uncommitted": 5}
    })
    dashboard = agent.build_dashboard(
        agent_output_root=agent_root,
        scratchpad_root=scratchpad,
        missions_root=tmp_path / "m",
    )
    assert not any("uncommitted" in a.lower() for a in dashboard["alerts"])

    # 60 uncommitted files => alert
    _write_json(scratchpad / "workspace_health.json", {
        "summary": {"total_repos": 10, "dirty": 3, "stale": 0, "total_uncommitted": 60}
    })
    dashboard2 = agent.build_dashboard(
        agent_output_root=agent_root,
        scratchpad_root=scratchpad,
        missions_root=tmp_path / "m",
    )
    assert any("uncommitted" in a.lower() for a in dashboard2["alerts"])
