"""Tests for AgentSupervisor lifecycle management."""

from __future__ import annotations

import signal
from pathlib import Path
from unittest.mock import patch

from afs.agents.supervisor import AgentSupervisor
from afs.schema import AFSConfig, AgentConfig, GeneralConfig


def test_supervisor_list_empty(tmp_path: Path) -> None:
    supervisor = AgentSupervisor(state_dir=tmp_path / "state")
    assert supervisor.list_running() == []


def test_supervisor_status_unknown(tmp_path: Path) -> None:
    supervisor = AgentSupervisor(state_dir=tmp_path / "state")
    assert supervisor.status("nonexistent") is None


def test_supervisor_stop_unknown(tmp_path: Path) -> None:
    supervisor = AgentSupervisor(state_dir=tmp_path / "state")
    assert supervisor.stop("nonexistent") is False


def test_supervisor_spawn_and_status(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    supervisor = AgentSupervisor(state_dir=state_dir)

    # Mock Popen to avoid actually launching a process
    mock_proc = type("MockProc", (), {"pid": 99999})()
    with patch("subprocess.Popen", return_value=mock_proc):
        agent = supervisor.spawn("test-agent", "afs.agents.context_warm")

    assert agent.name == "test-agent"
    assert agent.pid == 99999
    assert agent.state == "running"
    assert agent.module == "afs.agents.context_warm"

    # State file should exist
    assert (state_dir / "test-agent.json").exists()

    # Status should reflect running (mock pid_alive)
    with patch.object(supervisor, "_pid_alive", return_value=True):
        status = supervisor.status("test-agent")
    assert status is not None
    assert status.state == "running"


def test_supervisor_spawn_and_stop(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    supervisor = AgentSupervisor(state_dir=state_dir)

    mock_proc = type("MockProc", (), {"pid": 88888})()
    with patch("subprocess.Popen", return_value=mock_proc):
        supervisor.spawn("stop-test", "some.module")

    def kill_side_effect(pid, sig):
        if sig == 0 and kill_side_effect.alive:
            return None  # pid_alive: process exists
        if sig == signal.SIGTERM:
            kill_side_effect.alive = False
            return None  # SIGTERM sent
        raise ProcessLookupError("no such process")

    kill_side_effect.alive = True
    with patch("os.kill", side_effect=kill_side_effect):
        stopped = supervisor.stop("stop-test")

    assert stopped is True
    status = supervisor.status("stop-test")
    assert status is not None
    assert status.state == "stopped"
    assert status.manually_stopped is True


def test_supervisor_list_detects_dead_pid(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    supervisor = AgentSupervisor(state_dir=state_dir)

    mock_proc = type("MockProc", (), {"pid": 77777})()
    with patch("subprocess.Popen", return_value=mock_proc):
        supervisor.spawn("dead-agent", "some.module")

    # Simulate dead PID — list_agents refreshes state, marking it failed
    with patch.object(supervisor, "_pid_alive", return_value=False):
        agents = supervisor.list_agents()

    failed = [a for a in agents if a.state == "failed"]
    assert len(failed) == 1
    assert failed[0].name == "dead-agent"
    # list_running should be empty since dead agents aren't "running"
    with patch.object(supervisor, "_pid_alive", return_value=False):
        running = supervisor.list_running()
    assert len(running) == 0


def test_supervisor_auto_start(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    supervisor = AgentSupervisor(state_dir=state_dir)

    configs = [
        AgentConfig(name="auto1", auto_start=True, module="afs.agents.context_warm"),
        AgentConfig(name="skip1", auto_start=False, module="afs.agents.context_warm"),
        AgentConfig(name="no-module", auto_start=True, module=""),
    ]

    mock_proc = type("MockProc", (), {"pid": 11111})()
    with patch("subprocess.Popen", return_value=mock_proc):
        started = supervisor.auto_start(configs)

    assert len(started) == 1
    assert started[0].name == "auto1"


def test_supervisor_evaluate_triggers(tmp_path: Path) -> None:
    supervisor = AgentSupervisor(state_dir=tmp_path / "state")
    configs = [
        AgentConfig(name="a1", triggers=["on_mount", "on_profile_switch"]),
        AgentConfig(name="a2", triggers=["file_watch"]),
        AgentConfig(name="a3", triggers=[]),
    ]
    matched = supervisor.evaluate_triggers_from("on_mount", configs)
    assert len(matched) == 1
    assert matched[0].name == "a1"


def test_supervisor_state_persistence(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"

    # Create supervisor and spawn
    sup1 = AgentSupervisor(state_dir=state_dir)
    mock_proc = type("MockProc", (), {"pid": 55555})()
    with patch("subprocess.Popen", return_value=mock_proc):
        sup1.spawn("persist-test", "some.module")

    # New supervisor instance reads state from disk
    sup2 = AgentSupervisor(state_dir=state_dir)
    with patch.object(sup2, "_pid_alive", return_value=True):
        status = sup2.status("persist-test")
    assert status is not None
    assert status.pid == 55555
    assert status.state == "running"


def test_supervisor_uses_context_scoped_state_dir(tmp_path: Path) -> None:
    context_root = tmp_path / "context"
    supervisor = AgentSupervisor(
        config=AFSConfig(general=GeneralConfig(context_root=context_root))
    )
    assert supervisor._state_dir == context_root / "scratchpad" / "afs_agents" / "supervisor"


def test_supervisor_due_schedules(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    supervisor = AgentSupervisor(state_dir=state_dir)
    configs = [
        AgentConfig(name="scheduled", module="pkg.agent", schedule="5m"),
        AgentConfig(name="disabled", module="pkg.agent", schedule=""),
    ]
    matched = supervisor.due_schedules(configs)
    assert [config.name for config in matched] == ["scheduled"]


def test_supervisor_watch_path_matching(tmp_path: Path) -> None:
    supervisor = AgentSupervisor(state_dir=tmp_path / "state")
    watch_path = tmp_path / "workspace"
    watch_path.mkdir()
    configs = [
        AgentConfig(name="watcher", module="pkg.agent", watch_paths=[watch_path]),
        AgentConfig(name="other", module="pkg.agent", watch_paths=[tmp_path / "other"]),
    ]
    matched = supervisor.evaluate_watch_paths([watch_path / "changed.txt"], configs)
    assert [config.name for config in matched] == ["watcher"]


def test_supervisor_audit_reports_failed_and_manual_stop(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    supervisor = AgentSupervisor(state_dir=state_dir)

    failed = type("MockProc", (), {"pid": 12345})()
    stopped = type("MockProc", (), {"pid": 23456})()
    with patch("subprocess.Popen", side_effect=[failed, stopped]):
        supervisor.spawn("failed-agent", "pkg.failed")
        supervisor.spawn("stopped-agent", "pkg.stopped")

    with patch.object(supervisor, "_pid_alive", return_value=False):
        supervisor.status("failed-agent")
    with patch.object(supervisor, "_pid_alive", return_value=True):
        supervisor.stop("stopped-agent")
    audit = supervisor.audit()

    assert audit["counts"]["failed"] == 1
    assert audit["counts"]["manual_stop"] == 1
    assert "failed-agent" in audit["stale_pid_files"]
