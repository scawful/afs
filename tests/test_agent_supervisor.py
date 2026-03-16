"""Tests for AgentSupervisor lifecycle management."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

from afs.agents.supervisor import AgentSupervisor
from afs.schema import AgentConfig


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

    with patch("os.kill") as mock_kill:
        mock_kill.side_effect = [None, None]  # pid_alive check + SIGTERM
        stopped = supervisor.stop("stop-test")

    assert stopped is True
    assert not (state_dir / "stop-test.json").exists()


def test_supervisor_list_detects_dead_pid(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    supervisor = AgentSupervisor(state_dir=state_dir)

    mock_proc = type("MockProc", (), {"pid": 77777})()
    with patch("subprocess.Popen", return_value=mock_proc):
        supervisor.spawn("dead-agent", "some.module")

    # Simulate dead PID
    with patch.object(supervisor, "_pid_alive", return_value=False):
        agents = supervisor.list_running()

    assert len(agents) == 1
    assert agents[0].state == "failed"


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
