"""Tests for review gate integration with supervisor."""

from __future__ import annotations

import signal
from pathlib import Path
from unittest.mock import patch

from afs.agents.supervisor import AgentSupervisor
from afs.schema import AgentConfig


def test_supervisor_awaiting_review_state(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    supervisor = AgentSupervisor(state_dir=state_dir)

    mock_proc = type("MockProc", (), {"pid": 99999})()
    with patch("afs.agents.supervisor.subprocess.Popen", return_value=mock_proc):
        supervisor.spawn("review-agent", "some.module")

    with (
        patch("afs.agents.supervisor.os.kill") as kill,
        patch.object(supervisor, "_pid_alive", side_effect=[True, False]),
    ):
        result = supervisor.set_awaiting_review("review-agent")
    assert result is True
    kill.assert_called_once_with(99999, signal.SIGTERM)

    status = supervisor.status("review-agent")
    assert status is not None
    assert status.state == "awaiting_review"
    assert status.pid is None  # PID cleared when entering review


def test_supervisor_approve_review(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    supervisor = AgentSupervisor(state_dir=state_dir)

    mock_proc = type("MockProc", (), {"pid": 99999})()
    with patch("afs.agents.supervisor.subprocess.Popen", return_value=mock_proc):
        supervisor.spawn("review-agent", "some.module")

    with (
        patch("afs.agents.supervisor.os.kill"),
        patch.object(supervisor, "_pid_alive", side_effect=[True, False]),
    ):
        supervisor.set_awaiting_review("review-agent")
    approved = supervisor.approve_review("review-agent")
    assert approved is True

    status = supervisor.status("review-agent")
    assert status is not None
    assert status.state == "stopped"
    assert status.last_event == "review_approved"


def test_supervisor_reject_review(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    supervisor = AgentSupervisor(state_dir=state_dir)

    mock_proc = type("MockProc", (), {"pid": 99999})()
    with patch("afs.agents.supervisor.subprocess.Popen", return_value=mock_proc):
        supervisor.spawn("review-agent", "some.module")

    with (
        patch("afs.agents.supervisor.os.kill"),
        patch.object(supervisor, "_pid_alive", side_effect=[True, False]),
    ):
        supervisor.set_awaiting_review("review-agent")
    rejected = supervisor.reject_review("review-agent")
    assert rejected is True

    status = supervisor.status("review-agent")
    assert status is not None
    assert status.state == "failed"
    assert status.last_error == "review_rejected"


def test_approve_non_review_agent_fails(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    supervisor = AgentSupervisor(state_dir=state_dir)

    mock_proc = type("MockProc", (), {"pid": 99999})()
    with patch("afs.agents.supervisor.subprocess.Popen", return_value=mock_proc):
        supervisor.spawn("running-agent", "some.module")

    # Agent is running (or failed if PID dead) — either way not awaiting_review
    approved = supervisor.approve_review("running-agent")
    assert approved is False


def test_reconcile_skips_awaiting_review(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    supervisor = AgentSupervisor(state_dir=state_dir)

    mock_proc = type("MockProc", (), {"pid": 99999})()
    with patch("afs.agents.supervisor.subprocess.Popen", return_value=mock_proc):
        supervisor.spawn("review-agent", "some.module")

    with (
        patch("afs.agents.supervisor.os.kill"),
        patch.object(supervisor, "_pid_alive", side_effect=[True, False]),
    ):
        supervisor.set_awaiting_review("review-agent")
    configs = [
        AgentConfig(name="review-agent", auto_start=True, module="some.module"),
    ]
    with patch("afs.agents.supervisor.subprocess.Popen", return_value=mock_proc):
        started = supervisor.reconcile(configs)

    # Should not restart an agent that's awaiting review
    assert len(started) == 0


def test_auto_start_passes_agent_scope_env(tmp_path: Path) -> None:
    state_dir = tmp_path / "state"
    supervisor = AgentSupervisor(state_dir=state_dir)
    captured: dict[str, object] = {}

    def _fake_popen(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["env"] = kwargs.get("env")
        return type("MockProc", (), {"pid": 12345})()

    config = AgentConfig(
        name="scoped-agent",
        auto_start=True,
        module="some.module",
        allowed_mounts=["scratchpad"],
        allowed_tools=["task.list", "task.create"],
        workspace_isolated=True,
    )

    with patch("afs.agents.supervisor.subprocess.Popen", side_effect=_fake_popen):
        started = supervisor.auto_start([config])

    assert len(started) == 1
    env = captured["env"]
    assert isinstance(env, dict)
    assert env["AFS_ALLOWED_MOUNTS"] == "scratchpad"
    assert env["AFS_ALLOWED_TOOLS"] == "task.list,task.create"
    assert env["AFS_WORKSPACE_ISOLATED"] == "1"
    assert env["AFS_PREFER_REPO_CONFIG"] == "1"
    assert env["AFS_PREFER_USER_CONFIG"] == "0"
