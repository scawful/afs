from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from afs.agents.supervisor import RunningAgent
from afs.cli.core import agents_monitor_command, agents_wait_command
from afs.schema import AFSConfig, GeneralConfig


def test_agents_wait_filters_to_session_agents(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    context_root = tmp_path / "context"
    context_root.mkdir()
    manager = SimpleNamespace(config=AFSConfig(general=GeneralConfig(context_root=context_root)))

    class FakeSupervisor:
        session_calls = 0

        def __init__(self, *args, **kwargs) -> None:
            pass

        def list_agents(self):
            return [
                RunningAgent(name="session-agent", state="running", session_id="sess-1"),
                RunningAgent(name="other-agent", state="running", session_id="sess-2"),
            ]

        def status(self, name: str):
            if name == "session-agent":
                FakeSupervisor.session_calls += 1
                if FakeSupervisor.session_calls == 1:
                    return RunningAgent(name=name, state="running", session_id="sess-1")
                return RunningAgent(name=name, state="stopped", session_id="sess-1")
            return RunningAgent(name=name, state="running", session_id="sess-2")

    monkeypatch.setattr("afs.cli.core.load_manager", lambda _config_path=None: manager)
    monkeypatch.setattr("afs.cli.core._resolve_command_context", lambda _args: context_root)
    monkeypatch.setattr("afs.cli.core.time.sleep", lambda _seconds: None)
    monkeypatch.setattr("afs.cli.core.read_agent_events", lambda *args, **kwargs: [])

    args = Namespace(
        name=None,
        all=True,
        session_id="sess-1",
        timeout=1.0,
        poll_interval=0.01,
        limit=10,
        json=False,
        config=None,
        path=None,
        context_root=None,
        context_dir=None,
    )

    with patch("afs.agents.supervisor.AgentSupervisor", FakeSupervisor):
        rc = agents_wait_command(args)

    captured = capsys.readouterr()
    assert rc == 0
    assert "waiting for 1 agent(s) session=sess-1: session-agent" in captured.out
    assert "session-agent" in captured.out
    assert "other-agent" not in captured.out
    assert "state=stopped" in captured.out


def test_agents_monitor_streams_session_events(
    monkeypatch,
    tmp_path: Path,
    capsys,
) -> None:
    context_root = tmp_path / "context"
    history_root = context_root / "history"
    history_root.mkdir(parents=True)
    manager = SimpleNamespace(config=AFSConfig(general=GeneralConfig(context_root=context_root)))

    timestamps = iter([0.0, 0.005, 0.02])
    calls = {"count": 0}

    def fake_query_events(*args, **kwargs):
        calls["count"] += 1
        if calls["count"] == 1:
            return [
                {
                    "id": "evt-1",
                    "timestamp": "2026-03-31T21:40:00+00:00",
                    "type": "agent_lifecycle",
                    "source": "afs.agents",
                    "op": "spawned",
                    "metadata": {
                        "agent_name": "session-agent",
                        "session_id": "sess-1",
                    },
                }
            ]
        return []

    monkeypatch.setattr("afs.cli.core.load_manager", lambda _config_path=None: manager)
    monkeypatch.setattr("afs.cli.core._resolve_command_context", lambda _args: context_root)
    monkeypatch.setattr("afs.cli.core.query_events", fake_query_events)
    monkeypatch.setattr("afs.cli.core.time.sleep", lambda _seconds: None)
    monkeypatch.setattr("afs.cli.core.time.monotonic", lambda: next(timestamps))

    args = Namespace(
        name=None,
        all=False,
        session_id="sess-1",
        timeout=0.01,
        poll_interval=0.01,
        json=False,
        config=None,
        path=None,
        context_root=None,
        context_dir=None,
    )

    rc = agents_monitor_command(args)

    captured = capsys.readouterr()
    assert rc == 0
    assert "monitoring background agent events session=sess-1" in captured.out
    assert "session-agent  spawned" in captured.out
