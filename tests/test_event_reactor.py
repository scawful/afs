"""Tests for the event reactor and on_event supervisor triggers."""

from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

from afs.agent_defaults import default_agent_configs
from afs.agents.event_reactor import (
    ReactorEvent,
    collect_new_events,
    load_cursor,
    match_event_rules,
    pattern_matches,
)
from afs.agents.supervisor import AgentSupervisor
from afs.schema import AFSConfig, AgentConfig, GeneralConfig

NOW = datetime(2026, 7, 15, 12, 0, tzinfo=timezone.utc)


def _event(kind: str, detail: str = "", *, offset_seconds: int = 0) -> ReactorEvent:
    stamp = (NOW + timedelta(seconds=offset_seconds)).isoformat()
    return ReactorEvent(kind=kind, detail=detail, timestamp=stamp)


def _write_history_event(
    context_path: Path,
    *,
    event_type: str,
    op: str,
    timestamp: datetime,
) -> None:
    history = context_path / "history"
    history.mkdir(parents=True, exist_ok=True)
    log = history / f"events_{timestamp:%Y%m%d}.jsonl"
    record = {
        "id": "abc123",
        "timestamp": timestamp.isoformat(),
        "type": event_type,
        "op": op,
        "source": "test",
        "metadata": {},
    }
    with log.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


# ---------------------------------------------------------------------------
# Pattern matching
# ---------------------------------------------------------------------------


def test_pattern_matches_kind_and_detail() -> None:
    event = _event("mcp_tool", "call")
    assert pattern_matches("mcp_tool", event)
    assert pattern_matches("mcp_tool:call", event)
    assert pattern_matches("mcp_tool:*", event)
    assert pattern_matches("*", event)
    assert not pattern_matches("mcp_tool:error", event)
    assert not pattern_matches("agent_lifecycle", event)
    assert not pattern_matches("", event)


def test_pattern_matches_hivemind_topic_with_colon() -> None:
    event = _event("hivemind", "context:repair")
    assert pattern_matches("hivemind:context:repair", event)
    assert pattern_matches("hivemind:context:*", event)
    assert not pattern_matches("hivemind:other-topic", event)


def test_match_event_rules_reports_first_match() -> None:
    config = AgentConfig(name="reactor", module="x.y", on_event=["error", "hivemind:context:*"])
    quiet = AgentConfig(name="quiet", module="x.y")
    events = [_event("mcp_tool", "call"), _event("hivemind", "context:repair")]
    matched = match_event_rules(events, [config, quiet])
    assert [(cfg.name, reason) for cfg, reason in matched] == [
        ("reactor", "event:hivemind:context:repair"),
    ]


# ---------------------------------------------------------------------------
# Cursor + collection
# ---------------------------------------------------------------------------


def test_collect_primes_cursor_without_replaying_history(tmp_path: Path) -> None:
    context = tmp_path / "context"
    state = tmp_path / "state"
    _write_history_event(
        context, event_type="error", op="boom", timestamp=NOW - timedelta(hours=2)
    )

    first = collect_new_events(context, state, now=NOW)
    assert first == []
    assert load_cursor(state) == NOW.isoformat()

    _write_history_event(
        context, event_type="error", op="boom", timestamp=NOW + timedelta(seconds=30)
    )
    second = collect_new_events(context, state, now=NOW + timedelta(seconds=60))
    assert [event.label() for event in second] == ["error:boom"]
    # Cursor advanced to the newest seen event, not to "now".
    assert load_cursor(state) == (NOW + timedelta(seconds=30)).isoformat()

    third = collect_new_events(context, state, now=NOW + timedelta(seconds=120))
    assert third == []


def test_collect_normalizes_hivemind_messages(tmp_path: Path) -> None:
    from afs.hivemind import HivemindBus

    context = tmp_path / "context"
    (context / "hivemind").mkdir(parents=True)
    state = tmp_path / "state"
    collect_new_events(context, state, now=NOW - timedelta(seconds=60))

    HivemindBus(context).send(
        "afs-watch",
        "status",
        {"changed_count": 3},
        topic="context:repair",
    )
    events = collect_new_events(context, state, now=NOW + timedelta(days=1))
    labels = [event.label() for event in events]
    assert "hivemind:context:repair" in labels


# ---------------------------------------------------------------------------
# Supervisor integration
# ---------------------------------------------------------------------------


def _supervisor(tmp_path: Path) -> AgentSupervisor:
    return AgentSupervisor(
        state_dir=tmp_path / "supervisor-state",
        config=AFSConfig(general=GeneralConfig(context_root=tmp_path / "context")),
    )


def test_evaluate_event_records_applies_debounce(tmp_path: Path, monkeypatch) -> None:
    supervisor = _supervisor(tmp_path)
    config = AgentConfig(name="reactor", module="x.y", on_event=["error"])
    events = [_event("error", "boom")]

    assert [
        cfg.name for cfg, _ in supervisor.evaluate_event_records(events, [config], now=NOW)
    ] == ["reactor"]

    class _Stub:
        started_at = (NOW - timedelta(seconds=30)).isoformat()

    monkeypatch.setattr(supervisor, "status", lambda name: _Stub())
    assert supervisor.evaluate_event_records(events, [config], now=NOW) == []

    # Custom debounce window shorter than the elapsed time allows a restart.
    config_fast = AgentConfig(
        name="reactor", module="x.y", on_event=["error"], event_debounce="10s"
    )
    assert [
        cfg.name
        for cfg, _ in supervisor.evaluate_event_records(events, [config_fast], now=NOW)
    ] == ["reactor"]


def test_reconcile_spawns_for_event_records(tmp_path: Path, monkeypatch) -> None:
    supervisor = _supervisor(tmp_path)
    spawned: list[tuple[str, str]] = []

    def _fake_spawn(name, module, args=None, *, reason="", agent_config=None):
        spawned.append((name, reason))
        raise RuntimeError("stop after recording")

    monkeypatch.setattr(supervisor, "spawn", _fake_spawn)
    config = AgentConfig(name="reactor", module="x.y", on_event=["hivemind:context:*"])
    supervisor.reconcile(
        [config],
        event_records=[_event("hivemind", "context:repair")],
        now=NOW,
    )
    assert spawned == [("reactor", "event:hivemind:context:repair")]


def test_reconcile_skips_job_action_configs(tmp_path: Path, monkeypatch) -> None:
    supervisor = _supervisor(tmp_path)
    spawned: list[str] = []
    monkeypatch.setattr(
        supervisor,
        "spawn",
        lambda name, module, args=None, *, reason="", agent_config=None: spawned.append(name),
    )
    config = AgentConfig(
        name="jobber", module="x.y", on_event=["error"], on_event_action="job"
    )
    supervisor.reconcile([config], event_records=[_event("error")], now=NOW)
    assert spawned == []


def test_enqueue_event_jobs_dedupes_while_queued(tmp_path: Path) -> None:
    from afs.agent_jobs import AgentJobQueue

    supervisor = _supervisor(tmp_path)
    config = AgentConfig(
        name="jobber", module="x.y", on_event=["error"], on_event_action="job"
    )
    events = [_event("error", "boom")]

    created = supervisor.enqueue_event_jobs(events, [config], now=NOW)
    assert len(created) == 1
    assert supervisor.enqueue_event_jobs(events, [config], now=NOW) == []

    queue = AgentJobQueue((tmp_path / "context"))
    jobs = queue.list(status="queue")
    assert len(jobs) == 1
    assert jobs[0].dedupe_key == "on_event:jobber"
    assert "event:error:boom" in jobs[0].title


def test_default_index_rebuild_reacts_to_watch_topic(tmp_path: Path) -> None:
    config = AFSConfig(general=GeneralConfig(context_root=tmp_path / "context"))
    defaults = {agent.name: agent for agent in default_agent_configs(config)}
    assert defaults["index-rebuild"].on_event == ["hivemind:context:repair"]

    supervisor = _supervisor(tmp_path)
    matched = supervisor.evaluate_event_records(
        [_event("hivemind", "context:repair")],
        list(defaults.values()),
        now=NOW,
    )
    assert [cfg.name for cfg, _ in matched] == ["index-rebuild"]
