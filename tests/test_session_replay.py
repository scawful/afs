"""Tests for session replay."""

from __future__ import annotations

import json

import pytest

from afs.event_log import (
    _describe_timeline_event,
    build_session_timeline,
    list_sessions,
)


@pytest.fixture
def history_context(tmp_path, monkeypatch):
    monkeypatch.setenv("AFS_ALLOWED_MOUNTS", "memory,history,scratchpad,hivemind,knowledge,tools,global,items,monorepo")
    context = tmp_path / ".context"
    context.mkdir()
    history = context / "history"
    history.mkdir()
    return context


def _write_events(context, events):
    """Write events to history as JSONL.

    Groups events by date stamp (YYYYMMDD) into separate files matching
    the ``events_*.jsonl`` glob pattern used by ``iter_history_events``.
    """
    history = context / "history"
    by_date: dict[str, list[dict]] = {}
    for event in events:
        ts = str(event.get("timestamp", ""))
        date_key = ts[:10].replace("-", "") if len(ts) >= 10 else "00000000"
        by_date.setdefault(date_key, []).append(event)
    for date_key, date_events in by_date.items():
        events_file = history / f"events_{date_key}.jsonl"
        with events_file.open("a", encoding="utf-8") as f:
            for event in date_events:
                f.write(json.dumps(event) + "\n")


def test_list_sessions_groups_by_date(history_context):
    _write_events(history_context, [
        {"id": "1", "timestamp": "2024-06-01T10:00:00Z", "type": "cli", "op": "invoke"},
        {"id": "2", "timestamp": "2024-06-01T11:00:00Z", "type": "fs", "op": "write"},
        {"id": "3", "timestamp": "2024-06-02T09:00:00Z", "type": "context", "op": "mount"},
    ])
    sessions = list_sessions(history_context)
    assert len(sessions) == 2
    dates = [s["session_id"] for s in sessions]
    assert "2024-06-02" in dates
    assert "2024-06-01" in dates
    # Check the first is most recent
    assert sessions[0]["session_id"] == "2024-06-02"


def test_build_timeline_chronological(history_context):
    _write_events(history_context, [
        {"id": "2", "timestamp": "2024-06-01T11:00:00Z", "type": "fs", "op": "write"},
        {"id": "1", "timestamp": "2024-06-01T10:00:00Z", "type": "cli", "op": "invoke"},
    ])
    result = build_session_timeline(history_context)
    assert result["event_count"] == 2
    assert result["timeline"][0]["timestamp"] < result["timeline"][1]["timestamp"]


def test_build_timeline_filters_by_session_id(history_context):
    _write_events(history_context, [
        {"id": "1", "timestamp": "2024-06-01T10:00:00Z", "type": "cli", "op": "invoke"},
        {"id": "2", "timestamp": "2024-06-02T09:00:00Z", "type": "context", "op": "mount"},
    ])
    result = build_session_timeline(history_context, session_id="2024-06-01")
    assert result["event_count"] == 1
    assert result["timeline"][0]["id"] == "1"


def test_build_timeline_empty_history(history_context):
    result = build_session_timeline(history_context)
    assert result["event_count"] == 0
    assert result["timeline"] == []


def test_describe_timeline_event_formats():
    # CLI event
    event = {"type": "cli", "op": "invoke", "source": "", "metadata": {"argv": ["status", "--json"]}}
    desc = _describe_timeline_event(event)
    assert "CLI" in desc
    assert "status" in desc

    # Session bootstrap
    event2 = {"type": "session", "op": "bootstrap", "source": "", "metadata": {}}
    assert "bootstrap" in _describe_timeline_event(event2).lower()

    # FS event
    event3 = {"type": "fs", "op": "write", "source": "", "metadata": {"mount_type": "knowledge", "relative_path": "doc.md"}}
    desc3 = _describe_timeline_event(event3)
    assert "knowledge" in desc3
    assert "doc.md" in desc3

    # Agent progress
    event4 = {"type": "agent_progress", "op": "started", "source": "", "metadata": {"agent": "context-warm", "detail": "audit"}}
    desc4 = _describe_timeline_event(event4)
    assert "context-warm" in desc4

    # Unknown
    event5 = {"type": "", "op": "", "source": "", "metadata": {}}
    assert _describe_timeline_event(event5) == "unknown event"
