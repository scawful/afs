"""Tests for structured event log (Feature 3)."""

from __future__ import annotations

from pathlib import Path

from afs.history import (
    EVENT_EMBEDDING,
    EVENT_HIVEMIND,
    EVENT_MCP_TOOL,
    EVENT_SESSION,
    append_history_event,
    log_embedding_event,
    log_hivemind_event,
    log_mcp_tool_call,
    log_session_event,
    query_events,
)


def test_query_events_returns_recent(tmp_path: Path) -> None:
    history_root = tmp_path / "history"
    history_root.mkdir()
    append_history_event(history_root, "cli", "afs.cli", op="invoke", metadata={"argv": ["status"]})
    append_history_event(history_root, "mcp_tool", "afs.mcp", op="call", metadata={"tool_name": "fs.read"})
    events = query_events(history_root, limit=10)
    assert len(events) == 2
    assert events[0]["type"] == "cli"
    assert events[1]["type"] == "mcp_tool"


def test_query_events_filters_by_type(tmp_path: Path) -> None:
    history_root = tmp_path / "history"
    history_root.mkdir()
    append_history_event(history_root, "cli", "afs.cli", op="invoke")
    append_history_event(history_root, "mcp_tool", "afs.mcp", op="call")
    events = query_events(history_root, event_types={"mcp_tool"}, limit=10)
    assert len(events) == 1
    assert events[0]["type"] == "mcp_tool"


def test_query_events_filters_by_source(tmp_path: Path) -> None:
    history_root = tmp_path / "history"
    history_root.mkdir()
    append_history_event(history_root, "cli", "afs.cli", op="invoke")
    append_history_event(history_root, "mcp_tool", "afs.mcp", op="call")
    events = query_events(history_root, source="afs.mcp", limit=10)
    assert len(events) == 1
    assert events[0]["source"] == "afs.mcp"


def test_query_events_respects_limit(tmp_path: Path) -> None:
    history_root = tmp_path / "history"
    history_root.mkdir()
    for i in range(10):
        append_history_event(history_root, "cli", "afs.cli", op="invoke", metadata={"i": i})
    events = query_events(history_root, limit=3)
    assert len(events) == 3


def test_event_type_constants() -> None:
    assert EVENT_MCP_TOOL == "mcp_tool"
    assert EVENT_HIVEMIND == "hivemind"
    assert EVENT_EMBEDDING == "embedding"
    assert EVENT_SESSION == "session"


def test_convenience_wrappers_return_none_when_disabled(monkeypatch) -> None:
    monkeypatch.setenv("AFS_HISTORY_DISABLED", "1")
    assert log_mcp_tool_call("test", {}, {}, duration_ms=10) is None
    assert log_hivemind_event("send", "agent-a") is None
    assert log_embedding_event("index_build") is None
    assert log_session_event("bootstrap") is None
