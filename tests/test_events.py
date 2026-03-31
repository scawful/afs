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
    log_event,
    log_hivemind_event,
    log_mcp_tool_call,
    log_session_event,
    query_events,
)


def test_query_events_returns_recent(tmp_path: Path) -> None:
    history_root = tmp_path / "history"
    history_root.mkdir()
    append_history_event(history_root, "cli", "afs.cli", op="invoke", metadata={"argv": ["status"]})
    append_history_event(
        history_root,
        "mcp_tool",
        "afs.mcp",
        op="call",
        metadata={"tool_name": "context.read"},
    )
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


def test_query_events_filters_by_session_id(tmp_path: Path) -> None:
    history_root = tmp_path / "history"
    history_root.mkdir()
    append_history_event(
        history_root,
        "session",
        "afs.session",
        op="bootstrap",
        metadata={"session_id": "session-a"},
    )
    append_history_event(
        history_root,
        "mcp_tool",
        "afs.mcp",
        op="call",
        metadata={"tool_name": "context.status", "session_id": "session-b"},
    )
    events = query_events(history_root, session_id="session-b", limit=10)
    assert len(events) == 1
    assert events[0]["metadata"]["session_id"] == "session-b"


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


def test_log_event_injects_env_session_id(tmp_path: Path, monkeypatch) -> None:
    context_root = tmp_path / ".context"
    history_root = context_root / "history"
    history_root.mkdir(parents=True)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("AFS_SESSION_ID", "session-env")

    event_id = log_event(
        "cli",
        "afs.cli",
        op="invoke",
        metadata={"argv": ["status"]},
        context_root=context_root,
    )

    assert event_id is not None
    events = query_events(history_root, session_id="session-env", limit=10)
    assert len(events) == 1
    assert events[0]["metadata"]["session_id"] == "session-env"


def test_log_session_event_can_include_payload(tmp_path: Path) -> None:
    history_root = tmp_path / "history"
    history_root.mkdir()

    append_history_event(
        history_root,
        "session",
        "afs.session",
        op="user_prompt_submit",
        metadata={"session_id": "session-a"},
        payload={"prompt": "Investigate monitor sidecar."},
        include_payloads=True,
    )

    events = query_events(history_root, session_id="session-a", limit=10)
    assert events[0]["payload"]["prompt"] == "Investigate monitor sidecar."
