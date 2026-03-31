from __future__ import annotations

from pathlib import Path

from afs.event_log import (
    build_session_replay,
    build_session_timeline,
    list_sessions,
    list_recorded_sessions,
    summarize_event_analytics,
)
from afs.history import append_history_event


def _make_context(tmp_path: Path) -> tuple[Path, Path]:
    context_path = tmp_path / ".context"
    history_root = context_path / "history"
    history_root.mkdir(parents=True)
    return context_path, history_root


def test_summarize_event_analytics_reports_mcp_metrics(tmp_path: Path) -> None:
    context_path, history_root = _make_context(tmp_path)
    append_history_event(
        history_root,
        "mcp_tool",
        "afs.mcp",
        op="call",
        metadata={
            "tool_name": "context.status",
            "duration_ms": 10,
            "ok": True,
            "session_id": "session-a",
        },
    )
    append_history_event(
        history_root,
        "mcp_tool",
        "afs.mcp",
        op="call",
        metadata={
            "tool_name": "context.status",
            "duration_ms": 30,
            "ok": False,
            "error": "boom",
            "session_id": "session-b",
        },
    )

    summary = summarize_event_analytics(context_path, lookback_hours=24)

    assert summary["total_events"] == 2
    assert summary["event_types"]["mcp_tool"] == 2
    assert summary["sessions"]["count"] == 2
    assert summary["mcp_tools"]["context.status"]["count"] == 2
    assert summary["mcp_tools"]["context.status"]["errors"] == 1
    assert summary["mcp_tools"]["context.status"]["avg_duration_ms"] == 20.0


def test_build_session_replay_returns_matching_timeline(tmp_path: Path) -> None:
    context_path, history_root = _make_context(tmp_path)
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
        metadata={"tool_name": "context.query", "session_id": "session-a"},
        payload={"ok": True},
        include_payloads=True,
    )
    append_history_event(
        history_root,
        "session",
        "afs.session",
        op="bootstrap",
        metadata={"session_id": "session-b"},
    )

    replay = build_session_replay(context_path, session_id="session-a", include_payloads=True)

    assert replay["session_id"] == "session-a"
    assert replay["count"] == 2
    assert replay["event_types"]["session"] == 1
    assert replay["event_types"]["mcp_tool"] == 1
    assert replay["events"][1]["payload"]["ok"] is True


def test_list_recorded_sessions_groups_by_metadata_session_id(tmp_path: Path) -> None:
    context_path, history_root = _make_context(tmp_path)
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
        metadata={"tool_name": "context.query", "session_id": "session-a"},
    )
    append_history_event(
        history_root,
        "session",
        "afs.session",
        op="bootstrap",
        metadata={"session_id": "session-b"},
    )

    sessions = list_recorded_sessions(context_path)

    assert [session["session_id"] for session in sessions] == ["session-b", "session-a"]
    assert sessions[1]["event_count"] == 2


def test_build_session_timeline_filters_by_explicit_session_id_metadata(tmp_path: Path) -> None:
    context_path, history_root = _make_context(tmp_path)
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
        metadata={"tool_name": "context.query", "session_id": "session-a"},
    )
    append_history_event(
        history_root,
        "mcp_tool",
        "afs.mcp",
        op="call",
        metadata={"tool_name": "context.status", "session_id": "session-b"},
    )

    timeline = build_session_timeline(context_path, session_id="session-a")

    assert timeline["event_count"] == 2
    assert all(event["id"] for event in timeline["timeline"])


def test_build_session_timeline_describes_session_activity_events(tmp_path: Path) -> None:
    context_path, history_root = _make_context(tmp_path)
    append_history_event(
        history_root,
        "session",
        "afs.session",
        op="user_prompt_submit",
        metadata={
            "session_id": "session-a",
            "prompt_preview": "Investigate monitor sidecar.",
        },
    )
    append_history_event(
        history_root,
        "session",
        "afs.session",
        op="task_completed",
        metadata={
            "session_id": "session-a",
            "task_id": "task-1",
        },
    )

    timeline = build_session_timeline(context_path, session_id="session-a")

    assert timeline["event_count"] == 2
    assert timeline["timeline"][0]["summary"] == "Prompt submitted: Investigate monitor sidecar."
    assert timeline["timeline"][1]["summary"] == "Task task-1 completed"


def test_list_sessions_prefers_explicit_recorded_session_ids(tmp_path: Path) -> None:
    context_path, history_root = _make_context(tmp_path)
    append_history_event(
        history_root,
        "session",
        "afs.session",
        op="session_start",
        metadata={"session_id": "session-a"},
    )
    append_history_event(
        history_root,
        "session",
        "afs.session",
        op="session_end",
        metadata={"session_id": "session-a"},
    )

    sessions = list_sessions(context_path)

    assert sessions[0]["session_id"] == "session-a"
    assert sessions[0]["event_count"] == 2
