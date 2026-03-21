"""Shared event-log helpers for CLI, MCP, and health surfaces."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Iterable
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .context_paths import resolve_mount_root
from .history import iter_history_events
from .models import MountType


def read_agent_events(
    context_path: Path,
    *,
    agent_name: str,
    limit: int = 20,
    config: Any = None,
) -> list[dict[str, Any]]:
    """Read recent history events emitted by a named agent."""
    history_root = resolve_mount_root(context_path, MountType.HISTORY, config=config)
    prefix = f"agent.{agent_name}"
    events = [
        event
        for event in iter_history_events(
            history_root,
            include_payloads=False,
        )
        if str(event.get("source", "")).startswith(prefix)
    ]
    if limit > 0 and len(events) > limit:
        events = events[-limit:]
    return events


def summarize_mcp_tool_usage(
    context_path: Path,
    *,
    tool_names: Iterable[str],
    lookback_hours: int = 24,
    config: Any = None,
) -> dict[str, Any]:
    """Summarize recent MCP tool usage for a context."""
    history_root = resolve_mount_root(context_path, MountType.HISTORY, config=config)
    tracked = {
        str(name).strip(): {
            "count": 0,
            "last_used": None,
            "used_recently": False,
        }
        for name in tool_names
        if str(name).strip()
    }
    if not tracked:
        return {"lookback_hours": lookback_hours, "tools": {}, "proactive": False}

    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(hours=max(1, lookback_hours))

    for event in iter_history_events(
        history_root,
        event_types={"mcp_tool"},
        include_payloads=False,
    ):
        metadata = event.get("metadata")
        if not isinstance(metadata, dict):
            continue
        tool_name = str(metadata.get("tool_name", "")).strip()
        if tool_name not in tracked:
            continue
        timestamp = str(event.get("timestamp", "")).strip()
        tracked[tool_name]["count"] += 1
        if not tracked[tool_name]["last_used"] or timestamp > tracked[tool_name]["last_used"]:
            tracked[tool_name]["last_used"] = timestamp
        parsed = _parse_timestamp(timestamp)
        if parsed is not None and parsed >= cutoff:
            tracked[tool_name]["used_recently"] = True

    proactive = bool(
        tracked.get("afs.session.bootstrap", {}).get("used_recently")
        or (
            tracked.get("context.status", {}).get("used_recently")
            and tracked.get("context.diff", {}).get("used_recently")
        )
    )
    return {
        "lookback_hours": lookback_hours,
        "tools": tracked,
        "proactive": proactive,
    }


def build_session_replay(
    context_path: Path,
    *,
    session_id: str,
    limit: int = 200,
    include_payloads: bool = False,
    config: Any = None,
) -> dict[str, Any]:
    """Reconstruct the event timeline for a recorded AFS session."""
    history_root = resolve_mount_root(context_path, MountType.HISTORY, config=config)
    events: list[dict[str, Any]] = []
    for event in iter_history_events(
        history_root,
        include_payloads=include_payloads,
    ):
        metadata = event.get("metadata")
        if not isinstance(metadata, dict):
            continue
        if str(metadata.get("session_id", "")).strip() != session_id:
            continue
        events.append(event)

    total_events = len(events)
    truncated = 0
    if limit > 0 and total_events > limit:
        truncated = total_events - limit
        events = events[-limit:]

    event_types = Counter(str(event.get("type", "")) for event in events)
    sources = Counter(str(event.get("source", "")) for event in events)
    started_at = events[0]["timestamp"] if events else None
    ended_at = events[-1]["timestamp"] if events else None
    return {
        "session_id": session_id,
        "count": len(events),
        "total_events": total_events,
        "truncated": truncated,
        "started_at": started_at,
        "ended_at": ended_at,
        "event_types": dict(sorted(event_types.items())),
        "sources": dict(sorted(sources.items())),
        "events": events,
    }


def list_recorded_sessions(
    context_path: Path,
    *,
    config: Any = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Return recorded sessions grouped by explicit ``metadata.session_id``."""
    history_root = resolve_mount_root(context_path, MountType.HISTORY, config=config)
    sessions_by_id: dict[str, dict[str, Any]] = {}

    for event in iter_history_events(history_root, include_payloads=False):
        metadata = event.get("metadata") if isinstance(event.get("metadata"), dict) else {}
        session_id = str(metadata.get("session_id", "")).strip()
        timestamp = str(event.get("timestamp", "")).strip()
        if not session_id or not timestamp:
            continue
        session = sessions_by_id.setdefault(
            session_id,
            {
                "session_id": session_id,
                "start": timestamp,
                "end": timestamp,
                "event_count": 0,
                "event_types": set(),
            },
        )
        session["start"] = min(str(session["start"]), timestamp)
        session["end"] = max(str(session["end"]), timestamp)
        session["event_count"] = int(session["event_count"]) + 1
        session["event_types"].add(str(event.get("type", "")).strip())

    sessions: list[dict[str, Any]] = []
    for payload in sorted(
        sessions_by_id.values(),
        key=lambda item: str(item["end"]),
        reverse=True,
    ):
        sessions.append(
            {
                "session_id": str(payload["session_id"]),
                "start": str(payload["start"]),
                "end": str(payload["end"]),
                "event_count": int(payload["event_count"]),
                "event_types": sorted(
                    event_type
                    for event_type in payload["event_types"]
                    if event_type
                ),
            }
        )
        if len(sessions) >= limit:
            break

    return sessions


def summarize_event_analytics(
    context_path: Path,
    *,
    lookback_hours: int = 24,
    event_types: Iterable[str] | None = None,
    config: Any = None,
) -> dict[str, Any]:
    """Summarize recent event volume, MCP usage, and error rates."""
    history_root = resolve_mount_root(context_path, MountType.HISTORY, config=config)
    normalized_types = {
        str(item).strip()
        for item in (event_types or [])
        if str(item).strip()
    }
    cutoff = datetime.now(timezone.utc) - timedelta(hours=max(1, lookback_hours))

    total_events = 0
    oldest: str | None = None
    newest: str | None = None
    type_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    op_counts: Counter[str] = Counter()
    session_ids: set[str] = set()
    tool_metrics: dict[str, dict[str, Any]] = defaultdict(
        lambda: {
            "count": 0,
            "errors": 0,
            "last_used": None,
            "avg_duration_ms": None,
            "_duration_total": 0,
            "_duration_count": 0,
        }
    )

    for event in iter_history_events(history_root, include_payloads=False):
        timestamp = _parse_timestamp(str(event.get("timestamp", "")))
        if timestamp < cutoff:
            continue
        event_type = str(event.get("type", "")).strip()
        if normalized_types and event_type not in normalized_types:
            continue
        source = str(event.get("source", "")).strip()
        op = str(event.get("op", "")).strip()
        metadata = event.get("metadata") if isinstance(event.get("metadata"), dict) else {}

        total_events += 1
        if oldest is None or str(event.get("timestamp", "")) < oldest:
            oldest = str(event.get("timestamp", ""))
        if newest is None or str(event.get("timestamp", "")) > newest:
            newest = str(event.get("timestamp", ""))
        type_counts[event_type] += 1
        if source:
            source_counts[source] += 1
        if op:
            op_counts[f"{source}:{op}" if source else op] += 1
        session_id = str(metadata.get("session_id", "")).strip()
        if session_id:
            session_ids.add(session_id)

        if event_type == "mcp_tool":
            tool_name = str(metadata.get("tool_name", "")).strip() or "(unknown)"
            metrics = tool_metrics[tool_name]
            metrics["count"] += 1
            metrics["last_used"] = max(
                filter(
                    None,
                    [metrics.get("last_used"), str(event.get("timestamp", ""))],
                )
            )
            if metadata.get("ok") is False or metadata.get("error"):
                metrics["errors"] += 1
            duration_raw = metadata.get("duration_ms")
            if isinstance(duration_raw, (int, float)):
                metrics["_duration_total"] += int(duration_raw)
                metrics["_duration_count"] += 1

    tools_summary: dict[str, Any] = {}
    for name, metrics in sorted(tool_metrics.items()):
        duration_count = int(metrics.pop("_duration_count"))
        duration_total = int(metrics.pop("_duration_total"))
        metrics["avg_duration_ms"] = (
            round(duration_total / duration_count, 2) if duration_count else None
        )
        metrics["error_rate"] = (
            round(metrics["errors"] / metrics["count"], 3) if metrics["count"] else 0.0
        )
        tools_summary[name] = metrics

    return {
        "lookback_hours": max(1, lookback_hours),
        "cutoff": cutoff.isoformat(),
        "total_events": total_events,
        "time_range": {"oldest": oldest, "newest": newest},
        "event_types": dict(sorted(type_counts.items())),
        "sources": dict(sorted(source_counts.items())),
        "ops": dict(sorted(op_counts.items())),
        "sessions": {
            "count": len(session_ids),
            "latest": None,
        },
        "mcp_tools": tools_summary,
    }


def list_sessions(
    context_path: Path,
    *,
    config: Any = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Group events by date and return session summaries."""
    history_root = resolve_mount_root(context_path, MountType.HISTORY, config=config)
    events_by_date: dict[str, list[dict[str, Any]]] = {}

    for event in iter_history_events(history_root, include_payloads=False):
        timestamp = str(event.get("timestamp", "")).strip()
        if not timestamp:
            continue
        date_key = timestamp[:10]  # YYYY-MM-DD
        events_by_date.setdefault(date_key, []).append(event)

    sessions: list[dict[str, Any]] = []
    for date_key in sorted(events_by_date.keys(), reverse=True):
        events = events_by_date[date_key]
        if not events:
            continue
        timestamps = [str(e.get("timestamp", "")) for e in events if e.get("timestamp")]
        event_types = list({str(e.get("type", "")) for e in events if e.get("type")})
        sessions.append({
            "session_id": date_key,
            "start": min(timestamps) if timestamps else "",
            "end": max(timestamps) if timestamps else "",
            "event_count": len(events),
            "event_types": sorted(event_types),
        })
        if len(sessions) >= limit:
            break

    return sessions


def build_session_timeline(
    context_path: Path,
    *,
    session_id: str | None = None,
    since: str | None = None,
    limit: int = 100,
    config: Any = None,
) -> dict[str, Any]:
    """Build a chronological timeline of events."""
    history_root = resolve_mount_root(context_path, MountType.HISTORY, config=config)
    events: list[dict[str, Any]] = []

    for event in iter_history_events(history_root, include_payloads=False):
        timestamp = str(event.get("timestamp", "")).strip()
        if not timestamp:
            continue
        metadata = event.get("metadata") if isinstance(event.get("metadata"), dict) else {}
        event_session_id = str(metadata.get("session_id", "")).strip()
        if session_id:
            if event_session_id:
                if event_session_id != session_id:
                    continue
            elif not timestamp.startswith(session_id):
                continue
        if since and timestamp < since:
            continue
        events.append(event)

    events.sort(key=lambda e: str(e.get("timestamp", "")))
    if limit and len(events) > limit:
        events = events[:limit]

    timeline: list[dict[str, Any]] = []
    for event in events:
        timeline.append({
            "timestamp": str(event.get("timestamp", "")),
            "type": str(event.get("type", "")),
            "op": str(event.get("op", "")),
            "source": str(event.get("source", "")),
            "id": str(event.get("id", "")),
            "summary": _describe_timeline_event(event),
        })

    return {
        "session_id": session_id,
        "since": since,
        "event_count": len(timeline),
        "timeline": timeline,
    }


def _describe_timeline_event(event: dict[str, Any]) -> str:
    """Human-readable one-liner per event type."""
    event_type = str(event.get("type", "")).strip()
    op = str(event.get("op", "")).strip()
    source = str(event.get("source", "")).strip()
    metadata = event.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    if event_type == "session" and op == "bootstrap":
        return "Session bootstrap"
    if event_type == "cli":
        argv = metadata.get("argv")
        if isinstance(argv, list):
            return f"CLI: {' '.join(str(a) for a in argv[:5])}"
        return f"CLI invocation ({op})"
    if event_type == "fs":
        mount_type = metadata.get("mount_type", "")
        rel = metadata.get("relative_path", "")
        return f"{op} {mount_type}/{rel}" if rel else f"fs {op}"
    if event_type == "context":
        mount_type = metadata.get("mount_type", "")
        alias = metadata.get("alias", "")
        if mount_type and alias:
            return f"{op} {mount_type}/{alias}"
        return f"context {op}"
    if event_type == "hivemind":
        agent = metadata.get("agent_name", source)
        return f"hivemind {op} by {agent}"
    if event_type == "agent_progress":
        agent = metadata.get("agent", "")
        detail = metadata.get("detail", "")
        return f"agent {agent} {op}: {detail}" if detail else f"agent {agent} {op}"
    if event_type == "mcp_tool":
        tool = metadata.get("tool_name", "")
        return f"MCP tool: {tool}"
    if event_type == "handoff":
        return f"Handoff {op}"

    parts = [event_type, op, source]
    return " ".join(p for p in parts if p) or "unknown event"


def _parse_timestamp(raw: str) -> datetime | None:
    if not raw:
        return None
    value = raw.strip()
    if value.endswith("Z"):
        value = value[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
