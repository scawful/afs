"""Shared event-log helpers for CLI, MCP, and health surfaces."""

from __future__ import annotations

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
