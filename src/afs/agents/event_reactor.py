"""React to new event-log entries and hivemind messages.

The supervisor's reconcile loop calls :func:`collect_new_events` each cycle to
read history events and hivemind messages that arrived since the stored
cursor, then routes matches against ``AgentConfig.on_event`` patterns into its
normal spawn path (or the agent-jobs queue for ``on_event_action = "job"``).

Pattern grammar: ``"<kind>"`` or ``"<kind>:<detail>"``, both sides fnmatch
globs. ``kind`` is the history event ``type`` (``mcp_tool``, ``error``,
``agent_lifecycle``, ...) or the literal ``hivemind``; ``detail`` is the
history event ``op`` or the hivemind topic. Only the first ``:`` splits, so
``hivemind:context:repair`` matches topic ``context:repair``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

from ..history import iter_history_events
from ..context_paths import resolve_mount_root
from ..models import MountType
from ..schema import AgentConfig

_log = logging.getLogger(__name__)

CURSOR_FILE_NAME = "event_reactor_cursor.json"
DEFAULT_EVENT_DEBOUNCE_SECONDS = 300.0
HIVEMIND_KIND = "hivemind"


@dataclass(frozen=True)
class ReactorEvent:
    kind: str
    detail: str = ""
    source: str = ""
    timestamp: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def label(self) -> str:
        return f"{self.kind}:{self.detail}" if self.detail else self.kind


def pattern_matches(pattern: str, event: ReactorEvent) -> bool:
    """Return whether one on_event pattern matches a reactor event."""
    text = pattern.strip()
    if not text:
        return False
    kind_pattern, _, detail_pattern = text.partition(":")
    if not fnmatchcase(event.kind, kind_pattern or "*"):
        return False
    return not detail_pattern or fnmatchcase(event.detail, detail_pattern)


def match_event_rules(
    events: list[ReactorEvent],
    agent_configs: list[AgentConfig],
) -> list[tuple[AgentConfig, str]]:
    """Return (config, reason) for each config whose on_event patterns match.

    One entry per config, keyed to the first matching event so the launch
    reason names what actually fired.
    """
    matched: list[tuple[AgentConfig, str]] = []
    for config in agent_configs:
        if not config.on_event:
            continue
        for event in events:
            if any(pattern_matches(pattern, event) for pattern in config.on_event):
                matched.append((config, f"event:{event.label()}"))
                break
    return matched


def _parse_timestamp(raw: str) -> datetime | None:
    text = (raw or "").strip()
    if not text:
        return None
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def load_cursor(state_dir: Path) -> str:
    try:
        payload = json.loads((state_dir / CURSOR_FILE_NAME).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ""
    value = payload.get("cursor", "") if isinstance(payload, dict) else ""
    return value if isinstance(value, str) else ""


def save_cursor(state_dir: Path, cursor: str) -> None:
    try:
        state_dir.mkdir(parents=True, exist_ok=True)
        (state_dir / CURSOR_FILE_NAME).write_text(
            json.dumps({"cursor": cursor}) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:  # cursor loss degrades to re-priming, never crashes
        _log.warning("Could not persist event reactor cursor: %s", exc)


def _history_events_since(
    context_path: Path,
    cursor: datetime,
    *,
    config: Any = None,
) -> list[ReactorEvent]:
    try:
        history_root = resolve_mount_root(context_path, MountType.HISTORY, config=config)
    except Exception:  # noqa: BLE001 - missing mounts must not break reconcile
        return []
    events: list[ReactorEvent] = []
    for event in iter_history_events(history_root, include_payloads=False):
        stamp = _parse_timestamp(str(event.get("timestamp", "")))
        if stamp is None or stamp <= cursor:
            continue
        metadata = event.get("metadata")
        events.append(
            ReactorEvent(
                kind=str(event.get("type", "")),
                detail=str(event.get("op", "") or ""),
                source=str(event.get("source", "")),
                timestamp=str(event.get("timestamp", "")),
                metadata=metadata if isinstance(metadata, dict) else {},
            )
        )
    return events


def _hivemind_events_since(
    context_path: Path,
    cursor: datetime,
    *,
    config: Any = None,
) -> list[ReactorEvent]:
    try:
        from ..hivemind import HivemindBus

        messages = HivemindBus(context_path, config=config).read(since=cursor, limit=200)
    except Exception:  # noqa: BLE001 - hivemind is optional signal, not a dependency
        return []
    events: list[ReactorEvent] = []
    for message in messages:
        stamp = _parse_timestamp(message.timestamp)
        if stamp is None or stamp <= cursor:
            continue
        events.append(
            ReactorEvent(
                kind=HIVEMIND_KIND,
                detail=str(message.topic or message.msg_type or ""),
                source=message.from_agent,
                timestamp=message.timestamp,
                metadata={"msg_type": message.msg_type, "id": message.id},
            )
        )
    return events


def collect_new_events(
    context_path: Path,
    state_dir: Path,
    *,
    config: Any = None,
    now: datetime | None = None,
) -> list[ReactorEvent]:
    """Return events newer than the stored cursor, advancing it.

    The first call primes the cursor to *now* and returns nothing, so enabling
    the reactor never replays weeks of history as a spawn storm.
    """
    current = now or datetime.now(timezone.utc)
    stored = _parse_timestamp(load_cursor(state_dir))
    if stored is None:
        save_cursor(state_dir, current.isoformat())
        return []

    events = _history_events_since(context_path, stored, config=config)
    events.extend(_hivemind_events_since(context_path, stored, config=config))
    events.sort(key=lambda event: event.timestamp)

    if events:
        # Advance only to the newest event actually seen: anything written
        # mid-read with an earlier stamp than `current` stays in the next
        # window instead of being skipped forever.
        newest = stored
        for event in events:
            stamp = _parse_timestamp(event.timestamp)
            if stamp is not None and stamp > newest:
                newest = stamp
    else:
        newest = current
    save_cursor(state_dir, newest.isoformat())
    return events
