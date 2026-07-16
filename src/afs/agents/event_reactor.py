"""React to new event-log entries and hivemind messages.

The supervisor's reconcile loop opens an :func:`open_event_batch` each cycle:
events are read oldest-first from both sources under an exclusive lock, the
supervisor dispatches matches against ``AgentConfig.on_event`` patterns, and
only then acks the batch. Cursors never advance before dispatch, so a crash
mid-cycle redelivers instead of losing events (at-least-once delivery); the
persisted per-agent dispatch times make redelivery safe, and the lock closes
the read->dispatch->ack window against concurrent supervisors so a batch is
never double-dispatched.

Pattern grammar: ``"<kind>"`` or ``"<kind>:<detail>"``, both sides fnmatch
globs. ``kind`` is the history event ``type`` (``mcp_tool``, ``error``,
``agent_lifecycle``, ...) or the literal ``hivemind``; ``detail`` is the
history event ``op`` or the hivemind topic. Only the first ``:`` splits, so
``hivemind:context:repair`` matches topic ``context:repair``.
"""

from __future__ import annotations

import json
import logging
import re
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any, Iterator

from ..context_paths import resolve_mount_root
from ..history import EVENT_FILE_PREFIX
from ..models import MountType
from ..schema import AgentConfig
from .guardrails import _file_lock

_log = logging.getLogger(__name__)

CURSOR_FILE_NAME = "event_reactor_cursor.json"
STATE_VERSION = 2
DEFAULT_EVENT_DEBOUNCE_SECONDS = 300.0
HIVEMIND_KIND = "hivemind"
# Per-source per-cycle read bound. A backlog larger than this is drained
# across cycles (oldest first) instead of being dropped or read unbounded.
MAX_EVENTS_PER_CYCLE = 500
LOCK_TIMEOUT_SECONDS = 5.0

VALID_EVENT_ACTIONS = ("spawn", "job")

_LABEL_SAFE = re.compile(r"[^A-Za-z0-9_.:\-*]")
_LABEL_MAX = 80


class ReactorBusyError(RuntimeError):
    """Another process holds the reactor cursor lock for this context."""


@dataclass(frozen=True)
class ReactorEvent:
    kind: str
    detail: str = ""
    source: str = ""
    timestamp: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def label(self) -> str:
        return f"{self.kind}:{self.detail}" if self.detail else self.kind


def sanitize_label(label: str) -> str:
    """Make an event label safe to embed in prompts and launch reasons.

    Event kinds/details come from log and hivemind data an agent may have
    written; anything outside a conservative charset is replaced so event
    payloads cannot smuggle instructions into job prompts.
    """
    cleaned = _LABEL_SAFE.sub("_", label or "")
    return cleaned[:_LABEL_MAX]


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
    reason names what actually fired. Reasons carry sanitized labels only.
    """
    matched: list[tuple[AgentConfig, str]] = []
    for config in agent_configs:
        if not config.on_event:
            continue
        for event in events:
            if any(pattern_matches(pattern, event) for pattern in config.on_event):
                matched.append((config, f"event:{sanitize_label(event.label())}"))
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


# ---------------------------------------------------------------------------
# Reactor state: per-source cursors + per-agent dispatch times, one JSON file
# ---------------------------------------------------------------------------


def _state_path(state_dir: Path) -> Path:
    return state_dir / CURSOR_FILE_NAME


def _load_state(state_dir: Path) -> dict[str, Any]:
    try:
        payload = json.loads(_state_path(state_dir).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    if "version" not in payload and "cursor" in payload:
        # v1 state kept a single shared cursor; both sources start from it.
        legacy = payload.get("cursor")
        legacy = legacy if isinstance(legacy, str) else ""
        return {
            "version": STATE_VERSION,
            "history_cursor": legacy,
            "hivemind_cursor": legacy,
            "last_dispatch": {},
        }
    payload.setdefault("last_dispatch", {})
    if not isinstance(payload.get("last_dispatch"), dict):
        payload["last_dispatch"] = {}
    return payload


def _save_state(state_dir: Path, state: dict[str, Any]) -> None:
    try:
        state_dir.mkdir(parents=True, exist_ok=True)
        _state_path(state_dir).write_text(
            json.dumps({**state, "version": STATE_VERSION}) + "\n",
            encoding="utf-8",
        )
    except OSError as exc:  # state loss degrades to re-priming, never crashes
        _log.warning("Could not persist event reactor state: %s", exc)


# ---------------------------------------------------------------------------
# Bounded oldest-first source reads
# ---------------------------------------------------------------------------


def _cut_at_timestamp_boundary(
    events: list[ReactorEvent], limit: int
) -> tuple[list[ReactorEvent], bool]:
    """Truncate an oldest-first event list without splitting a timestamp group.

    Cursors are timestamps, so cutting in the middle of a group of events that
    share one timestamp would skip the rest of the group forever (the next
    cycle reads strictly-greater-than the cursor). The cut lands on the last
    complete timestamp boundary; if every event shares one timestamp the whole
    group is kept even when it exceeds the limit.
    """
    if len(events) <= limit:
        return events, False
    boundary_stamp = events[limit - 1].timestamp
    if events[limit].timestamp != boundary_stamp:
        # The cut already falls on a timestamp boundary.
        return events[:limit], True
    # The cut would split a same-timestamp group: back off to the previous
    # distinct timestamp so the whole group arrives together next cycle.
    cut = limit
    while cut > 0 and events[cut - 1].timestamp == boundary_stamp:
        cut -= 1
    if cut == 0:
        # One giant same-timestamp group: keep it whole rather than lose it.
        cut = limit
        while cut < len(events) and events[cut].timestamp == boundary_stamp:
            cut += 1
    return events[:cut], True


def _history_events_since(
    context_path: Path,
    cursor: datetime,
    *,
    config: Any = None,
    limit: int = MAX_EVENTS_PER_CYCLE,
) -> tuple[list[ReactorEvent], bool, int]:
    """Oldest-first history events after ``cursor``: (events, truncated, skipped)."""
    try:
        history_root = resolve_mount_root(context_path, MountType.HISTORY, config=config)
    except Exception:  # noqa: BLE001 - missing mounts must not break reconcile
        return [], False, 0
    if not history_root.exists():
        return [], False, 0

    # History files are daily (events_YYYYMMDD.jsonl); files whose date is
    # before the cursor's day cannot contain newer events, so the scan is
    # bounded by the cursor instead of the full history.
    cursor_stamp = cursor.strftime("%Y%m%d")
    events: list[ReactorEvent] = []
    skipped = 0
    for path in sorted(history_root.glob(f"{EVENT_FILE_PREFIX}_*.jsonl")):
        file_stamp = path.stem.rpartition("_")[2]
        if file_stamp.isdigit() and len(file_stamp) == 8 and file_stamp < cursor_stamp:
            continue
        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            continue
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                skipped += 1
                continue
            if not isinstance(record, dict):
                skipped += 1
                continue
            try:
                stamp = _parse_timestamp(str(record.get("timestamp", "")))
                if stamp is None or stamp <= cursor:
                    continue
                metadata = record.get("metadata")
                events.append(
                    ReactorEvent(
                        kind=str(record.get("type", "")),
                        detail=str(record.get("op", "") or ""),
                        source=str(record.get("source", "")),
                        timestamp=str(record.get("timestamp", "")),
                        metadata=metadata if isinstance(metadata, dict) else {},
                    )
                )
            except Exception:  # noqa: BLE001 - one bad record never stops ingestion
                skipped += 1
                continue
    events.sort(key=lambda event: event.timestamp)
    events, truncated = _cut_at_timestamp_boundary(events, limit)
    return events, truncated, skipped


def _hivemind_events_since(
    context_path: Path,
    cursor: datetime,
    *,
    config: Any = None,
    limit: int = MAX_EVENTS_PER_CYCLE,
) -> tuple[list[ReactorEvent], bool, int]:
    """Oldest-first hivemind events after ``cursor``: (events, truncated, skipped)."""
    try:
        from ..hivemind import HivemindBus

        # limit=0 disables the bus's newest-N slice: the backlog must be
        # drained oldest-first, so the bound is applied here at a timestamp
        # boundary instead of dropping the oldest messages.
        messages = HivemindBus(context_path, config=config).read(since=cursor, limit=0)
    except Exception:  # noqa: BLE001 - hivemind is optional signal, not a dependency
        return [], False, 0
    events: list[ReactorEvent] = []
    skipped = 0
    for message in messages:
        try:
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
        except Exception:  # noqa: BLE001 - one bad message never stops ingestion
            skipped += 1
            continue
    events.sort(key=lambda event: event.timestamp)
    events, truncated = _cut_at_timestamp_boundary(events, limit)
    return events, truncated, skipped


# ---------------------------------------------------------------------------
# Transactional batch: read under lock, dispatch, then ack
# ---------------------------------------------------------------------------


@dataclass
class ReactorBatch:
    """One cycle's events plus the cursor advance that ack() will commit."""

    events: list[ReactorEvent]
    truncated: bool
    skipped_malformed: int
    _state_dir: Path
    _state: dict[str, Any]
    _pending_history_cursor: str
    _pending_hivemind_cursor: str
    _now: datetime
    acked: bool = False

    def last_dispatch(self, agent_name: str) -> datetime | None:
        """Persisted time of the last event-triggered dispatch for an agent."""
        raw = self._state.get("last_dispatch", {}).get(agent_name, "")
        return _parse_timestamp(raw) if isinstance(raw, str) else None

    def mark_dispatched(self, agent_name: str) -> None:
        """Record an actual dispatch (spawn or job enqueue) for debounce."""
        self._state.setdefault("last_dispatch", {})[agent_name] = self._now.isoformat()

    def ack(self) -> None:
        """Commit the cursor advance. Call only after dispatch succeeded.

        Skipping ack (or crashing before it) leaves the cursors unchanged, so
        the same events are redelivered next cycle instead of being lost.
        """
        self._state["history_cursor"] = self._pending_history_cursor
        self._state["hivemind_cursor"] = self._pending_hivemind_cursor
        _save_state(self._state_dir, self._state)
        self.acked = True


def _source_watermark(events: list[ReactorEvent], stored: datetime) -> str:
    """Cursor value a source may advance to after this batch is dispatched.

    The cursor only ever advances to the newest event actually delivered —
    never to ``now`` — so a truncated backlog arrives next cycle and a write
    that lands mid-read with an earlier stamp is picked up later instead of
    being skipped forever.
    """
    newest = stored
    for event in events:
        stamp = _parse_timestamp(event.timestamp)
        if stamp is not None and stamp > newest:
            newest = stamp
    return newest.isoformat()


@contextmanager
def open_event_batch(
    context_path: Path,
    state_dir: Path,
    *,
    config: Any = None,
    now: datetime | None = None,
    max_events: int = MAX_EVENTS_PER_CYCLE,
    lock_timeout: float = LOCK_TIMEOUT_SECONDS,
) -> Iterator[ReactorBatch]:
    """Read new events under an exclusive per-context lock.

    The lock is held for the whole with-block (read -> dispatch -> ack), so
    two supervisors reconciling the same context can never both deliver one
    batch. Raises :class:`ReactorBusyError` when the lock is contended — the
    caller skips event handling for that cycle and retries on the next.

    The first call primes the cursors to *now* and yields an empty batch, so
    enabling the reactor never replays weeks of history as a spawn storm.
    """
    current = now or datetime.now(timezone.utc)
    with ExitStack() as stack:
        try:
            stack.enter_context(_file_lock(_state_path(state_dir), timeout=lock_timeout))
        except TimeoutError as exc:
            raise ReactorBusyError(
                f"event reactor lock is held by another process ({exc})"
            ) from exc

        state = _load_state(state_dir)
        history_stored = _parse_timestamp(str(state.get("history_cursor", "")))
        hivemind_stored = _parse_timestamp(str(state.get("hivemind_cursor", "")))

        if history_stored is None or hivemind_stored is None:
            primed = current.isoformat()
            state["history_cursor"] = primed
            state["hivemind_cursor"] = primed
            _save_state(state_dir, state)
            yield ReactorBatch(
                events=[],
                truncated=False,
                skipped_malformed=0,
                _state_dir=state_dir,
                _state=state,
                _pending_history_cursor=primed,
                _pending_hivemind_cursor=primed,
                _now=current,
                acked=True,
            )
            return

        history_events, history_truncated, history_skipped = _history_events_since(
            context_path, history_stored, config=config, limit=max_events
        )
        hivemind_events, hivemind_truncated, hivemind_skipped = _hivemind_events_since(
            context_path, hivemind_stored, config=config, limit=max_events
        )
        events = sorted(
            history_events + hivemind_events, key=lambda event: event.timestamp
        )
        skipped = history_skipped + hivemind_skipped
        if skipped:
            _log.warning("Event reactor skipped %d malformed record(s)", skipped)

        yield ReactorBatch(
            events=events,
            truncated=history_truncated or hivemind_truncated,
            skipped_malformed=skipped,
            _state_dir=state_dir,
            _state=state,
            _pending_history_cursor=_source_watermark(history_events, history_stored),
            _pending_hivemind_cursor=_source_watermark(hivemind_events, hivemind_stored),
            _now=current,
            acked=False,
        )
