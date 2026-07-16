"""React to new event-log entries and hivemind messages.

The supervisor's reconcile loop opens an :func:`open_event_batch` each cycle:
events are read oldest-first from both sources under an exclusive lock, the
supervisor dispatches matches against ``AgentConfig.on_event`` patterns, and
only then acks the batch. Cursors never advance before dispatch, so a crash
mid-cycle redelivers instead of losing events (at-least-once delivery).
Redelivery is kept safe by three overlapping guards — dispatch times
persisted at ack, job dedupe keys that live as long as the job is queued or
running, and the started agent's own state — and the lock closes the
read->dispatch->ack window against concurrent supervisors holding the same
state directory, so a batch is never double-dispatched on one host.

Pattern grammar: ``"<kind>"`` or ``"<kind>:<detail>"``, both sides fnmatch
globs. ``kind`` is the history event ``type`` (``mcp_tool``, ``error``,
``agent_lifecycle``, ...) or the literal ``hivemind``; ``detail`` is the
history event ``op`` or the hivemind topic. Only the first ``:`` splits, so
``hivemind:context:repair`` matches topic ``context:repair``. The hivemind
bus is the canonical source for ``hivemind`` events: history records of type
``hivemind`` (the send-op mirror of each bus message) are excluded so one
message never yields two events.
"""

from __future__ import annotations

import json
import logging
import os
import re
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any, Iterable, Iterator

from ..context_paths import resolve_mount_root
from ..history import EVENT_FILE_PREFIX
from ..models import MountType
from ..schema import AgentConfig
from .guardrails import _file_lock

_log = logging.getLogger(__name__)

# Reactor state lives in its own subdirectory: the supervisor treats every
# top-level *.json in the state dir as an agent state file, so a sibling
# cursor file would surface as a phantom agent (and an agent with the same
# name would clobber the cursors).
STATE_DIR_NAME = "event_reactor"
CURSOR_FILE_NAME = "cursor.json"
LEGACY_CURSOR_FILE_NAME = "event_reactor_cursor.json"
STATE_VERSION = 2
DEFAULT_EVENT_DEBOUNCE_SECONDS = 300.0
HIVEMIND_KIND = "hivemind"
# Per-source per-cycle read bound. A backlog larger than this is drained
# across cycles (oldest first) instead of being dropped or read unbounded.
MAX_EVENTS_PER_CYCLE = 500
LOCK_TIMEOUT_SECONDS = 5.0
# Writers stamp events before the write lands (history hashes and may write a
# payload file first). Events younger than this are deferred one cycle so the
# watermark can never advance past a stamped-but-not-yet-visible write.
WATERMARK_GRACE_SECONDS = 5.0

VALID_EVENT_ACTIONS = ("spawn", "job")

_LABEL_SAFE = re.compile(r"[^A-Za-z0-9_.:\-*]")
_LABEL_MAX = 80


class ReactorBusyError(RuntimeError):
    """Another process holds the reactor cursor lock for this context."""


class ReactorStateError(RuntimeError):
    """Reactor cursor state could not be persisted.

    Raised instead of being swallowed: a failed cursor commit that looked
    successful would silently re-prime on the next cycle and drop the
    unacked backlog. Leaving the cursors unchanged redelivers instead.
    """


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
    return state_dir / STATE_DIR_NAME / CURSOR_FILE_NAME


def _read_state_file(path: Path) -> dict[str, Any] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _load_state(state_dir: Path) -> dict[str, Any]:
    payload = _read_state_file(_state_path(state_dir))
    if payload is None:
        # Migrate state written by earlier versions directly into the state
        # dir. The legacy file is removed on the next successful save, never
        # before, so a crash mid-migration cannot lose the cursors.
        payload = _read_state_file(state_dir / LEGACY_CURSOR_FILE_NAME)
    if payload is None:
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
    """Persist reactor state atomically; raise :class:`ReactorStateError` on failure.

    The write goes through a temp file + ``os.replace`` so a crash or full
    disk mid-write can never leave a truncated state file (which would read
    as first-run and silently re-prime the cursors past the backlog). A
    failure must propagate: an ack that silently didn't commit is the one
    way this module can still lose events.
    """
    path = _state_path(state_dir)
    tmp_path = path.with_suffix(".json.tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_text(
            json.dumps({**state, "version": STATE_VERSION}) + "\n",
            encoding="utf-8",
        )
        os.replace(tmp_path, path)
    except OSError as exc:
        try:
            tmp_path.unlink(missing_ok=True)
        except OSError:
            pass
        raise ReactorStateError(
            f"could not persist event reactor state to {path}: {exc}"
        ) from exc
    try:
        (state_dir / LEGACY_CURSOR_FILE_NAME).unlink(missing_ok=True)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# Bounded oldest-first source reads
# ---------------------------------------------------------------------------


def _cut_at_timestamp_boundary(
    events: list[ReactorEvent], limit: int
) -> tuple[list[ReactorEvent], bool]:
    """Truncate an oldest-first event list without splitting a timestamp group.

    Cursors are timestamps, so cutting in the middle of a group of events that
    share one instant would skip the rest of the group forever (the next
    cycle reads strictly-greater-than the cursor). Grouping compares parsed
    timestamps, not raw strings, so ``Z`` vs ``+00:00`` spellings of the same
    instant stay in one group. The cut lands on the last complete boundary;
    if every event shares one instant the whole group is kept even when it
    exceeds the limit. A non-positive limit means unbounded.
    """
    if limit <= 0 or len(events) <= limit:
        return events, False
    stamps = [_parse_timestamp(event.timestamp) for event in events]
    boundary_stamp = stamps[limit - 1]
    if stamps[limit] != boundary_stamp:
        # The cut already falls on a timestamp boundary.
        return events[:limit], True
    # The cut would split a same-timestamp group: back off to the previous
    # distinct timestamp so the whole group arrives together next cycle.
    cut = limit
    while cut > 0 and stamps[cut - 1] == boundary_stamp:
        cut -= 1
    if cut == 0:
        # One giant same-timestamp group: keep it whole rather than lose it.
        cut = limit
        while cut < len(stamps) and stamps[cut] == boundary_stamp:
            cut += 1
    return events[:cut], cut < len(events)


def _event_sort_key(event: ReactorEvent) -> datetime:
    # Every ingested event passed the parse filter, so this cannot be None;
    # parsed comparison keeps mixed Z/+00:00/fractional spellings in order
    # where raw string comparison would not.
    return _parse_timestamp(event.timestamp) or datetime.min.replace(tzinfo=timezone.utc)


def _history_events_since(
    context_path: Path,
    cursor: datetime,
    until: datetime,
    *,
    config: Any = None,
    limit: int = MAX_EVENTS_PER_CYCLE,
) -> tuple[list[ReactorEvent], bool, int]:
    """Oldest-first history events in ``(cursor, until]``: (events, truncated, skipped)."""
    try:
        history_root = resolve_mount_root(context_path, MountType.HISTORY, config=config)
    except Exception:  # noqa: BLE001 - missing mounts must not break reconcile
        return [], False, 0
    if not history_root.exists():
        return [], False, 0

    # History files are daily (events_YYYYMMDD.jsonl); files dated before the
    # cursor's day cannot contain newer events, so the scan is bounded by the
    # cursor instead of the full history. One day of grace covers filenames
    # stamped in a writer's local day that differs from the cursor's UTC day.
    cursor_stamp = (
        cursor.astimezone(timezone.utc) - timedelta(days=1)
    ).strftime("%Y%m%d")
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
                kind = str(record.get("type", ""))
                if kind == HIVEMIND_KIND:
                    # The bus is the canonical hivemind source; its history
                    # mirror (op "send") would double-deliver every message
                    # under a bare "hivemind" pattern with an op, not a
                    # topic, in the detail slot.
                    continue
                stamp = _parse_timestamp(str(record.get("timestamp", "")))
                if stamp is None or stamp <= cursor or stamp > until:
                    continue
                metadata = record.get("metadata")
                events.append(
                    ReactorEvent(
                        kind=kind,
                        detail=str(record.get("op", "") or ""),
                        source=str(record.get("source", "")),
                        timestamp=str(record.get("timestamp", "")),
                        metadata=metadata if isinstance(metadata, dict) else {},
                    )
                )
            except Exception:  # noqa: BLE001 - one bad record never stops ingestion
                skipped += 1
                continue
    events.sort(key=_event_sort_key)
    events, truncated = _cut_at_timestamp_boundary(events, limit)
    return events, truncated, skipped


def _hivemind_events_since(
    context_path: Path,
    cursor: datetime,
    until: datetime,
    *,
    config: Any = None,
    limit: int = MAX_EVENTS_PER_CYCLE,
) -> tuple[list[ReactorEvent], bool, int]:
    """Oldest-first hivemind events in ``(cursor, until]``: (events, truncated, skipped)."""
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
            if stamp is None or stamp <= cursor or stamp > until:
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
    events.sort(key=_event_sort_key)
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
    dispatch_failures: int = 0

    def last_dispatch(self, agent_name: str) -> datetime | None:
        """Persisted time of the last event-triggered dispatch for an agent."""
        raw = self._state.get("last_dispatch", {}).get(agent_name, "")
        return _parse_timestamp(raw) if isinstance(raw, str) else None

    def mark_dispatched(self, agent_name: str) -> None:
        """Record an actual dispatch (spawn or job enqueue) for debounce.

        In memory until ack() persists it: in the crash window before ack the
        job dedupe key and the started agent's own state are what suppress
        duplicates, not this table.
        """
        self._state.setdefault("last_dispatch", {})[agent_name] = self._now.isoformat()

    def mark_dispatch_failed(self, agent_name: str) -> None:
        """Record that an event-triggered dispatch failed.

        The caller must then skip ack() so the batch is redelivered: acking
        over a failed dispatch would consume the event with zero deliveries.
        """
        _log.warning(
            "Event dispatch for agent '%s' failed; deferring ack for redelivery",
            agent_name,
        )
        self.dispatch_failures += 1

    def prune_dispatch(self, active_names: Iterable[str]) -> None:
        """Drop persisted dispatch times for agents no longer configured."""
        table = self._state.get("last_dispatch", {})
        keep = set(active_names)
        for name in [key for key in table if key not in keep]:
            table.pop(name, None)

    def ack(self) -> None:
        """Commit the cursor advance. Call only after every dispatch succeeded.

        Skipping ack (or crashing before it) leaves the cursors unchanged, so
        the same events are redelivered next cycle instead of being lost.
        Raises :class:`ReactorStateError` when the commit itself cannot be
        persisted — the batch then counts as unacked and redelivers.
        """
        self._state["history_cursor"] = self._pending_history_cursor
        self._state["hivemind_cursor"] = self._pending_hivemind_cursor
        _save_state(self._state_dir, self._state)
        self.acked = True


def _source_watermark(events: list[ReactorEvent], stored: datetime) -> str:
    """Cursor value a source may advance to after this batch is dispatched.

    The cursor only ever advances to the newest event actually delivered —
    never to ``now`` — so a truncated backlog arrives next cycle. Because
    delivered events are bounded by the ripeness cutoff (``now`` minus
    :data:`WATERMARK_GRACE_SECONDS`), a writer that stamped an event before
    the read but landed it after is still ahead of the cursor as long as its
    stamp-to-land delay is under the grace window; without the cutoff such a
    write would be skipped forever.
    """
    newest = stored
    for event in events:
        stamp = _parse_timestamp(event.timestamp)
        if stamp is not None and stamp > newest:
            newest = stamp
    return newest.astimezone(timezone.utc).isoformat()


@contextmanager
def open_event_batch(
    context_path: Path,
    state_dir: Path,
    *,
    config: Any = None,
    now: datetime | None = None,
    max_events: int = MAX_EVENTS_PER_CYCLE,
    lock_timeout: float = LOCK_TIMEOUT_SECONDS,
    grace_seconds: float = WATERMARK_GRACE_SECONDS,
) -> Iterator[ReactorBatch]:
    """Read new events under an exclusive per-context lock.

    The lock is held for the whole with-block (read -> dispatch -> ack), so
    two supervisors sharing this state directory can never both deliver one
    batch (the lock is advisory and host-local: a second supervisor pointed
    at a different ``AFS_AGENT_STATE_DIR``, or another host on a synced
    filesystem, is outside its reach). Raises :class:`ReactorBusyError` when
    the lock is contended — the caller skips event handling for that cycle
    and retries on the next.

    Events stamped within ``grace_seconds`` of *now* are deferred one cycle:
    writers stamp before their write lands, so consuming right up to *now*
    could advance the watermark past a stamped-but-in-flight event.

    A missing or unreadable cursor is primed to *now* per source, so first
    enabling the reactor never replays weeks of history as a spawn storm and
    one corrupt cursor never drops the other source's backlog.
    """
    current = now or datetime.now(timezone.utc)
    ripe_until = current - timedelta(seconds=max(grace_seconds, 0.0))
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

        primed_sources: list[str] = []
        if history_stored is None:
            history_stored = current
            state["history_cursor"] = current.isoformat()
            primed_sources.append("history")
        if hivemind_stored is None:
            hivemind_stored = current
            state["hivemind_cursor"] = current.isoformat()
            primed_sources.append("hivemind")
        if primed_sources:
            if len(primed_sources) < 2:
                # A fresh install primes both; one missing cursor on an
                # initialized state means corruption or manual edits.
                _log.warning(
                    "Event reactor cursor for %s was missing or unreadable; "
                    "primed to now, its pre-cursor backlog is skipped",
                    ", ".join(primed_sources),
                )
            _save_state(state_dir, state)
        if len(primed_sources) == 2:
            yield ReactorBatch(
                events=[],
                truncated=False,
                skipped_malformed=0,
                _state_dir=state_dir,
                _state=state,
                _pending_history_cursor=state["history_cursor"],
                _pending_hivemind_cursor=state["hivemind_cursor"],
                _now=current,
                acked=True,
            )
            return

        history_events, history_truncated, history_skipped = _history_events_since(
            context_path, history_stored, ripe_until, config=config, limit=max_events
        )
        hivemind_events, hivemind_truncated, hivemind_skipped = _hivemind_events_since(
            context_path, hivemind_stored, ripe_until, config=config, limit=max_events
        )
        events = sorted(history_events + hivemind_events, key=_event_sort_key)
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
