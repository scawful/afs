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
from collections.abc import Iterable, Iterator
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from fnmatch import fnmatchcase
from heapq import nsmallest
from pathlib import Path
from typing import Any

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
INITIALIZED_FILE_NAME = "initialized"
LEGACY_CURSOR_FILE_NAME = "event_reactor_cursor.json"
STATE_VERSION = 3
DEFAULT_EVENT_DEBOUNCE_SECONDS = 300.0
HIVEMIND_KIND = "hivemind"
# Per-source per-cycle record bound. Positional checkpoints resume larger
# backlogs without materializing an entire daily log or message collection.
MAX_EVENTS_PER_CYCLE = 500
MAX_HISTORY_SCAN_BYTES = 1024 * 1024
MAX_HISTORY_RECORD_BYTES = 256 * 1024
MAX_HIVEMIND_RECORD_BYTES = 256 * 1024
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


def _initialized_path(state_dir: Path) -> Path:
    return state_dir / STATE_DIR_NAME / INITIALIZED_FILE_NAME


def _read_state_file(path: Path) -> dict[str, Any] | None:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ReactorStateError(f"could not read event reactor state from {path}: {exc}") from exc
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ReactorStateError(f"event reactor state is invalid JSON: {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise ReactorStateError(
            f"event reactor state must be a JSON object, got {type(payload).__name__}: {path}"
        )
    return payload


def _load_state(state_dir: Path) -> dict[str, Any]:
    state_path = _state_path(state_dir)
    payload = _read_state_file(state_path)
    if payload is None:
        # Migrate state written by earlier versions directly into the state
        # dir. The legacy file is removed on the next successful save, never
        # before, so a crash mid-migration cannot lose the cursors.
        legacy_path = state_dir / LEGACY_CURSOR_FILE_NAME
        payload = _read_state_file(legacy_path)
    if payload is None:
        if _initialized_path(state_dir).exists():
            raise ReactorStateError(
                f"event reactor cursor is missing after initialization: {state_path}; "
                "restore the cursor or remove the initialized marker only when an "
                "explicit re-prime is intended"
            )
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
    if not payload:
        raise ReactorStateError(f"event reactor state is empty: {state_path}")
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
        _initialized_path(state_dir).write_text(f"{STATE_VERSION}\n", encoding="utf-8")
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


def _history_files(
    context_path: Path,
    *,
    config: Any = None,
) -> list[Path]:
    try:
        history_root = resolve_mount_root(context_path, MountType.HISTORY, config=config)
    except Exception:  # noqa: BLE001 - missing mounts must not break reconcile
        return []
    if not history_root.exists():
        return []
    return sorted(history_root.glob(f"{EVENT_FILE_PREFIX}_*.jsonl"))


def _history_offset_snapshot(
    context_path: Path,
    *,
    config: Any = None,
) -> dict[str, int]:
    """Current append offsets used when priming or migrating state."""
    snapshot: dict[str, int] = {}
    for path in _history_files(context_path, config=config):
        try:
            snapshot[path.name] = path.stat().st_size
        except OSError:
            continue
    return snapshot


def _offset_map(value: Any, *, key: str) -> dict[str, int]:
    if not isinstance(value, dict):
        raise ReactorStateError(f"event reactor state field {key!r} must be an object")
    result: dict[str, int] = {}
    for raw_name, raw_offset in value.items():
        if (
            not isinstance(raw_name, str)
            or not raw_name
            or isinstance(raw_offset, bool)
            or not isinstance(raw_offset, int)
            or raw_offset < 0
        ):
            raise ReactorStateError(
                f"event reactor state field {key!r} contains an invalid file offset"
            )
        result[raw_name] = raw_offset
    return result


def _history_events_since(
    context_path: Path,
    cursor: datetime,
    until: datetime,
    *,
    config: Any = None,
    offsets: dict[str, int] | None = None,
    migration_cutoffs: dict[str, int] | None = None,
    limit: int = MAX_EVENTS_PER_CYCLE,
    max_scan_bytes: int = MAX_HISTORY_SCAN_BYTES,
) -> tuple[list[ReactorEvent], bool, int, dict[str, int], dict[str, int]]:
    """Stream one bounded, resumable slice of append-only history files.

    File offsets, not timestamps, are the delivery checkpoint. That keeps the
    scan bounded without assuming callers append timestamp-monotonic records.
    ``migration_cutoffs`` marks bytes that predate an older timestamp-only
    cursor; those records are filtered once while offsets catch up. Offsets are
    returned separately and persist only when the enclosing batch is acked.
    """
    paths = _history_files(context_path, config=config)
    pending_offsets = dict(offsets or {})
    pending_cutoffs = dict(migration_cutoffs or {})
    existing_names = {path.name for path in paths}
    for name in [name for name in pending_offsets if name not in existing_names]:
        pending_offsets.pop(name, None)
        pending_cutoffs.pop(name, None)

    events: list[ReactorEvent] = []
    skipped = 0
    scanned_records = 0
    scanned_bytes = 0
    record_limit = limit if limit > 0 else MAX_EVENTS_PER_CYCLE
    byte_limit = max(max_scan_bytes, 1)
    truncated = False

    for path_index, path in enumerate(paths):
        name = path.name
        offset = pending_offsets.get(name, 0)
        try:
            size = path.stat().st_size
            if size < offset:
                raise ReactorStateError(
                    f"history file shrank below its reactor checkpoint: {path} "
                    f"({size} < {offset})"
                )
            handle = path.open("rb")
        except ReactorStateError:
            raise
        except OSError:
            continue
        with handle:
            handle.seek(offset)
            while handle.tell() < size:
                if len(events) >= record_limit or scanned_records >= record_limit:
                    truncated = True
                    break
                remaining = byte_limit - scanned_bytes
                if remaining <= 0:
                    truncated = True
                    break
                line_start = handle.tell()
                read_cap = min(MAX_HISTORY_RECORD_BYTES + 1, remaining + 1)
                raw = handle.readline(read_cap)
                if not raw:
                    break
                if not raw.endswith(b"\n"):
                    if len(raw) > MAX_HISTORY_RECORD_BYTES:
                        raise ReactorStateError(
                            f"history record exceeds {MAX_HISTORY_RECORD_BYTES} bytes: "
                            f"{path} at offset {line_start}"
                        )
                    # The cycle byte budget ended inside a valid line. Leave
                    # the offset at its start so the next cycle reads it whole.
                    handle.seek(line_start)
                    truncated = True
                    break

                pending_offsets[name] = handle.tell()
                scanned_records += 1
                scanned_bytes += len(raw)
                line = raw.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    skipped += 1
                    continue
                if not isinstance(record, dict):
                    skipped += 1
                    continue
                try:
                    kind = str(record.get("type", ""))
                    if kind == HIVEMIND_KIND:
                        # The bus is the canonical hivemind source; its history
                        # mirror would double-deliver each message.
                        continue
                    stamp = _parse_timestamp(str(record.get("timestamp", "")))
                    if stamp is None:
                        continue
                    if stamp > until:
                        # Do not checkpoint past a grace-window event. Writers
                        # stamp before append; advancing the file offset here
                        # would make the event disappear permanently.
                        pending_offsets[name] = line_start
                        handle.seek(line_start)
                        truncated = True
                        break
                    cutoff = pending_cutoffs.get(name, 0)
                    if line_start < cutoff and stamp <= cursor:
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
            if pending_offsets.get(name, offset) < size:
                truncated = True
            cutoff = pending_cutoffs.get(name)
            if cutoff is not None and pending_offsets.get(name, offset) >= cutoff:
                pending_cutoffs.pop(name, None)
        if truncated:
            break
        if path_index < len(paths) - 1 and (
            len(events) >= record_limit
            or scanned_records >= record_limit
            or scanned_bytes >= byte_limit
        ):
            truncated = True
            break

    events.sort(key=_event_sort_key)
    return events, truncated, skipped, pending_offsets, pending_cutoffs


def _string_checkpoint_map(value: Any, *, key: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ReactorStateError(f"event reactor state field {key!r} must be an object")
    result: dict[str, str] = {}
    for raw_name, raw_checkpoint in value.items():
        if (
            not isinstance(raw_name, str)
            or not raw_name
            or not isinstance(raw_checkpoint, str)
        ):
            raise ReactorStateError(
                f"event reactor state field {key!r} contains an invalid checkpoint"
            )
        _decode_hivemind_checkpoint(raw_checkpoint)
        result[raw_name] = raw_checkpoint
    return result


def _hivemind_root(context_path: Path, *, config: Any = None) -> Path | None:
    try:
        root = resolve_mount_root(context_path, MountType.HIVEMIND, config=config)
    except Exception:  # noqa: BLE001 - hivemind is an optional signal source
        return None
    return root if root.exists() else None


def _hivemind_checkpoint_snapshot(
    context_path: Path,
    *,
    config: Any = None,
) -> dict[str, str]:
    root = _hivemind_root(context_path, config=config)
    if root is None:
        return {}
    snapshot: dict[str, str] = {}
    try:
        directories = sorted(
            path for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")
        )
    except OSError:
        return {}
    for directory in directories:
        try:
            with os.scandir(directory) as entries:
                latest = max(
                    (
                        (entry.stat().st_mtime_ns, entry.name)
                        for entry in entries
                        if entry.is_file() and entry.name.endswith(".json")
                    ),
                    default=None,
                )
        except OSError:
            continue
        if latest is not None:
            snapshot[directory.name] = _encode_hivemind_checkpoint(*latest)
    return snapshot


def _encode_hivemind_checkpoint(mtime_ns: int, filename: str) -> str:
    return f"{mtime_ns}:{filename}"


def _decode_hivemind_checkpoint(value: str) -> tuple[int, str]:
    if not value:
        return -1, ""
    raw_mtime, separator, filename = value.partition(":")
    if not separator or not raw_mtime.isdigit() or not filename:
        raise ReactorStateError("event reactor hivemind checkpoint is malformed")
    return int(raw_mtime), filename


def _hivemind_candidates(
    root: Path,
    checkpoints: dict[str, str],
) -> Iterator[tuple[int, str, str, Path]]:
    """Yield uncheckpointed message paths without retaining directory contents."""
    try:
        directories = sorted(
            path for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")
        )
    except OSError:
        return
    for directory in directories:
        checkpoint = _decode_hivemind_checkpoint(
            checkpoints.get(directory.name, "")
        )
        try:
            with os.scandir(directory) as entries:
                for entry in entries:
                    if not entry.is_file() or not entry.name.endswith(".json"):
                        continue
                    candidate = (entry.stat().st_mtime_ns, entry.name)
                    if candidate > checkpoint:
                        yield candidate[0], candidate[1], directory.name, Path(entry.path)
        except OSError:
            continue


def _hivemind_events_since(
    context_path: Path,
    cursor: datetime,
    until: datetime,
    *,
    config: Any = None,
    checkpoints: dict[str, str] | None = None,
    migration_cutoffs: dict[str, str] | None = None,
    limit: int = MAX_EVENTS_PER_CYCLE,
) -> tuple[list[ReactorEvent], bool, int, dict[str, str], dict[str, str]]:
    """Read a bounded oldest-first slice of immutable hivemind message files."""
    from ..hivemind import HivemindMessage

    root = _hivemind_root(context_path, config=config)
    pending_checkpoints = dict(checkpoints or {})
    pending_cutoffs = dict(migration_cutoffs or {})
    if root is None:
        return [], False, 0, pending_checkpoints, pending_cutoffs

    try:
        active_directories = {
            path.name
            for path in root.iterdir()
            if path.is_dir() and not path.name.startswith(".")
        }
    except OSError:
        active_directories = set()
    for name in [name for name in pending_checkpoints if name not in active_directories]:
        pending_checkpoints.pop(name, None)
        pending_cutoffs.pop(name, None)

    record_limit = limit if limit > 0 else MAX_EVENTS_PER_CYCLE
    selected = nsmallest(
        record_limit + 1,
        _hivemind_candidates(root, pending_checkpoints),
        key=lambda item: (item[0], item[1], item[2]),
    )
    truncated = len(selected) > record_limit
    events: list[ReactorEvent] = []
    skipped = 0
    expiry_now = datetime.now(timezone.utc)
    for mtime_ns, filename, directory_name, path in selected[:record_limit]:
        checkpoint = _encode_hivemind_checkpoint(mtime_ns, filename)
        try:
            with path.open("rb") as handle:
                raw = handle.read(MAX_HIVEMIND_RECORD_BYTES + 1)
            if len(raw) > MAX_HIVEMIND_RECORD_BYTES:
                skipped += 1
                pending_checkpoints[directory_name] = checkpoint
                continue
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError("message is not an object")
            message = HivemindMessage.from_dict(data)
            stamp = _parse_timestamp(message.timestamp)
            if stamp is None:
                skipped += 1
                pending_checkpoints[directory_name] = checkpoint
                continue
            if stamp > until:
                # Filename ordering follows the send timestamp. Leave this and
                # later files uncheckpointed until the grace window elapses.
                truncated = True
                break
            expires = _parse_timestamp(message.expires_at or "")
            pending_checkpoints[directory_name] = checkpoint
            if expires is not None and expires <= expiry_now:
                continue
            cutoff = pending_cutoffs.get(directory_name, "")
            if cutoff and (mtime_ns, filename) <= _decode_hivemind_checkpoint(cutoff) and stamp <= cursor:
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
            pending_checkpoints[directory_name] = checkpoint
            continue
    for directory_name, cutoff in list(pending_cutoffs.items()):
        checkpoint = pending_checkpoints.get(directory_name, "")
        if checkpoint and _decode_hivemind_checkpoint(checkpoint) >= _decode_hivemind_checkpoint(cutoff):
            pending_cutoffs.pop(directory_name, None)
    events.sort(key=_event_sort_key)
    return events, truncated, skipped, pending_checkpoints, pending_cutoffs


# ---------------------------------------------------------------------------
# Transactional batch: read under lock, dispatch, then ack
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReactorDispatchOutcome:
    """One event route's terminal result for the current batch."""

    state: str
    reason: str


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
    _pending_history_offsets: dict[str, int] = field(default_factory=dict)
    _pending_history_migration_cutoffs: dict[str, int] = field(default_factory=dict)
    _pending_hivemind_files: dict[str, str] = field(default_factory=dict)
    _pending_hivemind_migration_cutoffs: dict[str, str] = field(default_factory=dict)
    acked: bool = False
    dispatch_failures: int = 0
    dispatch_outcomes: dict[str, ReactorDispatchOutcome] = field(default_factory=dict)

    def last_dispatch(self, agent_name: str) -> datetime | None:
        """Persisted time of the last event-triggered dispatch for an agent."""
        raw = self._state.get("last_dispatch", {}).get(agent_name, "")
        return _parse_timestamp(raw) if isinstance(raw, str) else None

    def _record_dispatch_outcome(
        self,
        agent_name: str,
        state: str,
        reason: str,
    ) -> bool:
        """Record one explicit route outcome and maintain the defer count."""
        previous = self.dispatch_outcomes.get(agent_name)
        if previous == ReactorDispatchOutcome(state=state, reason=reason):
            return False
        # Once work reached its destination, later duplicate/coalescing checks
        # in the same cycle must not downgrade that successful delivery.
        if previous is not None and previous.state == "dispatched":
            return False
        if previous is not None and previous.state == "deferred":
            self.dispatch_failures = max(0, self.dispatch_failures - 1)
        if state == "deferred":
            self.dispatch_failures += 1
        self.dispatch_outcomes[agent_name] = ReactorDispatchOutcome(
            state=state,
            reason=reason,
        )
        return True

    def mark_dispatched(self, agent_name: str, *, reason: str = "dispatch") -> None:
        """Record an actual dispatch (spawn or job enqueue) for debounce.

        In memory until ack() persists it: in the crash window before ack the
        job dedupe key and the started agent's own state are what suppress
        duplicates, not this table.
        """
        self._state.setdefault("last_dispatch", {})[agent_name] = self._now.isoformat()
        self._record_dispatch_outcome(agent_name, "dispatched", reason)

    def mark_dispatch_deferred(self, agent_name: str, *, reason: str) -> None:
        """Record a retryable gate and require batch redelivery."""
        if not self._record_dispatch_outcome(agent_name, "deferred", reason):
            return
        _log.warning(
            "Event delivery for agent '%s' deferred (%s); batch will redeliver",
            agent_name,
            reason,
        )

    def mark_coalesced(self, agent_name: str, *, reason: str) -> None:
        """Record an intentional delivery coalescing decision."""
        self._record_dispatch_outcome(agent_name, "coalesced", reason)

    def mark_rejected(self, agent_name: str, *, reason: str) -> bool:
        """Record a terminal fail-closed rejection and permit cursor advance."""
        return self._record_dispatch_outcome(agent_name, "rejected", reason)

    def mark_dispatch_failed(
        self,
        agent_name: str,
        *,
        reason: str = "dispatch_failed",
    ) -> None:
        """Record that an event-triggered dispatch failed.

        The caller must then skip ack() so the batch is redelivered: acking
        over a failed dispatch would consume the event with zero deliveries.
        """
        self.mark_dispatch_deferred(agent_name, reason=reason)

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
        self._state["history_offsets"] = self._pending_history_offsets
        if self._pending_history_migration_cutoffs:
            self._state["history_migration_cutoffs"] = (
                self._pending_history_migration_cutoffs
            )
        else:
            self._state.pop("history_migration_cutoffs", None)
        self._state["hivemind_files"] = self._pending_hivemind_files
        if self._pending_hivemind_migration_cutoffs:
            self._state["hivemind_migration_cutoffs"] = (
                self._pending_hivemind_migration_cutoffs
            )
        else:
            self._state.pop("hivemind_migration_cutoffs", None)
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
    max_history_scan_bytes: int = MAX_HISTORY_SCAN_BYTES,
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

    A genuinely absent state is primed to *now* once, so first enabling the
    reactor never replays weeks of history as a spawn storm. Once initialized,
    missing, unreadable, malformed, or partial cursor state raises
    :class:`ReactorStateError` instead of silently skipping backlog.
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
        fresh_state = not state
        history_stored = _parse_timestamp(str(state.get("history_cursor", "")))
        hivemind_stored = _parse_timestamp(str(state.get("hivemind_cursor", "")))

        if fresh_state:
            history_stored = current
            state["history_cursor"] = current.isoformat()
            hivemind_stored = current
            state["hivemind_cursor"] = current.isoformat()
            state["history_offsets"] = _history_offset_snapshot(
                context_path, config=config
            )
            state["hivemind_files"] = _hivemind_checkpoint_snapshot(
                context_path, config=config
            )
            _save_state(state_dir, state)
            yield ReactorBatch(
                events=[],
                truncated=False,
                skipped_malformed=0,
                _state_dir=state_dir,
                _state=state,
                _pending_history_cursor=state["history_cursor"],
                _pending_hivemind_cursor=state["hivemind_cursor"],
                _now=current,
                _pending_history_offsets=dict(state["history_offsets"]),
                _pending_hivemind_files=dict(state["hivemind_files"]),
                acked=True,
            )
            return
        if history_stored is None or hivemind_stored is None:
            invalid = []
            if history_stored is None:
                invalid.append("history_cursor")
            if hivemind_stored is None:
                invalid.append("hivemind_cursor")
            raise ReactorStateError(
                "event reactor state has missing or invalid cursor(s): "
                + ", ".join(invalid)
                + "; repair the state explicitly instead of skipping backlog"
            )

        if "history_offsets" in state:
            history_offsets = _offset_map(
                state["history_offsets"], key="history_offsets"
            )
            history_migration_cutoffs = _offset_map(
                state.get("history_migration_cutoffs", {}),
                key="history_migration_cutoffs",
            )
        else:
            # Upgrade timestamp-only state without an unbounded catch-up read.
            # Existing bytes are scanned in bounded slices and filtered against
            # the old timestamp cursor; bytes appended after this snapshot are
            # always delivered, even if their caller supplied an older stamp.
            history_offsets = {}
            history_migration_cutoffs = _history_offset_snapshot(
                context_path, config=config
            )

        if "hivemind_files" in state:
            hivemind_files = _string_checkpoint_map(
                state["hivemind_files"], key="hivemind_files"
            )
            hivemind_migration_cutoffs = _string_checkpoint_map(
                state.get("hivemind_migration_cutoffs", {}),
                key="hivemind_migration_cutoffs",
            )
        else:
            hivemind_files = {}
            hivemind_migration_cutoffs = _hivemind_checkpoint_snapshot(
                context_path, config=config
            )

        (
            history_events,
            history_truncated,
            history_skipped,
            pending_history_offsets,
            pending_history_migration_cutoffs,
        ) = _history_events_since(
            context_path,
            history_stored,
            ripe_until,
            config=config,
            offsets=history_offsets,
            migration_cutoffs=history_migration_cutoffs,
            limit=max_events,
            max_scan_bytes=max_history_scan_bytes,
        )
        (
            hivemind_events,
            hivemind_truncated,
            hivemind_skipped,
            pending_hivemind_files,
            pending_hivemind_migration_cutoffs,
        ) = _hivemind_events_since(
            context_path,
            hivemind_stored,
            ripe_until,
            config=config,
            checkpoints=hivemind_files,
            migration_cutoffs=hivemind_migration_cutoffs,
            limit=max_events,
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
            _pending_history_offsets=pending_history_offsets,
            _pending_history_migration_cutoffs=(
                pending_history_migration_cutoffs
            ),
            _pending_hivemind_files=pending_hivemind_files,
            _pending_hivemind_migration_cutoffs=(
                pending_hivemind_migration_cutoffs
            ),
            acked=False,
        )
