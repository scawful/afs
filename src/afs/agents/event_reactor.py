"""React to new event-log entries and hivemind messages.

The supervisor's reconcile loop opens an :func:`open_event_batch` each cycle:
events are read in bounded source order under an exclusive lock, the
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

import hashlib
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
STATE_VERSION = 4
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


@dataclass(frozen=True)
class _DeferredHistoryRecord:
    """Reference to a visible history record whose timestamp is not ripe yet."""

    filename: str
    offset: int
    length: int
    digest: str
    timestamp: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "offset": self.offset,
            "length": self.length,
            "digest": self.digest,
            "timestamp": self.timestamp,
        }


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


def _read_initialized_version(state_dir: Path) -> int | None:
    path = _initialized_path(state_dir)
    try:
        raw = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return None
    except OSError as exc:
        raise ReactorStateError(
            f"could not read event reactor initialized marker from {path}: {exc}"
        ) from exc
    if not raw.isdigit():
        raise ReactorStateError(f"event reactor initialized marker is malformed: {path}")
    version = int(raw)
    if version not in (2, 3, STATE_VERSION):
        raise ReactorStateError(
            f"event reactor initialized marker version {version} is unsupported; "
            f"expected 2, 3, or {STATE_VERSION}: {path}"
        )
    return version


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


def _state_version(payload: dict[str, Any], *, path: Path) -> int:
    """Validate and return an explicitly supported persisted-state version."""
    if "version" not in payload:
        if "cursor" in payload:
            return 1
        raise ReactorStateError(
            f"event reactor state has no supported version or legacy cursor: {path}"
        )
    raw_version = payload["version"]
    if isinstance(raw_version, bool) or not isinstance(raw_version, int):
        raise ReactorStateError(f"event reactor state version must be an integer: {path}")
    if raw_version not in (1, 2, 3, STATE_VERSION):
        raise ReactorStateError(
            f"event reactor state version {raw_version} is unsupported; "
            f"supported versions are 1, 2, 3, and {STATE_VERSION}: {path}"
        )
    return raw_version


def _normalize_state(payload: dict[str, Any], *, path: Path) -> dict[str, Any]:
    """Normalize legacy cursors while rejecting ambiguous current state."""
    if not payload:
        raise ReactorStateError(f"event reactor state is empty: {path}")
    version = _state_version(payload, path=path)
    state = dict(payload)
    if version == 1:
        legacy = state.get("cursor")
        if not isinstance(legacy, str):
            raise ReactorStateError(f"event reactor version-1 state has an invalid cursor: {path}")
        state = {
            "version": 1,
            "history_cursor": legacy,
            "hivemind_cursor": legacy,
            "last_dispatch": {},
        }
    else:
        missing_cursors = [key for key in ("history_cursor", "hivemind_cursor") if key not in state]
        if missing_cursors:
            raise ReactorStateError(
                "event reactor state is partial; missing cursor field(s): "
                + ", ".join(missing_cursors)
                + f": {path}"
            )

    last_dispatch = state.setdefault("last_dispatch", {})
    if not isinstance(last_dispatch, dict):
        raise ReactorStateError(
            f"event reactor state field 'last_dispatch' must be an object: {path}"
        )

    if version == 3:
        missing_maps = [key for key in ("history_offsets", "hivemind_files") if key not in state]
        if missing_maps:
            raise ReactorStateError(
                "event reactor version-3 state is partial; missing positional map(s): "
                + ", ".join(missing_maps)
                + f": {path}"
            )
        for cutoff_key, watermark_key in (
            ("history_migration_cutoffs", "history_migration_watermark"),
            ("hivemind_migration_cutoffs", "hivemind_migration_watermark"),
        ):
            if (cutoff_key in state) != (watermark_key in state):
                raise ReactorStateError(
                    "event reactor version-3 migration state is partial; "
                    f"{cutoff_key!r} and {watermark_key!r} must appear together: {path}"
                )
    elif version == STATE_VERSION:
        missing_maps = [key for key in ("history_offsets", "hivemind_seen") if key not in state]
        if missing_maps:
            raise ReactorStateError(
                f"event reactor version-{STATE_VERSION} state is partial; "
                "missing positional map(s): " + ", ".join(missing_maps) + f": {path}"
            )
        if ("history_migration_cutoffs" in state) != ("history_migration_watermark" in state):
            raise ReactorStateError(
                f"event reactor version-{STATE_VERSION} migration state is partial; "
                "'history_migration_cutoffs' and 'history_migration_watermark' "
                f"must appear together: {path}"
            )
        has_hivemind_boundary = (
            "hivemind_migration_existing" in state or "hivemind_migration_cutoffs" in state
        )
        if has_hivemind_boundary != ("hivemind_migration_watermark" in state):
            raise ReactorStateError(
                f"event reactor version-{STATE_VERSION} migration state is partial; "
                "a hivemind migration boundary and 'hivemind_migration_watermark' "
                f"must appear together: {path}"
            )
    return state


def _load_state(state_dir: Path) -> dict[str, Any]:
    state_path = _state_path(state_dir)
    initialized_version = _read_initialized_version(state_dir)
    payload = _read_state_file(state_path)
    loaded_path = state_path
    if payload is None:
        # Migrate state written by earlier versions directly into the state
        # dir. The legacy file is removed on the next successful save, never
        # before, so a crash mid-migration cannot lose the cursors.
        legacy_path = state_dir / LEGACY_CURSOR_FILE_NAME
        payload = _read_state_file(legacy_path)
        loaded_path = legacy_path
    if payload is None:
        if initialized_version is not None:
            raise ReactorStateError(
                f"event reactor cursor is missing after initialization: {state_path}; "
                "restore the cursor or remove the initialized marker only when an "
                "explicit re-prime is intended"
            )
        return {}
    state = _normalize_state(payload, path=loaded_path)
    if loaded_path == state_path and state["version"] in (2, 3, STATE_VERSION):
        if initialized_version != state["version"]:
            raise ReactorStateError(
                f"event reactor version-{state['version']} state is missing its matching "
                f"initialized marker: {state_path}"
            )
    return state


def _save_state(state_dir: Path, state: dict[str, Any]) -> None:
    """Persist reactor state atomically; raise :class:`ReactorStateError` on failure.

    The write goes through a temp file + ``os.replace`` so a crash or full
    disk mid-write can never leave a truncated state file. The initialized
    marker is atomically installed first when needed; a marker failure leaves
    the old state authoritative, while a later first-state failure leaves the
    marker behind to force an explicit repair instead of a silent re-prime. A
    failure must propagate: an ack that silently didn't commit is the one way
    this module can still lose events.
    """
    path = _state_path(state_dir)
    marker_path = _initialized_path(state_dir)
    tmp_path = path.with_suffix(".json.tmp")
    marker_tmp_path = marker_path.with_suffix(".tmp")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path.write_text(
            json.dumps({**state, "version": STATE_VERSION}) + "\n",
            encoding="utf-8",
        )
        # Establish the durable initialized sentinel before committing cursor
        # offsets. If marker persistence fails, os.replace() below never runs
        # and the previous cursor remains authoritative. If the later state
        # replace fails during first use, the marker makes the next cycle fail
        # closed instead of silently re-priming.
        if _read_initialized_version(state_dir) != STATE_VERSION:
            marker_tmp_path.write_text(f"{STATE_VERSION}\n", encoding="utf-8")
            os.replace(marker_tmp_path, marker_path)
        os.replace(tmp_path, path)
    except OSError as exc:
        for candidate in (tmp_path, marker_tmp_path):
            try:
                candidate.unlink(missing_ok=True)
            except OSError:
                pass
        raise ReactorStateError(f"could not persist event reactor state to {path}: {exc}") from exc
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
    """Last complete-record offsets used when priming or migrating state.

    A writer may have appended only part of its next JSONL record. Capturing
    raw ``st_size`` would checkpoint inside that record; after completion the
    saved position would either skip its prefix or fail the boundary check.
    Inspect at most one record-cap window from EOF and round down to the last
    newline instead.
    """
    snapshot: dict[str, int] = {}
    for path in _history_files(context_path, config=config):
        try:
            with path.open("rb") as handle:
                size = os.fstat(handle.fileno()).st_size
                if size == 0:
                    snapshot[path.name] = 0
                    continue
                start = max(0, size - (MAX_HISTORY_RECORD_BYTES + 1))
                handle.seek(start)
                tail = handle.read(MAX_HISTORY_RECORD_BYTES + 1)
        except OSError as exc:
            raise ReactorStateError(
                f"could not snapshot history record boundary for {path}: {exc}"
            ) from exc
        newline = tail.rfind(b"\n")
        if newline < 0:
            if start > 0:
                raise ReactorStateError(
                    f"trailing partial history record exceeds "
                    f"{MAX_HISTORY_RECORD_BYTES} bytes: {path}"
                )
            snapshot[path.name] = 0
        else:
            snapshot[path.name] = start + newline + 1
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


def _history_deferred_records(value: Any) -> list[_DeferredHistoryRecord]:
    if not isinstance(value, list):
        raise ReactorStateError("event reactor state field 'history_deferred' must be a list")
    records: list[_DeferredHistoryRecord] = []
    for item in value:
        if not isinstance(item, dict):
            raise ReactorStateError("event reactor history_deferred contains a non-object")
        filename = item.get("filename")
        offset = item.get("offset")
        length = item.get("length")
        digest = item.get("digest")
        timestamp = item.get("timestamp")
        if (
            not isinstance(filename, str)
            or not filename
            or Path(filename).name != filename
            or isinstance(offset, bool)
            or not isinstance(offset, int)
            or offset < 0
            or isinstance(length, bool)
            or not isinstance(length, int)
            or length <= 0
            or length > MAX_HISTORY_RECORD_BYTES
            or not isinstance(digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", digest)
            or not isinstance(timestamp, str)
            or _parse_timestamp(timestamp) is None
        ):
            raise ReactorStateError(
                "event reactor history_deferred contains an invalid record reference"
            )
        records.append(
            _DeferredHistoryRecord(
                filename=filename,
                offset=offset,
                length=length,
                digest=digest,
                timestamp=timestamp,
            )
        )
    return records


def _history_record_event(raw: bytes) -> ReactorEvent | None:
    """Parse one complete history line, excluding the hivemind mirror."""
    record = json.loads(raw.strip())
    if not isinstance(record, dict):
        raise ValueError("history record is not an object")
    kind = str(record.get("type", ""))
    if kind == HIVEMIND_KIND:
        return None
    stamp = _parse_timestamp(str(record.get("timestamp", "")))
    if stamp is None:
        raise ValueError("history record timestamp is invalid")
    metadata = record.get("metadata")
    return ReactorEvent(
        kind=kind,
        detail=str(record.get("op", "") or ""),
        source=str(record.get("source", "")),
        timestamp=str(record.get("timestamp", "")),
        metadata=metadata if isinstance(metadata, dict) else {},
    )


def _read_deferred_history_event(
    history_root: Path,
    reference: _DeferredHistoryRecord,
) -> ReactorEvent | None:
    path = history_root / reference.filename
    try:
        with path.open("rb") as handle:
            handle.seek(reference.offset)
            raw = handle.read(reference.length)
    except OSError as exc:
        raise ReactorStateError(
            f"could not reread deferred history record {path} at offset {reference.offset}: {exc}"
        ) from exc
    if len(raw) != reference.length or hashlib.sha256(raw).hexdigest() != reference.digest:
        raise ReactorStateError(
            f"deferred history record changed before delivery: {path} at offset {reference.offset}"
        )
    try:
        return _history_record_event(raw)
    except (UnicodeDecodeError, json.JSONDecodeError, ValueError) as exc:
        raise ReactorStateError(
            f"deferred history record became malformed: {path} at offset {reference.offset}: {exc}"
        ) from exc


def _history_events_since(
    context_path: Path,
    migration_watermark: datetime | None,
    until: datetime,
    *,
    config: Any = None,
    offsets: dict[str, int] | None = None,
    migration_cutoffs: dict[str, int] | None = None,
    deferred: list[_DeferredHistoryRecord] | None = None,
    limit: int = MAX_EVENTS_PER_CYCLE,
    max_scan_bytes: int = MAX_HISTORY_SCAN_BYTES,
) -> tuple[
    list[ReactorEvent],
    bool,
    int,
    dict[str, int],
    dict[str, int],
    list[_DeferredHistoryRecord],
]:
    """Stream one payload-bounded, resumable slice of append-only history files.

    File offsets, not timestamps, are the delivery checkpoint. The byte budget
    bounds record payload reads after file discovery, without assuming callers
    append timestamp-monotonic records.
    ``migration_cutoffs`` marks bytes that predate an older timestamp-only
    cursor. ``migration_watermark`` is that cursor's immutable value: the
    normal delivery watermark may advance while a bounded migration drains,
    but later out-of-order legacy records must still be compared with the
    original cursor. Offsets are returned separately and persist only when the
    enclosing batch is acked.
    """
    paths = _history_files(context_path, config=config)
    path_by_name = {path.name: path for path in paths}
    pending_offsets = dict(offsets or {})
    pending_cutoffs = dict(migration_cutoffs or {})
    pending_deferred = list(deferred or [])
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
    deferred_pending = False

    # A future-stamped record must not pin the append offset forever. Once its
    # containing batch is acked, the durable reference below lets the byte
    # checkpoint advance past it; it is reread and delivered when ripe.
    retained_deferred: list[_DeferredHistoryRecord] = []
    for reference in sorted(
        pending_deferred,
        key=lambda item: _parse_timestamp(item.timestamp)
        or datetime.min.replace(tzinfo=timezone.utc),
    ):
        stamp = _parse_timestamp(reference.timestamp)
        if stamp is None:
            raise ReactorStateError("deferred history record has an invalid timestamp")
        if stamp > until:
            retained_deferred.append(reference)
            deferred_pending = True
            continue
        if scanned_records >= record_limit:
            retained_deferred.append(reference)
            truncated = True
            continue
        path = path_by_name.get(reference.filename)
        if path is None:
            raise ReactorStateError(
                f"history file containing a deferred event is missing: {reference.filename}"
            )
        event = _read_deferred_history_event(path.parent, reference)
        scanned_records += 1
        if event is not None:
            events.append(event)
    pending_deferred = retained_deferred

    for path_index, path in enumerate(paths):
        if scanned_records >= record_limit:
            truncated = True
            break
        name = path.name
        offset = pending_offsets.get(name, 0)
        handle = None
        try:
            handle = path.open("rb")
            size = os.fstat(handle.fileno()).st_size
            if size < offset:
                handle.close()
                raise ReactorStateError(
                    f"history file shrank below its reactor checkpoint: {path} ({size} < {offset})"
                )
            if offset:
                handle.seek(offset - 1)
                if handle.read(1) != b"\n":
                    handle.close()
                    raise ReactorStateError(
                        f"history reactor checkpoint is not a record boundary: "
                        f"{path} at offset {offset}"
                    )
        except ReactorStateError:
            raise
        except OSError:
            if handle is not None:
                handle.close()
            continue
        assert handle is not None
        with handle:
            handle.seek(offset)
            while handle.tell() < size:
                if len(events) >= record_limit or scanned_records >= record_limit:
                    truncated = True
                    break
                if scanned_bytes >= byte_limit:
                    truncated = True
                    break
                line_start = handle.tell()
                # The cycle budget controls whether another record starts; a
                # started record may finish up to the per-record cap. Splitting
                # it at the cycle boundary would retry the same line forever
                # whenever its valid size exceeds ``max_scan_bytes``.
                raw = handle.readline(MAX_HISTORY_RECORD_BYTES + 1)
                if not raw:
                    break
                if len(raw) > MAX_HISTORY_RECORD_BYTES:
                    raise ReactorStateError(
                        f"history record exceeds {MAX_HISTORY_RECORD_BYTES} bytes: "
                        f"{path} at offset {line_start}"
                    )
                if not raw.endswith(b"\n"):
                    # A writer has not published a complete JSONL record yet.
                    # Leave the offset at its start so completion is retried.
                    handle.seek(line_start)
                    truncated = True
                    break

                pending_offsets[name] = handle.tell()
                scanned_records += 1
                scanned_bytes += len(raw)
                if not raw.strip():
                    continue
                try:
                    event = _history_record_event(raw)
                except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
                    skipped += 1
                    continue
                if event is None:
                    continue
                stamp = _parse_timestamp(event.timestamp)
                if stamp is None:
                    skipped += 1
                    continue
                cutoff = pending_cutoffs.get(name, 0)
                if (
                    line_start < cutoff
                    and migration_watermark is not None
                    and stamp <= migration_watermark
                ):
                    continue
                if stamp > until:
                    pending_deferred.append(
                        _DeferredHistoryRecord(
                            filename=name,
                            offset=line_start,
                            length=len(raw),
                            digest=hashlib.sha256(raw).hexdigest(),
                            timestamp=event.timestamp,
                        )
                    )
                    continue
                events.append(event)
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
    truncated = truncated or deferred_pending or bool(pending_deferred)
    return (
        events,
        truncated,
        skipped,
        pending_offsets,
        pending_cutoffs,
        pending_deferred,
    )


def _string_checkpoint_map(value: Any, *, key: str) -> dict[str, str]:
    if not isinstance(value, dict):
        raise ReactorStateError(f"event reactor state field {key!r} must be an object")
    result: dict[str, str] = {}
    for raw_name, raw_checkpoint in value.items():
        if not isinstance(raw_name, str) or not raw_name or not isinstance(raw_checkpoint, str):
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


@dataclass(frozen=True)
class _HivemindFile:
    directory: str
    filename: str
    path: Path
    signature: str
    mtime_ns: int


def _hivemind_file_signature(stat: os.stat_result) -> str:
    """Identity for one immutable final file in the current filesystem model."""
    return ":".join(
        str(value)
        for value in (
            stat.st_dev,
            stat.st_ino,
            stat.st_size,
            stat.st_mtime_ns,
            stat.st_ctime_ns,
        )
    )


def _valid_hivemind_signature(value: str) -> bool:
    return bool(re.fullmatch(r"-?\d+(?::-?\d+){4}", value))


def _hivemind_inventory(root: Path) -> dict[str, dict[str, _HivemindFile]]:
    """Return a metadata inventory, failing closed on an incomplete scan.

    Discovery necessarily visits every extant final ``*.json`` file so a file
    copied in later with an old name or mtime cannot hide behind a high-water
    mark. Message *contents* remain bounded and are read only for candidates.
    """
    inventory: dict[str, dict[str, _HivemindFile]] = {}
    try:
        directories = sorted(
            path for path in root.iterdir() if path.is_dir() and not path.name.startswith(".")
        )
        for directory in directories:
            files: dict[str, _HivemindFile] = {}
            with os.scandir(directory) as entries:
                for entry in entries:
                    if not entry.name.endswith(".json") or not entry.is_file(follow_symlinks=False):
                        continue
                    stat = entry.stat(follow_symlinks=False)
                    files[entry.name] = _HivemindFile(
                        directory=directory.name,
                        filename=entry.name,
                        path=Path(entry.path),
                        signature=_hivemind_file_signature(stat),
                        mtime_ns=stat.st_mtime_ns,
                    )
            inventory[directory.name] = files
    except OSError as exc:
        raise ReactorStateError(
            f"could not inventory hivemind message files under {root}: {exc}"
        ) from exc
    return inventory


def _hivemind_signature_snapshot(
    context_path: Path,
    *,
    config: Any = None,
) -> dict[str, dict[str, str]]:
    root = _hivemind_root(context_path, config=config)
    if root is None:
        return {}
    return {
        directory: {name: item.signature for name, item in files.items()}
        for directory, files in _hivemind_inventory(root).items()
        if files
    }


def _decode_hivemind_checkpoint(value: str) -> tuple[int, str]:
    if not value:
        return -1, ""
    raw_mtime, separator, filename = value.partition(":")
    if not separator or not raw_mtime.isdigit() or not filename:
        raise ReactorStateError("event reactor hivemind checkpoint is malformed")
    return int(raw_mtime), filename


def _nested_string_map(value: Any, *, key: str) -> dict[str, dict[str, str]]:
    if not isinstance(value, dict):
        raise ReactorStateError(f"event reactor state field {key!r} must be an object")
    result: dict[str, dict[str, str]] = {}
    for raw_directory, raw_files in value.items():
        if (
            not isinstance(raw_directory, str)
            or not raw_directory
            or not isinstance(raw_files, dict)
        ):
            raise ReactorStateError(
                f"event reactor state field {key!r} contains an invalid directory map"
            )
        files: dict[str, str] = {}
        for raw_filename, raw_signature in raw_files.items():
            if (
                not isinstance(raw_filename, str)
                or not raw_filename
                or Path(raw_filename).name != raw_filename
                or not isinstance(raw_signature, str)
                or not _valid_hivemind_signature(raw_signature)
            ):
                raise ReactorStateError(
                    f"event reactor state field {key!r} contains an invalid file identity"
                )
            files[raw_filename] = raw_signature
        if files:
            result[raw_directory] = files
    return result


def _hivemind_deferred_map(
    value: Any,
) -> dict[str, dict[str, dict[str, str]]]:
    if not isinstance(value, dict):
        raise ReactorStateError("event reactor state field 'hivemind_deferred' must be an object")
    result: dict[str, dict[str, dict[str, str]]] = {}
    for raw_directory, raw_files in value.items():
        if (
            not isinstance(raw_directory, str)
            or not raw_directory
            or not isinstance(raw_files, dict)
        ):
            raise ReactorStateError("event reactor hivemind_deferred is malformed")
        files: dict[str, dict[str, str]] = {}
        for raw_filename, raw_entry in raw_files.items():
            if (
                not isinstance(raw_filename, str)
                or not raw_filename
                or Path(raw_filename).name != raw_filename
                or not isinstance(raw_entry, dict)
            ):
                raise ReactorStateError("event reactor hivemind_deferred is malformed")
            signature = raw_entry.get("signature")
            timestamp = raw_entry.get("timestamp")
            if (
                not isinstance(signature, str)
                or not _valid_hivemind_signature(signature)
                or not isinstance(timestamp, str)
                or _parse_timestamp(timestamp) is None
            ):
                raise ReactorStateError("event reactor hivemind_deferred is malformed")
            files[raw_filename] = {
                "signature": signature,
                "timestamp": timestamp,
            }
        if files:
            result[raw_directory] = files
    return result


def _exact_inventory_subset(
    values: dict[str, dict[str, str]],
    inventory: dict[str, dict[str, _HivemindFile]],
) -> dict[str, dict[str, str]]:
    """Keep only identities still present exactly as previously observed."""
    result: dict[str, dict[str, str]] = {}
    for directory, files in values.items():
        current = inventory.get(directory, {})
        kept = {
            filename: signature
            for filename, signature in files.items()
            if filename in current and current[filename].signature == signature
        }
        if kept:
            result[directory] = kept
    return result


def _exact_deferred_subset(
    values: dict[str, dict[str, dict[str, str]]],
    inventory: dict[str, dict[str, _HivemindFile]],
) -> dict[str, dict[str, dict[str, str]]]:
    result: dict[str, dict[str, dict[str, str]]] = {}
    for directory, files in values.items():
        current = inventory.get(directory, {})
        kept = {
            filename: dict(entry)
            for filename, entry in files.items()
            if filename in current and current[filename].signature == entry.get("signature")
        }
        if kept:
            result[directory] = kept
    return result


def _seen_from_legacy_checkpoints(
    inventory: dict[str, dict[str, _HivemindFile]],
    checkpoints: dict[str, str],
) -> dict[str, dict[str, str]]:
    """Translate version-3 checkpoints without preserving their tuple-loss bug.

    The old ``(mtime, filename)`` high-water mark cannot distinguish a newly
    copied lower filename from one already delivered at the same mtime. Replay
    the ambiguous same-mtime group (except the checkpoint file itself): an
    at-least-once duplicate is safer than silently consuming a new message.
    """
    seen: dict[str, dict[str, str]] = {}
    for directory, files in inventory.items():
        checkpoint = _decode_hivemind_checkpoint(checkpoints.get(directory, ""))
        migrated = {}
        for filename, item in files.items():
            if item.mtime_ns < checkpoint[0] or (
                item.mtime_ns == checkpoint[0] and filename == checkpoint[1]
            ):
                migrated[filename] = item.signature
        if migrated:
            seen[directory] = migrated
    return seen


def _hivemind_candidates(
    inventory: dict[str, dict[str, _HivemindFile]],
    seen: dict[str, dict[str, str]],
    deferred: dict[str, dict[str, dict[str, str]]],
    until: datetime,
) -> tuple[Iterator[_HivemindFile], bool]:
    waiting = False
    candidates: list[_HivemindFile] = []
    for directory, files in inventory.items():
        for filename, item in files.items():
            if seen.get(directory, {}).get(filename) == item.signature:
                continue
            deferred_entry = deferred.get(directory, {}).get(filename)
            if deferred_entry and deferred_entry.get("signature") == item.signature:
                stamp = _parse_timestamp(deferred_entry.get("timestamp", ""))
                if stamp is not None and stamp > until:
                    waiting = True
                    continue
            candidates.append(item)
    return iter(candidates), waiting


def _hivemind_events_since(
    context_path: Path,
    migration_watermark: datetime | None,
    until: datetime,
    *,
    config: Any = None,
    seen: dict[str, dict[str, str]] | None = None,
    malformed: dict[str, dict[str, str]] | None = None,
    deferred: dict[str, dict[str, dict[str, str]]] | None = None,
    migration_existing: dict[str, dict[str, str]] | None = None,
    legacy_migration_cutoffs: dict[str, str] | None = None,
    limit: int = MAX_EVENTS_PER_CYCLE,
) -> tuple[
    list[ReactorEvent],
    bool,
    int,
    dict[str, dict[str, str]],
    dict[str, dict[str, str]],
    dict[str, dict[str, dict[str, str]]],
    dict[str, dict[str, str]],
    dict[str, str],
]:
    """Read bounded candidate payloads from a complete filesystem inventory."""
    from ..hivemind import HivemindMessage

    root = _hivemind_root(context_path, config=config)
    if root is None:
        return (
            [],
            False,
            0,
            dict(seen or {}),
            dict(malformed or {}),
            dict(deferred or {}),
            dict(migration_existing or {}),
            dict(legacy_migration_cutoffs or {}),
        )

    inventory = _hivemind_inventory(root)
    pending_seen = _exact_inventory_subset(seen or {}, inventory)
    pending_malformed = _exact_inventory_subset(malformed or {}, inventory)
    pending_deferred = _exact_deferred_subset(deferred or {}, inventory)
    pending_migration = _exact_inventory_subset(migration_existing or {}, inventory)
    pending_legacy_cutoffs = dict(legacy_migration_cutoffs or {})

    record_limit = limit if limit > 0 else MAX_EVENTS_PER_CYCLE
    candidates, waiting = _hivemind_candidates(inventory, pending_seen, pending_deferred, until)
    selected = nsmallest(
        record_limit + 1,
        candidates,
        key=lambda item: (item.mtime_ns, item.filename, item.directory),
    )
    truncated = waiting or len(selected) > record_limit
    events: list[ReactorEvent] = []
    skipped = 0
    expiry_now = datetime.now(timezone.utc)

    def mark_seen(item: _HivemindFile) -> None:
        pending_seen.setdefault(item.directory, {})[item.filename] = item.signature
        pending_malformed.get(item.directory, {}).pop(item.filename, None)
        pending_deferred.get(item.directory, {}).pop(item.filename, None)
        pending_migration.get(item.directory, {}).pop(item.filename, None)

    def observe_malformed(item: _HivemindFile, reason: str) -> None:
        nonlocal skipped, truncated
        previous = pending_malformed.get(item.directory, {}).get(item.filename)
        if previous == item.signature:
            skipped += 1
            _log.warning(
                "Event reactor permanently skipped stable malformed hivemind record %s: %s",
                item.path,
                reason,
            )
            mark_seen(item)
            return
        pending_malformed.setdefault(item.directory, {})[item.filename] = item.signature
        truncated = True

    for item in selected[:record_limit]:
        try:
            with item.path.open("rb") as handle:
                raw = handle.read(MAX_HIVEMIND_RECORD_BYTES + 1)
            current_stat = os.stat(item.path, follow_symlinks=False)
        except OSError as exc:
            # The directory inventory raced a copy/rename/permission change.
            # Nothing about this file is acknowledged; retry a later cycle.
            truncated = True
            _log.debug("Transient hivemind read failure for %s: %s", item.path, exc)
            continue
        if _hivemind_file_signature(current_stat) != item.signature:
            truncated = True
            continue
        if len(raw) > MAX_HIVEMIND_RECORD_BYTES:
            observe_malformed(item, f"record exceeds {MAX_HIVEMIND_RECORD_BYTES} bytes")
            continue
        try:
            data = json.loads(raw)
            if not isinstance(data, dict):
                raise ValueError("message is not an object")
            message = HivemindMessage.from_dict(data)
            stamp = _parse_timestamp(message.timestamp)
            if stamp is None:
                raise ValueError("message timestamp is invalid")
        except (UnicodeDecodeError, json.JSONDecodeError, TypeError, ValueError) as exc:
            # A non-atomic external writer may have exposed a partial final
            # file. Require the exact malformed identity to survive one acked
            # cycle before treating it as durable and advancing past it.
            observe_malformed(item, str(exc))
            continue

        pending_malformed.get(item.directory, {}).pop(item.filename, None)
        if stamp > until:
            pending_deferred.setdefault(item.directory, {})[item.filename] = {
                "signature": item.signature,
                "timestamp": message.timestamp,
            }
            truncated = True
            continue

        expires = _parse_timestamp(message.expires_at or "")
        legacy_cutoff = pending_legacy_cutoffs.get(item.directory, "")
        predates_legacy_cursor = bool(
            legacy_cutoff
            and (item.mtime_ns, item.filename) <= _decode_hivemind_checkpoint(legacy_cutoff)
            and migration_watermark is not None
            and stamp <= migration_watermark
        )
        existed_at_migration = (
            pending_migration.get(item.directory, {}).get(item.filename) == item.signature
            and migration_watermark is not None
            and stamp <= migration_watermark
        )
        mark_seen(item)
        if expires is not None and expires <= expiry_now:
            continue
        if predates_legacy_cursor or existed_at_migration:
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

    # A v3 migration cutoff is done only after every extant identity at or
    # below it has reached a terminal state. This legacy path cannot recover a
    # backdated file that v3 had already skipped before upgrading; v4 exact
    # identities prevent that loss for all subsequent discoveries.
    for directory, cutoff in list(pending_legacy_cutoffs.items()):
        boundary = _decode_hivemind_checkpoint(cutoff)
        remaining = any(
            (item.mtime_ns, item.filename) <= boundary
            and pending_seen.get(directory, {}).get(item.filename) != item.signature
            for item in inventory.get(directory, {}).values()
        )
        if not remaining:
            pending_legacy_cutoffs.pop(directory, None)

    for mapping in (pending_seen, pending_malformed, pending_migration):
        for directory in [name for name, files in mapping.items() if not files]:
            mapping.pop(directory, None)
    for directory in [name for name, files in pending_deferred.items() if not files]:
        pending_deferred.pop(directory, None)

    events.sort(key=_event_sort_key)
    return (
        events,
        truncated,
        skipped,
        pending_seen,
        pending_malformed,
        pending_deferred,
        pending_migration,
        pending_legacy_cutoffs,
    )


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
    _pending_history_deferred: list[_DeferredHistoryRecord] = field(default_factory=list)
    _pending_hivemind_seen: dict[str, dict[str, str]] = field(default_factory=dict)
    _pending_hivemind_malformed: dict[str, dict[str, str]] = field(default_factory=dict)
    _pending_hivemind_deferred: dict[str, dict[str, dict[str, str]]] = field(default_factory=dict)
    _pending_hivemind_migration_existing: dict[str, dict[str, str]] = field(default_factory=dict)
    _pending_hivemind_legacy_cutoffs: dict[str, str] = field(default_factory=dict)
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
            self._state["history_migration_cutoffs"] = self._pending_history_migration_cutoffs
        else:
            self._state.pop("history_migration_cutoffs", None)
            self._state.pop("history_migration_watermark", None)
        if self._pending_history_deferred:
            self._state["history_deferred"] = [
                reference.to_dict() for reference in self._pending_history_deferred
            ]
        else:
            self._state.pop("history_deferred", None)
        self._state["hivemind_seen"] = self._pending_hivemind_seen
        if self._pending_hivemind_malformed:
            self._state["hivemind_malformed"] = self._pending_hivemind_malformed
        else:
            self._state.pop("hivemind_malformed", None)
        if self._pending_hivemind_deferred:
            self._state["hivemind_deferred"] = self._pending_hivemind_deferred
        else:
            self._state.pop("hivemind_deferred", None)
        if self._pending_hivemind_migration_existing:
            self._state["hivemind_migration_existing"] = self._pending_hivemind_migration_existing
        else:
            self._state.pop("hivemind_migration_existing", None)
        if self._pending_hivemind_legacy_cutoffs:
            self._state["hivemind_migration_cutoffs"] = self._pending_hivemind_legacy_cutoffs
        else:
            self._state.pop("hivemind_migration_cutoffs", None)
        if (
            not self._pending_hivemind_migration_existing
            and not self._pending_hivemind_legacy_cutoffs
        ):
            self._state.pop("hivemind_migration_watermark", None)
        self._state.pop("hivemind_files", None)
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


def _start_positional_migration(
    context_path: Path,
    state_dir: Path,
    state: dict[str, Any],
    *,
    history_watermark: datetime,
    hivemind_watermark: datetime,
    config: Any = None,
) -> dict[str, Any]:
    """Atomically establish immutable v1/v2 migration boundaries.

    Migration state is persisted before any events are exposed for dispatch.
    That makes the source snapshots survive an unacked first batch: a
    backdated append arriving between retries is after the fixed cutoff and
    can never be reclassified as legacy input on the next cycle.
    """
    migrated = dict(state)
    migrated["version"] = STATE_VERSION
    migrated["history_offsets"] = {}
    history_cutoffs = _history_offset_snapshot(context_path, config=config)
    if history_cutoffs:
        migrated["history_migration_cutoffs"] = history_cutoffs
        migrated["history_migration_watermark"] = history_watermark.astimezone(
            timezone.utc
        ).isoformat()
    else:
        migrated.pop("history_migration_cutoffs", None)
        migrated.pop("history_migration_watermark", None)

    migrated["hivemind_seen"] = {}
    migrated.pop("hivemind_files", None)
    migrated.pop("hivemind_migration_cutoffs", None)
    hivemind_existing = _hivemind_signature_snapshot(context_path, config=config)
    if hivemind_existing:
        migrated["hivemind_migration_existing"] = hivemind_existing
        migrated["hivemind_migration_watermark"] = hivemind_watermark.astimezone(
            timezone.utc
        ).isoformat()
    else:
        migrated.pop("hivemind_migration_existing", None)
        migrated.pop("hivemind_migration_watermark", None)
    _save_state(state_dir, migrated)
    return migrated


def _upgrade_v3_state(
    context_path: Path,
    state_dir: Path,
    state: dict[str, Any],
    *,
    config: Any = None,
) -> dict[str, Any]:
    """Persist a version-4 exact-identity shell before reading v3 state."""
    checkpoints = _string_checkpoint_map(state["hivemind_files"], key="hivemind_files")
    root = _hivemind_root(context_path, config=config)
    inventory = _hivemind_inventory(root) if root is not None else {}
    upgraded = dict(state)
    upgraded["version"] = STATE_VERSION
    upgraded["hivemind_seen"] = _seen_from_legacy_checkpoints(inventory, checkpoints)
    upgraded.pop("hivemind_files", None)
    _save_state(state_dir, upgraded)
    return upgraded


def _migration_watermark(
    state: dict[str, Any],
    *,
    source: str,
) -> datetime | None:
    watermark_key = f"{source}_migration_watermark"
    if source == "hivemind":
        has_boundary = (
            "hivemind_migration_existing" in state or "hivemind_migration_cutoffs" in state
        )
    else:
        has_boundary = f"{source}_migration_cutoffs" in state
    if not has_boundary:
        return None
    parsed = _parse_timestamp(str(state.get(watermark_key, "")))
    if parsed is None:
        raise ReactorStateError(
            f"event reactor state has an invalid {watermark_key!r}; "
            "repair the migration state explicitly"
        )
    return parsed


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
            state["version"] = STATE_VERSION
            state["history_cursor"] = current.isoformat()
            hivemind_stored = current
            state["hivemind_cursor"] = current.isoformat()
            state["last_dispatch"] = {}
            state["history_offsets"] = _history_offset_snapshot(context_path, config=config)
            state["hivemind_seen"] = _hivemind_signature_snapshot(context_path, config=config)
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
                _pending_hivemind_seen=dict(state["hivemind_seen"]),
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

        if state["version"] in (1, 2):
            state = _start_positional_migration(
                context_path,
                state_dir,
                state,
                history_watermark=history_stored,
                hivemind_watermark=hivemind_stored,
                config=config,
            )
        elif state["version"] == 3:
            state = _upgrade_v3_state(
                context_path,
                state_dir,
                state,
                config=config,
            )

        history_offsets = _offset_map(state["history_offsets"], key="history_offsets")
        history_migration_cutoffs = _offset_map(
            state.get("history_migration_cutoffs", {}),
            key="history_migration_cutoffs",
        )
        history_migration_watermark = _migration_watermark(state, source="history")
        history_deferred = _history_deferred_records(state.get("history_deferred", []))

        hivemind_seen = _nested_string_map(state["hivemind_seen"], key="hivemind_seen")
        hivemind_migration_existing = _nested_string_map(
            state.get("hivemind_migration_existing", {}),
            key="hivemind_migration_existing",
        )
        legacy_hivemind_cutoffs = _string_checkpoint_map(
            state.get("hivemind_migration_cutoffs", {}),
            key="hivemind_migration_cutoffs",
        )
        hivemind_migration_watermark = _migration_watermark(state, source="hivemind")
        hivemind_malformed = _nested_string_map(
            state.get("hivemind_malformed", {}), key="hivemind_malformed"
        )
        hivemind_deferred = _hivemind_deferred_map(state.get("hivemind_deferred", {}))

        (
            history_events,
            history_truncated,
            history_skipped,
            pending_history_offsets,
            pending_history_migration_cutoffs,
            pending_history_deferred,
        ) = _history_events_since(
            context_path,
            history_migration_watermark,
            ripe_until,
            config=config,
            offsets=history_offsets,
            migration_cutoffs=history_migration_cutoffs,
            deferred=history_deferred,
            limit=max_events,
            max_scan_bytes=max_history_scan_bytes,
        )
        (
            hivemind_events,
            hivemind_truncated,
            hivemind_skipped,
            pending_hivemind_seen,
            pending_hivemind_malformed,
            pending_hivemind_deferred,
            pending_hivemind_migration_existing,
            pending_hivemind_legacy_cutoffs,
        ) = _hivemind_events_since(
            context_path,
            hivemind_migration_watermark,
            ripe_until,
            config=config,
            seen=hivemind_seen,
            malformed=hivemind_malformed,
            deferred=hivemind_deferred,
            migration_existing=hivemind_migration_existing,
            legacy_migration_cutoffs=legacy_hivemind_cutoffs,
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
            _pending_history_migration_cutoffs=(pending_history_migration_cutoffs),
            _pending_history_deferred=pending_history_deferred,
            _pending_hivemind_seen=pending_hivemind_seen,
            _pending_hivemind_malformed=pending_hivemind_malformed,
            _pending_hivemind_deferred=pending_hivemind_deferred,
            _pending_hivemind_migration_existing=(pending_hivemind_migration_existing),
            _pending_hivemind_legacy_cutoffs=pending_hivemind_legacy_cutoffs,
            acked=False,
        )
