"""React to new event-log entries and hivemind messages.

The supervisor's reconcile loop opens an :func:`open_event_batch` each cycle:
events are read in bounded source order under an exclusive lock and matched
against ``AgentConfig.on_event`` patterns. Successful/coalesced routes leave
the outbox; retryable routes remain in one durable slot per agent. Source
checkpoints and that outbox commit in the same ack, so one blocked route never
pins unrelated input and a failed commit loses neither source nor route.
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
from dataclasses import dataclass, field, replace
from datetime import datetime, timezone
from fnmatch import fnmatchcase
from pathlib import Path
from typing import Any

from ..context_paths import resolve_mount_root
from ..history import EVENT_FILE_PREFIX
from ..models import MountType
from ..protocols.canonical_json import (
    CanonicalJSONError,
    sha256_canonical_json,
    strict_json_loads,
)
from ..schema import MAX_AGENT_RESTARTS, AgentConfig
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
# Compatibility bound for future-record references written by earlier v4
# builds. Current v4 readers deliver complete positional records on arrival
# regardless of untrusted payload timestamps and create no new references.
MAX_HISTORY_DEFERRED_RECORDS = MAX_EVENTS_PER_CYCLE
LOCK_TIMEOUT_SECONDS = 5.0
# Retained for API compatibility. Version-4 positional/exact-identity
# checkpoints deliver complete records on durable arrival, so caller clocks
# no longer decide whether a visible record is consumed.
WATERMARK_GRACE_SECONDS = 5.0

VALID_EVENT_ACTIONS = ("spawn", "job")
MAX_LAUNCH_FAILURE_COUNT = MAX_AGENT_RESTARTS

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
    """Compatibility reference written by an earlier version-4 reader."""

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


@dataclass(frozen=True)
class _OversizedHistoryScan:
    """Bounded-memory progress through one newline-complete oversized record."""

    start: int
    scan_offset: int

    def to_dict(self) -> dict[str, int]:
        return {"start": self.start, "scan_offset": self.scan_offset}


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
    except (OverflowError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    # ``fromisoformat`` accepts boundary values whose offset cannot be
    # normalized (for example year 1 with +14:00).  Later comparisons and
    # ``astimezone`` calls would raise a raw OverflowError, so reject them at
    # the parsing boundary like every other malformed source/state stamp.
    try:
        parsed.astimezone(timezone.utc)
    except (OverflowError, ValueError):
        return None
    return parsed


def _state_timestamp(value: Any, *, key: str) -> datetime:
    if not isinstance(value, str):
        raise ReactorStateError(
            f"event reactor state field {key!r} must be a timestamp string"
        )
    parsed = _parse_timestamp(value)
    if parsed is None:
        raise ReactorStateError(
            f"event reactor state field {key!r} has an invalid timestamp"
        )
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
    except (OSError, UnicodeDecodeError) as exc:
        raise ReactorStateError(
            f"could not read event reactor initialized marker from {path}: {exc}"
        ) from exc
    supported = {"2": 2, "3": 3, str(STATE_VERSION): STATE_VERSION}
    if raw not in supported:
        raise ReactorStateError(f"event reactor initialized marker is malformed: {path}")
    return supported[raw]


def _read_state_file(path: Path) -> dict[str, Any] | None:
    try:
        raw = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return None
    except (OSError, UnicodeDecodeError) as exc:
        raise ReactorStateError(f"could not read event reactor state from {path}: {exc}") from exc
    try:
        payload = strict_json_loads(raw)
    except (
        CanonicalJSONError,
        json.JSONDecodeError,
        UnicodeDecodeError,
        RecursionError,
        ValueError,
    ) as exc:
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
    for agent_name, dispatched_at in last_dispatch.items():
        if not isinstance(agent_name, str) or not agent_name.strip():
            raise ReactorStateError(
                f"event reactor state field 'last_dispatch' has an invalid agent name: {path}"
            )
        try:
            _state_timestamp(dispatched_at, key=f"last_dispatch.{agent_name}")
        except ReactorStateError as exc:
            raise ReactorStateError(f"{exc}: {path}") from exc

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
        if initialized_version is not None:
            raise ReactorStateError(
                f"event reactor cursor is missing after initialization: {state_path}; "
                "restore the cursor or remove the initialized marker only when an "
                "explicit re-prime is intended"
            )
        # Migrate state written by earlier versions directly into the state
        # dir. The legacy file is removed on the next successful save, never
        # before, so a crash mid-migration cannot lose the cursors.
        legacy_path = state_dir / LEGACY_CURSOR_FILE_NAME
        payload = _read_state_file(legacy_path)
        loaded_path = legacy_path
    if payload is None:
        return {}
    state = _normalize_state(payload, path=loaded_path)
    if loaded_path == state_path and initialized_version != state["version"]:
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


def _utf8_checkpoint_name(value: str) -> bool:
    """Return whether a filesystem name can round-trip through JSON state.

    POSIX may expose undecodable bytes as surrogateescape code points.  JSON
    can emit those as ``\\udcXX`` escapes, but the reactor's strict loader
    correctly rejects unpaired surrogates on the next cycle.  Such names
    therefore cannot be used as durable checkpoint keys.
    """
    try:
        value.encode("utf-8", errors="strict")
    except UnicodeEncodeError:
        return False
    return True


def _history_files(
    context_path: Path,
    *,
    config: Any = None,
) -> tuple[list[Path], bool]:
    """Return history logs and whether the configured source is available.

    An unavailable mount is distinct from an available empty directory. Once
    positional checkpoints exist, treating a transient mount failure as an
    empty inventory would prune them and replay restored files from byte zero.
    """
    try:
        history_root = resolve_mount_root(context_path, MountType.HISTORY, config=config)
        if not history_root.is_dir():
            return [], False
        paths: list[Path] = []
        for path in history_root.glob(f"{EVENT_FILE_PREFIX}_*.jsonl"):
            if not _utf8_checkpoint_name(path.name):
                raise ReactorStateError(
                    "history source contains a filename that cannot be represented "
                    f"safely in reactor state: {os.fsencode(path.name)!r}; rename it as UTF-8"
                )
            paths.append(path)
        return sorted(paths), True
    except ReactorStateError:
        raise
    except Exception:  # noqa: BLE001 - provider resolution may fail
        return [], False


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
    paths, _available = _history_files(context_path, config=config)
    for path in paths:
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


def _history_round(value: Any) -> list[str]:
    """Validate the durable finite file round used for history fairness."""
    if not isinstance(value, list):
        raise ReactorStateError("event reactor state field 'history_round' must be a list")
    names: list[str] = []
    for raw_name in value:
        if (
            not isinstance(raw_name, str)
            or not raw_name
            or Path(raw_name).name != raw_name
            or not _utf8_checkpoint_name(raw_name)
            or raw_name in names
        ):
            raise ReactorStateError(
                "event reactor state field 'history_round' is malformed"
            )
        names.append(raw_name)
    return names


def _history_deferred_records(value: Any) -> list[_DeferredHistoryRecord]:
    if not isinstance(value, list):
        raise ReactorStateError("event reactor state field 'history_deferred' must be a list")
    if len(value) > MAX_HISTORY_DEFERRED_RECORDS:
        raise ReactorStateError(
            "event reactor state field 'history_deferred' exceeds the bounded record limit"
        )
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


def _history_oversized_map(value: Any) -> dict[str, _OversizedHistoryScan]:
    if not isinstance(value, dict):
        raise ReactorStateError("event reactor state field 'history_oversized' must be an object")
    scans: dict[str, _OversizedHistoryScan] = {}
    for filename, raw in value.items():
        if (
            not isinstance(filename, str)
            or not filename
            or Path(filename).name != filename
            or not isinstance(raw, dict)
        ):
            raise ReactorStateError("event reactor state field 'history_oversized' is malformed")
        start = raw.get("start")
        scan_offset = raw.get("scan_offset")
        if (
            isinstance(start, bool)
            or not isinstance(start, int)
            or start < 0
            or isinstance(scan_offset, bool)
            or not isinstance(scan_offset, int)
            or scan_offset <= start
        ):
            raise ReactorStateError("event reactor state field 'history_oversized' is malformed")
        scans[filename] = _OversizedHistoryScan(start=start, scan_offset=scan_offset)
    return scans


def _history_record_event(raw: bytes) -> ReactorEvent | None:
    """Parse one complete history line, excluding the hivemind mirror."""
    record = strict_json_loads(raw.strip())
    if not isinstance(record, dict):
        raise ValueError("history record is not an object")
    raw_kind = record.get("type")
    if not isinstance(raw_kind, str) or not raw_kind.strip():
        raise ValueError("history record type must be a non-empty string")
    kind = raw_kind
    if kind == HIVEMIND_KIND:
        return None
    raw_timestamp = record.get("timestamp")
    if not isinstance(raw_timestamp, str):
        raise ValueError("history record timestamp must be a string")
    stamp = _parse_timestamp(raw_timestamp)
    if stamp is None:
        raise ValueError("history record timestamp is invalid")
    metadata = record.get("metadata")
    return ReactorEvent(
        kind=kind,
        detail=str(record.get("op", "") or ""),
        source=str(record.get("source", "")),
        timestamp=raw_timestamp,
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
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError, ValueError) as exc:
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
    oversized: dict[str, _OversizedHistoryScan] | None = None,
    round_files: list[str] | None = None,
    limit: int = MAX_EVENTS_PER_CYCLE,
    max_scan_bytes: int = MAX_HISTORY_SCAN_BYTES,
) -> tuple[
    list[ReactorEvent],
    bool,
    int,
    dict[str, int],
    dict[str, int],
    list[_DeferredHistoryRecord],
    dict[str, _OversizedHistoryScan],
    list[str],
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
    paths, history_available = _history_files(context_path, config=config)
    path_by_name = {path.name: path for path in paths}
    pending_offsets = dict(offsets or {})
    pending_cutoffs = dict(migration_cutoffs or {})
    pending_deferred = list(deferred or [])
    pending_oversized = dict(oversized or {})
    if not history_available:
        if (
            pending_offsets
            or pending_cutoffs
            or pending_deferred
            or pending_oversized
            or round_files
        ):
            raise ReactorStateError(
                "history source is unavailable while reactor checkpoints exist; "
                "preserving state for retry"
            )
        return (
            [],
            False,
            0,
            pending_offsets,
            pending_cutoffs,
            pending_deferred,
            pending_oversized,
            list(round_files or []),
        )
    existing_names = {path.name for path in paths}
    for name in [name for name in pending_offsets if name not in existing_names]:
        pending_offsets.pop(name, None)
        pending_cutoffs.pop(name, None)
        pending_oversized.pop(name, None)
    pending_round = [name for name in (round_files or []) if name in existing_names]
    eligible_names: set[str] = set()
    for path in paths:
        try:
            has_unread_bytes = path.stat().st_size > pending_offsets.get(path.name, 0)
        except OSError:
            # Keep a transiently unstatable path in the finite round so later
            # files still get a turn and this identity is retried next round.
            has_unread_bytes = True
        if has_unread_bytes or path.name in pending_oversized:
            eligible_names.add(path.name)
    if not pending_round:
        pending_round = [path.name for path in paths if path.name in eligible_names]
    round_waiting = bool(eligible_names - set(pending_round))
    paths = [path_by_name[name] for name in pending_round]

    events: list[ReactorEvent] = []
    skipped = 0
    scanned_records = 0
    scanned_bytes = 0
    record_limit = limit if limit > 0 else MAX_EVENTS_PER_CYCLE
    byte_limit = max(max_scan_bytes, 1)
    truncated = False

    # Drain compatibility references from earlier v4 builds immediately.
    # Positional visibility, not the untrusted timestamp, now defines arrival.
    retained_deferred: list[_DeferredHistoryRecord] = []
    for reference in sorted(
        pending_deferred,
        key=lambda item: _parse_timestamp(item.timestamp)
        or datetime.min.replace(tzinfo=timezone.utc),
    ):
        path = path_by_name.get(reference.filename)
        if path is None:
            raise ReactorStateError(
                f"history file containing a deferred event is missing: {reference.filename}"
            )
        stamp = _parse_timestamp(reference.timestamp)
        if stamp is None:
            raise ReactorStateError("deferred history record has an invalid timestamp")
        if scanned_records >= record_limit:
            retained_deferred.append(reference)
            truncated = True
            continue
        event = _read_deferred_history_event(path.parent, reference)
        scanned_records += 1
        if event is not None:
            events.append(event)
    pending_deferred = retained_deferred

    for path in paths:
        if (
            len(events) >= record_limit
            or scanned_records >= record_limit
            or scanned_bytes >= byte_limit
        ):
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
        except OSError as exc:
            if handle is not None:
                try:
                    handle.close()
                except OSError:
                    pass
            _log.debug("Transient history open/checkpoint failure for %s: %s", path, exc)
            pending_round.remove(name)
            continue
        assert handle is not None
        try:
            with handle:
                handle.seek(offset)
                oversized_scan = pending_oversized.get(name)
                if oversized_scan is not None:
                    if oversized_scan.start != offset or oversized_scan.scan_offset > size:
                        raise ReactorStateError(
                            f"history oversized-record checkpoint is inconsistent: {path}"
                        )
                    handle.seek(oversized_scan.scan_offset)
                    while handle.tell() < size and scanned_bytes < byte_limit:
                        chunk_start = handle.tell()
                        chunk = handle.read(min(64 * 1024, byte_limit - scanned_bytes))
                        if not chunk:
                            break
                        newline = chunk.find(b"\n")
                        consumed = len(chunk) if newline < 0 else newline + 1
                        scanned_bytes += consumed
                        if newline >= 0:
                            record_end = chunk_start + consumed
                            pending_offsets[name] = record_end
                            pending_oversized.pop(name, None)
                            handle.seek(record_end)
                            scanned_records += 1
                            skipped += 1
                            _log.warning(
                                "Event reactor skipped oversized history record %s at offset %d",
                                path,
                                oversized_scan.start,
                            )
                            break
                        pending_oversized[name] = _OversizedHistoryScan(
                            start=oversized_scan.start,
                            scan_offset=handle.tell(),
                        )
                    if name in pending_oversized:
                        truncated = True
                while name not in pending_oversized and handle.tell() < size:
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
                        scanned_bytes += len(raw)
                        if raw.endswith(b"\n"):
                            pending_offsets[name] = handle.tell()
                            scanned_records += 1
                            skipped += 1
                            _log.warning(
                                "Event reactor skipped oversized history record %s at offset %d",
                                path,
                                line_start,
                            )
                            continue
                        pending_offsets[name] = line_start
                        pending_oversized[name] = _OversizedHistoryScan(
                            start=line_start,
                            scan_offset=handle.tell(),
                        )
                        truncated = True
                        break
                    if not raw.endswith(b"\n"):
                        # A writer has not published a complete JSONL record yet.
                        # Leave the offset at its start so completion is retried,
                        # but charge the bytes and continue to independent files
                        # while this cycle still has capacity.
                        scanned_bytes += len(raw)
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
                    except (
                        UnicodeDecodeError,
                        json.JSONDecodeError,
                        RecursionError,
                        ValueError,
                    ):
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
                    events.append(event)
                if pending_offsets.get(name, offset) < size:
                    truncated = True
                cutoff = pending_cutoffs.get(name)
                if cutoff is not None and pending_offsets.get(name, offset) >= cutoff:
                    pending_cutoffs.pop(name, None)
        except OSError as exc:
            raise ReactorStateError(
                f"could not scan history file {path} from offset {offset}: {exc}"
            ) from exc
        pending_round.remove(name)
        if (
            len(events) >= record_limit
            or scanned_records >= record_limit
            or scanned_bytes >= byte_limit
        ):
            truncated = True
            break

    events.sort(key=_event_sort_key)
    truncated = truncated or bool(pending_deferred) or bool(pending_round) or round_waiting
    return (
        events,
        truncated,
        skipped,
        pending_offsets,
        pending_cutoffs,
        pending_deferred,
        pending_oversized,
        pending_round,
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
        directories: list[Path] = []
        for path in root.iterdir():
            if not path.is_dir() or path.name.startswith("."):
                continue
            if not _utf8_checkpoint_name(path.name):
                raise ReactorStateError(
                    "hivemind source contains a directory name that cannot be represented "
                    f"safely in reactor state: {os.fsencode(path.name)!r}; rename it as UTF-8"
                )
            directories.append(path)
        directories.sort()
        for directory in directories:
            files: dict[str, _HivemindFile] = {}
            with os.scandir(directory) as entries:
                for entry in entries:
                    if not entry.name.endswith(".json") or not entry.is_file(follow_symlinks=False):
                        continue
                    if not _utf8_checkpoint_name(entry.name):
                        raise ReactorStateError(
                            "hivemind source contains a filename that cannot be represented "
                            "safely in reactor state: "
                            f"{os.fsencode(entry.name)!r}; rename it as UTF-8"
                        )
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
    if (
        not separator
        or not re.fullmatch(r"[0-9]{1,20}", raw_mtime)
        or not filename
    ):
        raise ReactorStateError("event reactor hivemind checkpoint is malformed")
    try:
        return int(raw_mtime), filename
    except ValueError as exc:
        raise ReactorStateError("event reactor hivemind checkpoint is malformed") from exc


def _hivemind_retry_cursor(value: Any, *, key: str) -> tuple[int, str, str] | None:
    """Validate the bounded round-robin cursor used for retry candidates."""
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ReactorStateError(f"event reactor state field {key!r} is malformed")
    mtime_ns = value.get("mtime_ns")
    filename = value.get("filename")
    directory = value.get("directory")
    if (
        isinstance(mtime_ns, bool)
        or not isinstance(mtime_ns, int)
        or not -(2**63) <= mtime_ns < 2**63
        or not isinstance(filename, str)
        or not filename
        or Path(filename).name != filename
        or not _utf8_checkpoint_name(filename)
        or not isinstance(directory, str)
        or not directory
        or Path(directory).name != directory
        or not _utf8_checkpoint_name(directory)
    ):
        raise ReactorStateError(f"event reactor state field {key!r} is malformed")
    return mtime_ns, filename, directory


def _hivemind_retry_cursor_dict(
    value: tuple[int, str, str] | None,
) -> dict[str, Any] | None:
    if value is None:
        return None
    mtime_ns, filename, directory = value
    return {"mtime_ns": mtime_ns, "filename": filename, "directory": directory}


def _hivemind_retry_phase(value: Any) -> bool:
    if value is None:
        return False
    if not isinstance(value, bool):
        raise ReactorStateError(
            "event reactor state field 'hivemind_retry_phase' must be boolean"
        )
    return value


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


def _hivemind_candidates(
    inventory: dict[str, dict[str, _HivemindFile]],
    seen: dict[str, dict[str, str]],
    deferred: dict[str, dict[str, dict[str, str]]],
    until: datetime,
) -> tuple[list[_HivemindFile], bool]:
    del until  # v4 exact identities deliver complete files on durable arrival.
    candidates: list[_HivemindFile] = []
    for directory, files in inventory.items():
        for filename, item in files.items():
            if seen.get(directory, {}).get(filename) == item.signature:
                continue
            candidates.append(item)
    return candidates, False


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
    discovery: dict[str, dict[str, str]] | None = None,
    retry: dict[str, dict[str, str]] | None = None,
    retry_cursor: tuple[int, str, str] | None = None,
    retry_phase: bool = False,
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
    dict[str, dict[str, str]],
    dict[str, dict[str, str]],
    tuple[int, str, str] | None,
    bool,
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
            dict(discovery or {}),
            dict(retry or {}),
            retry_cursor,
            retry_phase,
        )

    inventory = _hivemind_inventory(root)
    pending_seen = _exact_inventory_subset(seen or {}, inventory)
    pending_malformed = _exact_inventory_subset(malformed or {}, inventory)
    pending_deferred = _exact_deferred_subset(deferred or {}, inventory)
    pending_migration = _exact_inventory_subset(migration_existing or {}, inventory)
    pending_legacy_cutoffs = dict(legacy_migration_cutoffs or {})
    pending_discovery = _exact_inventory_subset(discovery or {}, inventory)
    pending_retry = _exact_inventory_subset(retry or {}, inventory)

    record_limit = limit if limit > 0 else MAX_EVENTS_PER_CYCLE
    candidates, waiting = _hivemind_candidates(inventory, pending_seen, pending_deferred, until)

    def candidate_key(item: _HivemindFile) -> tuple[int, str, str]:
        return item.mtime_ns, item.filename, item.directory

    def is_retry(item: _HivemindFile) -> bool:
        return (
            pending_retry.get(item.directory, {}).get(item.filename) == item.signature
            or pending_malformed.get(item.directory, {}).get(item.filename)
            == item.signature
            or pending_deferred.get(item.directory, {})
            .get(item.filename, {})
            .get("signature")
            == item.signature
        )

    candidate_identities = {
        (item.directory, item.filename): item.signature for item in candidates
    }
    for directory, files in list(pending_discovery.items()):
        pending_discovery[directory] = {
            filename: signature
            for filename, signature in files.items()
            if candidate_identities.get((directory, filename)) == signature
            and not is_retry(inventory[directory][filename])
        }
        if not pending_discovery[directory]:
            pending_discovery.pop(directory, None)
    if not pending_discovery:
        for item in candidates:
            if not is_retry(item):
                pending_discovery.setdefault(item.directory, {})[
                    item.filename
                ] = item.signature

    # Discovery and retry progress are independent.  Exact inventory finds
    # every identity, while a durable retry set records only identities whose
    # payload read/parse needs another attempt.  Reserving capacity for both
    # classes prevents a persistent unreadable oldest file from starving new
    # input and prevents sustained ingress from starving retries.
    unattempted = sorted(
        (
            item
            for item in candidates
            if pending_discovery.get(item.directory, {}).get(item.filename)
            == item.signature
        ),
        key=candidate_key,
    )
    retry_candidates = sorted((item for item in candidates if is_retry(item)), key=candidate_key)
    if retry_cursor is not None:
        retry_candidates = [
            *(item for item in retry_candidates if candidate_key(item) > retry_cursor),
            *(item for item in retry_candidates if candidate_key(item) <= retry_cursor),
        ]

    pending_retry_phase = retry_phase
    if record_limit == 1:
        if unattempted and retry_candidates:
            selected = [retry_candidates[0] if retry_phase else unattempted[0]]
            pending_retry_phase = not retry_phase
        elif unattempted:
            selected = [unattempted[0]]
            # A failure creates the retry class after selection.  Keep the
            # discovery phase once so the next unseen identity gets a chance;
            # the following mixed cycle then reserves the retry turn.
            pending_retry_phase = False
        elif retry_candidates:
            selected = [retry_candidates[0]]
            pending_retry_phase = False
        else:
            selected = []
            pending_retry_phase = False
    elif unattempted and retry_candidates:
        unattempted_slots = min(len(unattempted), record_limit - 1)
        selected = unattempted[:unattempted_slots] + retry_candidates[:1]
        remaining = record_limit - len(selected)
        if remaining:
            selected.extend(unattempted[unattempted_slots : unattempted_slots + remaining])
            remaining = record_limit - len(selected)
        if remaining:
            selected.extend(retry_candidates[1 : 1 + remaining])
    elif unattempted:
        selected = unattempted[:record_limit]
    else:
        selected = retry_candidates[:record_limit]

    truncated = waiting or len(candidates) > len(selected)
    pending_retry_cursor = retry_cursor
    events: list[ReactorEvent] = []
    skipped = 0
    expiry_now = datetime.now(timezone.utc)

    def mark_seen(item: _HivemindFile) -> None:
        pending_seen.setdefault(item.directory, {})[item.filename] = item.signature
        pending_discovery.get(item.directory, {}).pop(item.filename, None)
        pending_retry.get(item.directory, {}).pop(item.filename, None)
        pending_malformed.get(item.directory, {}).pop(item.filename, None)
        pending_deferred.get(item.directory, {}).pop(item.filename, None)
        pending_migration.get(item.directory, {}).pop(item.filename, None)

    def observe_malformed(item: _HivemindFile, reason: str) -> None:
        nonlocal skipped, truncated
        pending_discovery.get(item.directory, {}).pop(item.filename, None)
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

    for item in selected:
        if is_retry(item):
            pending_retry_cursor = candidate_key(item)
        try:
            with item.path.open("rb") as handle:
                raw = handle.read(MAX_HIVEMIND_RECORD_BYTES + 1)
            current_stat = os.stat(item.path, follow_symlinks=False)
        except OSError as exc:
            # The directory inventory raced a copy/rename/permission change.
            # Nothing about this file is acknowledged; retry a later cycle.
            truncated = True
            pending_discovery.get(item.directory, {}).pop(item.filename, None)
            pending_retry.setdefault(item.directory, {})[item.filename] = item.signature
            _log.debug("Transient hivemind read failure for %s: %s", item.path, exc)
            continue
        if _hivemind_file_signature(current_stat) != item.signature:
            truncated = True
            pending_discovery.get(item.directory, {}).pop(item.filename, None)
            pending_retry.get(item.directory, {}).pop(item.filename, None)
            continue
        pending_retry.get(item.directory, {}).pop(item.filename, None)
        if len(raw) > MAX_HIVEMIND_RECORD_BYTES:
            observe_malformed(item, f"record exceeds {MAX_HIVEMIND_RECORD_BYTES} bytes")
            continue
        try:
            data = strict_json_loads(raw)
            if not isinstance(data, dict):
                raise ValueError("message is not an object")
            if not isinstance(data.get("timestamp"), str):
                raise ValueError("message timestamp must be a string")
            message = HivemindMessage.from_dict(data)
            stamp = _parse_timestamp(message.timestamp)
            if stamp is None:
                raise ValueError("message timestamp is invalid")
            raw_expires = message.expires_at
            if raw_expires is None or raw_expires == "":
                expires = None
            else:
                if not isinstance(raw_expires, str):
                    raise ValueError("message expires_at must be a timestamp string")
                expires = _parse_timestamp(raw_expires)
                if expires is None:
                    raise ValueError("message expires_at is invalid")
        except (
            CanonicalJSONError,
            UnicodeDecodeError,
            json.JSONDecodeError,
            RecursionError,
            TypeError,
            ValueError,
        ) as exc:
            # A non-atomic external writer may have exposed a partial final
            # file. Require the exact malformed identity to survive one acked
            # cycle before treating it as durable and advancing past it.
            observe_malformed(item, str(exc))
            continue

        pending_malformed.get(item.directory, {}).pop(item.filename, None)
        if expires is not None and expires <= expiry_now:
            mark_seen(item)
            continue
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

    for mapping in (
        pending_seen,
        pending_malformed,
        pending_migration,
        pending_discovery,
        pending_retry,
    ):
        for directory in [name for name, files in mapping.items() if not files]:
            mapping.pop(directory, None)
    for directory in [name for name, files in pending_deferred.items() if not files]:
        pending_deferred.pop(directory, None)

    has_retry_work = bool(pending_retry or pending_malformed or pending_deferred)
    if not has_retry_work:
        pending_retry_cursor = None
        pending_retry_phase = False

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
        pending_discovery,
        pending_retry,
        pending_retry_cursor,
        pending_retry_phase,
    )


# ---------------------------------------------------------------------------
# Transactional batch: read under lock, dispatch, then ack
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ReactorDispatchOutcome:
    """One event route's terminal result for the current batch."""

    state: str
    reason: str


@dataclass(frozen=True)
class ReactorPendingRoute:
    """Durable coalesced delivery intent for one configured agent."""

    agent_name: str
    action: str
    reason: str
    config_digest: str
    queued_at: str
    launch_failure_count: int = 0
    last_launch_failure_at: str = ""
    launch_circuit_opened_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "action": self.action,
            "reason": self.reason,
            "config_digest": self.config_digest,
            "queued_at": self.queued_at,
        }
        if self.launch_failure_count:
            payload["launch_failure_count"] = self.launch_failure_count
            payload["last_launch_failure_at"] = self.last_launch_failure_at
        if self.launch_circuit_opened_at:
            payload["launch_circuit_opened_at"] = self.launch_circuit_opened_at
        return payload


def _valid_pending_route_reason(value: Any) -> bool:
    if not isinstance(value, str) or not value.startswith("event:"):
        return False
    label = value.removeprefix("event:")
    return bool(label) and sanitize_label(label) == label


def _pending_route_map(value: Any) -> dict[str, ReactorPendingRoute]:
    if not isinstance(value, dict):
        raise ReactorStateError("event reactor state field 'pending_routes' must be an object")
    routes: dict[str, ReactorPendingRoute] = {}
    for agent_name, raw in value.items():
        if (
            not isinstance(agent_name, str)
            or not agent_name.strip()
            or not isinstance(raw, dict)
        ):
            raise ReactorStateError("event reactor state field 'pending_routes' is malformed")
        action = raw.get("action")
        reason = raw.get("reason")
        config_digest = raw.get("config_digest")
        queued_at = raw.get("queued_at")
        launch_failure_count = raw.get("launch_failure_count", 0)
        last_launch_failure_at = raw.get("last_launch_failure_at", "")
        launch_circuit_opened_at = raw.get("launch_circuit_opened_at", "")
        if (
            action not in VALID_EVENT_ACTIONS
            or not _valid_pending_route_reason(reason)
            or not isinstance(config_digest, str)
            or not re.fullmatch(r"[0-9a-f]{64}", config_digest)
            or not isinstance(queued_at, str)
            or _parse_timestamp(queued_at) is None
            or isinstance(launch_failure_count, bool)
            or not isinstance(launch_failure_count, int)
            or launch_failure_count < 0
            or launch_failure_count > MAX_LAUNCH_FAILURE_COUNT
            or not isinstance(last_launch_failure_at, str)
            or bool(last_launch_failure_at)
            != bool(launch_failure_count)
            or (
                bool(last_launch_failure_at)
                and _parse_timestamp(last_launch_failure_at) is None
            )
            or not isinstance(launch_circuit_opened_at, str)
            or (
                bool(launch_circuit_opened_at)
                and (
                    launch_failure_count == 0
                    or _parse_timestamp(launch_circuit_opened_at) is None
                )
            )
        ):
            raise ReactorStateError("event reactor state field 'pending_routes' is malformed")
        routes[agent_name] = ReactorPendingRoute(
            agent_name=agent_name,
            action=action,
            reason=reason,
            config_digest=config_digest,
            queued_at=queued_at,
            launch_failure_count=launch_failure_count,
            last_launch_failure_at=last_launch_failure_at,
            launch_circuit_opened_at=launch_circuit_opened_at,
        )
    return routes


def event_config_digest(config: AgentConfig) -> str:
    """Hash stable trigger authorization, not retry-recovery mechanics.

    Module/dependency/debounce edits commonly *fix* a parked route and must
    not erase it. Rule/action changes alter what was authorized and therefore
    terminally invalidate the old route.
    """
    try:
        return sha256_canonical_json(
            {
                "name": config.name,
                "on_event": sorted(
                    {pattern.strip() for pattern in config.on_event if pattern.strip()}
                ),
                "on_event_action": config.on_event_action,
            }
        )
    except (CanonicalJSONError, TypeError, ValueError) as exc:
        raise ReactorStateError(
            f"agent config {config.name!r} cannot be hashed for event delivery: {exc}"
        ) from exc


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
    _pending_history_oversized: dict[str, _OversizedHistoryScan] = field(default_factory=dict)
    _pending_history_round: list[str] = field(default_factory=list)
    _pending_hivemind_seen: dict[str, dict[str, str]] = field(default_factory=dict)
    _pending_hivemind_malformed: dict[str, dict[str, str]] = field(default_factory=dict)
    _pending_hivemind_deferred: dict[str, dict[str, dict[str, str]]] = field(default_factory=dict)
    _pending_hivemind_migration_existing: dict[str, dict[str, str]] = field(default_factory=dict)
    _pending_hivemind_legacy_cutoffs: dict[str, str] = field(default_factory=dict)
    _pending_hivemind_discovery: dict[str, dict[str, str]] = field(default_factory=dict)
    _pending_hivemind_retry: dict[str, dict[str, str]] = field(default_factory=dict)
    _pending_hivemind_retry_cursor: tuple[int, str, str] | None = None
    _pending_hivemind_retry_phase: bool = False
    _pending_routes: dict[str, ReactorPendingRoute] = field(default_factory=dict)
    _active_routes: set[str] = field(default_factory=set)
    acked: bool = False
    dispatch_failures: int = 0
    dispatch_outcomes: dict[str, ReactorDispatchOutcome] = field(default_factory=dict)

    def pending_routes(self, *, action: str | None = None) -> list[ReactorPendingRoute]:
        """Return durable routes, oldest first, optionally for one action."""
        routes = (
            route
            for route in self._pending_routes.values()
            if action is None or route.action == action
        )
        return sorted(routes, key=lambda route: (route.queued_at, route.agent_name))

    def pending_route(self, agent_name: str) -> ReactorPendingRoute | None:
        """Return one durable route, including its launch retry metadata."""
        return self._pending_routes.get(agent_name)

    def prune_pending_routes(
        self,
        valid_routes: dict[str, tuple[str, str]],
    ) -> None:
        """Terminally reject routes whose config disappeared or changed."""
        for name, route in list(self._pending_routes.items()):
            valid = valid_routes.get(name)
            if valid == (route.action, route.config_digest):
                continue
            self._pending_routes.pop(name, None)
            self._active_routes.discard(name)
            reason = "config_removed" if valid is None else "config_changed"
            self._record_dispatch_outcome(name, "rejected", reason)

    def register_route(
        self,
        agent_name: str,
        *,
        action: str,
        reason: str,
        config_digest: str,
    ) -> ReactorPendingRoute:
        """Coalesce one source match into the agent's durable pending slot."""
        if (
            not isinstance(agent_name, str)
            or not agent_name.strip()
            or action not in VALID_EVENT_ACTIONS
            or not _valid_pending_route_reason(reason)
            or not re.fullmatch(r"[0-9a-f]{64}", config_digest)
        ):
            raise ReactorStateError(
                f"cannot register malformed event route for {agent_name!r}"
            )
        existing = self._pending_routes.get(agent_name)
        if existing is not None and (
            existing.action != action or existing.config_digest != config_digest
        ):
            self._pending_routes.pop(agent_name, None)
            self._record_dispatch_outcome(agent_name, "rejected", "config_changed")
            existing = None
        if existing is None:
            existing = ReactorPendingRoute(
                agent_name=agent_name,
                action=action,
                reason=reason,
                config_digest=config_digest,
                queued_at=self._now.isoformat(),
            )
            self._pending_routes[agent_name] = existing
        self._active_routes.add(agent_name)
        return existing

    def record_launch_failure(self, agent_name: str, *, at: datetime) -> None:
        """Persist one failed process launch in the route's retry policy state."""
        route = self._pending_routes.get(agent_name)
        if route is None:
            raise ReactorStateError(
                f"cannot record launch failure for unregistered route {agent_name!r}"
            )
        if route.launch_failure_count >= MAX_LAUNCH_FAILURE_COUNT:
            raise ReactorStateError(
                f"launch failure count exceeds safe bound for route {agent_name!r}"
            )
        self._pending_routes[agent_name] = replace(
            route,
            launch_failure_count=route.launch_failure_count + 1,
            last_launch_failure_at=at.isoformat(),
            launch_circuit_opened_at="",
        )

    def open_launch_circuit(self, agent_name: str, *, at: datetime) -> None:
        """Persist the start of a launch circuit cooldown without dropping work."""
        route = self._pending_routes.get(agent_name)
        if route is None or route.launch_failure_count <= 0:
            raise ReactorStateError(
                f"cannot open launch circuit for route {agent_name!r} without a failure"
            )
        self._pending_routes[agent_name] = replace(
            route,
            launch_circuit_opened_at=at.isoformat(),
        )

    def reset_launch_failures(self, agent_name: str) -> None:
        """Reset retry metadata after circuit cooldown while retaining the route."""
        route = self._pending_routes.get(agent_name)
        if route is None:
            raise ReactorStateError(
                f"cannot reset launch failures for unregistered route {agent_name!r}"
            )
        self._pending_routes[agent_name] = replace(
            route,
            launch_failure_count=0,
            last_launch_failure_at="",
            launch_circuit_opened_at="",
        )

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
        self._pending_routes.pop(agent_name, None)
        self._active_routes.discard(agent_name)
        self._record_dispatch_outcome(agent_name, "dispatched", reason)

    def mark_dispatch_deferred(self, agent_name: str, *, reason: str) -> None:
        """Record a retryable gate while retaining its durable pending route."""
        if agent_name not in self._pending_routes:
            raise ReactorStateError(
                f"cannot defer event route for {agent_name!r} before it is registered"
            )
        if not self._record_dispatch_outcome(agent_name, "deferred", reason):
            return
        _log.warning(
            "Event delivery for agent '%s' deferred (%s); route parked for retry",
            agent_name,
            reason,
        )

    def mark_coalesced(self, agent_name: str, *, reason: str) -> None:
        """Record an intentional delivery coalescing decision."""
        self._pending_routes.pop(agent_name, None)
        self._active_routes.discard(agent_name)
        self._record_dispatch_outcome(agent_name, "coalesced", reason)

    def mark_rejected(self, agent_name: str, *, reason: str) -> bool:
        """Record a terminal fail-closed rejection and permit cursor advance."""
        self._pending_routes.pop(agent_name, None)
        self._active_routes.discard(agent_name)
        return self._record_dispatch_outcome(agent_name, "rejected", reason)

    def mark_dispatch_failed(
        self,
        agent_name: str,
        *,
        reason: str = "dispatch_failed",
    ) -> None:
        """Record that an event-triggered dispatch failed.

        The registered route remains in the durable outbox while source
        offsets can advance independently.
        """
        self.mark_dispatch_deferred(agent_name, reason=reason)

    def prune_dispatch(self, active_names: Iterable[str]) -> None:
        """Drop persisted dispatch times for agents no longer configured."""
        table = self._state.get("last_dispatch", {})
        keep = set(active_names)
        for name in [key for key in table if key not in keep]:
            table.pop(name, None)

    def ack(self) -> None:
        """Atomically commit source cursors and the durable route outbox.

        A retryable dispatch does not pin unrelated source backlog: its
        coalesced route is parked in ``pending_routes`` in the same atomic
        state write as source offsets. A failed state save commits neither.
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
        if self._pending_history_oversized:
            self._state["history_oversized"] = {
                name: scan.to_dict()
                for name, scan in self._pending_history_oversized.items()
            }
        else:
            self._state.pop("history_oversized", None)
        if self._pending_history_round:
            self._state["history_round"] = self._pending_history_round
        else:
            self._state.pop("history_round", None)
        self._state.pop("history_scan_cursor", None)
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
        if self._pending_hivemind_discovery:
            self._state["hivemind_discovery"] = self._pending_hivemind_discovery
        else:
            self._state.pop("hivemind_discovery", None)
        if self._pending_hivemind_retry:
            self._state["hivemind_retry"] = self._pending_hivemind_retry
        else:
            self._state.pop("hivemind_retry", None)
        retry_cursor = _hivemind_retry_cursor_dict(self._pending_hivemind_retry_cursor)
        if retry_cursor is not None:
            self._state["hivemind_retry_cursor"] = retry_cursor
        else:
            self._state.pop("hivemind_retry_cursor", None)
        if self._pending_hivemind_retry_phase:
            self._state["hivemind_retry_phase"] = True
        else:
            self._state.pop("hivemind_retry_phase", None)
        # Compatibility with the first v4 fairness cursor.  Exact retry state
        # supersedes its global candidate rotation on the next successful ack.
        self._state.pop("hivemind_scan_cursor", None)
        self._state.pop("hivemind_scan_wrap", None)
        self._state.pop("hivemind_files", None)
        if self._pending_routes:
            self._state["pending_routes"] = {
                name: route.to_dict() for name, route in self._pending_routes.items()
            }
        else:
            self._state.pop("pending_routes", None)
        _save_state(self._state_dir, self._state)
        self.acked = True


def _source_watermark(
    events: list[ReactorEvent],
    stored: datetime,
    *,
    ceiling: datetime | None = None,
) -> str:
    """Cursor value a source may advance to after this batch is dispatched.

    Version-4 delivery uses positional/exact-identity checkpoints, not this
    diagnostic timestamp. Clamp caller-future stamps so a bad source clock
    does not make legacy metadata jump arbitrarily far ahead.
    """
    newest = stored
    for event in events:
        stamp = _parse_timestamp(event.timestamp)
        if stamp is not None and ceiling is not None and stamp > ceiling:
            stamp = ceiling
        if stamp is not None and stamp > newest:
            newest = stamp
    return newest.astimezone(timezone.utc).isoformat()


def _start_positional_migration(
    state_dir: Path,
    state: dict[str, Any],
) -> dict[str, Any]:
    """Persist a fail-safe v4 replay shell for timestamp-only v1/v2 state.

    A timestamp cursor cannot prove when a backdated record actually landed.
    Filesystem mtimes cannot close that gap either: history files mix many
    append times and copied hivemind files may preserve caller-controlled
    mtimes. Conservatively replay extant source content once under positional
    and exact-identity checkpoints rather than silently acknowledge a record
    that arrived after the legacy state was written.
    """
    migrated = dict(state)
    migrated["version"] = STATE_VERSION
    migrated["history_offsets"] = {}
    migrated.pop("history_migration_cutoffs", None)
    migrated.pop("history_migration_watermark", None)
    migrated["hivemind_seen"] = {}
    migrated.pop("hivemind_files", None)
    migrated.pop("hivemind_migration_existing", None)
    migrated.pop("hivemind_migration_cutoffs", None)
    migrated.pop("hivemind_migration_watermark", None)
    _save_state(state_dir, migrated)
    return migrated


def _upgrade_v3_state(
    state_dir: Path,
    state: dict[str, Any],
) -> dict[str, Any]:
    """Persist a fail-safe version-4 shell before reading tuple-based v3 state.

    Version 3 recorded only ``(mtime, filename)`` high-water marks. It cannot
    prove whether any currently extant file at or below a checkpoint is the
    identity previously delivered or a post-snapshot backdated copy. Replay
    the extant inventory once under v4 exact identities rather than translate
    an ambiguous tuple into a silent acknowledgement.
    """
    _string_checkpoint_map(state["hivemind_files"], key="hivemind_files")
    if "hivemind_migration_cutoffs" in state:
        _string_checkpoint_map(
            state["hivemind_migration_cutoffs"],
            key="hivemind_migration_cutoffs",
        )
        _state_timestamp(
            state.get("hivemind_migration_watermark"),
            key="hivemind_migration_watermark",
        )
    upgraded = dict(state)
    upgraded["version"] = STATE_VERSION
    upgraded["hivemind_seen"] = {}
    upgraded.pop("hivemind_files", None)
    upgraded.pop("hivemind_migration_cutoffs", None)
    upgraded.pop("hivemind_migration_watermark", None)
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
    return _state_timestamp(state.get(watermark_key), key=watermark_key)


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

    ``grace_seconds`` is retained for call compatibility. Version-4
    positional and exact-identity checkpoints deliver complete records when
    they are durably visible; untrusted payload timestamps do not gate them.

    A genuinely absent state is primed to *now* once, so first enabling the
    reactor never replays weeks of history as a spawn storm. Once initialized,
    missing, unreadable, malformed, or partial cursor state raises
    :class:`ReactorStateError` instead of silently skipping backlog.
    """
    current = now or datetime.now(timezone.utc)
    del grace_seconds
    with ExitStack() as stack:
        try:
            stack.enter_context(_file_lock(_state_path(state_dir), timeout=lock_timeout))
        except TimeoutError as exc:
            raise ReactorBusyError(
                f"event reactor lock is held by another process ({exc})"
            ) from exc
        except OSError as exc:
            raise ReactorStateError(
                f"could not set up event reactor lock under {state_dir}: {exc}"
            ) from exc

        state = _load_state(state_dir)
        fresh_state = not state
        history_stored = (
            _state_timestamp(state.get("history_cursor"), key="history_cursor")
            if state
            else None
        )
        hivemind_stored = (
            _state_timestamp(state.get("hivemind_cursor"), key="hivemind_cursor")
            if state
            else None
        )

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
        assert history_stored is not None and hivemind_stored is not None

        if state["version"] in (1, 2):
            state = _start_positional_migration(
                state_dir,
                state,
            )
        elif state["version"] == 3:
            state = _upgrade_v3_state(
                state_dir,
                state,
            )

        history_offsets = _offset_map(state["history_offsets"], key="history_offsets")
        history_migration_cutoffs = _offset_map(
            state.get("history_migration_cutoffs", {}),
            key="history_migration_cutoffs",
        )
        history_migration_watermark = _migration_watermark(state, source="history")
        history_deferred = _history_deferred_records(state.get("history_deferred", []))
        history_oversized = _history_oversized_map(state.get("history_oversized", {}))
        history_round = _history_round(state.get("history_round", []))
        # The development-only partial-tail cursor is superseded by finite
        # file rounds. Validate its type before dropping it on the next ack.
        legacy_history_cursor = state.get("history_scan_cursor")
        if legacy_history_cursor is not None and (
            not isinstance(legacy_history_cursor, str)
            or not legacy_history_cursor
            or Path(legacy_history_cursor).name != legacy_history_cursor
            or not _utf8_checkpoint_name(legacy_history_cursor)
        ):
            raise ReactorStateError(
                "event reactor state field 'history_scan_cursor' is malformed"
            )

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
        hivemind_discovery = _nested_string_map(
            state.get("hivemind_discovery", {}), key="hivemind_discovery"
        )
        hivemind_retry = _nested_string_map(
            state.get("hivemind_retry", {}), key="hivemind_retry"
        )
        legacy_scan_cursor = _hivemind_retry_cursor(
            state.get("hivemind_scan_cursor"), key="hivemind_scan_cursor"
        )
        hivemind_retry_cursor = _hivemind_retry_cursor(
            state.get("hivemind_retry_cursor"), key="hivemind_retry_cursor"
        )
        if hivemind_retry_cursor is None:
            hivemind_retry_cursor = legacy_scan_cursor
        # Validate and discard the development-only global wrap flag if an
        # intermediate state file contains it.
        if "hivemind_scan_wrap" in state and not isinstance(
            state["hivemind_scan_wrap"], bool
        ):
            raise ReactorStateError(
                "event reactor state field 'hivemind_scan_wrap' must be boolean"
            )
        hivemind_retry_phase = _hivemind_retry_phase(
            state.get("hivemind_retry_phase")
        )
        pending_routes = _pending_route_map(state.get("pending_routes", {}))

        (
            history_events,
            history_truncated,
            history_skipped,
            pending_history_offsets,
            pending_history_migration_cutoffs,
            pending_history_deferred,
            pending_history_oversized,
            pending_history_round,
        ) = _history_events_since(
            context_path,
            history_migration_watermark,
            current,
            config=config,
            offsets=history_offsets,
            migration_cutoffs=history_migration_cutoffs,
            deferred=history_deferred,
            oversized=history_oversized,
            round_files=history_round,
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
            pending_hivemind_discovery,
            pending_hivemind_retry,
            pending_hivemind_retry_cursor,
            pending_hivemind_retry_phase,
        ) = _hivemind_events_since(
            context_path,
            hivemind_migration_watermark,
            current,
            config=config,
            seen=hivemind_seen,
            malformed=hivemind_malformed,
            deferred=hivemind_deferred,
            migration_existing=hivemind_migration_existing,
            legacy_migration_cutoffs=legacy_hivemind_cutoffs,
            discovery=hivemind_discovery,
            retry=hivemind_retry,
            retry_cursor=hivemind_retry_cursor,
            retry_phase=hivemind_retry_phase,
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
            _pending_history_cursor=_source_watermark(
                history_events, history_stored, ceiling=current
            ),
            _pending_hivemind_cursor=_source_watermark(
                hivemind_events, hivemind_stored, ceiling=current
            ),
            _now=current,
            _pending_history_offsets=pending_history_offsets,
            _pending_history_migration_cutoffs=(pending_history_migration_cutoffs),
            _pending_history_deferred=pending_history_deferred,
            _pending_history_oversized=pending_history_oversized,
            _pending_history_round=pending_history_round,
            _pending_hivemind_seen=pending_hivemind_seen,
            _pending_hivemind_malformed=pending_hivemind_malformed,
            _pending_hivemind_deferred=pending_hivemind_deferred,
            _pending_hivemind_migration_existing=(pending_hivemind_migration_existing),
            _pending_hivemind_legacy_cutoffs=pending_hivemind_legacy_cutoffs,
            _pending_hivemind_discovery=pending_hivemind_discovery,
            _pending_hivemind_retry=pending_hivemind_retry,
            _pending_hivemind_retry_cursor=pending_hivemind_retry_cursor,
            _pending_hivemind_retry_phase=pending_hivemind_retry_phase,
            _pending_routes=pending_routes,
            acked=False,
        )
