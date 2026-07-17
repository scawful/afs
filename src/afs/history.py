"""History logging helpers for AFS."""

from __future__ import annotations

import hashlib
import json
import os
import re
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import load_config_model
from .context_layout import LAYOUT_VERSION, detect_layout_version
from .context_paths import resolve_mount_root
from .core import find_root, resolve_context_root
from .index_storage import fsync_directory
from .models import MountType
from .path_safety import assert_no_linklike_components, lexical_absolute

SENSITIVE_MARKERS = ("key", "token", "secret", "password")
EVENT_FILE_PREFIX = "events"
DEFAULT_MAX_INLINE_CHARS = 4000
SAFE_HISTORY_NAME = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")

# Structured event type constants
EVENT_MCP_TOOL = "mcp_tool"
EVENT_HIVEMIND = "hivemind"
EVENT_EMBEDDING = "embedding"
EVENT_AGENT_LIFECYCLE = "agent_lifecycle"
EVENT_SESSION = "session"


@dataclass(frozen=True)
class HistoryEvent:
    """Immutable history event record."""

    event_id: str
    timestamp: str
    event_type: str
    source: str
    op: str | None
    context_root: str | None
    metadata: dict[str, Any]
    payload: dict[str, Any] | None
    payload_ref: str | None
    payload_sha256: str | None
    payload_preview: str | None

    def to_dict(self) -> dict:
        payload: dict[str, Any] = {
            "id": self.event_id,
            "timestamp": self.timestamp,
            "type": self.event_type,
            "source": self.source,
            "op": self.op,
            "context_root": self.context_root,
            "metadata": self.metadata,
        }
        if self.payload is not None:
            payload["payload"] = self.payload
        if self.payload_ref:
            payload["payload_ref"] = self.payload_ref
        if self.payload_sha256:
            payload["payload_sha256"] = self.payload_sha256
        if self.payload_preview:
            payload["payload_preview"] = self.payload_preview
        return payload


@dataclass(frozen=True)
class _HistoryRouting:
    """Resolved storage roots for one history namespace."""

    write_root: Path
    context_root: Path | None = None
    legacy_root: Path | None = None

    @property
    def is_v2(self) -> bool:
        return self.context_root is not None


def _should_redact(arg: str) -> bool:
    lowered = arg.lower()
    return any(marker in lowered for marker in SENSITIVE_MARKERS)


def sanitize_argv(argv: Iterable[str]) -> list[str]:
    """Redact sensitive arguments from argv."""
    sanitized: list[str] = []
    redact_next = False

    for arg in argv:
        if redact_next:
            sanitized.append("[redacted]")
            redact_next = False
            continue

        if "=" in arg:
            key, value = arg.split("=", 1)
            if _should_redact(key):
                sanitized.append(f"{key}=[redacted]")
                continue
        if arg.startswith("--") and _should_redact(arg):
            sanitized.append(arg)
            redact_next = True
            continue
        sanitized.append(arg)

    return sanitized


def _resolve_context_root(start_dir: Path | None = None) -> Path | None:
    config = load_config_model()
    root = find_root(start_dir)
    try:
        return resolve_context_root(config, root)
    except Exception:
        return None


def _assert_v2_context_root(context_root: Path) -> Path:
    """Validate the v2 namespace root itself without following a replacement."""

    root = lexical_absolute(context_root)
    return assert_no_linklike_components(
        root,
        boundary=root.parent,
        allow_missing=False,
    )


def _v2_routing_for_context(context_root: Path) -> _HistoryRouting | None:
    """Return fixed v2 history routing, or ``None`` for a genuine v1 root."""

    lexical_root = lexical_absolute(context_root)
    if detect_layout_version(lexical_root) != LAYOUT_VERSION:
        return None
    safe_context = _assert_v2_context_root(lexical_root)
    legacy_root = assert_no_linklike_components(
        safe_context / "history",
        boundary=safe_context,
    )
    write_root = assert_no_linklike_components(
        legacy_root / "common",
        boundary=safe_context,
    )
    return _HistoryRouting(
        write_root=write_root,
        context_root=safe_context,
        legacy_root=legacy_root,
    )


def _v2_routing_for_history_root(history_root: Path) -> _HistoryRouting | None:
    """Recognize either the v2 category root or its canonical common ledger."""

    root = lexical_absolute(history_root)
    candidates: list[Path] = []
    if root.name == "history":
        candidates.append(root.parent)
    if root.name == "common" and root.parent.name == "history":
        candidates.append(root.parent.parent)

    for context_root in candidates:
        routing = _v2_routing_for_context(context_root)
        if routing is None:
            continue
        if root in {routing.legacy_root, routing.write_root}:
            return routing
    return None


def resolve_history_root(
    context_root: Path,
    *,
    config: Any | None = None,
) -> Path:
    """Resolve the live event ledger, using ``history/common`` for v2.

    Version 1 keeps its metadata/config remapping behavior, including external
    history roots.  Version 2 is intentionally fixed beneath the central
    namespace and cannot be redirected through compatibility metadata.
    """

    routing = _v2_routing_for_context(context_root)
    if routing is not None:
        return routing.write_root
    return resolve_mount_root(
        context_root,
        MountType.HISTORY,
        config=config,
    )


def _routing_for_history_root(history_root: Path) -> _HistoryRouting:
    routing = _v2_routing_for_history_root(history_root)
    if routing is not None:
        return routing
    return _HistoryRouting(write_root=history_root.expanduser())


def _ensure_history_root(routing: _HistoryRouting) -> Path:
    root = routing.write_root
    if not routing.is_v2:
        root.mkdir(parents=True, exist_ok=True)
        return root

    assert routing.context_root is not None
    assert_no_linklike_components(root, boundary=routing.context_root)
    root.mkdir(parents=True, exist_ok=True, mode=0o700)
    return assert_no_linklike_components(
        root,
        boundary=routing.context_root,
        allow_missing=False,
    )


def _safe_v2_leaf(
    path: Path,
    routing: _HistoryRouting,
    *,
    allow_missing: bool,
    boundary: Path | None = None,
) -> Path:
    assert routing.context_root is not None
    _assert_v2_context_root(routing.context_root)
    trusted_boundary = boundary or routing.context_root
    assert_no_linklike_components(
        trusted_boundary,
        boundary=routing.context_root,
        allow_missing=allow_missing,
    )
    return assert_no_linklike_components(
        path,
        boundary=trusted_boundary,
        allow_missing=allow_missing,
    )


def _open_text_no_follow(path: Path, flags: int, *, mode: int = 0o600):
    """Open a text file without following its leaf link where supported."""

    descriptor = os.open(path, flags | getattr(os, "O_NOFOLLOW", 0), mode)
    return os.fdopen(descriptor, "r" if flags == os.O_RDONLY else "w", encoding="utf-8")


def _seed_v2_legacy_log(routing: _HistoryRouting, legacy_path: Path) -> None:
    """Publish one byte-identical legacy log into the canonical ledger once.

    Existing positional reactor offsets are keyed by the daily filename.  A
    byte-identical copy therefore lets an initialized pre-v2 reactor resume at
    the exact unread byte after the layout upgrade instead of pruning its
    checkpoint when ``history/common`` first appears.
    """

    assert routing.is_v2
    assert routing.context_root is not None
    assert routing.legacy_root is not None
    legacy_path = _safe_v2_leaf(
        legacy_path,
        routing,
        allow_missing=False,
        boundary=routing.legacy_root,
    )
    if not legacy_path.is_file():
        return
    canonical_path = _safe_v2_leaf(
        routing.write_root / legacy_path.name,
        routing,
        allow_missing=True,
        boundary=routing.write_root,
    )
    if canonical_path.exists():
        _safe_v2_leaf(
            canonical_path,
            routing,
            allow_missing=False,
            boundary=routing.write_root,
        )
        return

    temp_path = _safe_v2_leaf(
        routing.write_root / f".{legacy_path.name}.{uuid.uuid4().hex}.tmp",
        routing,
        allow_missing=True,
        boundary=routing.write_root,
    )
    source_fd = os.open(legacy_path, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
    destination_fd: int | None = None
    try:
        destination_fd = os.open(
            temp_path,
            os.O_WRONLY
            | os.O_CREAT
            | os.O_EXCL
            | getattr(os, "O_NOFOLLOW", 0),
            0o600,
        )
        while True:
            chunk = os.read(source_fd, 1024 * 1024)
            if not chunk:
                break
            view = memoryview(chunk)
            while view:
                written = os.write(destination_fd, view)
                view = view[written:]
        os.fsync(destination_fd)
        os.close(destination_fd)
        destination_fd = None
        try:
            os.link(temp_path, canonical_path, follow_symlinks=False)
            fsync_directory(routing.write_root)
        except FileExistsError:
            _safe_v2_leaf(
                canonical_path,
                routing,
                allow_missing=False,
                boundary=routing.write_root,
            )
    finally:
        os.close(source_fd)
        if destination_fd is not None:
            os.close(destination_fd)
        temp_path.unlink(missing_ok=True)


def prepare_history_reactor_root(
    context_root: Path,
    *,
    config: Any | None = None,
) -> Path:
    """Return the live ledger after safely seeding pre-fix v2 daily logs."""

    routing = _v2_routing_for_context(context_root)
    if routing is None:
        return resolve_mount_root(
            context_root,
            MountType.HISTORY,
            config=config,
        )
    _ensure_history_root(routing)
    assert routing.legacy_root is not None
    for legacy_path in sorted(
        routing.legacy_root.glob(f"{EVENT_FILE_PREFIX}_*.jsonl")
    ):
        _seed_v2_legacy_log(routing, legacy_path)
    return routing.write_root


def _write_text(path: Path, text: str, *, append: bool, routing: _HistoryRouting) -> None:
    if not routing.is_v2:
        with path.open("a" if append else "w", encoding="utf-8") as handle:
            handle.write(text)
        return

    _safe_v2_leaf(path, routing, allow_missing=True)
    flags = os.O_WRONLY | os.O_CREAT | (os.O_APPEND if append else os.O_TRUNC)
    with _open_text_no_follow(path, flags) as handle:
        handle.write(text)


def _read_text(path: Path, *, routing: _HistoryRouting) -> str:
    if not routing.is_v2:
        return path.read_text(encoding="utf-8")
    _safe_v2_leaf(path, routing, allow_missing=False)
    with _open_text_no_follow(path, os.O_RDONLY) as handle:
        return handle.read()


def _open_text_reader(path: Path, *, routing: _HistoryRouting):
    if not routing.is_v2:
        return path.open("r", encoding="utf-8")
    _safe_v2_leaf(path, routing, allow_missing=False)
    return _open_text_no_follow(path, os.O_RDONLY)


def _history_log_path(context_root: Path | None) -> Path | None:
    if context_root is None:
        return None
    history_dir = resolve_history_root(context_root)
    routing = _routing_for_history_root(history_dir)
    history_dir = _ensure_history_root(routing)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    path = history_dir / f"{EVENT_FILE_PREFIX}_{stamp}.jsonl"
    if routing.is_v2:
        assert routing.legacy_root is not None
        legacy_path = _safe_v2_leaf(
            routing.legacy_root / path.name,
            routing,
            allow_missing=True,
            boundary=routing.legacy_root,
        )
        if legacy_path.exists():
            _seed_v2_legacy_log(routing, legacy_path)
        path = _safe_v2_leaf(
            path,
            routing,
            allow_missing=True,
            boundary=history_dir,
        )
    return path


def _parse_timestamp(timestamp: str | None) -> datetime:
    if not timestamp:
        return datetime.now(timezone.utc)
    try:
        normalized = timestamp.strip()
        if normalized.endswith("Z"):
            normalized = normalized[:-1] + "+00:00"
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed
    except ValueError:
        return datetime.now(timezone.utc)


def _current_session_id() -> str | None:
    raw = os.getenv("AFS_SESSION_ID", "").strip()
    return raw or None


def _history_log_path_for_root(history_root: Path, event_time: datetime) -> Path:
    routing = _routing_for_history_root(history_root)
    history_root = _ensure_history_root(routing)
    stamp = event_time.strftime("%Y%m%d")
    path = history_root / f"{EVENT_FILE_PREFIX}_{stamp}.jsonl"
    if routing.is_v2:
        assert routing.legacy_root is not None
        legacy_path = _safe_v2_leaf(
            routing.legacy_root / path.name,
            routing,
            allow_missing=True,
            boundary=routing.legacy_root,
        )
        if legacy_path.exists():
            _seed_v2_legacy_log(routing, legacy_path)
        path = _safe_v2_leaf(
            path,
            routing,
            allow_missing=True,
            boundary=history_root,
        )
    return path


def _history_payload_path(
    history_dir: Path,
    payload_dir_name: str,
    event_id: str,
    timestamp: str,
) -> Path:
    routing = _routing_for_history_root(history_dir)
    history_dir = _ensure_history_root(routing)
    if routing.is_v2 and not SAFE_HISTORY_NAME.fullmatch(payload_dir_name):
        raise ValueError(
            "v2 history payload_dir_name must be one safe filesystem segment"
        )
    if routing.is_v2 and not SAFE_HISTORY_NAME.fullmatch(event_id):
        raise ValueError("v2 history event_id contains unsafe filesystem characters")
    payload_dir = history_dir / payload_dir_name
    if routing.is_v2:
        payload_dir = _safe_v2_leaf(
            payload_dir,
            routing,
            allow_missing=True,
            boundary=history_dir,
        )
        payload_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        payload_dir = _safe_v2_leaf(
            payload_dir,
            routing,
            allow_missing=False,
            boundary=history_dir,
        )
    else:
        payload_dir.mkdir(parents=True, exist_ok=True)
    stamp = (
        _parse_timestamp(timestamp).strftime("%Y%m%dT%H%M%S%fZ")
        if routing.is_v2
        else timestamp.replace(":", "").replace("-", "")
    )
    path = payload_dir / f"{stamp}_{event_id}.json"
    if routing.is_v2:
        path = _safe_v2_leaf(
            path,
            routing,
            allow_missing=True,
            boundary=payload_dir,
        )
    return path


def _normalize_payload(payload: Any) -> dict[str, Any]:
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, list):
        return {"items": payload}
    return {"text": str(payload)}


def _redact_payload(payload: Any) -> Any:
    if isinstance(payload, dict):
        redacted = {}
        for key, value in payload.items():
            if isinstance(key, str) and _should_redact(key):
                redacted[key] = "[redacted]"
            else:
                redacted[key] = _redact_payload(value)
        return redacted
    if isinstance(payload, list):
        return [_redact_payload(item) for item in payload]
    return payload


def _maybe_enrich_work_assistant(
    context_root: Path | None,
    event: dict[str, Any],
) -> None:
    if context_root is None:
        return
    try:
        from .work_assistant import enrich_logged_event

        enrich_logged_event(context_root, event)
    except Exception:
        # Work-assistant enrichment is a best-effort background side effect of
        # logging. A failure here must never block the canonical history event.
        return


def append_history_event(
    history_root: Path,
    event_type: str,
    source: str,
    *,
    op: str | None = None,
    metadata: dict[str, Any] | None = None,
    payload: Any | None = None,
    timestamp: str | None = None,
    context_root: Path | None = None,
    event_id: str | None = None,
    include_payloads: bool | None = None,
    max_inline_chars: int | None = None,
    payload_dir_name: str | None = None,
    redact_sensitive: bool | None = None,
) -> str:
    """Append an event directly to a history root."""
    config = load_config_model()
    history_cfg = config.history

    include_payloads = (
        history_cfg.include_payloads
        if include_payloads is None
        else include_payloads
    )
    max_inline = (
        history_cfg.max_inline_chars
        if max_inline_chars is None
        else max_inline_chars
    )
    payload_dir_name = payload_dir_name or history_cfg.payload_dir_name
    redact = (
        history_cfg.redact_sensitive
        if redact_sensitive is None
        else redact_sensitive
    )

    event_time = _parse_timestamp(timestamp)
    log_path = _history_log_path_for_root(history_root, event_time)
    routing = _routing_for_history_root(log_path.parent)

    event_id = event_id or uuid.uuid4().hex[:12]
    timestamp_value = timestamp or event_time.isoformat()

    metadata_payload = dict(metadata or {})
    session_id = _current_session_id()
    if session_id and not metadata_payload.get("session_id"):
        metadata_payload["session_id"] = session_id
    payload_data: dict[str, Any] | None = None
    payload_ref = None
    payload_sha256 = None
    payload_preview = None

    if payload is not None and include_payloads:
        normalized = _normalize_payload(payload)
        if redact:
            normalized = _redact_payload(normalized)

        serialized = json.dumps(normalized, ensure_ascii=False)
        max_inline = max_inline if max_inline is not None else DEFAULT_MAX_INLINE_CHARS

        if len(serialized) > max_inline:
            history_dir = log_path.parent
            payload_path = _history_payload_path(
                history_dir,
                payload_dir_name,
                event_id,
                timestamp_value,
            )
            _write_text(
                payload_path,
                serialized,
                append=False,
                routing=routing,
            )
            payload_ref = str(payload_path.relative_to(history_dir))
            payload_preview = serialized[:max_inline]
            payload_sha256 = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        else:
            payload_data = normalized
            payload_sha256 = hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    context_value = context_root or routing.context_root
    if context_value is None:
        candidate_context = history_root.parent
        if history_root == resolve_mount_root(candidate_context, MountType.HISTORY, config=config):
            context_value = candidate_context

    event = HistoryEvent(
        event_id=event_id,
        timestamp=timestamp_value,
        event_type=str(event_type),
        source=str(source),
        op=str(op) if op else None,
        context_root=str(context_value) if context_value else None,
        metadata=metadata_payload,
        payload=payload_data,
        payload_ref=payload_ref,
        payload_sha256=payload_sha256,
        payload_preview=payload_preview,
    )

    event_payload = event.to_dict()
    _write_text(
        log_path,
        json.dumps(event_payload, ensure_ascii=False) + "\n",
        append=True,
        routing=routing,
    )

    _maybe_enrich_work_assistant(context_value, event_payload)

    return event_id


def log_event(
    event_type: str,
    source: str,
    *,
    op: str | None = None,
    metadata: dict[str, Any] | None = None,
    payload: Any | None = None,
    context_root: Path | None = None,
    include_payloads: bool | None = None,
    redact_sensitive: bool | None = None,
) -> str | None:
    """Append an event to the history log.

    Returns the event ID if logged, otherwise None.
    """
    if os.getenv("AFS_HISTORY_DISABLED") == "1":
        return None

    config = load_config_model()
    if not config.history.enabled:
        return None

    resolved_root = context_root or _resolve_context_root(Path.cwd())
    log_path = _history_log_path(resolved_root)
    if log_path is None:
        return None
    routing = _routing_for_history_root(log_path.parent)

    event_id = uuid.uuid4().hex[:12]
    timestamp = datetime.now(timezone.utc).isoformat()

    metadata_payload = dict(metadata or {})
    session_id = _current_session_id()
    if session_id and not metadata_payload.get("session_id"):
        metadata_payload["session_id"] = session_id
    payload_data: dict[str, Any] | None = None
    payload_ref = None
    payload_sha256 = None
    payload_preview = None
    payloads_enabled = (
        config.history.include_payloads
        if include_payloads is None
        else include_payloads
    )

    redact = (
        config.history.redact_sensitive
        if redact_sensitive is None
        else redact_sensitive
    )

    if payload is not None and payloads_enabled:
        normalized = _normalize_payload(payload)
        if redact:
            normalized = _redact_payload(normalized)

        serialized = json.dumps(normalized, ensure_ascii=False)
        max_inline = config.history.max_inline_chars or DEFAULT_MAX_INLINE_CHARS

        if len(serialized) > max_inline:
            history_dir = log_path.parent
            payload_path = _history_payload_path(
                history_dir,
                config.history.payload_dir_name,
                event_id,
                timestamp,
            )
            _write_text(
                payload_path,
                serialized,
                append=False,
                routing=routing,
            )
            payload_ref = str(payload_path.relative_to(history_dir))
            payload_preview = serialized[:max_inline]
            payload_sha256 = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        else:
            payload_data = normalized
            payload_sha256 = hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    event = HistoryEvent(
        event_id=event_id,
        timestamp=timestamp,
        event_type=str(event_type),
        source=str(source),
        op=str(op) if op else None,
        context_root=str(resolved_root) if resolved_root else None,
        metadata=metadata_payload,
        payload=payload_data,
        payload_ref=payload_ref,
        payload_sha256=payload_sha256,
        payload_preview=payload_preview,
    )

    event_payload = event.to_dict()
    _write_text(
        log_path,
        json.dumps(event_payload, ensure_ascii=False) + "\n",
        append=True,
        routing=routing,
    )

    _maybe_enrich_work_assistant(resolved_root, event_payload)

    return event_id


def log_cli_invocation(argv: Iterable[str], exit_code: int) -> None:
    """Append a CLI invocation to the history log."""
    sanitized = sanitize_argv(argv)
    log_event(
        "cli",
        "afs.cli",
        op="invoke",
        metadata={
            "argv": sanitized,
            "exit_code": exit_code,
            "cwd": str(Path.cwd()),
        },
    )


def iter_history_events(
    history_root: Path,
    *,
    event_types: set[str] | None = None,
    include_payloads: bool = True,
) -> Iterable[dict[str, Any]]:
    """Iterate history events from a history root."""
    routing = _routing_for_history_root(history_root)
    if routing.is_v2:
        assert routing.context_root is not None
        roots = tuple(
            dict.fromkeys(
                root
                for root in (routing.legacy_root, routing.write_root)
                if root is not None
            )
        )
        safe_roots: list[Path] = []
        for root in roots:
            try:
                safe_root = _safe_v2_leaf(root, routing, allow_missing=True)
            except (OSError, ValueError):
                continue
            if safe_root.is_dir():
                safe_roots.append(safe_root)
        log_paths: list[tuple[str, int, Path, Path]] = []
        for priority, root in enumerate(safe_roots):
            for path in root.glob(f"{EVENT_FILE_PREFIX}_*.jsonl"):
                try:
                    path = _safe_v2_leaf(path, routing, allow_missing=False)
                except (OSError, ValueError):
                    continue
                if path.is_file():
                    log_paths.append((path.name, priority, path, root))
        ordered_paths = [
            (path, root)
            for _, _, path, root in sorted(log_paths, key=lambda item: (item[0], item[1]))
        ]
        canonical_event_ids: set[str] = set()
        for path, root in ordered_paths:
            if root != routing.write_root:
                continue
            try:
                with _open_text_reader(path, routing=routing) as handle:
                    for line in handle:
                        try:
                            event = json.loads(line)
                        except json.JSONDecodeError:
                            continue
                        if isinstance(event, dict):
                            event_id = str(event.get("id", "")).strip()
                            if event_id:
                                canonical_event_ids.add(event_id)
            except (OSError, ValueError):
                continue
    else:
        if not history_root.exists():
            return
        ordered_paths = [
            (path, history_root)
            for path in sorted(history_root.glob(f"{EVENT_FILE_PREFIX}_*.jsonl"))
        ]
        canonical_event_ids = set()

    seen_event_ids: set[str] = set()
    for path, source_root in ordered_paths:
        try:
            with _open_text_reader(path, routing=routing) as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    if not isinstance(event, dict):
                        continue
                    event_type = str(event.get("type", ""))
                    if event_types and event_type not in event_types:
                        continue
                    event_id = str(event.get("id", "")).strip()
                    if routing.is_v2 and event_id:
                        if (
                            source_root == routing.legacy_root
                            and event_id in canonical_event_ids
                        ):
                            continue
                        if event_id in seen_event_ids:
                            continue
                        seen_event_ids.add(event_id)
                    if include_payloads:
                        payload = _load_history_payload_from_root(
                            event,
                            source_root,
                            routing=routing,
                        )
                        if (
                            payload is None
                            and routing.is_v2
                            and source_root == routing.write_root
                            and routing.legacy_root is not None
                        ):
                            payload = _load_history_payload_from_root(
                                event,
                                routing.legacy_root,
                                routing=routing,
                                require_sha256=True,
                            )
                        if payload is not None:
                            event["payload"] = payload
                    yield event
        except (OSError, ValueError):
            continue


def query_events(
    history_root: Path,
    *,
    event_types: set[str] | None = None,
    since: str | None = None,
    limit: int = 50,
    source: str | None = None,
    session_id: str | None = None,
) -> list[dict[str, Any]]:
    """Query history events with filtering and return most recent N."""
    since_dt = _parse_timestamp(since) if since else None
    results: list[dict[str, Any]] = []
    for event in iter_history_events(history_root, event_types=event_types):
        if since_dt:
            event_ts = _parse_timestamp(event.get("timestamp"))
            if event_ts < since_dt:
                continue
        if source and event.get("source") != source:
            continue
        if session_id:
            metadata = event.get("metadata")
            if not isinstance(metadata, dict):
                continue
            if str(metadata.get("session_id", "")).strip() != session_id:
                continue
        results.append(event)
    if limit and len(results) > limit:
        results = results[-limit:]
    return results


def log_mcp_tool_call(
    tool_name: str,
    arguments: dict[str, Any] | None = None,
    result: dict[str, Any] | None = None,
    duration_ms: int | None = None,
    context_root: Path | None = None,
) -> str | None:
    """Log an MCP tool call event."""
    metadata: dict[str, Any] = {"tool_name": tool_name}
    if duration_ms is not None:
        metadata["duration_ms"] = duration_ms
    if arguments:
        metadata["arguments"] = arguments
    if isinstance(result, dict):
        has_error = "error" in result or result.get("ok") is False or result.get("isError") is True
        metadata["ok"] = not has_error
        if has_error:
            metadata["error"] = str(result.get("error", "tool call failed"))
    return log_event(
        EVENT_MCP_TOOL,
        "afs.mcp",
        op="call",
        metadata=metadata,
        payload=result,
        context_root=context_root,
    )


def log_hivemind_event(
    op: str,
    agent_name: str,
    msg_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    context_root: Path | None = None,
) -> str | None:
    """Log a hivemind bus event."""
    meta: dict[str, Any] = {"agent_name": agent_name}
    if msg_id:
        meta["msg_id"] = msg_id
    if metadata:
        meta.update(metadata)
    return log_event(
        EVENT_HIVEMIND,
        "afs.hivemind",
        op=op,
        metadata=meta,
        context_root=context_root,
    )


def log_embedding_event(
    op: str,
    metadata: dict[str, Any] | None = None,
    context_root: Path | None = None,
) -> str | None:
    """Log an embedding index/search event."""
    return log_event(
        EVENT_EMBEDDING,
        "afs.embeddings",
        op=op,
        metadata=metadata or {},
        context_root=context_root,
    )


def log_agent_lifecycle(
    agent_name: str,
    op: str,
    metadata: dict[str, Any] | None = None,
    context_root: Path | None = None,
) -> str | None:
    """Log an agent lifecycle event."""
    meta: dict[str, Any] = {"agent_name": agent_name}
    if metadata:
        meta.update(metadata)
    return log_event(
        EVENT_AGENT_LIFECYCLE,
        "afs.agents",
        op=op,
        metadata=meta,
        context_root=context_root,
    )


def log_session_event(
    op: str,
    session_id: str | None = None,
    metadata: dict[str, Any] | None = None,
    payload: Any | None = None,
    context_root: Path | None = None,
) -> str | None:
    """Log a session event."""
    meta: dict[str, Any] = {}
    if session_id:
        meta["session_id"] = session_id
    if metadata:
        meta.update(metadata)
    return log_event(
        EVENT_SESSION,
        "afs.session",
        op=op,
        metadata=meta,
        payload=payload,
        context_root=context_root,
    )


def _load_history_payload_from_root(
    event: dict[str, Any],
    source_root: Path,
    *,
    routing: _HistoryRouting,
    require_sha256: bool = False,
) -> dict[str, Any] | None:
    if "payload" in event and isinstance(event["payload"], dict):
        return event["payload"]
    payload_ref = event.get("payload_ref")
    if not isinstance(payload_ref, str) or not payload_ref:
        return None
    payload_path = Path(payload_ref)
    if routing.is_v2:
        if payload_path.is_absolute():
            return None
        try:
            payload_path = assert_no_linklike_components(
                source_root / payload_path,
                boundary=source_root,
                allow_missing=False,
            )
        except (OSError, ValueError):
            return None
    elif not payload_path.is_absolute():
        payload_path = source_root / payload_ref
    try:
        raw_payload = _read_text(payload_path, routing=routing)
        if require_sha256:
            expected_sha256 = event.get("payload_sha256")
            if (
                not isinstance(expected_sha256, str)
                or not re.fullmatch(r"[0-9a-f]{64}", expected_sha256)
                or hashlib.sha256(raw_payload.encode("utf-8")).hexdigest()
                != expected_sha256
            ):
                return None
        payload = json.loads(raw_payload)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    return payload


def load_history_payload(event: dict[str, Any], history_root: Path) -> dict[str, Any] | None:
    """Load a payload referenced by a history event."""

    routing = _routing_for_history_root(history_root)
    if not routing.is_v2:
        return _load_history_payload_from_root(
            event,
            history_root,
            routing=routing,
        )

    # Prefer the canonical ledger, while retaining a safe read-through for
    # payload references attached to pre-v2-fix events at ``history/``.
    for index, source_root in enumerate((routing.write_root, routing.legacy_root)):
        if source_root is None:
            continue
        payload = _load_history_payload_from_root(
            event,
            source_root,
            routing=routing,
            require_sha256=index > 0,
        )
        if payload is not None:
            return payload
    return None
