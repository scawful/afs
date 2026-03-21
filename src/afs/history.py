"""History logging helpers for AFS."""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import load_config_model
from .context_paths import resolve_mount_root
from .core import find_root, resolve_context_root
from .models import MountType

SENSITIVE_MARKERS = ("key", "token", "secret", "password")
EVENT_FILE_PREFIX = "events"
DEFAULT_MAX_INLINE_CHARS = 4000

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


def _history_log_path(context_root: Path | None) -> Path | None:
    if context_root is None:
        return None
    history_dir = resolve_mount_root(context_root, MountType.HISTORY)
    history_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    return history_dir / f"{EVENT_FILE_PREFIX}_{stamp}.jsonl"


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
    history_root.mkdir(parents=True, exist_ok=True)
    stamp = event_time.strftime("%Y%m%d")
    return history_root / f"{EVENT_FILE_PREFIX}_{stamp}.jsonl"


def _history_payload_path(
    history_dir: Path,
    payload_dir_name: str,
    event_id: str,
    timestamp: str,
) -> Path:
    payload_dir = history_dir / payload_dir_name
    payload_dir.mkdir(parents=True, exist_ok=True)
    stamp = timestamp.replace(":", "").replace("-", "")
    return payload_dir / f"{stamp}_{event_id}.json"


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
            payload_path.write_text(serialized, encoding="utf-8")
            payload_ref = str(payload_path.relative_to(history_dir))
            payload_preview = serialized[:max_inline]
            payload_sha256 = hashlib.sha256(serialized.encode("utf-8")).hexdigest()
        else:
            payload_data = normalized
            payload_sha256 = hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    context_value = context_root
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

    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")

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
            payload_path.write_text(serialized, encoding="utf-8")
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

    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event.to_dict(), ensure_ascii=False) + "\n")

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
    if not history_root.exists():
        return

    for path in sorted(history_root.glob(f"{EVENT_FILE_PREFIX}_*.jsonl")):
        try:
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        event = json.loads(line)
                    except json.JSONDecodeError:
                        continue
                    event_type = str(event.get("type", ""))
                    if event_types and event_type not in event_types:
                        continue
                    if include_payloads:
                        payload = load_history_payload(event, history_root)
                        if payload is not None:
                            event["payload"] = payload
                    yield event
        except OSError:
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
        context_root=context_root,
    )


def load_history_payload(event: dict[str, Any], history_root: Path) -> dict[str, Any] | None:
    """Load a payload referenced by a history event."""
    if "payload" in event and isinstance(event["payload"], dict):
        return event["payload"]
    payload_ref = event.get("payload_ref")
    if not payload_ref:
        return None
    payload_path = Path(payload_ref)
    if not payload_path.is_absolute():
        payload_path = history_root / payload_ref
    try:
        return json.loads(payload_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
