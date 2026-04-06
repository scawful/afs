"""Consolidate AFS history events into durable memory entries."""

from __future__ import annotations

import json
import uuid
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import load_config_model
from .context_paths import resolve_agent_output_root, resolve_mount_root
from .history import iter_history_events
from .models import MountType
from .schema import AFSConfig

# Type alias for the optional LLM summarizer callable.
# Signature: (events, context_root) -> str | None
SummarizerCallable = Callable[[list[dict[str, Any]], Path], str | None]

_SELF_SOURCES = {"afs.memory_consolidation", "agent.history-memory"}
_MAX_HIGHLIGHTS = 8
_MAX_TOUCHED_PATHS = 10


@dataclass(frozen=True)
class ConsolidationCursor:
    """Checkpoint for incremental history processing."""

    timestamp: str | None = None
    event_id: str | None = None


@dataclass
class MemoryConsolidationResult:
    """Summary of a history-to-memory consolidation run."""

    context_root: Path
    history_root: Path
    memory_root: Path
    entries_path: Path
    checkpoint_path: Path
    scanned_events: int = 0
    consolidated_events: int = 0
    entries_written: int = 0
    markdown_written: int = 0
    markdown_paths: list[Path] = field(default_factory=list)
    last_timestamp: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_root": str(self.context_root),
            "history_root": str(self.history_root),
            "memory_root": str(self.memory_root),
            "entries_path": str(self.entries_path),
            "checkpoint_path": str(self.checkpoint_path),
            "scanned_events": self.scanned_events,
            "consolidated_events": self.consolidated_events,
            "entries_written": self.entries_written,
            "markdown_written": self.markdown_written,
            "markdown_paths": [str(path) for path in self.markdown_paths],
            "last_timestamp": self.last_timestamp,
            "notes": list(self.notes),
        }


def consolidate_history_to_memory(
    context_root: Path,
    *,
    config: AFSConfig | None = None,
    history_root: Path | None = None,
    memory_root: Path | None = None,
    checkpoint_path: Path | None = None,
    max_events_per_run: int | None = None,
    max_events_per_entry: int | None = None,
    include_event_types: list[str] | None = None,
    write_markdown: bool | None = None,
    summarizer: SummarizerCallable | None = None,
) -> MemoryConsolidationResult:
    """Summarize new history events into durable memory entries."""
    config = config or load_config_model()
    context_root = context_root.expanduser().resolve()
    consolidation_cfg = config.memory_consolidation
    history_root = (
        history_root.expanduser().resolve()
        if history_root is not None
        else resolve_mount_root(context_root, MountType.HISTORY, config=config)
    )
    memory_root = (
        memory_root.expanduser().resolve()
        if memory_root is not None
        else resolve_mount_root(context_root, MountType.MEMORY, config=config)
    )
    agent_output_root = resolve_agent_output_root(context_root, config=config)
    checkpoint_path = (
        checkpoint_path.expanduser().resolve()
        if checkpoint_path is not None
        else agent_output_root / consolidation_cfg.checkpoint_filename
    )
    entries_path = memory_root / consolidation_cfg.entries_filename
    summary_dir = memory_root / consolidation_cfg.summary_dir_name
    max_events_per_run = max_events_per_run or consolidation_cfg.max_events_per_run
    max_events_per_entry = max_events_per_entry or consolidation_cfg.max_events_per_entry
    write_markdown = (
        consolidation_cfg.write_markdown
        if write_markdown is None
        else write_markdown
    )
    normalized_event_types = _normalize_event_types(
        include_event_types
        if include_event_types is not None
        else consolidation_cfg.include_event_types
    )

    result = MemoryConsolidationResult(
        context_root=context_root,
        history_root=history_root,
        memory_root=memory_root,
        entries_path=entries_path,
        checkpoint_path=checkpoint_path,
    )

    if not history_root.exists():
        result.notes.append("history root does not exist")
        return result

    cursor = _load_cursor(checkpoint_path)
    pending_events: list[dict[str, Any]] = []

    for event in iter_history_events(
        history_root,
        event_types=normalized_event_types or None,
        include_payloads=False,
    ):
        if not isinstance(event, dict):
            continue
        if not _is_event_after_cursor(event, cursor):
            continue
        if _should_skip_event(event):
            continue
        pending_events.append(event)

    result.scanned_events = len(pending_events)
    if not pending_events:
        result.notes.append("no new events")
        return result

    pending_events.sort(key=_event_sort_key)
    if max_events_per_run and len(pending_events) > max_events_per_run:
        result.notes.append(f"run capped at {max_events_per_run} events")
        pending_events = pending_events[:max_events_per_run]

    memory_root.mkdir(parents=True, exist_ok=True)
    entries_path.parent.mkdir(parents=True, exist_ok=True)
    if write_markdown:
        summary_dir.mkdir(parents=True, exist_ok=True)

    batches = [
        pending_events[index : index + max_events_per_entry]
        for index in range(0, len(pending_events), max_events_per_entry)
    ]

    with entries_path.open("a", encoding="utf-8") as handle:
        for batch_index, batch in enumerate(batches, start=1):
            entry_id = _entry_id(batch[0], batch_index=batch_index)
            entry = _build_memory_entry(
                batch,
                context_root=context_root,
                entry_id=entry_id,
            )
            # When an LLM summarizer is provided, attempt to replace the
            # mechanical counter-based output with a natural-language summary.
            if summarizer is not None:
                try:
                    llm_summary = summarizer(batch, context_root)
                    if isinstance(llm_summary, str) and llm_summary.strip():
                        entry["output"] = llm_summary.strip()
                        entry["_metadata"]["summarizer"] = "llm"
                except Exception:
                    # Never crash — keep the counter-based entry on failure.
                    pass
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
            result.entries_written += 1
            result.consolidated_events += len(batch)
            result.last_timestamp = _event_timestamp(batch[-1])
            if write_markdown:
                markdown_path = summary_dir / f"{entry_id}.md"
                markdown_path.write_text(_render_markdown(entry), encoding="utf-8")
                result.markdown_written += 1
                result.markdown_paths.append(markdown_path)

    _save_cursor(checkpoint_path, pending_events[-1])
    return result


def _normalize_event_types(event_types: list[str] | None) -> set[str]:
    if not event_types:
        return set()
    return {
        str(event_type).strip()
        for event_type in event_types
        if isinstance(event_type, str) and str(event_type).strip()
    }


def _load_cursor(path: Path) -> ConsolidationCursor:
    if not path.exists():
        return ConsolidationCursor()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return ConsolidationCursor()
    if not isinstance(raw, dict):
        return ConsolidationCursor()
    timestamp = raw.get("timestamp")
    event_id = raw.get("event_id")
    return ConsolidationCursor(
        timestamp=str(timestamp) if isinstance(timestamp, str) and timestamp.strip() else None,
        event_id=str(event_id) if isinstance(event_id, str) and event_id.strip() else None,
    )


def _save_cursor(path: Path, event: dict[str, Any]) -> None:
    timestamp = _event_timestamp(event)
    payload = {
        "timestamp": timestamp,
        "event_id": _event_id(event),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _is_event_after_cursor(event: dict[str, Any], cursor: ConsolidationCursor) -> bool:
    if cursor.timestamp is None:
        return True
    event_timestamp = _event_timestamp(event)
    if event_timestamp > cursor.timestamp:
        return True
    if event_timestamp < cursor.timestamp:
        return False

    event_id = _event_id(event)
    if cursor.event_id is None:
        return False
    return event_id > cursor.event_id


def _event_timestamp(event: dict[str, Any]) -> str:
    timestamp = event.get("timestamp")
    if isinstance(timestamp, str) and timestamp.strip():
        return timestamp
    return datetime.now(timezone.utc).isoformat()


def _event_sort_key(event: dict[str, Any]) -> tuple[datetime, str]:
    return (_parse_timestamp(_event_timestamp(event)), _event_id(event))


def _parse_timestamp(timestamp: str) -> datetime:
    try:
        normalized = timestamp.strip()
        if normalized.endswith("Z"):
            normalized = normalized[:-1] + "+00:00"
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except ValueError:
        return datetime.min.replace(tzinfo=timezone.utc)


def _event_id(event: dict[str, Any]) -> str:
    event_id = event.get("id")
    if isinstance(event_id, str) and event_id.strip():
        return event_id
    return ""


def _should_skip_event(event: dict[str, Any]) -> bool:
    source = event.get("source")
    if not isinstance(source, str):
        return False
    return any(source.startswith(prefix) for prefix in _SELF_SOURCES)


def _entry_id(event: dict[str, Any], *, batch_index: int) -> str:
    timestamp = _event_timestamp(event)
    compact = "".join(ch for ch in timestamp if ch.isdigit())[:14]
    return f"history-memory-{compact or uuid.uuid4().hex[:12]}-{batch_index:02d}"


def _build_memory_entry(
    events: list[dict[str, Any]],
    *,
    context_root: Path,
    entry_id: str,
) -> dict[str, Any]:
    start_ts = _event_timestamp(events[0])
    end_ts = _event_timestamp(events[-1])
    context_label = (
        context_root.parent.name
        if context_root.name == ".context" and context_root.parent.name
        else context_root.name
    )
    type_counts = Counter(
        event_type
        for event in events
        if isinstance((event_type := event.get("type")), str) and event_type
    )
    op_counts = Counter(
        op
        for event in events
        if isinstance((op := event.get("op")), str) and op
    )
    source_counts = Counter(
        source
        for event in events
        if isinstance((source := event.get("source")), str) and source
    )
    touched_paths = _collect_touched_paths(events)
    highlights = _collect_highlights(events)
    output_lines = [
        f"AFS activity for `{context_label}` from {start_ts} to {end_ts}.",
        f"Events: {_render_counter(type_counts)}.",
    ]
    if op_counts:
        output_lines.append(f"Operations: {_render_counter(op_counts)}.")
    if source_counts:
        output_lines.append(f"Sources: {_render_counter(source_counts, limit=3)}.")
    if touched_paths:
        output_lines.append("Touched paths: " + ", ".join(touched_paths) + ".")
    if highlights:
        output_lines.append("Highlights:")
        output_lines.extend(f"- {item}" for item in highlights)

    return {
        "id": entry_id,
        "instruction": (
            f"Recall the durable AFS context changes for `{context_label}` "
            f"between {start_ts} and {end_ts}."
        ),
        "output": "\n".join(output_lines).strip(),
        "domain": "memory",
        "source": "history.consolidation",
        "tags": ["afs", "history-consolidated", *sorted(type_counts.keys())],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "_metadata": {
            "context_root": str(context_root),
            "window_start": start_ts,
            "window_end": end_ts,
            "event_count": len(events),
            "event_types": dict(type_counts),
            "operations": dict(op_counts),
            "sources": dict(source_counts),
            "touched_paths": touched_paths,
            "event_ids": [
                event_id
                for event in events
                if isinstance((event_id := event.get("id")), str) and event_id
            ],
        },
    }


def _render_counter(counter: Counter[str], *, limit: int = 5) -> str:
    if not counter:
        return "none"
    return ", ".join(f"{name}={count}" for name, count in counter.most_common(limit))


def _collect_touched_paths(events: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    touched: list[str] = []
    for event in events:
        metadata = event.get("metadata")
        if not isinstance(metadata, dict):
            continue
        label = _event_path_label(metadata)
        if label and label not in seen:
            seen.add(label)
            touched.append(label)
        if len(touched) >= _MAX_TOUCHED_PATHS:
            break
    return touched


def _event_path_label(metadata: dict[str, Any]) -> str | None:
    mount_type = metadata.get("mount_type")
    relative_path = metadata.get("relative_path")
    alias = metadata.get("alias")
    filename = metadata.get("filename")
    category = metadata.get("category")

    if isinstance(mount_type, str) and isinstance(relative_path, str) and relative_path.strip():
        return f"{mount_type}/{relative_path}"
    if isinstance(mount_type, str) and isinstance(alias, str) and alias.strip():
        return f"{mount_type}/{alias}"
    if isinstance(category, str) and isinstance(filename, str):
        return f"review/{category}/{filename}"
    return None


def _collect_highlights(events: list[dict[str, Any]]) -> list[str]:
    highlights: list[str] = []
    for event in events:
        highlight = _describe_event(event)
        if not highlight:
            continue
        highlights.append(highlight)
        if len(highlights) >= _MAX_HIGHLIGHTS:
            break
    remaining = len(events) - len(highlights)
    if remaining > 0:
        highlights.append(f"{remaining} additional events omitted from highlights")
    return highlights


def _describe_event(event: dict[str, Any]) -> str | None:
    event_type = event.get("type")
    op = event.get("op")
    source = event.get("source")
    metadata = event.get("metadata")
    if not isinstance(metadata, dict):
        metadata = {}

    if event_type == "fs":
        path = _event_path_label(metadata)
        if path:
            return f"{op or 'touch'} {path}"
    if event_type == "context":
        mount_type = metadata.get("mount_type")
        alias = metadata.get("alias")
        if isinstance(mount_type, str) and isinstance(alias, str) and alias:
            return f"{op or 'update'} {mount_type}/{alias}"
        if isinstance(op, str):
            return f"context {op}"
    if event_type == "hook":
        status = metadata.get("status")
        if isinstance(op, str):
            suffix = f" ({status})" if isinstance(status, str) and status else ""
            return f"hook {op}{suffix}"
    if event_type == "review":
        path = _event_path_label(metadata)
        if path:
            return f"review {op or 'update'} {path}"
    if event_type == "agent_progress":
        detail = metadata.get("detail")
        agent = metadata.get("agent")
        if isinstance(agent, str) and isinstance(op, str):
            suffix = f": {detail}" if isinstance(detail, str) and detail else ""
            return f"agent {agent} {op}{suffix}"
    if isinstance(source, str) and isinstance(op, str):
        return f"{source} {op}"
    return None


def _render_markdown(entry: dict[str, Any]) -> str:
    output = entry.get("output", "")
    metadata = entry.get("_metadata", {})
    lines = [
        f"# {entry.get('id', 'history-memory')}",
        "",
        f"- Instruction: {entry.get('instruction', '')}",
        f"- Source: {entry.get('source', '')}",
        f"- Domain: {entry.get('domain', '')}",
    ]
    if isinstance(metadata, dict):
        lines.extend(
            [
                f"- Window: {metadata.get('window_start', '')} -> {metadata.get('window_end', '')}",
                f"- Event count: {metadata.get('event_count', 0)}",
            ]
        )
    lines.extend(["", "## Summary", "", str(output).strip(), ""])
    return "\n".join(lines).strip() + "\n"


def memory_status(
    context_root: Path,
    *,
    config: AFSConfig | None = None,
) -> dict[str, Any]:
    """Return memory pipeline health: entry count, cursor position, staleness."""
    config = config or load_config_model()
    context_root = context_root.expanduser().resolve()
    consolidation_cfg = config.memory_consolidation
    memory_root = resolve_mount_root(context_root, MountType.MEMORY, config=config)
    agent_output_root = resolve_agent_output_root(context_root, config=config)
    entries_path = memory_root / consolidation_cfg.entries_filename
    checkpoint_path = agent_output_root / consolidation_cfg.checkpoint_filename
    summary_dir = memory_root / consolidation_cfg.summary_dir_name

    entries_count = 0
    if entries_path.exists():
        try:
            entries_count = sum(
                1 for line in entries_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            )
        except OSError:
            pass

    cursor = _load_cursor(checkpoint_path)
    cursor_age_seconds: float | None = None
    if checkpoint_path.exists():
        try:
            cursor_age_seconds = max(
                0.0,
                datetime.now(timezone.utc).timestamp() - checkpoint_path.stat().st_mtime,
            )
        except OSError:
            pass

    latest_summary_path: str | None = None
    if summary_dir.exists():
        try:
            candidates = [p for p in summary_dir.glob("*.md") if p.is_file()]
            if candidates:
                latest = max(candidates, key=lambda p: p.stat().st_mtime)
                latest_summary_path = str(latest)
        except OSError:
            pass

    stale = cursor_age_seconds is not None and cursor_age_seconds > 3600.0

    return {
        "entries_count": entries_count,
        "entries_path": str(entries_path),
        "cursor_timestamp": cursor.timestamp,
        "cursor_event_id": cursor.event_id,
        "cursor_age_seconds": cursor_age_seconds,
        "latest_summary_path": latest_summary_path,
        "stale": stale,
        "checkpoint_path": str(checkpoint_path),
    }


@dataclass(frozen=True)
class GateResult:
    """Result of a consolidation gate check."""

    passed: bool
    gate: str
    reason: str


def check_consolidation_gates(
    context_root: Path,
    *,
    config: AFSConfig | None = None,
    lock_path: Path | None = None,
) -> GateResult:
    """Run the auto-consolidation gate chain (cheapest check first).

    Gate order (inspired by Claude Code's auto-dream pattern):
      1. Lock gate   — skip if another consolidation is in progress
      2. Time gate   — skip if last run was less than gate_min_hours ago
      3. Volume gate — skip if fewer than gate_min_events new events exist
      4. Session gate — skip if fewer than gate_min_sessions bootstrap events

    Returns a GateResult indicating whether consolidation should proceed.
    """
    config = config or load_config_model()
    context_root = context_root.expanduser().resolve()
    consolidation_cfg = config.memory_consolidation
    agent_output_root = resolve_agent_output_root(context_root, config=config)
    checkpoint_path = agent_output_root / consolidation_cfg.checkpoint_filename

    # --- Gate 1: Lock ---
    _lock_path = lock_path or (agent_output_root / "history_memory.lock")
    if _lock_path.exists():
        try:
            lock_age = datetime.now(timezone.utc).timestamp() - _lock_path.stat().st_mtime
            # Stale lock (>1 hour) is ignored
            if lock_age < 3600:
                return GateResult(False, "lock", "another consolidation is in progress")
        except OSError:
            pass

    # --- Gate 2: Time ---
    min_hours = consolidation_cfg.gate_min_hours
    if min_hours > 0 and checkpoint_path.exists():
        try:
            age_seconds = datetime.now(timezone.utc).timestamp() - checkpoint_path.stat().st_mtime
            age_hours = age_seconds / 3600
            if age_hours < min_hours:
                return GateResult(
                    False,
                    "time",
                    f"last run was {age_hours:.1f}h ago (minimum {min_hours}h)",
                )
        except OSError:
            pass

    # --- Gate 3: Volume ---
    min_events = consolidation_cfg.gate_min_events
    if min_events > 0:
        history_root = resolve_mount_root(context_root, MountType.HISTORY, config=config)
        if history_root.exists():
            cursor = _load_cursor(checkpoint_path)
            normalized_event_types = _normalize_event_types(
                consolidation_cfg.include_event_types
            )
            count = 0
            for event in iter_history_events(
                history_root,
                event_types=normalized_event_types or None,
                include_payloads=False,
            ):
                if not isinstance(event, dict):
                    continue
                if not _is_event_after_cursor(event, cursor):
                    continue
                if _should_skip_event(event):
                    continue
                count += 1
                if count >= min_events:
                    break
            if count < min_events:
                return GateResult(
                    False,
                    "volume",
                    f"only {count} new events (minimum {min_events})",
                )
        else:
            return GateResult(False, "volume", "history root does not exist")

    # --- Gate 4: Session ---
    min_sessions = consolidation_cfg.gate_min_sessions
    if min_sessions > 0:
        history_root = resolve_mount_root(context_root, MountType.HISTORY, config=config)
        session_dir = history_root / "session"
        session_count = 0
        if session_dir.exists():
            cursor = _load_cursor(checkpoint_path)
            try:
                for event_file in sorted(session_dir.iterdir()):
                    if not event_file.is_file() or not event_file.suffix == ".json":
                        continue
                    try:
                        event = json.loads(event_file.read_text(encoding="utf-8"))
                    except (OSError, json.JSONDecodeError):
                        continue
                    if not isinstance(event, dict):
                        continue
                    if event.get("type") != "session" or event.get("op") != "bootstrap":
                        continue
                    if not _is_event_after_cursor(event, cursor):
                        continue
                    session_count += 1
                    if session_count >= min_sessions:
                        break
            except OSError:
                pass
        if session_count < min_sessions:
            return GateResult(
                False,
                "session",
                f"only {session_count} sessions since last run (minimum {min_sessions})",
            )

    return GateResult(True, "all", "all gates passed")


def search_memory(
    context_root: Path,
    query: str,
    *,
    config: AFSConfig | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Keyword search across memory entries in entries.jsonl."""
    config = config or load_config_model()
    context_root = context_root.expanduser().resolve()
    consolidation_cfg = config.memory_consolidation
    memory_root = resolve_mount_root(context_root, MountType.MEMORY, config=config)
    entries_path = memory_root / consolidation_cfg.entries_filename

    if not entries_path.exists():
        return []

    query_lower = query.lower()
    results: list[dict[str, Any]] = []
    try:
        for line in entries_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            searchable = " ".join([
                str(entry.get("instruction", "")),
                str(entry.get("output", "")),
                " ".join(str(t) for t in entry.get("tags", []) if isinstance(t, str)),
            ]).lower()
            if query_lower in searchable:
                results.append(entry)
                if len(results) >= limit:
                    break
    except OSError:
        pass

    return results
