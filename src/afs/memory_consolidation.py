"""Consolidate AFS history events into durable memory entries."""

from __future__ import annotations

import json
import os
import stat
import uuid
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .config import load_config_model
from .context_layout import (
    LAYOUT_VERSION,
    _atomic_write_text,
    detect_layout_version,
)
from .context_paths import resolve_agent_output_root, resolve_mount_root
from .history import iter_history_events, resolve_history_root
from .models import MountType
from .path_safety import assert_no_linklike_components
from .schema import AFSConfig

# Type alias for the optional LLM summarizer callable.
# Signature: (events, context_root) -> str | None
SummarizerCallable = Callable[[list[dict[str, Any]], Path], str | None]

_SELF_SOURCES = {"afs.memory_consolidation", "agent.history-memory"}
_MAX_HIGHLIGHTS = 8
_MAX_TOUCHED_PATHS = 10


def _is_v2(context_root: Path) -> bool:
    return detect_layout_version(context_root) == LAYOUT_VERSION


def _managed_memory_roots(
    context_root: Path,
    *,
    config: AFSConfig,
) -> tuple[Path, Path | None]:
    """Return the canonical memory root and optional pre-fix v2 root."""

    category_root = resolve_mount_root(
        context_root,
        MountType.MEMORY,
        config=config,
    )
    if not _is_v2(context_root):
        return category_root, None
    canonical = assert_no_linklike_components(
        category_root / "common",
        boundary=context_root,
    )
    return canonical, category_root


def _validate_managed_path(
    path: Path,
    *,
    context_root: Path | None,
    allow_missing: bool = True,
    require_directory: bool = False,
) -> Path:
    """Validate one default-managed path without following links."""

    if context_root is None:
        return path
    safe = assert_no_linklike_components(
        path,
        boundary=context_root,
        allow_missing=allow_missing,
    )
    try:
        path_stat = os.lstat(safe)
    except FileNotFoundError:
        if allow_missing:
            return safe
        raise
    expected = stat.S_ISDIR(path_stat.st_mode) if require_directory else stat.S_ISREG(
        path_stat.st_mode
    )
    if not expected:
        kind = "directory" if require_directory else "regular file"
        raise ValueError(f"managed memory path is not a safe {kind}: {safe}")
    return safe


def _ensure_managed_directory(path: Path, *, context_root: Path | None) -> Path:
    path = _validate_managed_path(
        path,
        context_root=context_root,
        allow_missing=True,
        require_directory=True,
    )
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    return _validate_managed_path(
        path,
        context_root=context_root,
        allow_missing=False,
        require_directory=True,
    )


def _read_managed_text(path: Path, *, context_root: Path | None) -> str:
    if context_root is None:
        return path.read_text(encoding="utf-8")
    path = _validate_managed_path(
        path,
        context_root=context_root,
        allow_missing=False,
    )
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    with os.fdopen(descriptor, encoding="utf-8") as handle:
        opened = os.fstat(handle.fileno())
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError(f"managed memory path is not a safe regular file: {path}")
        return handle.read()


def _append_managed_text(
    path: Path,
    text: str,
    *,
    context_root: Path | None,
) -> None:
    if context_root is None:
        with path.open("a", encoding="utf-8") as handle:
            handle.write(text)
        return
    path = _validate_managed_path(path, context_root=context_root, allow_missing=True)
    flags = (
        os.O_WRONLY
        | os.O_CREAT
        | os.O_APPEND
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    descriptor = os.open(path, flags, 0o600)
    with os.fdopen(descriptor, "a", encoding="utf-8") as handle:
        opened = os.fstat(handle.fileno())
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError(f"managed memory path is not a safe regular file: {path}")
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def _write_managed_text(
    path: Path,
    text: str,
    *,
    context_root: Path | None,
) -> None:
    if context_root is None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return
    _validate_managed_path(path, context_root=context_root, allow_missing=True)
    _atomic_write_text(path, text)
    _validate_managed_path(path, context_root=context_root, allow_missing=False)


def _readable_memory_roots(
    context_root: Path,
    *,
    config: AFSConfig,
) -> tuple[Path, ...]:
    canonical, legacy = _managed_memory_roots(context_root, config=config)
    if legacy is None:
        return (canonical,)
    # The category-root source is read-only compatibility for data written by
    # early v2 builds. Migration itself already places v1 data in ``common``.
    return (canonical, legacy)


def _default_checkpoint_read_path(
    context_root: Path,
    canonical_path: Path,
    *,
    config: AFSConfig,
) -> Path:
    """Prefer the canonical cursor, then an early-v2 compatibility cursor."""

    if not _is_v2(context_root):
        return canonical_path
    scratchpad_root = resolve_mount_root(
        context_root,
        MountType.SCRATCHPAD,
        config=config,
    )
    pre_fix_path = (
        scratchpad_root
        / "afs_agents"
        / config.memory_consolidation.checkpoint_filename
    )
    for candidate in dict.fromkeys((canonical_path, pre_fix_path)):
        try:
            return _validate_managed_path(
                candidate,
                context_root=context_root,
                allow_missing=False,
            )
        except FileNotFoundError:
            continue
    return canonical_path


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
    managed_boundary = context_root if _is_v2(context_root) else None
    history_root = (
        history_root.expanduser().resolve()
        if history_root is not None
        else resolve_history_root(context_root, config=config)
    )
    if memory_root is not None:
        memory_root = memory_root.expanduser().resolve()
        memory_boundary: Path | None = None
    else:
        memory_root, _legacy_memory_root = _managed_memory_roots(
            context_root,
            config=config,
        )
        memory_boundary = managed_boundary
    agent_output_root = resolve_agent_output_root(context_root, config=config)
    if checkpoint_path is not None:
        checkpoint_path = checkpoint_path.expanduser().resolve()
        checkpoint_boundary: Path | None = None
        checkpoint_read_path = checkpoint_path
    else:
        checkpoint_path = agent_output_root / consolidation_cfg.checkpoint_filename
        checkpoint_boundary = managed_boundary
        checkpoint_read_path = _default_checkpoint_read_path(
            context_root,
            checkpoint_path,
            config=config,
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

    cursor = _load_cursor(checkpoint_read_path, context_root=checkpoint_boundary)
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

    memory_root = _ensure_managed_directory(
        memory_root,
        context_root=memory_boundary,
    )
    entries_path.parent.mkdir(parents=True, exist_ok=True)
    if write_markdown:
        summary_dir = _ensure_managed_directory(
            summary_dir,
            context_root=memory_boundary,
        )

    batches = [
        pending_events[index : index + max_events_per_entry]
        for index in range(0, len(pending_events), max_events_per_entry)
    ]

    rendered_entries: list[str] = []
    rendered_markdown: list[tuple[Path, str]] = []
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
        rendered_entries.append(json.dumps(entry, ensure_ascii=False) + "\n")
        result.entries_written += 1
        result.consolidated_events += len(batch)
        result.last_timestamp = _event_timestamp(batch[-1])
        if write_markdown:
            markdown_path = summary_dir / f"{entry_id}.md"
            rendered_markdown.append((markdown_path, _render_markdown(entry)))

    _append_managed_text(
        entries_path,
        "".join(rendered_entries),
        context_root=memory_boundary,
    )
    for markdown_path, rendered in rendered_markdown:
        _write_managed_text(
            markdown_path,
            rendered,
            context_root=memory_boundary,
        )
        result.markdown_written += 1
        result.markdown_paths.append(markdown_path)

    _save_cursor(
        checkpoint_path,
        pending_events[-1],
        context_root=checkpoint_boundary,
    )
    return result


def _normalize_event_types(event_types: list[str] | None) -> set[str]:
    if not event_types:
        return set()
    return {
        str(event_type).strip()
        for event_type in event_types
        if isinstance(event_type, str) and str(event_type).strip()
    }


def _load_cursor(
    path: Path,
    *,
    context_root: Path | None = None,
) -> ConsolidationCursor:
    try:
        raw_text = _read_managed_text(path, context_root=context_root)
    except FileNotFoundError:
        return ConsolidationCursor()
    except OSError:
        if context_root is not None:
            raise
        return ConsolidationCursor()
    try:
        raw = json.loads(raw_text)
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


def _save_cursor(
    path: Path,
    event: dict[str, Any],
    *,
    context_root: Path | None = None,
) -> None:
    timestamp = _event_timestamp(event)
    payload = {
        "timestamp": timestamp,
        "event_id": _event_id(event),
    }
    _ensure_managed_directory(path.parent, context_root=context_root)
    _write_managed_text(
        path,
        json.dumps(payload, indent=2) + "\n",
        context_root=context_root,
    )


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
    managed_boundary = context_root if _is_v2(context_root) else None
    memory_roots = _readable_memory_roots(context_root, config=config)
    memory_root = memory_roots[0]
    agent_output_root = resolve_agent_output_root(context_root, config=config)
    entries_path = memory_root / consolidation_cfg.entries_filename
    checkpoint_path = agent_output_root / consolidation_cfg.checkpoint_filename
    checkpoint_read_path = _default_checkpoint_read_path(
        context_root,
        checkpoint_path,
        config=config,
    )

    entries_count = 0
    seen_entry_ids: set[str] = set()
    for source_root in memory_roots:
        source_path = source_root / consolidation_cfg.entries_filename
        try:
            raw_entries = _read_managed_text(
                source_path,
                context_root=managed_boundary,
            ).splitlines()
        except FileNotFoundError:
            continue
        except OSError:
            if managed_boundary is not None:
                raise
            continue
        for line in raw_entries:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                entry = None
            entry_id = entry.get("id") if isinstance(entry, dict) else None
            identity = str(entry_id) if entry_id else line
            if identity not in seen_entry_ids:
                seen_entry_ids.add(identity)
                entries_count += 1

    cursor = _load_cursor(checkpoint_read_path, context_root=managed_boundary)
    cursor_age_seconds: float | None = None
    try:
        safe_checkpoint = _validate_managed_path(
            checkpoint_read_path,
            context_root=managed_boundary,
            allow_missing=False,
        )
    except FileNotFoundError:
        safe_checkpoint = None
    if safe_checkpoint is not None:
        try:
            cursor_age_seconds = max(
                0.0,
                datetime.now(timezone.utc).timestamp()
                - os.lstat(safe_checkpoint).st_mtime,
            )
        except OSError:
            pass

    latest_summary_path: str | None = None
    summary_candidates: list[Path] = []
    for source_root in memory_roots:
        source_summary_dir = source_root / consolidation_cfg.summary_dir_name
        try:
            safe_summary_dir = _validate_managed_path(
                source_summary_dir,
                context_root=managed_boundary,
                allow_missing=False,
                require_directory=True,
            )
        except FileNotFoundError:
            continue
        for candidate in safe_summary_dir.glob("*.md"):
            try:
                summary_candidates.append(
                    _validate_managed_path(
                        candidate,
                        context_root=managed_boundary,
                        allow_missing=False,
                    )
                )
            except (FileNotFoundError, ValueError):
                continue
    if summary_candidates:
        latest = max(summary_candidates, key=lambda path: os.lstat(path).st_mtime)
        latest_summary_path = str(latest)

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
    managed_boundary = context_root if _is_v2(context_root) else None
    agent_output_root = resolve_agent_output_root(context_root, config=config)
    checkpoint_path = agent_output_root / consolidation_cfg.checkpoint_filename
    checkpoint_read_path = _default_checkpoint_read_path(
        context_root,
        checkpoint_path,
        config=config,
    )

    # --- Gate 1: Lock ---
    _lock_path = lock_path or (agent_output_root / "history_memory.lock")
    lock_boundary = managed_boundary if lock_path is None else None
    try:
        safe_lock = _validate_managed_path(
            _lock_path,
            context_root=lock_boundary,
            allow_missing=False,
        )
    except FileNotFoundError:
        safe_lock = None
    if safe_lock is not None:
        try:
            lock_age = (
                datetime.now(timezone.utc).timestamp() - os.lstat(safe_lock).st_mtime
            )
            # Stale lock (>1 hour) is ignored
            if lock_age < 3600:
                return GateResult(False, "lock", "another consolidation is in progress")
        except OSError:
            pass

    # --- Gate 2: Time ---
    min_hours = consolidation_cfg.gate_min_hours
    try:
        safe_checkpoint = _validate_managed_path(
            checkpoint_read_path,
            context_root=managed_boundary,
            allow_missing=False,
        )
    except FileNotFoundError:
        safe_checkpoint = None
    if min_hours > 0 and safe_checkpoint is not None:
        try:
            age_seconds = (
                datetime.now(timezone.utc).timestamp()
                - os.lstat(safe_checkpoint).st_mtime
            )
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
        history_root = resolve_history_root(context_root, config=config)
        if history_root.exists():
            cursor = _load_cursor(checkpoint_read_path, context_root=managed_boundary)
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
        history_root = resolve_history_root(context_root, config=config)
        session_count = 0
        cursor = _load_cursor(
            checkpoint_read_path,
            context_root=managed_boundary,
        )
        for event in iter_history_events(
            history_root,
            event_types={"session"},
            include_payloads=False,
        ):
            if not isinstance(event, dict) or event.get("op") != "bootstrap":
                continue
            if not _is_event_after_cursor(event, cursor):
                continue
            session_count += 1
            if session_count >= min_sessions:
                break
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
    managed_boundary = context_root if _is_v2(context_root) else None
    memory_roots = _readable_memory_roots(context_root, config=config)

    query_lower = query.lower()
    results: list[dict[str, Any]] = []
    seen_entries: set[tuple[str, str]] = set()
    for memory_root in memory_roots:
        entries_path = memory_root / consolidation_cfg.entries_filename
        try:
            lines = _read_managed_text(
                entries_path,
                context_root=managed_boundary,
            ).splitlines()
        except FileNotFoundError:
            continue
        except OSError:
            if managed_boundary is not None:
                raise
            continue
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(entry, dict):
                continue
            entry_id = str(entry.get("id", ""))
            identity = (
                ("id", entry_id)
                if entry_id
                else (
                    "record",
                    json.dumps(entry, sort_keys=True, separators=(",", ":")),
                )
            )
            if identity in seen_entries:
                continue
            seen_entries.add(identity)
            searchable = " ".join([
                str(entry.get("instruction", "")),
                str(entry.get("output", "")),
                " ".join(str(t) for t in entry.get("tags", []) if isinstance(t, str)),
            ]).lower()
            if query_lower in searchable:
                results.append(entry)
                if len(results) >= limit:
                    return results

    return results
