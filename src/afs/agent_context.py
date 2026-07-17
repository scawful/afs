"""Agent context snapshot — the bridge between AFS context infrastructure and agents.

Builds lightweight context snapshots that agents receive at spawn time,
provides a mixin for query-first agent patterns, and auto-indexes agent
output back into the context system.

This module closes the loop:
    context index → agent bootstrap → agent execution → agent output → context index
"""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .codebase_explorer import build_codebase_summary, build_scoped_codebase_summary
from .context_layout import LAYOUT_VERSION, _atomic_write_text, detect_layout_version
from .path_safety import assert_no_linklike_components, iter_regular_files_no_links
from .scopes import ResolvedScope, resolve_scope, visible_mount_roots

logger = logging.getLogger(__name__)

# Environment variable for passing context snapshot path to spawned agents
AGENT_CONTEXT_ENV = "AFS_AGENT_CONTEXT_PATH"


def _scoped_codebase_summary(scoped: ResolvedScope) -> dict[str, Any]:
    """Summarize only the authorized project checkout for one scope."""

    if scoped.layout_version != LAYOUT_VERSION:
        return build_codebase_summary(scoped.context_root)
    if scoped.requester_path is None or not scoped.project_id:
        return {}
    return build_scoped_codebase_summary(
        scoped.context_root,
        scoped.requester_path,
        project_id=scoped.project_id,
    )


# ---------------------------------------------------------------------------
# 1. Agent Context Snapshot — built by supervisor, consumed by agents
# ---------------------------------------------------------------------------

@dataclass
class AgentContextSnapshot:
    """Lightweight context state passed to agents at spawn time.

    Contains enough for an agent to understand what's indexed, what memory
    topics exist, what recently happened, and what other agents are doing —
    without loading the full session bootstrap.
    """

    # Who and when
    agent_name: str = ""
    built_at: str = ""

    # Index summary: {mount_type: count}
    index_summary: dict[str, int] = field(default_factory=dict)
    index_total: int = 0

    # Memory topics: list of memory entry filenames/titles
    memory_topics: list[str] = field(default_factory=list)
    memory_entry_count: int = 0

    # Recent events (last N from history)
    recent_events: list[dict[str, Any]] = field(default_factory=list)

    # Active agents from supervisor state
    active_agents: list[str] = field(default_factory=list)

    # Mount freshness: {mount_type: {file_count, newest_mtime, stale}}
    mount_freshness: dict[str, dict[str, Any]] = field(default_factory=dict)

    # Cheap project structure summary for quick codebase orientation
    codebase_summary: dict[str, Any] = field(default_factory=dict)

    # Context root path
    context_root: str = ""
    scope_id: str = "common"
    project_id: str = ""
    requester_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent_name": self.agent_name,
            "built_at": self.built_at,
            "index_summary": self.index_summary,
            "index_total": self.index_total,
            "memory_topics": self.memory_topics,
            "memory_entry_count": self.memory_entry_count,
            "recent_events": self.recent_events,
            "active_agents": self.active_agents,
            "mount_freshness": self.mount_freshness,
            "codebase_summary": self.codebase_summary,
            "context_root": self.context_root,
            "scope_id": self.scope_id,
            "project_id": self.project_id,
            "requester_path": self.requester_path,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> AgentContextSnapshot:
        return cls(
            agent_name=data.get("agent_name", ""),
            built_at=data.get("built_at", ""),
            index_summary=data.get("index_summary", {}),
            index_total=data.get("index_total", 0),
            memory_topics=data.get("memory_topics", []),
            memory_entry_count=data.get("memory_entry_count", 0),
            recent_events=data.get("recent_events", []),
            active_agents=data.get("active_agents", []),
            mount_freshness=data.get("mount_freshness", {}),
            codebase_summary=data.get("codebase_summary", {}),
            context_root=data.get("context_root", ""),
            scope_id=str(data.get("scope_id", "common") or "common"),
            project_id=str(data.get("project_id", "") or ""),
            requester_path=str(data.get("requester_path", "") or ""),
        )


def build_agent_context_snapshot(
    agent_name: str,
    context_root: Path,
    *,
    config: Any | None = None,
    requester_path: Path | None = None,
    max_events: int = 20,
    max_memory_topics: int = 30,
) -> AgentContextSnapshot:
    """Build a context snapshot for an agent about to be spawned.

    This is intentionally cheap — reads the SQLite index summary, lists
    memory files, and grabs recent history events. No full-text content
    loading, no embedding queries, no LLM calls.
    """
    context_path = context_root.expanduser().resolve()
    scoped = resolve_scope(
        context_path,
        requester_path=requester_path,
        common=requester_path is None,
    )
    snapshot = AgentContextSnapshot(
        agent_name=agent_name,
        built_at=datetime.now(timezone.utc).isoformat(),
        context_root=str(context_path),
        scope_id=scoped.scope_id,
        project_id=scoped.project_id,
        requester_path=str(scoped.requester_path or ""),
    )

    # 1. Index summary
    try:
        from .config import load_config_model
        from .context_index import ContextSQLiteIndex
        from .manager import AFSManager

        resolved_config = config or load_config_model(merge_user=True)
        manager = AFSManager(config=resolved_config)
        index = ContextSQLiteIndex(manager, context_path)
        if scoped.layout_version == LAYOUT_VERSION:
            from .models import ContextCategory, MountType

            snapshot.index_summary = {
                mount_type.value: count
                for mount_type in MountType
                if ContextCategory.from_mount_type(mount_type) is not None
                and (
                    count := index.count_entries_scoped(
                        scoped,
                        mount_types=[mount_type],
                    )
                )
            }
            snapshot.index_total = sum(snapshot.index_summary.values())
        else:
            summary = index.summary()
            snapshot.index_summary = summary.by_mount_type
            snapshot.index_total = summary.rows_written
    except Exception:
        logger.debug("Failed to read index summary for agent context", exc_info=True)

    # 2. Memory topics
    try:
        from .models import MountType

        memory_dir = context_path / "memory"
        if memory_dir.is_dir():
            if scoped.layout_version == LAYOUT_VERSION:
                entries = sorted(
                    entry
                    for root in visible_mount_roots(
                        memory_dir,
                        mount_type=MountType.MEMORY,
                        scoped=scoped,
                    )
                    if root.is_dir()
                    for entry in iter_regular_files_no_links(root)
                    if entry.suffix in (".md", ".json", ".txt")
                )
            else:
                entries = sorted(
                    entry
                    for entry in memory_dir.iterdir()
                    if entry.is_file() and entry.suffix in (".md", ".json", ".txt")
                )
            snapshot.memory_entry_count = len(entries)
            snapshot.memory_topics = [
                entry.stem for entry in entries[:max_memory_topics]
            ]
    except Exception:
        logger.debug("Failed to read memory topics for agent context", exc_info=True)

    # 3. Recent events
    try:
        from .history import iter_history_events

        history_root = context_path / "history"
        if history_root.is_dir():
            events = []
            for event in iter_history_events(history_root):
                events.append({
                    "type": event.get("event_type", ""),
                    "source": event.get("source", ""),
                    "op": event.get("op", ""),
                    "ts": event.get("timestamp", ""),
                })
                if len(events) >= max_events:
                    break
            snapshot.recent_events = events
    except Exception:
        logger.debug("Failed to read recent events for agent context", exc_info=True)

    # 4. Mount freshness
    try:
        from .context_freshness import mount_freshness

        freshness = mount_freshness(context_path, scoped=scoped)
        snapshot.mount_freshness = {
            name: {
                "file_count": mf.file_count,
                "freshness_score": round(mf.freshness_score, 3),
                "stale": mf.stale,
            }
            for name, mf in freshness.items()
        }
    except Exception:
        logger.debug("Failed to read mount freshness for agent context", exc_info=True)

    # 5. Codebase structure
    try:
        snapshot.codebase_summary = _scoped_codebase_summary(scoped)
    except Exception:
        logger.debug("Failed to read codebase summary for agent context", exc_info=True)

    # 6. Active agents
    try:
        from .agent_registry import AgentRegistry

        registry = AgentRegistry()
        for entry in registry.list_all():
            status = entry.get("status", "")
            if status in ("running", "awaiting_review"):
                snapshot.active_agents.append(entry.get("name", ""))
    except Exception:
        logger.debug("Failed to read active agents for agent context", exc_info=True)

    return snapshot


def write_agent_context_snapshot(
    snapshot: AgentContextSnapshot,
    output_dir: Path,
    *,
    trusted_root: Path | None = None,
) -> Path:
    """Write the snapshot to a JSON file and return its path."""
    if trusted_root is not None:
        if not re.fullmatch(
            r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}",
            snapshot.agent_name,
        ):
            raise ValueError("agent name must be one safe filesystem segment")
        trusted_root = assert_no_linklike_components(
            trusted_root,
            allow_missing=False,
        )
        output_dir = assert_no_linklike_components(
            output_dir,
            boundary=trusted_root,
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"agent_context_{snapshot.agent_name}.json"
    rendered = json.dumps(snapshot.to_dict(), indent=2) + "\n"
    if trusted_root is not None:
        output_dir = assert_no_linklike_components(
            output_dir,
            boundary=trusted_root,
            allow_missing=False,
        )
        path = assert_no_linklike_components(path, boundary=output_dir)
        _atomic_write_text(path, rendered)
    else:
        path.write_text(rendered, encoding="utf-8")
    return path


def load_agent_context_snapshot() -> AgentContextSnapshot | None:
    """Load the context snapshot from the environment (called by agents at startup).

    Returns None if no snapshot path is set or the file doesn't exist.
    """
    path_str = os.environ.get(AGENT_CONTEXT_ENV, "").strip()
    if not path_str:
        return None
    path = Path(path_str)
    if not path.is_file():
        logger.debug("Agent context snapshot not found at %s", path)
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        return AgentContextSnapshot.from_dict(data)
    except Exception:
        logger.debug("Failed to load agent context snapshot from %s", path, exc_info=True)
        return None


# ---------------------------------------------------------------------------
# 2. Auto-index agent output — called from emit_result()
# ---------------------------------------------------------------------------

def index_agent_output(
    result_path: Path,
    agent_name: str,
    result_data: dict[str, Any],
) -> bool:
    """Index an agent's result into the context index.

    Called automatically by emit_result() so that agent output is
    discoverable via context.query and by other agents.

    Returns True if successfully indexed, False otherwise.
    """
    # Resolve context root from environment or config
    context_root_str = os.environ.get("AFS_CONTEXT_ROOT", "").strip()
    if not context_root_str:
        try:
            from .config import load_config_model
            cfg = load_config_model(merge_user=True)
            context_root_str = str(cfg.general.context_root.expanduser().resolve())
        except Exception:
            return False

    context_path = Path(context_root_str).expanduser().resolve()
    if not context_path.is_dir():
        return False

    try:
        from .config import load_config_model
        from .context_index import ContextSQLiteIndex
        from .context_layout import LAYOUT_VERSION, detect_layout_version
        from .context_paths import resolve_mount_root
        from .manager import AFSManager
        from .models import MountType
        from .path_safety import assert_no_linklike_components, lexical_absolute
        from .project_registry import ProjectRegistry

        config = load_config_model(merge_user=True)
        manager = AFSManager(config=config)
        index = ContextSQLiteIndex(manager, context_path)

        # Build a content summary from the result
        parts = [f"Agent: {agent_name}", f"Status: {result_data.get('status', '')}"]
        if result_data.get("task"):
            parts.append(f"Task: {result_data['task']}")
        for note in result_data.get("notes", []):
            parts.append(f"- {note}")
        payload = result_data.get("payload", {})
        if isinstance(payload, dict):
            for key, value in list(payload.items())[:10]:
                if isinstance(value, (str, int, float, bool)):
                    parts.append(f"{key}: {value}")
        content_text = "\n".join(parts)

        # A v2 row must carry a mount-relative scope prefix (``common/...``
        # or ``projects/<id>/...``). External and raw category-root outputs
        # have no authorized scope and must not enter the shared index.
        indexed_result_path = result_path.resolve()
        if detect_layout_version(context_path) == LAYOUT_VERSION:
            scratchpad_root = resolve_mount_root(
                context_path,
                MountType.SCRATCHPAD,
                config=config,
            )
            lexical_result = lexical_absolute(result_path)
            try:
                indexed_result_path = assert_no_linklike_components(
                    lexical_result,
                    boundary=scratchpad_root,
                    allow_missing=False,
                )
                relative = indexed_result_path.relative_to(scratchpad_root)
            except (OSError, ValueError):
                return False
            parts = relative.parts
            if len(parts) < 2:
                return False
            if parts[0] == "common":
                pass
            elif parts[0] == "projects" and len(parts) >= 3:
                registered_ids = {
                    record.project_id for record in ProjectRegistry(context_path).all_records()
                }
                if parts[1] not in registered_ids:
                    return False
            else:
                return False
            relative_str = relative.as_posix()
        else:
            try:
                relative = indexed_result_path.relative_to(context_path)
                relative_str = str(relative)
            except ValueError:
                relative_str = f"agents/{agent_name}/{result_path.name}"

        # Insert directly into the index
        now = datetime.now(timezone.utc).isoformat()
        stat = indexed_result_path.stat() if indexed_result_path.exists() else None

        with index._connect() as conn:
            # Remove any existing entry for this path
            conn.execute(
                "DELETE FROM file_index "
                "WHERE context_path = ? AND mount_type = ? AND relative_path = ?",
                (str(context_path), "scratchpad", relative_str),
            )
            conn.execute(
                """INSERT INTO file_index (
                    context_path, mount_type, relative_path, absolute_path,
                    is_dir, size_bytes, modified_at, indexed_at, content_text
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    str(context_path),
                    "scratchpad",
                    relative_str,
                    str(indexed_result_path),
                    0,
                    stat.st_size if stat else len(content_text),
                    now,
                    now,
                    content_text[:12000],
                ),
            )

        logger.debug("Indexed agent output: %s → %s", agent_name, relative_str)
        return True

    except Exception:
        logger.debug("Failed to index agent output for %s", agent_name, exc_info=True)
        return False


# ---------------------------------------------------------------------------
# 3. ContextAwareAgent mixin — gives agents query-first capabilities
# ---------------------------------------------------------------------------

class ContextAwareAgent:
    """Mixin that gives agents structured access to AFS context.

    Agents that inherit from this get:
    - self.context_snapshot: The bootstrap snapshot from spawn time
    - self.query_context(): Search the SQLite index
    - self.search_memory(): Find memory entries by keyword
    - self.get_recent_events(): Read recent history events
    - self.get_mount_freshness(): Check which mounts are stale
    - self.get_codebase_summary(): Read a cheap project structure summary

    Usage:
        class MyAgent(ContextAwareAgent):
            def run(self):
                self.load_context()  # call once at startup
                results = self.query_context("deployment config")
                memory = self.search_memory("last training run")
    """

    context_snapshot: AgentContextSnapshot | None = None
    _context_path: Path | None = None
    _config: Any = None

    def load_context(self) -> AgentContextSnapshot | None:
        """Load the context snapshot from environment. Call once at agent startup."""
        self.context_snapshot = load_agent_context_snapshot()
        if self.context_snapshot and self.context_snapshot.context_root:
            self._context_path = Path(self.context_snapshot.context_root)
        elif os.environ.get("AFS_CONTEXT_ROOT"):
            self._context_path = Path(os.environ["AFS_CONTEXT_ROOT"]).expanduser().resolve()
        return self.context_snapshot

    def _get_index(self):
        """Lazily create a context index instance."""
        if self._context_path is None:
            return None
        try:
            from .config import load_config_model
            from .context_index import ContextSQLiteIndex
            from .manager import AFSManager

            if self._config is None:
                self._config = load_config_model(merge_user=True)
            manager = AFSManager(config=self._config)
            return ContextSQLiteIndex(manager, self._context_path)
        except Exception:
            logger.debug("Failed to create context index", exc_info=True)
            return None

    def _get_scope(self) -> ResolvedScope | None:
        """Resolve snapshot authority, falling back to common-only in v2."""

        if self._context_path is None:
            return None
        try:
            if detect_layout_version(self._context_path) != LAYOUT_VERSION:
                return resolve_scope(self._context_path)
            if (
                self.context_snapshot is not None
                and self.context_snapshot.scope_id != "common"
                and self.context_snapshot.requester_path
            ):
                try:
                    scoped = resolve_scope(
                        self._context_path,
                        requester_path=Path(
                            self.context_snapshot.requester_path
                        ).expanduser().resolve(),
                    )
                except (OSError, PermissionError, ValueError):
                    scoped = None
                if (
                    scoped is not None
                    and scoped.scope_id == self.context_snapshot.scope_id
                ):
                    return scoped
            return resolve_scope(self._context_path, common=True)
        except (OSError, PermissionError, ValueError):
            logger.debug("Failed to resolve agent context scope", exc_info=True)
            return None

    def query_context(
        self,
        query: str,
        *,
        mount_types: list[str] | None = None,
        limit: int = 10,
        include_content: bool = False,
    ) -> list[dict[str, Any]]:
        """Search the context index. Returns matching entries with paths and metadata."""
        index = self._get_index()
        if index is None:
            return []
        try:
            from .models import MountType
            mt = [MountType(t) for t in mount_types] if mount_types else None
            scoped = self._get_scope()
            if scoped is None:
                return []
            return index.query_scoped(
                scoped,
                query=query,
                mount_types=mt,
                limit=limit,
                include_content=include_content,
            )
        except Exception:
            logger.debug("Context query failed: %s", query, exc_info=True)
            return []

    def search_memory(
        self,
        keyword: str,
        *,
        limit: int = 10,
    ) -> list[dict[str, str]]:
        """Search memory entries by keyword in filename or content."""
        if self._context_path is None:
            return []
        from .models import MountType

        memory_dir = self._context_path / "memory"
        if not memory_dir.is_dir():
            return []

        results = []
        keyword_lower = keyword.lower()
        try:
            scoped = self._get_scope()
            if scoped is None:
                return []
            if scoped.layout_version == LAYOUT_VERSION:
                entries = sorted(
                    entry
                    for root in visible_mount_roots(
                        memory_dir,
                        mount_type=MountType.MEMORY,
                        scoped=scoped,
                    )
                    if root.is_dir()
                    for entry in iter_regular_files_no_links(root)
                )
            else:
                entries = sorted(memory_dir.iterdir())
            for entry in entries:
                if not entry.is_file() or entry.suffix not in (".md", ".json", ".txt"):
                    continue

                # Check filename match
                name_match = keyword_lower in entry.stem.lower()

                # Check content match (first 2000 chars only)
                content_match = False
                content = ""
                try:
                    content = entry.read_text(encoding="utf-8")[:2000]
                    content_match = keyword_lower in content.lower()
                except Exception:
                    pass

                if name_match or content_match:
                    results.append({
                        "path": str(entry),
                        "name": entry.stem,
                        "preview": content[:200] if content else "",
                    })
                    if len(results) >= limit:
                        break
        except Exception:
            logger.debug("Memory search failed: %s", keyword, exc_info=True)

        return results

    def get_recent_events(
        self,
        *,
        event_types: set[str] | None = None,
        limit: int = 20,
    ) -> list[dict[str, Any]]:
        """Get recent events from the history log."""
        if self._context_path is None:
            return []
        try:
            from .history import iter_history_events

            history_root = self._context_path / "history"
            if not history_root.is_dir():
                return []

            events = []
            for event in iter_history_events(history_root, event_types=event_types):
                events.append(event)
                if len(events) >= limit:
                    break
            return events
        except Exception:
            logger.debug("Event query failed", exc_info=True)
            return []

    def get_mount_freshness(self) -> dict[str, dict[str, Any]]:
        """Get mount freshness data (from snapshot or live)."""
        if self.context_snapshot and self.context_snapshot.mount_freshness:
            return self.context_snapshot.mount_freshness
        if self._context_path is None:
            return {}
        try:
            from .context_freshness import mount_freshness
            scoped = self._get_scope()
            if scoped is None:
                return {}
            freshness = mount_freshness(self._context_path, scoped=scoped)
            return {
                name: {
                    "file_count": mf.file_count,
                    "freshness_score": round(mf.freshness_score, 3),
                    "stale": mf.stale,
                }
                for name, mf in freshness.items()
            }
        except Exception:
            return {}

    def get_codebase_summary(self) -> dict[str, Any]:
        """Get a cheap codebase structure summary (from snapshot or live)."""
        if self.context_snapshot and self.context_snapshot.codebase_summary:
            return self.context_snapshot.codebase_summary
        if self._context_path is None:
            return {}
        try:
            scoped = self._get_scope()
            if scoped is None:
                return {}
            return _scoped_codebase_summary(scoped)
        except Exception:
            logger.debug("Codebase summary failed", exc_info=True)
            return {}
