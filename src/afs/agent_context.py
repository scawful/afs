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
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Environment variable for passing context snapshot path to spawned agents
AGENT_CONTEXT_ENV = "AFS_AGENT_CONTEXT_PATH"


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

    # Context root path
    context_root: str = ""

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
            "context_root": self.context_root,
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
            context_root=data.get("context_root", ""),
        )


def build_agent_context_snapshot(
    agent_name: str,
    context_root: Path,
    *,
    config: Any | None = None,
    max_events: int = 20,
    max_memory_topics: int = 30,
) -> AgentContextSnapshot:
    """Build a context snapshot for an agent about to be spawned.

    This is intentionally cheap — reads the SQLite index summary, lists
    memory files, and grabs recent history events. No full-text content
    loading, no embedding queries, no LLM calls.
    """
    context_path = context_root.expanduser().resolve()
    snapshot = AgentContextSnapshot(
        agent_name=agent_name,
        built_at=datetime.now(timezone.utc).isoformat(),
        context_root=str(context_path),
    )

    # 1. Index summary
    try:
        from .config import load_config_model
        from .context_index import ContextSQLiteIndex
        from .manager import AFSManager

        resolved_config = config or load_config_model(merge_user=True)
        manager = AFSManager(config=resolved_config)
        index = ContextSQLiteIndex(manager, context_path)
        summary = index.summary()
        snapshot.index_summary = summary.by_mount_type
        snapshot.index_total = summary.rows_written
    except Exception:
        logger.debug("Failed to read index summary for agent context", exc_info=True)

    # 2. Memory topics
    try:
        memory_dir = context_path / "memory"
        if memory_dir.is_dir():
            entries = sorted(memory_dir.iterdir())
            snapshot.memory_entry_count = len(entries)
            snapshot.memory_topics = [
                e.stem for e in entries[:max_memory_topics]
                if e.is_file() and e.suffix in (".md", ".json", ".txt")
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

        freshness = mount_freshness(context_path)
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

    # 5. Active agents
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
) -> Path:
    """Write the snapshot to a JSON file and return its path."""
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"agent_context_{snapshot.agent_name}.json"
    path.write_text(
        json.dumps(snapshot.to_dict(), indent=2) + "\n",
        encoding="utf-8",
    )
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
        from .manager import AFSManager

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

        # Determine relative path from context root
        try:
            relative = result_path.resolve().relative_to(context_path)
            relative_str = str(relative)
        except ValueError:
            relative_str = f"agents/{agent_name}/{result_path.name}"

        # Insert directly into the index
        now = datetime.now(timezone.utc).isoformat()
        stat = result_path.stat() if result_path.exists() else None

        with index._connect() as conn:
            # Remove any existing entry for this path
            conn.execute(
                "DELETE FROM file_index WHERE context_path = ? AND relative_path = ?",
                (str(context_path), relative_str),
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
                    str(result_path.resolve()),
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
            return index.query(
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
        memory_dir = self._context_path / "memory"
        if not memory_dir.is_dir():
            return []

        results = []
        keyword_lower = keyword.lower()
        try:
            for entry in sorted(memory_dir.iterdir()):
                if not entry.is_file():
                    continue
                if entry.suffix not in (".md", ".json", ".txt"):
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
            freshness = mount_freshness(self._context_path)
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
