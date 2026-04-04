"""Token-budgeted context packs for Gemini, Claude, Codex, and generic clients."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .context_index import ContextSQLiteIndex
from .context_paths import load_context_metadata, resolve_agent_output_root, resolve_mount_root
from .embeddings import search_embedding_index
from .manager import AFSManager
from .models import MountType
from .sensitivity import matches_path_rules
from .session_bootstrap import _build_recommendations, build_session_bootstrap
from .session_workflows import build_session_execution_profile

logger = logging.getLogger(__name__)

DEFAULT_CONTEXT_PACK_TOKENS = {
    "generic": 8000,
    "gemini": 16000,
    "claude": 12000,
    "codex": 12000,
}
PACK_MODE_CHOICES = (
    "focused",
    "retrieval",
    "full_slice",
)
DEFAULT_SEARCH_MOUNTS = (
    MountType.SCRATCHPAD,
    MountType.MEMORY,
    MountType.KNOWLEDGE,
    MountType.ITEMS,
)
CONTEXT_PACK_CACHE_VERSION = 1


@dataclass
class ContextPackSection:
    title: str
    body: str
    priority: int
    sources: list[str] = field(default_factory=list)

    def estimated_tokens(self) -> int:
        return estimate_tokens(self.body) + max(4, estimate_tokens(self.title))

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "body": self.body,
            "priority": self.priority,
            "sources": list(self.sources),
            "estimated_tokens": self.estimated_tokens(),
        }


def estimate_tokens(text: str) -> int:
    """Approximate token count cheaply for context budgeting."""
    if not text or not text.strip():
        return 0
    return max(1, (len(text) + 3) // 4)


def build_context_pack(
    manager: AFSManager,
    context_path: Path,
    *,
    query: str = "",
    task: str = "",
    model: str = "generic",
    workflow: str = "general",
    tool_profile: str = "default",
    pack_mode: str = "focused",
    token_budget: int | None = None,
    include_content: bool = False,
    max_query_results: int = 6,
    max_embedding_results: int = 4,
) -> dict[str, Any]:
    """Build a model-aware context pack from AFS state."""
    context_path = context_path.expanduser().resolve()
    normalized_model = _normalize_model(model)
    normalized_pack_mode = _normalize_pack_mode(pack_mode)
    resolved_budget = token_budget or DEFAULT_CONTEXT_PACK_TOKENS[normalized_model]
    resolved_include_content, resolved_max_query_results, resolved_max_embedding_results = (
        _resolve_pack_mode_settings(
            normalized_pack_mode,
            include_content=include_content,
            max_query_results=max_query_results,
            max_embedding_results=max_embedding_results,
        )
    )

    # --- Session pack cache: early return before expensive bootstrap/scan ---
    session_cached = _load_session_pack_cache(
        manager,
        context_path,
        query=query,
        task=task,
        model=normalized_model,
        workflow=workflow,
        tool_profile=tool_profile,
        pack_mode=normalized_pack_mode,
        token_budget=resolved_budget,
        include_content=resolved_include_content,
        max_query_results=resolved_max_query_results,
        max_embedding_results=resolved_max_embedding_results,
    )
    if session_cached is not None:
        return session_cached

    execution_profile = build_session_execution_profile(
        model=normalized_model,
        workflow=workflow,
        tool_profile=tool_profile,
    )
    bootstrap = build_session_bootstrap(manager, context_path, record_event=False)
    cached_bootstrap = _cache_bootstrap(manager, context_path, bootstrap)
    cache_key = _context_pack_cache_key(
        context_path,
        bootstrap=cached_bootstrap,
        query=query,
        task=task,
        model=normalized_model,
        pack_mode=normalized_pack_mode,
        workflow=str(execution_profile["workflow"]),
        tool_profile=str((execution_profile.get("tool_profile") or {}).get("name", "default")),
        token_budget=resolved_budget,
        include_content=resolved_include_content,
        max_query_results=resolved_max_query_results,
        max_embedding_results=resolved_max_embedding_results,
    )
    cached = _load_cached_context_pack(
        manager,
        context_path,
        model=normalized_model,
        cache_key=cache_key,
    )
    if cached is not None:
        return cached
    execution_profile_text = _render_execution_profile_block(execution_profile)
    focus_block = _render_focus_block(task=task, query=query)
    reserved_tokens = 0
    if execution_profile_text:
        reserved_tokens += estimate_tokens(execution_profile_text) + estimate_tokens("Execution Profile")
    if focus_block:
        reserved_tokens += estimate_tokens(focus_block["body"]) + estimate_tokens(focus_block["title"])
    guidance = _model_guidance(normalized_model)
    sections = _build_sections(
        manager,
        context_path,
        bootstrap=bootstrap,
        query=query,
        model=normalized_model,
        pack_mode=normalized_pack_mode,
        include_content=resolved_include_content,
        max_query_results=resolved_max_query_results,
        max_embedding_results=resolved_max_embedding_results,
    )
    chosen, omitted = _select_sections(
        sections,
        token_budget=max(0, resolved_budget - reserved_tokens),
    )
    sources = sorted({source for section in chosen for source in section.sources})
    estimated = reserved_tokens + sum(section.estimated_tokens() for section in chosen)

    pack = {
        "context_path": str(context_path),
        "project": bootstrap["project"],
        "profile": bootstrap["profile"],
        "model": normalized_model,
        "pack_mode": normalized_pack_mode,
        "pack_mode_summary": _pack_mode_summary(normalized_pack_mode),
        "query": query,
        "task": task,
        "execution_profile": execution_profile,
        "token_budget": resolved_budget,
        "estimated_tokens": estimated,
        "guidance": guidance,
        "sections": [section.to_dict() for section in chosen],
        "sources": sources,
        "omitted_sections": [section.title for section in omitted],
        "cache": {
            "version": CONTEXT_PACK_CACHE_VERSION,
            "key": cache_key,
            "hit": False,
        },
    }
    pack["cache"]["prefix_hash"] = _context_pack_prefix_hash(pack)
    pack["cache"]["stable_prefix_hash"] = _context_pack_stable_prefix_hash(pack)
    _write_session_pack_cache(
        manager,
        context_path,
        pack,
        query=query,
        task=task,
        model=normalized_model,
        workflow=workflow,
        tool_profile=tool_profile,
        pack_mode=normalized_pack_mode,
        token_budget=resolved_budget,
        include_content=resolved_include_content,
        max_query_results=resolved_max_query_results,
        max_embedding_results=resolved_max_embedding_results,
    )
    return pack


def render_context_pack(pack: dict[str, Any]) -> str:
    """Render a context pack for direct model consumption."""
    lines = [
        f"# AFS Context Pack: {pack['project']}",
        f"Context: {pack['context_path']}",
        f"Profile: {pack['profile']}",
        f"Target model: {pack['model']}",
        f"Pack mode: {pack.get('pack_mode', 'focused')}",
        f"Token budget: {pack['estimated_tokens']}/{pack['token_budget']}",
    ]
    execution_profile = pack.get("execution_profile")
    execution_profile_text = _render_execution_profile_block(execution_profile)
    if execution_profile_text:
        lines.extend(["", "## Execution Profile", execution_profile_text])
    guidance = pack.get("guidance")
    if isinstance(guidance, str) and guidance.strip():
        lines.extend(["", "## Guidance", guidance.strip()])
    pack_mode_summary = str(pack.get("pack_mode_summary", "")).strip()
    if pack_mode_summary:
        lines.extend(["", "## Pack Mode", pack_mode_summary])

    for section in pack.get("sections", []):
        lines.extend(["", f"## {section['title']}", section["body"]])
        if section.get("sources"):
            lines.append("Sources:")
            for source in section["sources"]:
                lines.append(f"- {source}")

    omitted = pack.get("omitted_sections") or []
    if omitted:
        lines.extend(["", "## Omitted", ", ".join(str(item) for item in omitted)])
    focus_block = _render_focus_block(
        task=str(pack.get("task", "")),
        query=str(pack.get("query", "")),
    )
    if focus_block is not None:
        lines.extend(["", f"## {focus_block['title']}", focus_block["body"]])
    return "\n".join(lines)


def write_context_pack_artifacts(
    manager: AFSManager,
    context_path: Path,
    pack: dict[str, Any],
) -> dict[str, str]:
    """Persist the latest context pack for wrappers and agent handoff."""
    json_path, markdown_path = _context_pack_artifact_paths(
        manager,
        context_path,
        pack["model"],
    )
    json_path.parent.mkdir(parents=True, exist_ok=True)

    payload = dict(pack)
    cache = dict(payload.get("cache") or {})
    cache.setdefault("version", CONTEXT_PACK_CACHE_VERSION)
    cache["hit"] = False
    payload["cache"] = cache
    payload["generated_at"] = datetime.now(timezone.utc).isoformat()
    payload["artifact_paths"] = {
        "json": str(json_path),
        "markdown": str(markdown_path),
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_context_pack(payload) + "\n", encoding="utf-8")
    return payload["artifact_paths"]


def _context_pack_artifact_paths(
    manager: AFSManager,
    context_path: Path,
    model: str,
) -> tuple[Path, Path]:
    output_root = resolve_agent_output_root(context_path, config=manager.config)
    suffix = model.replace("/", "_")
    return (
        output_root / f"session_pack_{suffix}.json",
        output_root / f"session_pack_{suffix}.md",
    )


def _context_pack_cache_key(
    context_path: Path,
    *,
    bootstrap: dict[str, Any],
    query: str,
    task: str,
    model: str,
    pack_mode: str,
    workflow: str,
    tool_profile: str,
    token_budget: int,
    include_content: bool,
    max_query_results: int,
    max_embedding_results: int,
) -> str:
    payload = {
        "version": CONTEXT_PACK_CACHE_VERSION,
        "context_path": str(context_path),
        "query": query,
        "task": task,
        "model": model,
        "pack_mode": pack_mode,
        "workflow": workflow,
        "tool_profile": tool_profile,
        "token_budget": token_budget,
        "include_content": include_content,
        "max_query_results": max_query_results,
        "max_embedding_results": max_embedding_results,
        "bootstrap": bootstrap,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _cache_bootstrap(
    manager: AFSManager,
    context_path: Path,
    bootstrap: dict[str, Any],
) -> dict[str, Any]:
    result = json.loads(json.dumps(bootstrap, default=str))
    scratch = resolve_mount_root(context_path, MountType.SCRATCHPAD, config=manager.config)
    output = resolve_agent_output_root(context_path, config=manager.config)
    try:
        prefix = str(output.relative_to(scratch)).replace("\\", "/").strip("/")
    except ValueError:
        prefix = ""

    if not prefix:
        return result

    ignored = 0
    if output.exists():
        ignored = sum(1 for item in output.rglob("*") if item.is_file())

    status = result.get("status")
    if isinstance(status, dict):
        counts = status.get("mount_counts")
        if isinstance(counts, dict):
            counts.pop(MountType.HISTORY.value, None)
            counts.pop(MountType.GLOBAL.value, None)
            current = counts.get("scratchpad", 0)
            if isinstance(current, int):
                counts["scratchpad"] = max(0, current - ignored)
            status["total_files"] = sum(value for value in counts.values() if isinstance(value, int))
        index = status.get("index")
        if isinstance(index, dict):
            status["index"] = {"enabled": bool(index.get("enabled", False))}

    diff = result.get("diff")
    if isinstance(diff, dict):
        removed = 0
        for key in ("added", "modified", "deleted"):
            items = diff.get(key)
            if not isinstance(items, list):
                continue
            keep = []
            for item in items:
                if _is_agent_artifact(item, prefix):
                    removed += 1
                    continue
                keep.append(item)
            diff[key] = keep
        total = diff.get("total_changes")
        if isinstance(total, int):
            diff["total_changes"] = max(0, total - removed)
            diff["available"] = diff["total_changes"] > 0
        diff["error"] = ""
        diff.pop("truncated", None)

    stale = result.get("stale_mounts")
    if isinstance(stale, list):
        result["stale_mounts"] = [
            item for item in stale
            if item not in {MountType.HISTORY.value, MountType.GLOBAL.value}
        ]

    # Strip volatile fields that change between calls but don't affect
    # pack content (snapshot presence, mtime-based freshness scores).
    result.pop("session_changes", None)
    result.pop("mount_freshness", None)

    result["recommended_actions"] = _build_recommendations(result)
    return result


def _is_agent_artifact(item: Any, prefix: str) -> bool:
    if not isinstance(item, dict):
        return False
    mount = item.get("mount_type")
    if mount in {MountType.HISTORY.value, MountType.GLOBAL.value}:
        return True
    if mount != MountType.SCRATCHPAD.value:
        return False
    rel = item.get("relative_path")
    if not isinstance(rel, str):
        return False
    rel = rel.replace("\\", "/").strip("/")
    return rel == prefix or rel.startswith(f"{prefix}/")


def _load_cached_context_pack(
    manager: AFSManager,
    context_path: Path,
    *,
    model: str,
    cache_key: str,
) -> dict[str, Any] | None:
    json_path, markdown_path = _context_pack_artifact_paths(manager, context_path, model)
    if not json_path.exists():
        return None
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    cache = payload.get("cache")
    if not isinstance(cache, dict):
        return None
    if cache.get("version") != CONTEXT_PACK_CACHE_VERSION:
        return None
    if cache.get("key") != cache_key:
        return None
    result = dict(payload)
    result["artifact_paths"] = {
        "json": str(json_path),
        "markdown": str(markdown_path),
    }
    result["cache"] = {
        **cache,
        "hit": True,
    }
    return result


def _session_pack_cache_key(
    context_path: Path,
    *,
    query: str,
    task: str,
    model: str,
    workflow: str,
    tool_profile: str,
    pack_mode: str,
    token_budget: int,
    include_content: bool,
    max_query_results: int,
    max_embedding_results: int,
) -> str:
    """Compute a lightweight cache key from input parameters only (no bootstrap)."""
    payload = {
        "version": CONTEXT_PACK_CACHE_VERSION,
        "context_path": str(context_path),
        "query": query,
        "task": task,
        "model": model,
        "workflow": workflow,
        "tool_profile": tool_profile,
        "pack_mode": pack_mode,
        "token_budget": token_budget,
        "include_content": include_content,
        "max_query_results": max_query_results,
        "max_embedding_results": max_embedding_results,
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _session_pack_cache_path(manager: AFSManager, cache_key: str) -> Path:
    """Return the file path for a session pack cache entry."""
    cache_dir = _resolve_session_pack_cache_dir(manager.config)
    return cache_dir / f"{cache_key}.json"


def _nearest_existing_parent(path: Path) -> Path:
    current = path.expanduser().resolve()
    while not current.exists() and current != current.parent:
        current = current.parent
    return current


def _resolve_session_pack_cache_dir(config: Any) -> Path:
    """Return a writable cache directory for session-pack entries."""
    cache_dir = Path(config.session_pack_cache.cache_dir).expanduser().resolve()
    if os.access(_nearest_existing_parent(cache_dir), os.W_OK):
        return cache_dir
    return (Path(tempfile.gettempdir()) / "afs" / "session_pack_cache").resolve()


def _mount_fingerprint(context_path: Path, *, config: Any = None) -> str:
    """Compute a lightweight fingerprint from mount file mtimes.

    Walks the searchable mount directories and hashes (path, mtime, size)
    tuples. This is much cheaper than a full bootstrap but detects any
    file creation, deletion, or content change.

    Agent output artifacts (afs_agents/) are excluded because they are
    generated during bootstrap and should not invalidate the cache.
    """
    agent_output = resolve_agent_output_root(context_path, config=config)
    entries: list[str] = []
    # Include stable metadata content, but ignore dynamic timestamps that can
    # change as a side effect of bootstrap/index work.
    metadata = load_context_metadata(context_path)
    if metadata is not None:
        metadata_payload = metadata.to_dict()
        metadata_payload.pop("created_at", None)
        metadata_payload.pop("agents", None)
        mount_provenance = metadata_payload.get("mount_provenance")
        if isinstance(mount_provenance, dict):
            for entries_by_alias in mount_provenance.values():
                if not isinstance(entries_by_alias, dict):
                    continue
                for payload in entries_by_alias.values():
                    if isinstance(payload, dict):
                        payload.pop("updated_at", None)
        entries.append(
            json.dumps(
                {"metadata": metadata_payload},
                sort_keys=True,
                separators=(",", ":"),
            )
        )
    # Include all files in searchable mount roots
    for mount_type in DEFAULT_SEARCH_MOUNTS:
        mount_root = resolve_mount_root(context_path, mount_type, config=config)
        if not mount_root.exists():
            continue
        try:
            for path in sorted(mount_root.rglob("*")):
                if not path.is_file():
                    continue
                # Skip agent output artifacts (snapshots, reports, etc.)
                try:
                    path.relative_to(agent_output)
                    continue
                except ValueError:
                    pass
                try:
                    st = path.stat()
                    entries.append(f"{path}:{st.st_mtime}:{st.st_size}")
                except OSError:
                    pass
        except OSError:
            pass
    payload = "\n".join(entries).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _load_session_pack_cache(
    manager: AFSManager,
    context_path: Path,
    *,
    query: str,
    task: str,
    model: str,
    workflow: str,
    tool_profile: str,
    pack_mode: str,
    token_budget: int,
    include_content: bool,
    max_query_results: int,
    max_embedding_results: int,
) -> dict[str, Any] | None:
    """Attempt to load a fresh session pack from the file-based cache.

    Returns the cached pack dict on hit, or None on miss/stale/disabled.
    """
    cache_cfg = manager.config.session_pack_cache
    if not cache_cfg.enabled:
        logger.debug("session pack cache disabled")
        return None

    cache_key = _session_pack_cache_key(
        context_path,
        query=query,
        task=task,
        model=model,
        workflow=workflow,
        tool_profile=tool_profile,
        pack_mode=pack_mode,
        token_budget=token_budget,
        include_content=include_content,
        max_query_results=max_query_results,
        max_embedding_results=max_embedding_results,
    )
    cache_file = _session_pack_cache_path(manager, cache_key)
    if not cache_file.exists():
        logger.debug("session pack cache miss: no cache file for key %s", cache_key[:12])
        return None

    try:
        payload = json.loads(cache_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.debug("session pack cache miss: unreadable cache file %s", cache_file)
        return None

    meta = payload.get("_session_cache_meta")
    if not isinstance(meta, dict):
        logger.debug("session pack cache miss: missing metadata in %s", cache_file)
        return None

    # TTL check
    cached_at = meta.get("cached_at_epoch")
    if not isinstance(cached_at, (int, float)):
        logger.debug("session pack cache miss: no cached_at timestamp")
        return None
    now = datetime.now(timezone.utc).timestamp()
    age_seconds = now - cached_at
    if age_seconds > cache_cfg.ttl_seconds:
        logger.debug(
            "session pack cache miss: expired (age=%.1fs, ttl=%ds)",
            age_seconds,
            cache_cfg.ttl_seconds,
        )
        return None

    # Mount fingerprint invalidation: recompute a lightweight hash of mount
    # file mtimes/sizes and compare against the stored fingerprint.
    stored_fingerprint = meta.get("mount_fingerprint", "")
    current_fingerprint = _mount_fingerprint(context_path, config=manager.config)
    if current_fingerprint != stored_fingerprint:
        logger.debug(
            "session pack cache miss: mount fingerprint changed (stored=%s, current=%s)",
            stored_fingerprint[:12],
            current_fingerprint[:12],
        )
        return None

    # Valid cache hit
    pack = payload.get("pack")
    if not isinstance(pack, dict):
        logger.debug("session pack cache miss: no pack data in cache file")
        return None

    pack["cache"] = dict(pack.get("cache") or {})
    pack["cache"]["hit"] = True
    pack["cache"]["session_cache_hit"] = True
    pack["cache"]["session_cache_age_seconds"] = round(age_seconds, 1)

    # Populate artifact_paths from existing artifact files when available,
    # matching the behavior of the artifact-based cache layer.
    if "artifact_paths" not in pack:
        pack_model = pack.get("model", "generic")
        json_path, markdown_path = _context_pack_artifact_paths(manager, context_path, pack_model)
        if json_path.exists():
            pack["artifact_paths"] = {
                "json": str(json_path),
                "markdown": str(markdown_path),
            }

    logger.debug(
        "session pack cache hit: key=%s age=%.1fs",
        cache_key[:12],
        age_seconds,
    )
    return pack


def _write_session_pack_cache(
    manager: AFSManager,
    context_path: Path,
    pack: dict[str, Any],
    *,
    query: str,
    task: str,
    model: str,
    workflow: str,
    tool_profile: str,
    pack_mode: str,
    token_budget: int,
    include_content: bool,
    max_query_results: int,
    max_embedding_results: int,
) -> None:
    """Persist a context pack to the session cache for future reuse."""
    cache_cfg = manager.config.session_pack_cache
    if not cache_cfg.enabled:
        return

    cache_key = _session_pack_cache_key(
        context_path,
        query=query,
        task=task,
        model=model,
        workflow=workflow,
        tool_profile=tool_profile,
        pack_mode=pack_mode,
        token_budget=token_budget,
        include_content=include_content,
        max_query_results=max_query_results,
        max_embedding_results=max_embedding_results,
    )
    cache_file = _session_pack_cache_path(manager, cache_key)

    now = datetime.now(timezone.utc)
    fingerprint = _mount_fingerprint(context_path, config=manager.config)
    payload = {
        "_session_cache_meta": {
            "cached_at": now.isoformat(),
            "cached_at_epoch": now.timestamp(),
            "context_path": str(context_path),
            "cache_key": cache_key,
            "mount_fingerprint": fingerprint,
            "version": CONTEXT_PACK_CACHE_VERSION,
        },
        "pack": pack,
    }
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(
            json.dumps(payload, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
    except OSError:
        logger.debug("session pack cache: failed to write %s", cache_file)


def clear_pack_cache(context_path: Path | None = None, *, config: Any = None) -> int:
    """Remove session pack cache files.

    Args:
        context_path: If given, only clear caches for this context path.
            If None, clear all session pack cache files.
        config: Optional AFSConfig instance. Uses default if not provided.

    Returns:
        Number of cache files removed.
    """
    if config is None:
        from .config import load_config_model
        config = load_config_model()

    cache_dir = _resolve_session_pack_cache_dir(config)
    if not cache_dir.exists():
        return 0

    removed = 0
    for cache_file in sorted(cache_dir.glob("*.json")):
        if not cache_file.is_file():
            continue
        if context_path is not None:
            # Only remove files that belong to this context_path
            try:
                payload = json.loads(cache_file.read_text(encoding="utf-8"))
                meta = payload.get("_session_cache_meta", {})
                cached_context = meta.get("context_path", "")
                resolved = str(context_path.expanduser().resolve())
                if cached_context != resolved:
                    continue
            except (OSError, json.JSONDecodeError):
                pass
        try:
            cache_file.unlink()
            removed += 1
        except OSError:
            pass

    logger.debug("session pack cache: cleared %d files", removed)
    return removed


def _normalize_model(model: str) -> str:
    normalized = (model or "generic").strip().lower()
    if normalized not in DEFAULT_CONTEXT_PACK_TOKENS:
        return "generic"
    return normalized


def _normalize_pack_mode(pack_mode: str | None) -> str:
    normalized = (pack_mode or "focused").strip().lower()
    if normalized not in PACK_MODE_CHOICES:
        return "focused"
    return normalized


def _resolve_pack_mode_settings(
    pack_mode: str,
    *,
    include_content: bool,
    max_query_results: int,
    max_embedding_results: int,
) -> tuple[bool, int, int]:
    resolved_include_content = bool(include_content)
    resolved_query_results = max(1, max_query_results)
    resolved_embedding_results = max(1, max_embedding_results)
    if pack_mode == "full_slice":
        resolved_include_content = True
        resolved_query_results = max(resolved_query_results, 8)
        resolved_embedding_results = max(resolved_embedding_results, 6)
    elif pack_mode == "retrieval":
        resolved_query_results = max(resolved_query_results, 6)
        resolved_embedding_results = max(resolved_embedding_results, 4)
    return (
        resolved_include_content,
        resolved_query_results,
        resolved_embedding_results,
    )


def _pack_mode_summary(pack_mode: str) -> str:
    summaries = {
        "focused": "Balanced working set for a bounded task. Keeps the normal session context shape and retrieval breadth.",
        "retrieval": "Query-first working set that moves indexed and embedding hits ahead of broader session context.",
        "full_slice": "Broader long-context slice that expands retrieval breadth and adds a knowledge inventory section.",
    }
    return summaries[pack_mode]


def _model_guidance(model: str) -> str:
    guidance = {
        "generic": (
            "Prefer the highest-signal sections first. Cite file paths when acting on retrieved context. "
            "For follow-on indexed retrieval, use `afs query <text> --path <workspace>` or "
            "`afs context query <text> --path <workspace>`. If indexed search is stale or missing, "
            "run `afs index rebuild --path <workspace>`."
        ),
        "gemini": (
            "Gemini should start with the recommendations and source-backed sections, then ask for more retrieval "
            "only if the pack leaves gaps. For follow-on indexed retrieval, use `afs query <text> --path <workspace>` "
            "or `afs context query <text> --path <workspace>`. If indexed search is stale or missing, "
            "run `afs index rebuild --path <workspace>`."
        ),
        "claude": (
            "Claude should translate this pack into a short execution plan before editing, keeping cited paths "
            "attached to claims. For follow-on indexed retrieval, use `afs query <text> --path <workspace>` or "
            "`afs context query <text> --path <workspace>`. If indexed search is stale or missing, "
            "run `afs index rebuild --path <workspace>`."
        ),
        "codex": (
            "Codex should prioritize actionable files, recent drift, and concrete next steps before writing code. "
            "For follow-on indexed retrieval, use `afs query <text> --path <workspace>` or "
            "`afs context query <text> --path <workspace>`. If indexed search is stale or missing, "
            "run `afs index rebuild --path <workspace>`."
        ),
    }
    return guidance[model]


def _build_sections(
    manager: AFSManager,
    context_path: Path,
    *,
    bootstrap: dict[str, Any],
    query: str,
    model: str,
    pack_mode: str,
    include_content: bool,
    max_query_results: int,
    max_embedding_results: int,
) -> list[ContextPackSection]:
    sections: list[ContextPackSection] = []
    sections.append(
        ContextPackSection(
            title="Recommended Actions",
            body=_render_list(bootstrap.get("recommended_actions", []), fallback="No urgent actions."),
            priority=0,
        )
    )
    sections.append(
        ContextPackSection(
            title="Context Health",
            body=_render_status_block(bootstrap["status"], bootstrap["diff"]),
            priority=5,
        )
    )
    scratchpad = bootstrap.get("scratchpad", {})
    sections.append(
        ContextPackSection(
            title="Scratchpad State",
            body=_render_scratchpad_block(scratchpad),
            priority=10,
            sources=[scratchpad["path"]] if scratchpad.get("path") else [],
        )
    )

    handoff = bootstrap.get("handoff", {})
    if handoff.get("available"):
        sections.append(
            ContextPackSection(
                title="Latest Handoff",
                body=_render_handoff_block(handoff),
                priority=15,
            )
        )

    tasks = bootstrap.get("tasks", {})
    sections.append(
        ContextPackSection(
            title="Open Tasks",
            body=_render_tasks_block(tasks),
            priority=20,
        )
    )
    hivemind = bootstrap.get("hivemind", {})
    sections.append(
        ContextPackSection(
            title="Recent Hivemind",
            body=_render_hivemind_block(hivemind),
            priority=25,
        )
    )
    memory = bootstrap.get("memory", {})
    sections.append(
        ContextPackSection(
            title="Durable Memory",
            body=_render_memory_block(memory),
            priority=30,
            sources=[memory["latest_markdown_path"]] if memory.get("latest_markdown_path") else [],
        )
    )
    if pack_mode == "full_slice":
        knowledge_slice = _knowledge_slice_section(
            manager,
            context_path,
            limit=max_query_results,
        )
        if knowledge_slice is not None:
            sections.append(knowledge_slice)

    if query.strip():
        query_sections = _query_sections(
            manager,
            context_path,
            query=query,
            include_content=include_content,
            max_results=max_query_results,
            pack_mode=pack_mode,
        )
        sections.extend(query_sections)
        embedding_section = _embedding_section(
            manager,
            context_path,
            query=query,
            max_results=max_embedding_results,
            pack_mode=pack_mode,
        )
        if embedding_section is not None:
            sections.append(embedding_section)

    sections.append(
        ContextPackSection(
            title="Model Usage Notes",
            body=_model_guidance(model),
            priority=90,
        )
    )
    return [section for section in sections if section.body.strip()]


def _query_sections(
    manager: AFSManager,
    context_path: Path,
    *,
    query: str,
    include_content: bool,
    max_results: int,
    pack_mode: str,
) -> list[ContextPackSection]:
    settings = manager.config.context_index
    mount_types = list(DEFAULT_SEARCH_MOUNTS)
    index = ContextSQLiteIndex(manager, context_path)
    if settings.enabled:
        if not index.has_entries(mount_types=mount_types) and settings.auto_index:
            index.rebuild(
                mount_types=mount_types,
                include_content=settings.include_content,
                max_file_size_bytes=settings.max_file_size_bytes,
                max_content_chars=settings.max_content_chars,
            )
        elif settings.auto_refresh and index.has_entries(mount_types=mount_types) and index.needs_refresh(
            mount_types=mount_types
        ):
            index.rebuild(
                mount_types=mount_types,
                include_content=settings.include_content,
                max_file_size_bytes=settings.max_file_size_bytes,
                max_content_chars=settings.max_content_chars,
            )

    if not settings.enabled or not index.has_entries(mount_types=mount_types):
        return []

    entries = index.query(
        query=query,
        mount_types=mount_types,
        limit=max(1, max_results),
        include_content=include_content,
    )
    sections: list[ContextPackSection] = []
    for offset, entry in enumerate(entries, start=1):
        if _entry_blocked(entry, manager.config.sensitivity.never_export):
            continue
        source = str(entry.get("absolute_path", "")).strip()
        excerpt = ""
        if include_content and isinstance(entry.get("content"), str):
            excerpt = _trim_text(entry["content"], limit=1200)
        elif isinstance(entry.get("content_excerpt"), str):
            excerpt = _trim_text(entry["content_excerpt"], limit=500)
        body_lines = [
            f"Path: {entry['mount_type']}/{entry['relative_path']}",
            f"Modified: {entry.get('modified_at') or 'unknown'}",
        ]
        if excerpt:
            body_lines.extend(["", excerpt])
        sections.append(
            ContextPackSection(
                title=f"Indexed Hit {offset}",
                body="\n".join(body_lines),
                priority=_query_section_priority(pack_mode, offset),
                sources=[source] if source else [],
            )
        )
    return sections


def _embedding_section(
    manager: AFSManager,
    context_path: Path,
    *,
    query: str,
    max_results: int,
    pack_mode: str,
) -> ContextPackSection | None:
    knowledge_root = resolve_mount_root(context_path, MountType.KNOWLEDGE, config=manager.config)
    candidates: list[Path] = []
    root_index = knowledge_root / "embedding_index.json"
    if root_index.exists():
        candidates.append(knowledge_root)
    if knowledge_root.exists():
        for child in sorted(knowledge_root.iterdir()):
            if child.is_dir() and (child / "embedding_index.json").exists():
                candidates.append(child)

    hits: list[tuple[float, str, str]] = []
    for index_root in candidates:
        try:
            results = search_embedding_index(
                index_root,
                query,
                top_k=max(1, max_results),
                min_score=0.15,
            )
        except (FileNotFoundError, ValueError):
            continue
        for result in results:
            source_path = str(result.source_path)
            if _path_blocked(Path(source_path), relative_path="", patterns=manager.config.sensitivity.never_export):
                continue
            hits.append((result.score, source_path, result.text_preview))

    if not hits:
        return None

    hits.sort(key=lambda item: item[0], reverse=True)
    lines: list[str] = []
    sources: list[str] = []
    seen: set[str] = set()
    for score, source_path, preview in hits:
        if source_path in seen:
            continue
        seen.add(source_path)
        lines.append(f"- {source_path} (score={score:.3f})")
        if preview.strip():
            lines.append(f"  {preview.strip()}")
        sources.append(source_path)
        if len(sources) >= max_results:
            break
    return ContextPackSection(
        title="Embedding Hits",
        body="\n".join(lines),
        priority=_embedding_section_priority(pack_mode),
        sources=sources,
    )


def _knowledge_slice_section(
    manager: AFSManager,
    context_path: Path,
    *,
    limit: int,
) -> ContextPackSection | None:
    knowledge_root = resolve_mount_root(context_path, MountType.KNOWLEDGE, config=manager.config)
    if not knowledge_root.exists():
        return None

    index_path = knowledge_root / "INDEX.md"
    if index_path.exists() and not _path_blocked(
        index_path,
        relative_path="INDEX.md",
        patterns=manager.config.sensitivity.never_export,
    ):
        text = index_path.read_text(encoding="utf-8", errors="replace")
        return ContextPackSection(
            title="Knowledge Slice",
            body=_trim_text(text, limit=1600),
            priority=32,
            sources=[str(index_path)],
        )

    lines: list[str] = []
    sources: list[str] = []
    for path in sorted(knowledge_root.rglob("*.md")):
        rel = path.relative_to(knowledge_root).as_posix()
        if _path_blocked(path, relative_path=rel, patterns=manager.config.sensitivity.never_export):
            continue
        try:
            size = path.stat().st_size
        except OSError:
            size = 0
        lines.append(f"- {rel} ({size} bytes)")
        sources.append(str(path))
        if len(sources) >= max(1, limit):
            break
    if not lines:
        return None
    return ContextPackSection(
        title="Knowledge Slice",
        body="\n".join(lines),
        priority=32,
        sources=sources,
    )


def _query_section_priority(pack_mode: str, offset: int) -> int:
    if pack_mode == "retrieval":
        return 8 + offset
    if pack_mode == "full_slice":
        return 33 + offset
    return 35 + offset


def _embedding_section_priority(pack_mode: str) -> int:
    if pack_mode == "retrieval":
        return 14
    if pack_mode == "full_slice":
        return 50
    return 60


def _select_sections(
    sections: list[ContextPackSection],
    *,
    token_budget: int,
) -> tuple[list[ContextPackSection], list[ContextPackSection]]:
    chosen: list[ContextPackSection] = []
    omitted: list[ContextPackSection] = []
    used = 0
    for section in sorted(sections, key=lambda item: (item.priority, item.title)):
        cost = section.estimated_tokens()
        if chosen and used + cost > token_budget:
            omitted.append(section)
            continue
        chosen.append(section)
        used += cost
    return chosen, omitted


def _render_status_block(status: dict[str, Any], diff: dict[str, Any]) -> str:
    mount_health = status.get("mount_health", {})
    index_info = status.get("index", {})
    lines = [
        f"Mount files: {status.get('total_files', 0)}",
        f"Mount health: {'healthy' if mount_health.get('healthy') else 'needs repair'}",
    ]
    if index_info.get("enabled"):
        if index_info.get("built", index_info.get("has_entries", False)):
            freshness = "stale" if index_info.get("stale") else "fresh"
            lines.append(f"Index: {index_info.get('total_entries', 0)} entries ({freshness})")
        else:
            lines.append("Index: not built")
    else:
        lines.append("Index: disabled")
    if diff.get("available"):
        lines.append(f"Recent drift: {diff.get('total_changes', 0)} changes")
    elif diff.get("error"):
        lines.append(f"Recent drift: unavailable ({diff['error']})")
    return "\n".join(lines)


def _render_scratchpad_block(scratchpad: dict[str, Any]) -> str:
    chunks: list[str] = []
    state_text = str(scratchpad.get("state_text", "")).strip()
    deferred_text = str(scratchpad.get("deferred_text", "")).strip()
    if state_text:
        chunks.append("State:\n" + _trim_text(state_text, limit=1200))
    if deferred_text:
        chunks.append("Deferred:\n" + _trim_text(deferred_text, limit=1200))
    other_files = scratchpad.get("other_files") or []
    if other_files:
        chunks.append("Other files:\n" + "\n".join(f"- {name}" for name in other_files[:8]))
    return "\n\n".join(chunks) if chunks else "Scratchpad is empty."


def _render_handoff_block(handoff: dict[str, Any]) -> str:
    lines = [
        f"Session: {handoff.get('session_id', '')}",
        f"Agent: {handoff.get('agent_name', '')}",
        f"Timestamp: {handoff.get('timestamp', '')}",
    ]
    for label in ("accomplished", "blocked", "next_steps"):
        values = handoff.get(label) or []
        if values:
            lines.append(f"{label.replace('_', ' ').title()}:")
            lines.extend(f"- {value}" for value in values[:8])
    return "\n".join(lines)


def _render_tasks_block(tasks: dict[str, Any]) -> str:
    total = int(tasks.get("total", 0) or 0)
    if total <= 0:
        return "No queued tasks."
    counts = tasks.get("counts") or {}
    lines = [f"Total queued tasks: {total}"]
    if counts:
        lines.append(
            "Counts: " + ", ".join(f"{name}={count}" for name, count in sorted(counts.items()))
        )
    for item in (tasks.get("items") or [])[:8]:
        lines.append(
            f"- [{item.get('status', 'pending')}] p{item.get('priority', '?')} {item.get('title', '')}"
        )
    return "\n".join(lines)


def _render_hivemind_block(hivemind: dict[str, Any]) -> str:
    total = int(hivemind.get("recent_count", 0) or 0)
    if total <= 0:
        return "No recent hivemind messages."
    lines = [f"Recent messages: {total}"]
    for message in (hivemind.get("messages") or [])[:8]:
        target = f" -> {message.get('to')}" if message.get("to") else ""
        topic = f" #{message.get('topic')}" if message.get("topic") else ""
        lines.append(
            f"- {message.get('timestamp', '')[:19]} [{message.get('type', '')}] {message.get('from', '')}{target}{topic}"
        )
    return "\n".join(lines)


def _render_execution_profile_block(profile: Any) -> str:
    if not isinstance(profile, dict):
        return ""
    tool_profile = profile.get("tool_profile")
    lines = [
        f"Workflow: {profile.get('workflow', 'general')}",
        f"Summary: {profile.get('summary', '')}",
        f"Intent: {profile.get('intent', '')}",
    ]
    model_hint = str(profile.get("model_hint", "")).strip()
    if model_hint:
        lines.append(f"Model hint: {model_hint}")
    loop_policy = str(profile.get("loop_policy", "")).strip()
    if loop_policy:
        lines.append(f"Loop policy: {loop_policy}")
    retry_hint = str(profile.get("retry_hint", "")).strip()
    if retry_hint:
        lines.append(f"Retry hint: {retry_hint}")
    if isinstance(tool_profile, dict):
        lines.append(
            f"Tool profile: {tool_profile.get('name', 'default')} - {tool_profile.get('summary', '')}"
        )
        preferred = tool_profile.get("preferred_surfaces") or []
        if preferred:
            lines.append("Preferred surfaces:")
            lines.extend(f"- {value}" for value in preferred[:10])
        notes = tool_profile.get("notes") or []
        if notes:
            lines.append("Tool notes:")
            lines.extend(f"- {value}" for value in notes[:6])
    prompt_contract = profile.get("prompt_contract") or []
    if prompt_contract:
        lines.append("Prompt contract:")
        lines.extend(f"- {value}" for value in prompt_contract[:8])
    verification_contract = profile.get("verification_contract") or []
    if verification_contract:
        lines.append("Verification contract:")
        lines.extend(f"- {value}" for value in verification_contract[:8])
    retry_contract = profile.get("retry_contract") or []
    if retry_contract:
        lines.append("Retry contract:")
        lines.extend(f"- {value}" for value in retry_contract[:8])
    return "\n".join(line for line in lines if line.strip())


def _render_focus_block(*, task: str, query: str) -> dict[str, str] | None:
    task_text = task.strip()
    if task_text:
        return {"title": "Task", "body": task_text}
    query_text = query.strip()
    if query_text:
        return {"title": "Focus Query", "body": query_text}
    return None


def _context_pack_prefix_hash(pack: dict[str, Any]) -> str:
    lines = [
        f"project={pack.get('project', '')}",
        f"context={pack.get('context_path', '')}",
        f"profile={pack.get('profile', '')}",
        f"model={pack.get('model', '')}",
        f"pack_mode={pack.get('pack_mode', 'focused')}",
    ]
    execution_profile_text = _render_execution_profile_block(pack.get("execution_profile"))
    if execution_profile_text:
        lines.extend(["## Execution Profile", execution_profile_text])
    guidance = str(pack.get("guidance", "")).strip()
    if guidance:
        lines.extend(["## Guidance", guidance])
    pack_mode_summary = str(pack.get("pack_mode_summary", "")).strip()
    if pack_mode_summary:
        lines.extend(["## Pack Mode", pack_mode_summary])
    for section in pack.get("sections", []):
        if not isinstance(section, dict):
            continue
        title = str(section.get("title", "")).strip()
        if title == "Context Health":
            continue
        body = str(section.get("body", "")).strip()
        if not body:
            continue
        lines.extend(["## " + title, body])
        sources = section.get("sources") or []
        if sources:
            lines.append("Sources:")
            lines.extend(f"- {source}" for source in sources)
    payload = "\n".join(lines).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# Sections that change between sessions or on every scratchpad write.
# Excluded from stable_prefix_hash so Gemini context caching can hit on
# the knowledge-heavy prefix even when volatile state drifts.
_VOLATILE_SECTION_TITLES = frozenset({
    "Context Health",
    "Recommended Actions",
    "Scratchpad State",
    "Latest Handoff",
    "Open Tasks",
    "Recent Hivemind",
})


def _context_pack_stable_prefix_hash(pack: dict[str, Any]) -> str:
    """Hash only the stable corpus prefix (knowledge, execution profile, guidance).

    Volatile sections (scratchpad, tasks, hivemind, handoff, health) are
    excluded so Gemini context-cache adapters can match on the stable prefix
    even when session state drifts between calls.
    """
    lines = [
        f"project={pack.get('project', '')}",
        f"context={pack.get('context_path', '')}",
        f"profile={pack.get('profile', '')}",
        f"model={pack.get('model', '')}",
        f"pack_mode={pack.get('pack_mode', 'focused')}",
    ]
    execution_profile_text = _render_execution_profile_block(pack.get("execution_profile"))
    if execution_profile_text:
        lines.extend(["## Execution Profile", execution_profile_text])
    guidance = str(pack.get("guidance", "")).strip()
    if guidance:
        lines.extend(["## Guidance", guidance])
    pack_mode_summary = str(pack.get("pack_mode_summary", "")).strip()
    if pack_mode_summary:
        lines.extend(["## Pack Mode", pack_mode_summary])
    for section in pack.get("sections", []):
        if not isinstance(section, dict):
            continue
        title = str(section.get("title", "")).strip()
        if title in _VOLATILE_SECTION_TITLES:
            continue
        body = str(section.get("body", "")).strip()
        if not body:
            continue
        lines.extend(["## " + title, body])
        sources = section.get("sources") or []
        if sources:
            lines.append("Sources:")
            lines.extend(f"- {source}" for source in sorted(sources))
    payload = "\n".join(lines).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _render_memory_block(memory: dict[str, Any]) -> str:
    lines = [f"Entries: {memory.get('entries_count', 0)}"]
    latest = str(memory.get("latest_markdown_path", "")).strip()
    if latest:
        lines.append(f"Latest summary: {latest}")
    excerpt = str(memory.get("latest_markdown_excerpt", "")).strip()
    if excerpt:
        lines.extend(["", _trim_text(excerpt, limit=1200)])
    return "\n".join(lines)


def _render_list(items: list[str], *, fallback: str) -> str:
    cleaned = [str(item).strip() for item in items if str(item).strip()]
    if not cleaned:
        return fallback
    return "\n".join(f"- {item}" for item in cleaned[:10])


def _trim_text(text: str, *, limit: int) -> str:
    cleaned = text.strip()
    if len(cleaned) <= limit:
        return cleaned
    return cleaned[: max(0, limit - 3)].rstrip() + "..."


def _entry_blocked(entry: dict[str, Any], patterns: list[str]) -> bool:
    absolute_path = Path(str(entry.get("absolute_path", "")))
    relative_path = f"{entry.get('mount_type', '')}/{entry.get('relative_path', '')}".strip("/")
    return _path_blocked(absolute_path, relative_path=relative_path, patterns=patterns)


def _path_blocked(absolute_path: Path, *, relative_path: str, patterns: list[str]) -> bool:
    if not patterns:
        return False
    return matches_path_rules(
        absolute_path,
        relative_path=relative_path,
        patterns=patterns,
    )
