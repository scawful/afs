"""Tests for context pack caching — miss, hit, staleness, invalidation, clear."""

from __future__ import annotations

import json
from pathlib import Path

from afs.context_index import ContextSQLiteIndex
from afs.context_pack import (
    CONTEXT_PACK_CACHE_VERSION,
    _context_pack_artifact_paths,
    _context_pack_cache_key,
    _load_cached_context_pack,
    build_context_pack,
    write_context_pack_artifacts,
)
from afs.manager import AFSManager
from afs.models import MountType
from afs.schema import AFSConfig, GeneralConfig, SensitivityConfig, SessionPackCacheConfig

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_manager(tmp_path: Path) -> AFSManager:
    context_root = tmp_path / ".context"
    config = AFSConfig(
        general=GeneralConfig(context_root=context_root),
        sensitivity=SensitivityConfig(never_export=[]),
        session_pack_cache=SessionPackCacheConfig(cache_dir=tmp_path / "cache"),
    )
    manager = AFSManager(config=config)
    project_path = tmp_path / "project"
    project_path.mkdir()
    manager.ensure(path=project_path, context_root=context_root)
    return manager


def _seed_knowledge(manager: AFSManager, context_root: Path, content: str = "hello") -> None:
    """Create a minimal knowledge file so packs have something to index."""
    knowledge_root = manager.resolve_mount_root(context_root, MountType.KNOWLEDGE)
    knowledge_root.mkdir(parents=True, exist_ok=True)
    (knowledge_root / "doc.md").write_text(content, encoding="utf-8")


# ---------------------------------------------------------------------------
# Cache miss — writes a cache file
# ---------------------------------------------------------------------------


def test_cache_miss_writes_artifact_files(tmp_path: Path) -> None:
    """On first build + write, JSON and markdown artifacts are created."""
    manager = _make_manager(tmp_path)
    context_root = manager.config.general.context_root
    _seed_knowledge(manager, context_root)

    pack = build_context_pack(
        manager,
        context_root,
        query="hello",
        task="Test cache miss.",
        model="codex",
        token_budget=400,
    )
    assert pack["cache"]["hit"] is False

    artifact_paths = write_context_pack_artifacts(manager, context_root, pack)
    assert Path(artifact_paths["json"]).exists()
    assert Path(artifact_paths["markdown"]).exists()

    # The written JSON should contain a cache key
    written = json.loads(Path(artifact_paths["json"]).read_text(encoding="utf-8"))
    assert "key" in written["cache"]
    assert written["cache"]["version"] == CONTEXT_PACK_CACHE_VERSION


# ---------------------------------------------------------------------------
# Cache hit — returns without rebuilding
# ---------------------------------------------------------------------------


def test_cache_hit_returns_without_rebuild(tmp_path: Path, monkeypatch) -> None:
    """Second build with identical inputs should be a cache hit.

    Uses scratchpad (not knowledge) because write_context_pack_artifacts
    creates files under scratchpad/afs_agents — _cache_bootstrap strips
    those from the mount counts so the cache key stays stable.
    """
    manager = _make_manager(tmp_path)
    context_root = manager.config.general.context_root
    scratchpad_root = manager.resolve_mount_root(context_root, MountType.SCRATCHPAD)
    scratchpad_root.mkdir(parents=True, exist_ok=True)
    (scratchpad_root / "state.md").write_text("cached pack state", encoding="utf-8")

    kwargs = {
        "query": "cached pack",
        "task": "Keep the cached pack stable.",
        "model": "codex",
        "workflow": "general",
        "token_budget": 400,
    }

    first = build_context_pack(manager, context_root, **kwargs)
    assert first["cache"]["hit"] is False
    write_context_pack_artifacts(manager, context_root, first)

    # Monkey-patch _build_sections to explode if called — a cache hit must skip it
    monkeypatch.setattr(
        "afs.context_pack._build_sections",
        lambda *a, **kw: (_ for _ in ()).throw(AssertionError("should not rebuild on cache hit")),
    )

    second = build_context_pack(manager, context_root, **kwargs)
    assert second["cache"]["hit"] is True
    assert second["cache"]["key"] == first["cache"]["key"]


# ---------------------------------------------------------------------------
# Stale cache — triggers rebuild when inputs change
# ---------------------------------------------------------------------------


def test_stale_cache_triggers_rebuild_on_query_change(tmp_path: Path) -> None:
    """Changing the query should invalidate the cache key and force a rebuild."""
    manager = _make_manager(tmp_path)
    context_root = manager.config.general.context_root
    _seed_knowledge(manager, context_root)

    first = build_context_pack(
        manager, context_root,
        query="original query",
        task="same task",
        model="codex",
        token_budget=400,
    )
    write_context_pack_artifacts(manager, context_root, first)

    second = build_context_pack(
        manager, context_root,
        query="different query",
        task="same task",
        model="codex",
        token_budget=400,
    )
    assert second["cache"]["hit"] is False
    assert second["cache"]["key"] != first["cache"]["key"]


# ---------------------------------------------------------------------------
# Cache invalidation when sentinel file is newer
# ---------------------------------------------------------------------------


def test_cache_invalidation_when_bootstrap_content_changes(tmp_path: Path) -> None:
    """When a scratchpad/knowledge file changes, the bootstrap changes,
    which changes the cache key, causing a rebuild."""
    manager = _make_manager(tmp_path)
    context_root = manager.config.general.context_root
    scratchpad_root = manager.resolve_mount_root(context_root, MountType.SCRATCHPAD)
    scratchpad_root.mkdir(parents=True, exist_ok=True)
    state_path = scratchpad_root / "state.md"
    state_path.write_text("version A", encoding="utf-8")

    first = build_context_pack(
        manager, context_root,
        query="state check",
        task="Check state.",
        model="codex",
        token_budget=400,
    )
    write_context_pack_artifacts(manager, context_root, first)

    # Change the sentinel file
    state_path.write_text("version B", encoding="utf-8")

    second = build_context_pack(
        manager, context_root,
        query="state check",
        task="Check state.",
        model="codex",
        token_budget=400,
    )
    assert second["cache"]["hit"] is False


def test_session_cache_invalidates_when_sensitivity_changes(tmp_path: Path) -> None:
    context_root = tmp_path / ".context"
    cache_dir = tmp_path / "cache"
    base_general = GeneralConfig(context_root=context_root)
    cache_config = SessionPackCacheConfig(cache_dir=cache_dir)
    manager = AFSManager(
        config=AFSConfig(
            general=base_general,
            sensitivity=SensitivityConfig(never_export=[]),
            session_pack_cache=cache_config,
        )
    )
    project_path = tmp_path / "project"
    project_path.mkdir()
    manager.ensure(path=project_path, context_root=context_root)
    knowledge_root = manager.resolve_mount_root(context_root, MountType.KNOWLEDGE)
    (knowledge_root / "private").mkdir(parents=True, exist_ok=True)
    (knowledge_root / "private" / "secret.md").write_text(
        "private cache leak marker",
        encoding="utf-8",
    )
    ContextSQLiteIndex(manager, context_root).rebuild(
        mount_types=[MountType.KNOWLEDGE],
        include_content=True,
    )

    kwargs = {
        "query": "private cache",
        "task": "Check sensitivity cache invalidation.",
        "model": "codex",
        "token_budget": 1000,
        "include_content": True,
    }
    first = build_context_pack(manager, context_root, **kwargs)
    assert first["cache"]["hit"] is False
    assert "private cache leak marker" in json.dumps(first)

    restricted_manager = AFSManager(
        config=AFSConfig(
            general=base_general,
            sensitivity=SensitivityConfig(never_index=["knowledge/private/*"]),
            session_pack_cache=cache_config,
        )
    )
    second = build_context_pack(restricted_manager, context_root, **kwargs)
    assert second["cache"]["hit"] is False
    assert "private cache leak marker" not in json.dumps(second)


# ---------------------------------------------------------------------------
# clear_pack_cache — removing artifact files
# ---------------------------------------------------------------------------


def test_clear_pack_cache_removes_files(tmp_path: Path) -> None:
    """Deleting the artifact JSON file should cause subsequent loads to miss."""
    manager = _make_manager(tmp_path)
    context_root = manager.config.general.context_root
    _seed_knowledge(manager, context_root)

    pack = build_context_pack(
        manager, context_root,
        query="clear test",
        task="Verify cache clear.",
        model="codex",
        token_budget=400,
    )
    artifact_paths = write_context_pack_artifacts(manager, context_root, pack)
    json_path = Path(artifact_paths["json"])
    md_path = Path(artifact_paths["markdown"])
    assert json_path.exists()
    assert md_path.exists()

    # Simulate clearing cache by removing the files
    json_path.unlink()
    md_path.unlink()

    # _load_cached_context_pack should return None now
    result = _load_cached_context_pack(
        manager,
        context_root,
        model="codex",
        cache_key=pack["cache"]["key"],
    )
    assert result is None


# ---------------------------------------------------------------------------
# _load_cached_context_pack — edge cases
# ---------------------------------------------------------------------------


def test_load_cache_returns_none_on_version_mismatch(tmp_path: Path) -> None:
    """If the cached version differs, the cache should be treated as a miss."""
    manager = _make_manager(tmp_path)
    context_root = manager.config.general.context_root

    json_path, _ = _context_pack_artifact_paths(manager, context_root, "codex")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps({
            "cache": {
                "version": CONTEXT_PACK_CACHE_VERSION + 999,
                "key": "fake-key",
            }
        }),
        encoding="utf-8",
    )

    result = _load_cached_context_pack(
        manager, context_root, model="codex", cache_key="fake-key",
    )
    assert result is None


def test_load_cache_returns_none_on_key_mismatch(tmp_path: Path) -> None:
    """If the cached key doesn't match, the cache should miss."""
    manager = _make_manager(tmp_path)
    context_root = manager.config.general.context_root

    json_path, _ = _context_pack_artifact_paths(manager, context_root, "codex")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps({
            "cache": {
                "version": CONTEXT_PACK_CACHE_VERSION,
                "key": "old-key",
            }
        }),
        encoding="utf-8",
    )

    result = _load_cached_context_pack(
        manager, context_root, model="codex", cache_key="new-key",
    )
    assert result is None


def test_load_cache_returns_none_on_corrupt_json(tmp_path: Path) -> None:
    """Corrupt JSON should be treated as a cache miss, not an exception."""
    manager = _make_manager(tmp_path)
    context_root = manager.config.general.context_root

    json_path, _ = _context_pack_artifact_paths(manager, context_root, "codex")
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text("{{not valid json", encoding="utf-8")

    result = _load_cached_context_pack(
        manager, context_root, model="codex", cache_key="any-key",
    )
    assert result is None


# ---------------------------------------------------------------------------
# _context_pack_cache_key — determinism
# ---------------------------------------------------------------------------


def test_cache_key_is_deterministic(tmp_path: Path) -> None:
    """Same inputs should produce the same cache key."""
    kwargs = {
        "bootstrap": {"project": "test", "status": {}},
        "query": "q",
        "task": "t",
        "model": "codex",
        "pack_mode": "focused",
        "workflow": "general",
        "tool_profile": "default",
        "token_budget": 400,
        "include_content": False,
        "max_query_results": 6,
        "max_embedding_results": 4,
    }
    key_a = _context_pack_cache_key(tmp_path, **kwargs)
    key_b = _context_pack_cache_key(tmp_path, **kwargs)
    assert key_a == key_b


def test_cache_key_changes_with_different_task(tmp_path: Path) -> None:
    """Changing just the task should produce a different cache key."""
    base = {
        "bootstrap": {"project": "test"},
        "query": "q",
        "model": "codex",
        "pack_mode": "focused",
        "workflow": "general",
        "tool_profile": "default",
        "token_budget": 400,
        "include_content": False,
        "max_query_results": 6,
        "max_embedding_results": 4,
    }
    key_a = _context_pack_cache_key(tmp_path, task="task A", **base)
    key_b = _context_pack_cache_key(tmp_path, task="task B", **base)
    assert key_a != key_b
