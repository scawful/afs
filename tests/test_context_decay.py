"""Tests for context decay / per-file staleness scoring."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest

from afs.schema import AFSConfig


def _make_index_context(tmp_path, monkeypatch):
    """Create a minimal context with indexed files."""
    monkeypatch.setenv("AFS_ALLOWED_MOUNTS", "memory,history,scratchpad,hivemind,knowledge,tools,global,items,monorepo")
    context = tmp_path / ".context"
    context.mkdir()
    for mount in ("memory", "knowledge", "scratchpad", "history", "hivemind", "global", "tools", "items"):
        (context / mount).mkdir()
    # Create some knowledge files
    knowledge = context / "knowledge"
    (knowledge / "doc1.md").write_text("# Doc 1", encoding="utf-8")
    (knowledge / "doc2.md").write_text("# Doc 2", encoding="utf-8")
    return context


def test_index_scope_prefix_treats_sql_metacharacters_literally(tmp_path, monkeypatch):
    context = _make_index_context(tmp_path, monkeypatch)
    from afs.context_index import ContextSQLiteIndex
    from afs.manager import AFSManager
    from afs.models import MountType

    scratchpad = context / "scratchpad"
    literal = scratchpad / "projects" / "scope_a" / "note.md"
    wildcard_collision = scratchpad / "projects" / "scopeXa" / "note.md"
    prefix_collision = scratchpad / "projects" / "scope_a2" / "note.md"
    literal.parent.mkdir(parents=True)
    wildcard_collision.parent.mkdir(parents=True)
    prefix_collision.parent.mkdir(parents=True)
    literal.write_text("shared scope token", encoding="utf-8")
    wildcard_collision.write_text("shared scope token", encoding="utf-8")
    prefix_collision.write_text("shared scope token", encoding="utf-8")
    index = ContextSQLiteIndex(AFSManager(config=AFSConfig.from_dict({})), context)
    index.rebuild(mount_types=[MountType.SCRATCHPAD])

    assert index.count_entries(
        mount_types=[MountType.SCRATCHPAD],
        relative_prefixes=["projects/scope_a/"],
    ) == 1
    results = index.query(
        mount_types=[MountType.SCRATCHPAD],
        relative_prefix="projects/scope_a/",
        limit=10,
    )
    paths = {item["relative_path"] for item in results}
    assert "projects/scope_a/note.md" in paths
    assert not any("scopeXa" in path for path in paths)
    assert not any("scope_a2" in path for path in paths)
    fts_paths = {
        item["relative_path"]
        for item in index.query(
            query="shared scope token",
            mount_types=[MountType.SCRATCHPAD],
            relative_prefix="projects/scope_a/",
            limit=10,
        )
    }
    assert "projects/scope_a/note.md" in fts_paths
    assert not any("scope_a2" in path or "scopeXa" in path for path in fts_paths)

    assert index.delete_relative_prefix(
        MountType.SCRATCHPAD,
        "projects/scope_a",
    ) >= 1
    remaining = {
        item["relative_path"]
        for item in index.query(mount_types=[MountType.SCRATCHPAD], limit=20)
    }
    assert "projects/scope_a/note.md" not in remaining
    assert "projects/scope_a2/note.md" in remaining


def test_v2_index_never_reads_or_returns_linked_cross_scope_files(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "AFS_ALLOWED_MOUNTS",
        "memory,history,scratchpad,hivemind,knowledge,tools,global,items,monorepo",
    )
    from afs.context_index import ContextSQLiteIndex
    from afs.context_layout import scaffold_v2
    from afs.manager import AFSManager
    from afs.models import MountType
    from afs.project_registry import ProjectRegistry
    from afs.schema import GeneralConfig

    context = tmp_path / ".context"
    alpha = tmp_path / "alpha"
    beta = tmp_path / "beta"
    alpha.mkdir()
    beta.mkdir()
    scaffold_v2(context)
    registry = ProjectRegistry(context)
    alpha_record = registry.register(alpha)
    beta_record = registry.register(beta)
    manager = AFSManager(
        config=AFSConfig(general=GeneralConfig(context_root=context))
    )
    alpha_root = context / "knowledge" / "projects" / alpha_record.project_id
    beta_root = context / "knowledge" / "projects" / beta_record.project_id
    alpha_root.mkdir(parents=True)
    beta_root.mkdir(parents=True)
    (alpha_root / "safe.md").write_text("shared-index-token alpha-safe", encoding="utf-8")
    stale = alpha_root / "stale.md"
    stale.write_text("shared-index-token stale-private", encoding="utf-8")
    beta_secret = beta_root / "secret.md"
    beta_secret.write_text("shared-index-token beta-private", encoding="utf-8")
    outside_secret = tmp_path / "outside-secret.md"
    outside_secret.write_text("shared-index-token outside-private", encoding="utf-8")

    index = ContextSQLiteIndex(manager, context)
    index.rebuild(mount_types=[MountType.KNOWLEDGE], include_content=True)
    stale.unlink()
    try:
        stale.symlink_to(beta_secret)
        (alpha_root / "outside-link.md").symlink_to(outside_secret)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    def scoped_results() -> list[dict[str, object]]:
        return index.query(
            query="shared-index-token",
            mount_types=[MountType.KNOWLEDGE],
            relative_prefix=f"projects/{alpha_record.project_id}/",
            include_content=True,
            limit=20,
        )

    before_rebuild = scoped_results()
    assert [item["relative_path"] for item in before_rebuild] == [
        f"projects/{alpha_record.project_id}/safe.md"
    ]
    assert "beta-private" not in str(before_rebuild)
    assert "outside-private" not in str(before_rebuild)
    assert "stale-private" not in str(before_rebuild)

    index.rebuild(mount_types=[MountType.KNOWLEDGE], include_content=True)
    after_rebuild = scoped_results()
    assert [item["relative_path"] for item in after_rebuild] == [
        f"projects/{alpha_record.project_id}/safe.md"
    ]
    assert "beta-private" not in str(after_rebuild)
    assert "outside-private" not in str(after_rebuild)
    assert "stale-private" not in str(after_rebuild)


def test_upsert_purges_existing_never_index_rows(tmp_path: Path, monkeypatch) -> None:
    context = _make_index_context(tmp_path, monkeypatch)
    from afs.context_index import ContextSQLiteIndex
    from afs.manager import AFSManager
    from afs.models import MountType
    from afs.schema import GeneralConfig, SensitivityConfig

    private = context / "scratchpad" / "private" / "secret.md"
    private.parent.mkdir()
    private.write_text("upsert-never-index-secret", encoding="utf-8")
    permissive = ContextSQLiteIndex(
        AFSManager(config=AFSConfig(general=GeneralConfig(context_root=context))),
        context,
    )
    permissive.rebuild(mount_types=[MountType.SCRATCHPAD], include_content=True)
    assert permissive.query(query="upsert-never-index-secret")

    guarded = ContextSQLiteIndex(
        AFSManager(
            config=AFSConfig(
                general=GeneralConfig(context_root=context),
                sensitivity=SensitivityConfig(
                    never_index=["scratchpad/private/*"]
                ),
            )
        ),
        context,
    )
    assert not guarded.upsert_relative_path(
        MountType.SCRATCHPAD,
        "private/secret.md",
        include_content=True,
    )
    assert guarded.query(query="upsert-never-index-secret") == []


def test_freshness_scores_fresh_files_near_1(tmp_path, monkeypatch):
    context = _make_index_context(tmp_path, monkeypatch)
    from afs.context_index import ContextSQLiteIndex
    from afs.manager import AFSManager

    manager = AFSManager(config=AFSConfig.from_dict({}))
    index = ContextSQLiteIndex(manager, context)
    index.rebuild()

    result = index.freshness_scores(decay_hours=168.0)
    # Fresh files should have score near 1.0
    for _mount_key, files in result["files"].items():
        for f in files:
            if f["status"] == "indexed":
                assert f["score"] > 0.9, f"Fresh file {f['relative_path']} has low score {f['score']}"


def test_freshness_scores_deleted_files_are_0(tmp_path, monkeypatch):
    context = _make_index_context(tmp_path, monkeypatch)
    from afs.context_index import ContextSQLiteIndex
    from afs.manager import AFSManager

    manager = AFSManager(config=AFSConfig.from_dict({}))
    index = ContextSQLiteIndex(manager, context)
    index.rebuild()

    # Delete a file after indexing
    (context / "knowledge" / "doc1.md").unlink()

    result = index.freshness_scores()
    knowledge_files = result["files"].get("knowledge", [])
    deleted = [f for f in knowledge_files if f["relative_path"] == "doc1.md"]
    assert len(deleted) == 1
    assert deleted[0]["score"] == 0.0
    assert deleted[0]["status"] == "deleted"


def test_freshness_scores_modified_files_are_0(tmp_path, monkeypatch):
    context = _make_index_context(tmp_path, monkeypatch)
    from afs.context_index import ContextSQLiteIndex
    from afs.manager import AFSManager

    manager = AFSManager(config=AFSConfig.from_dict({}))
    index = ContextSQLiteIndex(manager, context)
    index.rebuild()

    # Modify a file after indexing (change mtime to future)
    doc = context / "knowledge" / "doc2.md"
    future_time = time.time() + 3600
    doc.write_text("# Modified", encoding="utf-8")
    os.utime(doc, (future_time, future_time))

    result = index.freshness_scores()
    knowledge_files = result["files"].get("knowledge", [])
    modified = [f for f in knowledge_files if f["relative_path"] == "doc2.md"]
    assert len(modified) == 1
    assert modified[0]["score"] == 0.0
    assert modified[0]["status"] == "modified"


def test_freshness_threshold_filters(tmp_path, monkeypatch):
    context = _make_index_context(tmp_path, monkeypatch)
    from afs.context_index import ContextSQLiteIndex
    from afs.manager import AFSManager

    manager = AFSManager(config=AFSConfig.from_dict({}))
    index = ContextSQLiteIndex(manager, context)
    index.rebuild()

    # Delete a file to create a 0.0 score entry
    (context / "knowledge" / "doc1.md").unlink()

    result = index.freshness_scores(threshold=0.5)
    knowledge_files = result["files"].get("knowledge", [])
    # Deleted file (score 0.0) should be filtered out
    deleted = [f for f in knowledge_files if f["relative_path"] == "doc1.md"]
    assert len(deleted) == 0


def test_freshness_scores_new_unindexed_files_are_0(tmp_path, monkeypatch):
    context = _make_index_context(tmp_path, monkeypatch)
    from afs.context_index import ContextSQLiteIndex
    from afs.manager import AFSManager

    manager = AFSManager(config=AFSConfig.from_dict({}))
    index = ContextSQLiteIndex(manager, context)
    index.rebuild()

    (context / "knowledge" / "doc3.md").write_text("# Doc 3", encoding="utf-8")

    result = index.freshness_scores()
    knowledge_files = result["files"].get("knowledge", [])
    added = [f for f in knowledge_files if f["relative_path"] == "doc3.md"]
    assert len(added) == 1
    assert added[0]["score"] == 0.0
    assert added[0]["status"] == "unindexed"


def test_decay_hours_config_loads():
    config = AFSConfig.from_dict({"context_index": {"decay_hours": 72.0}})
    assert config.context_index.decay_hours == 72.0

    default = AFSConfig.from_dict({})
    assert default.context_index.decay_hours == 168.0


def test_stale_mounts_in_bootstrap(tmp_path, monkeypatch):
    """Verify stale_mounts key appears in bootstrap output."""
    context = _make_index_context(tmp_path, monkeypatch)
    from afs.manager import AFSManager
    from afs.session_bootstrap import build_session_bootstrap

    manager = AFSManager(config=AFSConfig.from_dict({}))
    summary = build_session_bootstrap(manager, context)
    # stale_mounts should be a list (possibly empty for fresh index)
    assert isinstance(summary.get("stale_mounts"), list)


def test_mount_freshness_follows_symlinked_mount_directories(tmp_path, monkeypatch):
    context = _make_index_context(tmp_path, monkeypatch)
    from afs.context_freshness import mount_freshness

    shared_knowledge = tmp_path / "shared-knowledge"
    shared_knowledge.mkdir()
    fresh_doc = shared_knowledge / "fresh.md"
    fresh_doc.write_text("# Fresh", encoding="utf-8")

    mounted = context / "knowledge" / "shared"
    mounted.symlink_to(shared_knowledge, target_is_directory=True)

    stale_doc = context / "knowledge" / "doc1.md"
    old_time = time.time() - (30 * 24 * 3600)
    os.utime(stale_doc, (old_time, old_time))

    freshness = mount_freshness(context)

    assert freshness["knowledge"].file_count == 3
    assert freshness["knowledge"].stale is False
    assert freshness["knowledge"].newest_mtime is not None
    assert freshness["knowledge"].newest_mtime >= fresh_doc.stat().st_mtime


def test_context_diff_since_session_tracks_symlinked_mount_files(tmp_path, monkeypatch):
    context = _make_index_context(tmp_path, monkeypatch)
    from afs.context_freshness import context_diff_since_session, save_context_snapshot

    shared_knowledge = tmp_path / "shared-knowledge"
    shared_knowledge.mkdir()
    fresh_doc = shared_knowledge / "fresh.md"
    fresh_doc.write_text("# Fresh", encoding="utf-8")

    mounted = context / "knowledge" / "shared"
    mounted.symlink_to(shared_knowledge, target_is_directory=True)

    session_id = "symlink-test"
    save_context_snapshot(context, session_id)

    time.sleep(1.1)
    fresh_doc.write_text("# Fresh\nUpdated\n", encoding="utf-8")

    diff = context_diff_since_session(context, session_id)

    assert diff is not None
    modified_paths = {item["relative_path"] for item in diff.modified}
    assert "knowledge/shared/fresh.md" in modified_paths


def test_v2_freshness_snapshot_and_diff_skip_linked_descendants(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "AFS_ALLOWED_MOUNTS",
        "memory,history,scratchpad,hivemind,knowledge,tools,global,items,monorepo",
    )
    from afs.context_freshness import (
        context_diff_since_session,
        mount_freshness,
        save_context_snapshot,
    )
    from afs.context_layout import scaffold_v2
    from afs.project_registry import ProjectRegistry
    from afs.scopes import resolve_scope

    context = tmp_path / ".context"
    alpha = tmp_path / "alpha"
    beta = tmp_path / "beta"
    outside = tmp_path / "outside"
    alpha.mkdir()
    beta.mkdir()
    outside.mkdir()
    scaffold_v2(context)
    registry = ProjectRegistry(context)
    alpha_record = registry.register(alpha)
    beta_record = registry.register(beta)
    alpha_root = context / "knowledge" / "projects" / alpha_record.project_id
    beta_root = context / "knowledge" / "projects" / beta_record.project_id
    common_root = context / "knowledge" / "common"
    alpha_root.mkdir(parents=True)
    beta_root.mkdir(parents=True)
    common_root.mkdir(parents=True, exist_ok=True)
    (alpha_root / "safe.md").write_text("alpha safe", encoding="utf-8")
    (common_root / "shared.md").write_text("common safe", encoding="utf-8")
    beta_secret = beta_root / "secret.md"
    beta_secret.write_text("beta private", encoding="utf-8")
    outside_secret = outside / "secret.md"
    outside_secret.write_text("outside private", encoding="utf-8")
    try:
        (alpha_root / "linked-file.md").symlink_to(beta_secret)
        (alpha_root / "linked-dir").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    scope = resolve_scope(context, requester_path=alpha)
    freshness = mount_freshness(context, scoped=scope)
    assert freshness["knowledge"].file_count == 2

    snapshot_path = save_context_snapshot(context, "linked-v2", scoped=scope)
    snapshot = snapshot_path.read_text(encoding="utf-8")
    assert "safe.md" in snapshot
    assert "shared.md" in snapshot
    assert "linked-file.md" not in snapshot
    assert "linked-dir" not in snapshot

    future = time.time() + 10
    beta_secret.write_text("beta private changed", encoding="utf-8")
    outside_secret.write_text("outside private changed", encoding="utf-8")
    os.utime(beta_secret, (future, future))
    os.utime(outside_secret, (future, future))
    diff = context_diff_since_session(context, "linked-v2", scoped=scope)
    assert diff is not None
    serialized = str(diff.to_dict())
    assert "linked-file.md" not in serialized
    assert "linked-dir" not in serialized


def test_v2_index_diff_and_freshness_derive_scope_prefixes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv(
        "AFS_ALLOWED_MOUNTS",
        "memory,history,scratchpad,hivemind,knowledge,tools,global,items,monorepo",
    )
    from afs.context_index import ContextSQLiteIndex
    from afs.context_layout import scaffold_v2
    from afs.manager import AFSManager
    from afs.models import MountType
    from afs.project_registry import ProjectRegistry
    from afs.schema import GeneralConfig
    from afs.scopes import resolve_scope

    context = tmp_path / ".context"
    alpha = tmp_path / "alpha"
    beta = tmp_path / "beta"
    alpha.mkdir()
    beta.mkdir()
    scaffold_v2(context)
    registry = ProjectRegistry(context)
    alpha_record = registry.register(alpha)
    beta_record = registry.register(beta)
    roots = {
        "alpha": context / "knowledge" / "projects" / alpha_record.project_id,
        "beta": context / "knowledge" / "projects" / beta_record.project_id,
        "common": context / "knowledge" / "common",
    }
    for root in roots.values():
        root.mkdir(parents=True, exist_ok=True)
    files = {
        name: root / f"{name}.md"
        for name, root in roots.items()
    }
    for name, path in files.items():
        path.write_text(f"{name} initial", encoding="utf-8")

    manager = AFSManager(
        config=AFSConfig(general=GeneralConfig(context_root=context))
    )
    index = ContextSQLiteIndex(manager, context)
    index.rebuild(mount_types=[MountType.KNOWLEDGE])
    future = time.time() + 10
    for name, path in files.items():
        path.write_text(f"{name} changed", encoding="utf-8")
        os.utime(path, (future, future))

    scope = resolve_scope(context, requester_path=alpha)
    diff = index.diff(mount_types=[MountType.KNOWLEDGE], scoped=scope)
    diff_paths = {
        item["relative_path"]
        for key in ("added", "modified", "deleted")
        for item in diff[key]
    }
    assert f"projects/{alpha_record.project_id}/alpha.md" in diff_paths
    assert "common/common.md" in diff_paths
    assert f"projects/{beta_record.project_id}/beta.md" not in diff_paths

    beta_only = index.diff(
        mount_types=[MountType.KNOWLEDGE],
        relative_prefixes=[f"projects/{beta_record.project_id}"],
        scoped=scope,
    )
    assert beta_only["total_changes"] == 0

    freshness = index.freshness_scores(
        mount_types=[MountType.KNOWLEDGE],
        scoped=scope,
    )
    freshness_paths = {
        item["relative_path"]
        for item in freshness["files"][MountType.KNOWLEDGE.value]
    }
    assert f"projects/{alpha_record.project_id}/alpha.md" in freshness_paths
    assert "common/common.md" in freshness_paths
    assert f"projects/{beta_record.project_id}/beta.md" not in freshness_paths

    beta_freshness = index.freshness_scores(
        mount_types=[MountType.KNOWLEDGE],
        relative_prefixes=[f"projects/{beta_record.project_id}"],
        scoped=scope,
    )
    assert beta_freshness["files"][MountType.KNOWLEDGE.value] == []


def test_v2_snapshot_discovery_never_stats_link_targets(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from afs.context_freshness import (
        _load_latest_snapshot,
        _prune_snapshots,
        save_context_snapshot,
    )
    from afs.context_layout import scaffold_v2

    context = tmp_path / ".context"
    scaffold_v2(context)
    valid = save_context_snapshot(context, "valid")
    outside = tmp_path / "outside-snapshot.json"
    outside.write_text(
        json.dumps({"session_id": "poison", "entries": []}),
        encoding="utf-8",
    )
    linked = valid.parent / "snapshot_poison.json"
    try:
        linked.symlink_to(outside)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    original_stat = Path.stat

    def guarded_stat(path: Path, *args, **kwargs):
        if path == linked:
            raise AssertionError("linked snapshot must not be stat-followed")
        return original_stat(path, *args, **kwargs)

    monkeypatch.setattr(Path, "stat", guarded_stat)
    loaded = _load_latest_snapshot(context)
    _prune_snapshots(valid.parent, harden=True)

    assert loaded is not None
    assert loaded["session_id"] == "valid"
