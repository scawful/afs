"""Tests for context decay / per-file staleness scoring."""

from __future__ import annotations

import os
import time

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
