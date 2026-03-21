"""Tests for memory consolidation pipeline activation."""

from __future__ import annotations

import json
import time

import pytest

from afs.memory_consolidation import memory_status, search_memory


@pytest.fixture
def memory_context(tmp_path, monkeypatch):
    monkeypatch.setenv("AFS_ALLOWED_MOUNTS", "memory,history,scratchpad,hivemind,knowledge,tools,global,items,monorepo")
    context = tmp_path / ".context"
    context.mkdir()
    memory = context / "memory"
    memory.mkdir()
    history = context / "history"
    history.mkdir()
    scratchpad = context / "scratchpad"
    scratchpad.mkdir()
    (scratchpad / "afs_agents").mkdir()
    return context


def _write_entries(context, entries):
    entries_path = context / "memory" / "entries.jsonl"
    with entries_path.open("w", encoding="utf-8") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\n")


def _write_checkpoint(context, timestamp="2024-01-01T00:00:00"):
    checkpoint_dir = context / "scratchpad" / "afs_agents"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = checkpoint_dir / "history_memory_checkpoint.json"
    checkpoint_path.write_text(json.dumps({
        "timestamp": timestamp,
        "event_id": "test-001",
    }), encoding="utf-8")


def test_memory_status_returns_counts_and_cursor(memory_context):
    _write_entries(memory_context, [
        {"id": "e1", "instruction": "test", "output": "out", "tags": ["a"]},
        {"id": "e2", "instruction": "test2", "output": "out2", "tags": ["b"]},
    ])
    _write_checkpoint(memory_context, "2024-06-01T00:00:00")

    status = memory_status(memory_context)
    assert status["entries_count"] == 2
    assert status["cursor_timestamp"] == "2024-06-01T00:00:00"
    assert status["cursor_age_seconds"] is not None


def test_memory_status_empty_context(memory_context):
    status = memory_status(memory_context)
    assert status["entries_count"] == 0
    assert status["cursor_timestamp"] is None
    assert status["stale"] is False


def test_search_memory_finds_matching_entries(memory_context):
    _write_entries(memory_context, [
        {"id": "e1", "instruction": "context repair", "output": "fixed mounts", "tags": ["context"]},
        {"id": "e2", "instruction": "memory update", "output": "added entries", "tags": ["memory"]},
        {"id": "e3", "instruction": "context query", "output": "searched index", "tags": ["context"]},
    ])
    results = search_memory(memory_context, "context")
    assert len(results) == 2
    ids = [r["id"] for r in results]
    assert "e1" in ids
    assert "e3" in ids


def test_search_memory_empty_returns_empty(memory_context):
    results = search_memory(memory_context, "nonexistent")
    assert results == []


def test_auto_consolidation_when_stale(memory_context):
    """Verify memory_status detects staleness correctly."""
    _write_checkpoint(memory_context, "2024-01-01T00:00:00")
    # Checkpoint file is very old relative to "now" so stale should be True
    status = memory_status(memory_context)
    # cursor_age_seconds should be very large since checkpoint was just written
    # but the file mtime is "now", so stale depends on mtime
    # Since we just wrote it, cursor_age_seconds will be small (< 3600)
    # Let's make it old by backdating the file
    import os
    checkpoint_dir = memory_context / "scratchpad" / "afs_agents"
    checkpoint_path = checkpoint_dir / "history_memory_checkpoint.json"
    old_time = time.time() - 7200  # 2 hours ago
    os.utime(checkpoint_path, (old_time, old_time))
    status = memory_status(memory_context)
    assert status["stale"] is True


def test_auto_consolidation_skipped_when_fresh(memory_context):
    """Verify memory_status detects freshness correctly."""
    _write_checkpoint(memory_context, "2024-06-01T00:00:00")
    # File was just written so mtime is fresh
    status = memory_status(memory_context)
    assert status["stale"] is False
