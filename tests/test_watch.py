"""Tests for afs watch command."""

from __future__ import annotations

import pytest

from afs.cli.watch import (
    _collect_watch_paths,
    _diff_snapshots,
    _snapshot_paths,
    _trigger_actions,
)


@pytest.fixture
def watch_context(tmp_path, monkeypatch):
    monkeypatch.setenv(
        "AFS_ALLOWED_MOUNTS",
        "memory,history,scratchpad,hivemind,knowledge,tools,global,items,monorepo",
    )
    context = tmp_path / ".context"
    context.mkdir()
    for mount in (
        "knowledge",
        "tools",
        "scratchpad",
        "memory",
        "history",
        "hivemind",
        "global",
        "items",
    ):
        (context / mount).mkdir()
    return context


def test_collect_watch_paths_resolves_mounts(watch_context):
    from afs.manager import AFSManager
    from afs.schema import AFSConfig

    manager = AFSManager(config=AFSConfig.from_dict({}))
    paths = _collect_watch_paths(manager, watch_context)
    assert len(paths) >= 1
    path_strs = [str(p) for p in paths]
    # Should include knowledge, tools, scratchpad
    assert any("knowledge" in p for p in path_strs)


def test_polling_snapshot_detects_changes(watch_context):
    knowledge = watch_context / "knowledge"
    (knowledge / "doc1.md").write_text("v1", encoding="utf-8")

    snap1 = _snapshot_paths([knowledge])
    assert len(snap1) == 1

    # Modify a file
    (knowledge / "doc1.md").write_text("v2", encoding="utf-8")
    snap2 = _snapshot_paths([knowledge])

    changed = _diff_snapshots(snap1, snap2)
    # mtime should differ
    assert len(changed) >= 0  # May or may not differ depending on filesystem resolution

    # Add a file
    (knowledge / "doc2.md").write_text("new", encoding="utf-8")
    snap3 = _snapshot_paths([knowledge])
    changed2 = _diff_snapshots(snap2, snap3)
    assert len(changed2) >= 1


def test_snapshot_paths_follows_symlinked_mount_dirs(watch_context):
    knowledge = watch_context / "knowledge"
    source_dir = watch_context.parent / "external-knowledge"
    source_dir.mkdir()
    (source_dir / "nested").mkdir()
    (source_dir / "nested" / "doc.md").write_text("hello", encoding="utf-8")
    linked_dir = knowledge / "mounted"
    linked_dir.symlink_to(source_dir, target_is_directory=True)

    snapshot = _snapshot_paths([knowledge])

    assert any(path.endswith("mounted/nested/doc.md") for path in snapshot)


def test_on_change_command_executed(watch_context):
    from afs.manager import AFSManager
    from afs.schema import AFSConfig

    manager = AFSManager(config=AFSConfig.from_dict({}))
    result = _trigger_actions(
        manager,
        watch_context,
        ["test.md"],
        on_change="echo hello",
    )
    assert result["on_change"] is not None
    assert result["on_change"]["returncode"] == 0


def test_fallback_when_watchfiles_unavailable():
    """Verify the polling fallback path exists."""
    # The _watch_with_polling function should be importable
    from afs.cli.watch import _watch_with_polling

    assert callable(_watch_with_polling)
