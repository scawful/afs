"""Tests for task queue system."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from afs.context_layout import scaffold_v2
from afs.tasks import TaskQueue


def _symlink_or_skip(link: Path, target: Path, *, directory: bool = False) -> None:
    try:
        link.symlink_to(target, target_is_directory=directory)
    except OSError as exc:  # pragma: no cover - Windows without symlink privilege
        pytest.skip(f"symlinks unavailable: {exc}")


def test_task_create_and_list(tmp_path: Path) -> None:
    ctx = tmp_path / ".context"
    ctx.mkdir()
    (ctx / "items").mkdir()
    queue = TaskQueue(ctx)
    task = queue.create("Fix lint errors", created_by="agent-a", priority=2)
    assert task.title == "Fix lint errors"
    assert task.status == "pending"
    assert task.created_by == "agent-a"
    assert task.priority == 2

    tasks = queue.list()
    assert len(tasks) == 1
    assert tasks[0].id == task.id


def test_task_claim_and_complete(tmp_path: Path) -> None:
    ctx = tmp_path / ".context"
    ctx.mkdir()
    (ctx / "items").mkdir()
    queue = TaskQueue(ctx)
    task = queue.create("Build feature")
    claimed = queue.claim(task.id, "worker-1")
    assert claimed.status == "claimed"
    assert claimed.assigned_to == "worker-1"

    done = queue.complete(task.id, result={"output": "success"})
    assert done.status == "done"
    assert done.context["result"]["output"] == "success"


def test_task_claim_already_claimed(tmp_path: Path) -> None:
    ctx = tmp_path / ".context"
    ctx.mkdir()
    (ctx / "items").mkdir()
    queue = TaskQueue(ctx)
    task = queue.create("Unique task")
    queue.claim(task.id, "worker-1")

    with pytest.raises(ValueError, match="Cannot claim"):
        queue.claim(task.id, "worker-2")


def test_task_list_filter_by_status(tmp_path: Path) -> None:
    ctx = tmp_path / ".context"
    ctx.mkdir()
    (ctx / "items").mkdir()
    queue = TaskQueue(ctx)
    queue.create("Task A")
    task_b = queue.create("Task B")
    queue.claim(task_b.id, "worker")

    pending = queue.list(status="pending")
    assert len(pending) == 1
    assert pending[0].title == "Task A"


def test_task_get_missing(tmp_path: Path) -> None:
    ctx = tmp_path / ".context"
    ctx.mkdir()
    (ctx / "items").mkdir()
    queue = TaskQueue(ctx)
    assert queue.get("nonexistent") is None


def test_task_empty_list(tmp_path: Path) -> None:
    ctx = tmp_path / ".context"
    ctx.mkdir()
    (ctx / "items").mkdir()
    queue = TaskQueue(ctx)
    assert queue.list() == []


def test_task_queue_respects_allowed_mounts(tmp_path: Path, monkeypatch) -> None:
    ctx = tmp_path / ".context"
    ctx.mkdir()
    (ctx / "items").mkdir()
    monkeypatch.setenv("AFS_AGENT_NAME", "task-agent")
    monkeypatch.setenv("AFS_ALLOWED_MOUNTS", "scratchpad")

    with pytest.raises(PermissionError, match="not allowed to access items"):
        TaskQueue(ctx)


def test_task_queue_uses_remapped_items_mount(tmp_path: Path) -> None:
    ctx = tmp_path / ".context"
    queue_root = ctx / "queue"
    queue_root.mkdir(parents=True)
    (ctx / "metadata.json").write_text(
        json.dumps({"directories": {"items": "queue"}}),
        encoding="utf-8",
    )

    queue = TaskQueue(ctx)
    task = queue.create("Use remapped queue")

    assert (queue_root / f"task-{task.id}.json").exists()


def test_v1_task_queue_preserves_external_remapped_mount(tmp_path: Path) -> None:
    ctx = tmp_path / ".context"
    ctx.mkdir()
    external = tmp_path / "external-items"
    external.mkdir()
    (ctx / "metadata.json").write_text(
        json.dumps({"directories": {"items": str(external)}}),
        encoding="utf-8",
    )

    task = TaskQueue(ctx).create("Use external v1 queue")

    assert (external / f"task-{task.id}.json").exists()


def test_v2_task_queue_rejects_replaced_root_for_read_and_write(
    tmp_path: Path,
) -> None:
    ctx = tmp_path / ".context"
    scaffold_v2(ctx)
    queue = TaskQueue(ctx)
    items_root = ctx / ".afs" / "compat" / "items"
    items_root.rmdir()
    outside = tmp_path / "outside-items"
    outside.mkdir()
    _symlink_or_skip(items_root, outside, directory=True)

    with pytest.raises(ValueError, match="symbolic link or reparse point"):
        queue.list()
    with pytest.raises(ValueError, match="symbolic link or reparse point"):
        queue.create("Do not publish outside")

    assert list(outside.iterdir()) == []


def test_v2_task_queue_rejects_linked_record_for_read_and_update(
    tmp_path: Path,
) -> None:
    ctx = tmp_path / ".context"
    scaffold_v2(ctx)
    queue = TaskQueue(ctx)
    outside = tmp_path / "outside-task.json"
    outside.write_text(
        json.dumps(
            {
                "id": "outside",
                "title": "secret task",
                "status": "pending",
                "context": {"secret": True},
            }
        ),
        encoding="utf-8",
    )
    linked = ctx / ".afs" / "compat" / "items" / "task-linked.json"
    _symlink_or_skip(linked, outside)

    with pytest.raises(ValueError, match="symbolic link or reparse point"):
        queue.get("linked")
    with pytest.raises(ValueError, match="symbolic link or reparse point"):
        queue.list()
    with pytest.raises(ValueError, match="symbolic link or reparse point"):
        queue.claim("linked", "agent-a")

    payload = json.loads(outside.read_text(encoding="utf-8"))
    assert payload["status"] == "pending"
    assert payload["context"] == {"secret": True}


def test_v2_task_queue_rejects_linked_final_leaf_on_write(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ctx = tmp_path / ".context"
    scaffold_v2(ctx)
    queue = TaskQueue(ctx)
    task = queue.create("Keep the final write local")
    task_path = queue._task_path(task.id)  # type: ignore[attr-defined]
    task_path.unlink()
    outside = tmp_path / "outside-task.json"
    outside.write_text('{"status": "outside"}\n', encoding="utf-8")
    _symlink_or_skip(task_path, outside)
    # Force claim through to its write phase so this does not pass only because
    # the normal read path rejected the linked record first.
    monkeypatch.setattr(queue, "get", lambda _task_id: task)

    with pytest.raises(ValueError, match="symbolic link or reparse point"):
        queue.claim(task.id, "agent-a")

    assert json.loads(outside.read_text(encoding="utf-8")) == {"status": "outside"}


def test_v2_task_queue_rejects_traversal_before_cross_category_read(
    tmp_path: Path,
) -> None:
    ctx = tmp_path / ".context"
    scaffold_v2(ctx)
    queue = TaskQueue(ctx)
    canary = ctx / "memory" / "common" / "canary.json"
    canary.parent.mkdir(parents=True)
    canary.write_text('{"secret": true}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="one safe path segment"):
        queue.get("x/../../../../memory/common/canary")

    assert canary.read_text(encoding="utf-8") == '{"secret": true}\n'


def test_v2_task_queue_rejects_mismatched_persisted_identity(
    tmp_path: Path,
) -> None:
    ctx = tmp_path / ".context"
    scaffold_v2(ctx)
    queue = TaskQueue(ctx)
    task = queue.create("Original")
    task_path = queue._task_path(task.id)  # type: ignore[attr-defined]
    payload = json.loads(task_path.read_text(encoding="utf-8"))
    payload["id"] = "different-id"
    task_path.write_text(json.dumps(payload), encoding="utf-8")

    assert queue.get(task.id) is None
    assert queue.list() == []
    with pytest.raises(FileNotFoundError, match="Task not found"):
        queue.claim(task.id, "agent-a")
