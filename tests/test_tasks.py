"""Tests for task queue system."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from afs.tasks import TaskQueue


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
