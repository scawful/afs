"""Task queue backed by the items mount directory."""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .agent_scope import assert_mount_allowed
from .atomic_io import atomic_write_text
from .context_layout import LAYOUT_VERSION, detect_layout_version
from .context_paths import resolve_mount_root
from .file_io import _write_lock
from .models import MountType
from .path_safety import assert_no_linklike_components

VALID_STATUSES = ("pending", "claimed", "in_progress", "done", "failed")
_TASK_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_-]{0,127}$")


def _validate_task_id(task_id: str) -> str:
    normalized = str(task_id).strip()
    if not _TASK_ID_PATTERN.fullmatch(normalized):
        raise ValueError("task id must be one safe path segment")
    return normalized


@dataclass
class Task:
    id: str
    title: str
    status: str = "pending"
    assigned_to: str = ""
    created_by: str = ""
    priority: int = 5
    context: dict[str, Any] = field(default_factory=dict)
    created_at: str = ""
    updated_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "status": self.status,
            "assigned_to": self.assigned_to,
            "created_by": self.created_by,
            "priority": self.priority,
            "context": self.context,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Task:
        return cls(
            id=str(data.get("id", "")),
            title=str(data.get("title", "")),
            status=str(data.get("status", "pending")),
            assigned_to=str(data.get("assigned_to", "")),
            created_by=str(data.get("created_by", "")),
            priority=int(data.get("priority", 5)),
            context=data.get("context") or {},
            created_at=str(data.get("created_at", "")),
            updated_at=str(data.get("updated_at", "")),
        )


class TaskQueue:
    """File-based task queue using the items mount directory."""

    def __init__(self, context_path: Path) -> None:
        assert_mount_allowed(MountType.ITEMS, operation="access")
        self._context_path = context_path.expanduser().resolve()
        self._harden_paths = (
            detect_layout_version(self._context_path) == LAYOUT_VERSION
        )
        self._root = resolve_mount_root(self._context_path, MountType.ITEMS)
        self._safe_path(self._root)

    def _safe_path(self, path: Path, *, allow_missing: bool = True) -> Path:
        """Reject link-like v2 queue paths without changing v1 mount behavior."""

        if not self._harden_paths:
            return path
        assert_no_linklike_components(
            self._root,
            boundary=self._context_path,
            allow_missing=allow_missing,
        )
        return assert_no_linklike_components(
            path,
            boundary=self._root,
            allow_missing=allow_missing,
        )

    def _ensure_root(self) -> None:
        self._safe_path(self._root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._safe_path(self._root, allow_missing=False)

    def _write(self, task: Task) -> None:
        task.id = _validate_task_id(task.id)
        self._ensure_root()
        path = self._task_path(task.id)
        self._safe_path(path)
        atomic_write_text(path, json.dumps(task.to_dict(), indent=2))

    def _task_path(self, task_id: str) -> Path:
        safe_id = _validate_task_id(task_id)
        return self._safe_path(self._root / f"task-{safe_id}.json")

    def _now_iso(self) -> str:
        return datetime.now(timezone.utc).isoformat()

    def create(
        self,
        title: str,
        *,
        created_by: str = "",
        priority: int = 5,
        context: dict[str, Any] | None = None,
    ) -> Task:
        task_id = uuid.uuid4().hex[:12]
        now = self._now_iso()
        task = Task(
            id=task_id,
            title=title,
            status="pending",
            created_by=created_by,
            priority=priority,
            context=context or {},
            created_at=now,
            updated_at=now,
        )
        self._write(task)
        return task

    def get(self, task_id: str) -> Task | None:
        path = self._task_path(task_id)
        if not path.exists():
            return None
        self._safe_path(path)
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            task = Task.from_dict(data)
            if _validate_task_id(task.id) != _validate_task_id(task_id):
                return None
            return task
        except (json.JSONDecodeError, OSError, ValueError):
            return None

    def list(self, *, status: str | None = None) -> list[Task]:
        tasks: list[Task] = []
        self._safe_path(self._root)
        if not self._root.exists():
            return tasks
        self._safe_path(self._root)
        for path in sorted(self._root.glob("task-*.json")):
            self._safe_path(path)
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                task = Task.from_dict(data)
                filename_id = path.stem.removeprefix("task-")
                if _validate_task_id(task.id) != _validate_task_id(filename_id):
                    continue
            except (json.JSONDecodeError, OSError, ValueError):
                continue
            if status and task.status != status:
                continue
            tasks.append(task)
        tasks.sort(key=lambda t: (t.priority, t.created_at))
        return tasks

    @contextmanager
    def transaction(self, task_id: str) -> Iterator[None]:
        """Serialize task/opt-in agreement writers through the existing AFS lock."""
        self._ensure_root()
        with _write_lock(self._task_path(task_id)):
            yield

    def _assert_legacy_mutation_allowed(self, task: Task) -> None:
        from .commitments import assert_legacy_task_mutation_allowed

        assert_legacy_task_mutation_allowed(
            self._context_path, task.id, scope_id=task.context.get("scope_id", "common")
        )

    def claim(self, task_id: str, agent_name: str) -> Task:
        with self.transaction(task_id):
            return self._claim(task_id, agent_name)

    def _claim(self, task_id: str, agent_name: str) -> Task:
        task = self.get(task_id)
        if task is None:
            raise FileNotFoundError(f"Task not found: {task_id}")
        self._assert_legacy_mutation_allowed(task)
        if task.status not in ("pending",):
            raise ValueError(f"Cannot claim task in status: {task.status}")
        task.status = "claimed"
        task.assigned_to = agent_name
        task.updated_at = self._now_iso()
        self._write(task)
        return task

    def update_status(
        self,
        task_id: str,
        status: str,
        *,
        result: dict[str, Any] | None = None,
    ) -> Task:
        with self.transaction(task_id):
            return self._update_status(task_id, status, result=result)

    def _update_status(
        self, task_id: str, status: str, *, result: dict[str, Any] | None = None
    ) -> Task:
        task = self.get(task_id)
        if task is None:
            raise FileNotFoundError(f"Task not found: {task_id}")
        self._assert_legacy_mutation_allowed(task)
        if status not in VALID_STATUSES:
            raise ValueError(f"Invalid status: {status}")
        task.status = status
        task.updated_at = self._now_iso()
        if result:
            task.context["result"] = result
        self._write(task)
        return task

    def complete(self, task_id: str, result: dict[str, Any] | None = None) -> Task:
        return self.update_status(task_id, "done", result=result)
