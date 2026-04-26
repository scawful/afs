"""AFS-native agent run recorder."""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .agent_scope import assert_mount_allowed
from .context_paths import resolve_mount_root
from .models import MountType


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _run_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


@dataclass
class AgentRun:
    id: str
    task: str
    harness: str = ""
    workspace: str = ""
    status: str = "running"
    prompt: str = ""
    files_changed: list[str] = field(default_factory=list)
    commands: list[str] = field(default_factory=list)
    verification: list[dict[str, Any]] = field(default_factory=list)
    handoff_path: str = ""
    summary: str = ""
    events: list[dict[str, Any]] = field(default_factory=list)
    started_at: str = ""
    updated_at: str = ""
    finished_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "task": self.task,
            "harness": self.harness,
            "workspace": self.workspace,
            "status": self.status,
            "prompt": self.prompt,
            "files_changed": self.files_changed,
            "commands": self.commands,
            "verification": self.verification,
            "handoff_path": self.handoff_path,
            "summary": self.summary,
            "events": self.events,
            "started_at": self.started_at,
            "updated_at": self.updated_at,
            "finished_at": self.finished_at,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "AgentRun":
        return cls(
            id=str(data.get("id", "")),
            task=str(data.get("task", "")),
            harness=str(data.get("harness", "")),
            workspace=str(data.get("workspace", "")),
            status=str(data.get("status", "running")),
            prompt=str(data.get("prompt", "")),
            files_changed=[str(item) for item in data.get("files_changed", [])],
            commands=[str(item) for item in data.get("commands", [])],
            verification=[
                item for item in data.get("verification", []) if isinstance(item, dict)
            ],
            handoff_path=str(data.get("handoff_path", "")),
            summary=str(data.get("summary", "")),
            events=[item for item in data.get("events", []) if isinstance(item, dict)],
            started_at=str(data.get("started_at", "")),
            updated_at=str(data.get("updated_at", "")),
            finished_at=str(data.get("finished_at", "")),
        )


class AgentRunStore:
    """Store agent run records under scratchpad/agent_runs."""

    def __init__(self, context_path: Path) -> None:
        assert_mount_allowed(MountType.SCRATCHPAD, operation="access")
        self._root = resolve_mount_root(context_path, MountType.SCRATCHPAD) / "agent_runs"

    def _path(self, run_id: str) -> Path:
        return self._root / f"{run_id}.json"

    def _write(self, run: AgentRun) -> AgentRun:
        self._root.mkdir(parents=True, exist_ok=True)
        self._path(run.id).write_text(json.dumps(run.to_dict(), indent=2), encoding="utf-8")
        return run

    def start(
        self,
        task: str,
        *,
        harness: str = "",
        workspace: str = "",
        prompt: str = "",
    ) -> AgentRun:
        now = _now_iso()
        run = AgentRun(
            id=_run_id(),
            task=task,
            harness=harness,
            workspace=workspace,
            prompt=prompt,
            started_at=now,
            updated_at=now,
            events=[{"at": now, "type": "start", "summary": task}],
        )
        return self._write(run)

    def get(self, run_id: str) -> AgentRun | None:
        path = self._path(run_id)
        if not path.exists():
            return None
        try:
            return AgentRun.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except (OSError, json.JSONDecodeError):
            return None

    def list(self, *, status: str | None = None, limit: int = 20) -> list[AgentRun]:
        if not self._root.exists():
            return []
        runs: list[AgentRun] = []
        for path in sorted(self._root.glob("*.json")):
            try:
                run = AgentRun.from_dict(json.loads(path.read_text(encoding="utf-8")))
            except (OSError, json.JSONDecodeError):
                continue
            if status and run.status != status:
                continue
            runs.append(run)
        runs.sort(key=lambda item: item.updated_at or item.started_at, reverse=True)
        return runs[:limit] if limit > 0 else runs

    def record_event(
        self,
        run_id: str,
        event_type: str,
        *,
        summary: str = "",
        data: dict[str, Any] | None = None,
    ) -> AgentRun:
        run = self.get(run_id)
        if run is None:
            raise FileNotFoundError(f"Agent run not found: {run_id}")
        now = _now_iso()
        run.events.append(
            {
                "at": now,
                "type": event_type,
                "summary": summary,
                "data": data or {},
            }
        )
        run.updated_at = now
        return self._write(run)

    def finish(
        self,
        run_id: str,
        *,
        status: str = "done",
        summary: str = "",
        files_changed: list[str] | None = None,
        commands: list[str] | None = None,
        verification: list[dict[str, Any]] | None = None,
        handoff_path: str = "",
    ) -> AgentRun:
        if status not in {"done", "failed", "abandoned"}:
            raise ValueError("status must be done, failed, or abandoned")
        run = self.get(run_id)
        if run is None:
            raise FileNotFoundError(f"Agent run not found: {run_id}")
        now = _now_iso()
        run.status = status
        run.summary = summary
        if files_changed:
            run.files_changed.extend(files_changed)
        if commands:
            run.commands.extend(commands)
        if verification:
            run.verification.extend(verification)
        if handoff_path:
            run.handoff_path = handoff_path
        run.finished_at = now
        run.updated_at = now
        run.events.append({"at": now, "type": "finish", "summary": summary, "status": status})
        return self._write(run)
