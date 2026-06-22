"""Markdown-backed background agent job queue."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .agent_scope import assert_mount_allowed
from .context_paths import resolve_mount_root
from .models import MountType

JOB_STATES = ("queue", "running", "done", "failed", "archived")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _job_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


def _clean_scalar(value: str) -> str:
    return value.replace("\n", " ").strip()


@dataclass
class AgentJob:
    id: str
    title: str
    status: str
    prompt: str
    priority: int = 5
    created_by: str = ""
    assigned_to: str = ""
    scope: str = ""
    expected_output: str = ""
    allow_destructive: bool = False
    dedupe_key: str = ""
    run_id: str = ""
    result: str = ""
    created_at: str = ""
    updated_at: str = ""

    def metadata(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "title": self.title,
            "status": self.status,
            "priority": self.priority,
            "created_by": self.created_by,
            "assigned_to": self.assigned_to,
            "scope": self.scope,
            "expected_output": self.expected_output,
            "allow_destructive": self.allow_destructive,
            "dedupe_key": self.dedupe_key,
            "run_id": self.run_id,
            "result": self.result,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
        }

    def to_dict(self) -> dict[str, Any]:
        data = self.metadata()
        data["prompt"] = self.prompt
        return data


class AgentJobQueue:
    """Queue background agent jobs as markdown prompt files in items/agent_jobs."""

    def __init__(self, context_path: Path) -> None:
        assert_mount_allowed(MountType.ITEMS, operation="access")
        self._root = resolve_mount_root(context_path, MountType.ITEMS) / "agent_jobs"

    def _state_root(self, state: str) -> Path:
        if state not in JOB_STATES:
            raise ValueError(f"Invalid job state: {state}")
        return self._root / state

    def _path(self, state: str, job_id: str) -> Path:
        return self._state_root(state) / f"{job_id}.md"

    def ensure(self) -> None:
        for state in JOB_STATES:
            self._state_root(state).mkdir(parents=True, exist_ok=True)

    def _write(self, job: AgentJob) -> Path:
        self.ensure()
        path = self._path(job.status, job.id)
        lines = ["---"]
        for key, value in job.metadata().items():
            lines.append(f"{key}: {_clean_scalar(str(value))}")
        lines.extend(["---", "", job.prompt.rstrip(), ""])
        path.write_text("\n".join(lines), encoding="utf-8")
        return path

    def _parse(self, path: Path) -> AgentJob | None:
        text = path.read_text(encoding="utf-8")
        metadata: dict[str, str] = {}
        body = text
        if text.startswith("---\n"):
            match = re.match(r"---\n(.*?)\n---\n?(.*)", text, flags=re.DOTALL)
            if match:
                for line in match.group(1).splitlines():
                    if ":" not in line:
                        continue
                    key, value = line.split(":", 1)
                    metadata[key.strip()] = value.strip()
                body = match.group(2).lstrip("\n")
        job_id = metadata.get("id") or path.stem
        status = metadata.get("status") or path.parent.name
        if status not in JOB_STATES:
            status = path.parent.name if path.parent.name in JOB_STATES else "queue"
        try:
            priority = int(metadata.get("priority", "5"))
        except ValueError:
            priority = 5
        return AgentJob(
            id=job_id,
            title=metadata.get("title", job_id),
            status=status,
            prompt=body.rstrip(),
            priority=priority,
            created_by=metadata.get("created_by", ""),
            assigned_to=metadata.get("assigned_to", ""),
            scope=metadata.get("scope", ""),
            expected_output=metadata.get("expected_output", ""),
            allow_destructive=_parse_bool(metadata.get("allow_destructive", "")),
            dedupe_key=metadata.get("dedupe_key", ""),
            run_id=metadata.get("run_id", ""),
            result=metadata.get("result", ""),
            created_at=metadata.get("created_at", ""),
            updated_at=metadata.get("updated_at", ""),
        )

    def create(
        self,
        title: str,
        prompt: str,
        *,
        priority: int = 5,
        created_by: str = "",
        scope: str = "",
        expected_output: str = "",
        allow_destructive: bool = False,
        dedupe_key: str = "",
    ) -> AgentJob:
        now = _now_iso()
        job = AgentJob(
            id=_job_id(),
            title=title,
            status="queue",
            prompt=prompt,
            priority=priority,
            created_by=created_by,
            scope=scope,
            expected_output=expected_output,
            allow_destructive=allow_destructive,
            dedupe_key=dedupe_key,
            created_at=now,
            updated_at=now,
        )
        self._write(job)
        return job

    def get(self, job_id: str) -> AgentJob | None:
        for state in JOB_STATES:
            path = self._path(state, job_id)
            if path.exists():
                return self._parse(path)
        return None

    def list(self, *, status: str | None = None) -> list[AgentJob]:
        self.ensure()
        states = [status] if status else list(JOB_STATES)
        jobs: list[AgentJob] = []
        for state in states:
            if state not in JOB_STATES:
                raise ValueError(f"Invalid job state: {state}")
            for path in sorted(self._state_root(state).glob("*.md")):
                job = self._parse(path)
                if job is not None:
                    jobs.append(job)
        jobs.sort(key=lambda item: (item.status != "queue", item.priority, item.created_at))
        return jobs

    def move(
        self,
        job_id: str,
        status: str,
        *,
        assigned_to: str = "",
        result: str = "",
        run_id: str = "",
    ) -> AgentJob:
        if status not in JOB_STATES:
            raise ValueError(f"Invalid job state: {status}")
        job = self.get(job_id)
        if job is None:
            raise FileNotFoundError(f"Agent job not found: {job_id}")
        old_path = self._path(job.status, job.id)
        job.status = status
        if assigned_to:
            job.assigned_to = assigned_to
        if result:
            job.result = result
        if run_id:
            job.run_id = run_id
        job.updated_at = _now_iso()
        new_path = self._write(job)
        if old_path.exists() and old_path != new_path:
            old_path.unlink()
        return job

    def claim(self, job_id: str, agent_name: str) -> AgentJob:
        job = self.get(job_id)
        if job is None:
            raise FileNotFoundError(f"Agent job not found: {job_id}")
        if job.status != "queue":
            raise ValueError(f"Cannot claim job in state: {job.status}")
        return self.move(job_id, "running", assigned_to=agent_name)


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}
