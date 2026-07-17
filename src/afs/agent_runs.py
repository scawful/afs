"""AFS-native agent run recorder."""

from __future__ import annotations

import json
import os
import re
import stat
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .agent_scope import assert_mount_allowed
from .context_layout import LAYOUT_VERSION, detect_layout_version
from .context_paths import resolve_mount_root
from .models import MountType
from .path_safety import assert_no_linklike_components

_RUN_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


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
    def from_dict(cls, data: dict[str, Any]) -> AgentRun:
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
    """Store agent run records in the shared scratchpad control-plane scope."""

    def __init__(self, context_path: Path) -> None:
        assert_mount_allowed(MountType.SCRATCHPAD, operation="access")
        self._context_path = context_path.expanduser().resolve()
        scratchpad_root = resolve_mount_root(self._context_path, MountType.SCRATCHPAD)
        is_v2 = detect_layout_version(self._context_path) == LAYOUT_VERSION
        self._root_boundary = self._context_path if is_v2 else None
        if is_v2:
            self._root = scratchpad_root / "common" / "agent_runs"
            self._read_roots = (self._root, scratchpad_root / "agent_runs")
        else:
            self._root = scratchpad_root / "agent_runs"
            self._read_roots = (self._root,)
        self._root = assert_no_linklike_components(
            self._root,
            boundary=self._root_boundary,
        )

    def _path(self, run_id: str) -> Path:
        safe_id = self._validate_run_id(run_id)
        return assert_no_linklike_components(
            self._root / f"{safe_id}.json",
            boundary=self._root,
        )

    @staticmethod
    def _validate_run_id(run_id: str) -> str:
        if not isinstance(run_id, str):
            raise ValueError("agent run id must be a string")
        normalized = run_id.strip()
        if not _RUN_ID_PATTERN.fullmatch(normalized) or normalized in {".", ".."}:
            raise ValueError("agent run id must be a safe identifier")
        return normalized

    def _existing_root(self, root: Path) -> Path | None:
        try:
            safe_root = assert_no_linklike_components(
                root,
                boundary=self._root_boundary,
                allow_missing=False,
            )
        except FileNotFoundError:
            return None
        root_stat = os.lstat(safe_root)
        if not stat.S_ISDIR(root_stat.st_mode):
            raise ValueError(f"agent run root is not a safe directory: {safe_root}")
        return safe_root

    def _ensure_root(self) -> Path:
        root = assert_no_linklike_components(
            self._root,
            boundary=self._root_boundary,
        )
        root.mkdir(mode=0o700, parents=True, exist_ok=True)
        safe_root = self._existing_root(root)
        if safe_root is None:  # pragma: no cover - mkdir above guarantees this
            raise ValueError(f"agent run root is unavailable: {root}")
        return safe_root

    def _record_path(
        self,
        root: Path,
        run_id: str,
        *,
        allow_missing: bool,
    ) -> Path:
        safe_id = self._validate_run_id(run_id)
        return assert_no_linklike_components(
            root / f"{safe_id}.json",
            boundary=root,
            allow_missing=allow_missing,
        )

    def _read_from_root(self, root: Path, run_id: str) -> tuple[bool, AgentRun | None]:
        safe_root = self._existing_root(root)
        if safe_root is None:
            return False, None
        try:
            path = self._record_path(safe_root, run_id, allow_missing=False)
        except FileNotFoundError:
            return False, None
        path_stat = os.lstat(path)
        if not stat.S_ISREG(path_stat.st_mode):
            raise ValueError(f"agent run record is not a safe regular file: {path}")
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_NOFOLLOW", 0)
        try:
            fd = os.open(path, flags)
            with os.fdopen(fd, encoding="utf-8") as handle:
                opened_stat = os.fstat(handle.fileno())
                if (
                    not stat.S_ISREG(opened_stat.st_mode)
                    or (opened_stat.st_dev, opened_stat.st_ino)
                    != (path_stat.st_dev, path_stat.st_ino)
                ):
                    raise ValueError("agent run record changed while it was opened")
                data = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return True, None
        if not isinstance(data, dict):
            return True, None
        try:
            run = AgentRun.from_dict(data)
        except (TypeError, ValueError):
            return True, None
        if run.id != run_id:
            return True, None
        return True, run

    def _write(self, run: AgentRun) -> AgentRun:
        root = self._ensure_root()
        path = self._record_path(root, run.id, allow_missing=True)
        payload = json.dumps(run.to_dict(), indent=2, allow_nan=False) + "\n"
        root_stat = os.lstat(root)
        fd, temp_name = tempfile.mkstemp(prefix=f".{run.id}.", suffix=".tmp", dir=root)
        temp_path = Path(temp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write(payload)
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temp_path, 0o600)
            rebound_root = self._existing_root(root)
            if rebound_root is None:
                raise ValueError("agent run root disappeared during publication")
            rebound_stat = os.lstat(rebound_root)
            if (root_stat.st_dev, root_stat.st_ino) != (
                rebound_stat.st_dev,
                rebound_stat.st_ino,
            ):
                raise ValueError("agent run root changed during publication")
            self._record_path(root, run.id, allow_missing=True)
            os.replace(temp_path, path)
            if hasattr(os, "O_DIRECTORY"):
                directory_fd = os.open(
                    root,
                    os.O_RDONLY
                    | os.O_DIRECTORY
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0),
                )
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
        finally:
            temp_path.unlink(missing_ok=True)
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
        safe_id = self._validate_run_id(run_id)
        for root in self._read_roots:
            found, run = self._read_from_root(root, safe_id)
            if found:
                return run
        return None

    def list(self, *, status: str | None = None, limit: int = 20) -> list[AgentRun]:
        runs: list[AgentRun] = []
        seen: set[str] = set()
        for root in self._read_roots:
            safe_root = self._existing_root(root)
            if safe_root is None:
                continue
            for path in sorted(safe_root.glob("*.json")):
                run_id = path.stem
                if run_id in seen or not _RUN_ID_PATTERN.fullmatch(run_id):
                    continue
                found, run = self._read_from_root(safe_root, run_id)
                if not found:
                    continue
                seen.add(run_id)
                if run is None or (status and run.status != status):
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
