"""Markdown-backed background agent job queue."""

from __future__ import annotations

import os
import re
import stat
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .agent_scope import assert_mount_allowed
from .agents.guardrails import _file_lock
from .context_paths import resolve_mount_root
from .models import MountType

JOB_STATES = ("queue", "running", "done", "failed", "archived")
_WINDOWS_DURABILITY = os.name == "nt"
_WINDOWS_PATHS = os.name == "nt"
_JOB_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}\Z")


class AgentJobPublishError(OSError):
    """A durable job publish failed, possibly after the rename became visible."""

    def __init__(self, message: str, *, path: Path, installed: bool) -> None:
        super().__init__(message)
        self.path = path
        self.installed = installed
        self.job_id = ""


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _job_id() -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{stamp}-{uuid.uuid4().hex[:8]}"


def _clean_scalar(value: str) -> str:
    # ``str.splitlines`` covers CR, CRLF, NEL, and Unicode paragraph/line
    # separators in addition to LF. Leaving any parser-recognized boundary in
    # frontmatter would let a scalar inject a later metadata key.
    return " ".join(value.splitlines()).strip()


def _path_is_linklike(path: Path) -> bool:
    """Detect symlinks plus Windows directory junctions when available."""
    if path.is_symlink():
        return True
    is_junction = getattr(path, "is_junction", None)
    if callable(is_junction) and is_junction():
        return True
    if _WINDOWS_PATHS:
        try:
            attributes = path.lstat().st_file_attributes  # type: ignore[attr-defined]
        except FileNotFoundError:
            return False
        return bool(
            attributes
            & getattr(stat, "FILE_ATTRIBUTE_REPARSE_POINT", 0x00000400)
        )
    return False


def _fsync_directory(path: Path) -> None:
    """Persist directory-entry changes where the platform supports it."""
    if os.name == "nt":
        return
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _replace_durable(source: Path, destination: Path, *, replace: bool = True) -> None:
    """Rename with write-through semantics on Windows.

    POSIX callers pair ``os.replace`` with an explicit parent-directory
    fsync. Windows has no portable directory-fsync primitive, so use the
    platform's write-through move contract instead.
    """
    if os.name != "nt":
        os.replace(source, destination)
        return

    import ctypes

    move_file_ex = ctypes.windll.kernel32.MoveFileExW  # type: ignore[attr-defined]
    move_file_ex.argtypes = (ctypes.c_wchar_p, ctypes.c_wchar_p, ctypes.c_uint32)
    move_file_ex.restype = ctypes.c_int
    flags = 0x8  # MOVEFILE_WRITE_THROUGH
    if replace:
        flags |= 0x1  # MOVEFILE_REPLACE_EXISTING
    if not move_file_ex(str(source), str(destination), flags):
        raise ctypes.WinError()


def _mkdir_durable(path: Path, *, durable_base: Path) -> None:
    """Create ``path`` below a trusted base and re-sync every parent entry."""
    if not durable_base.is_dir():
        raise FileNotFoundError(f"durable directory base does not exist: {durable_base}")
    try:
        path.relative_to(durable_base)
    except ValueError as exc:
        raise ValueError(f"directory {path} is outside durable base {durable_base}") from exc
    if os.name == "nt":
        current = durable_base
        for part in path.relative_to(durable_base).parts:
            target = current / part
            if target.is_dir():
                current = target
                continue
            if target.exists():
                raise NotADirectoryError(f"durable directory path is not a directory: {target}")
            temporary = current / f".{part}.{uuid.uuid4().hex}.tmpdir"
            temporary.mkdir()
            try:
                _replace_durable(temporary, target, replace=False)
            except OSError:
                if not target.is_dir():
                    raise
            finally:
                try:
                    temporary.rmdir()
                except FileNotFoundError:
                    pass
            current = target
        return

    path.mkdir(parents=True, exist_ok=True)
    current = path
    while current != durable_base:
        parent = current.parent
        _fsync_directory(parent)
        current = parent


def _atomic_write_text(path: Path, content: str) -> None:
    """Install a complete, flushed file and durably publish its directory entry."""
    # The queue creates and syncs its directory hierarchy before calling this
    # helper; only the replacement itself belongs here.
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    installed = False
    try:
        try:
            with temporary.open("x", encoding="utf-8") as handle:
                handle.write(content)
                handle.flush()
                os.fsync(handle.fileno())
            _replace_durable(temporary, path)
            installed = True
            _fsync_directory(path.parent)
        except OSError as exc:
            raise AgentJobPublishError(
                str(exc),
                path=path,
                installed=installed,
            ) from exc
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
        except OSError:
            pass


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
    durable_publish: bool = False
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
            "durable_publish": self.durable_publish,
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
        self._items_root = resolve_mount_root(context_path, MountType.ITEMS)
        self._root = self._items_root / "agent_jobs"

    def _state_root(self, state: str) -> Path:
        if state not in JOB_STATES:
            raise ValueError(f"Invalid job state: {state}")
        return self._root / state

    def _path(self, state: str, job_id: str) -> Path:
        if not isinstance(job_id, str) or not _JOB_ID_PATTERN.fullmatch(job_id):
            raise ValueError(f"Invalid agent job id: {job_id!r}")
        state_root = self._state_root(state)
        self._assert_queue_directory(self._root)
        self._assert_queue_directory(state_root)
        path = state_root / f"{job_id}.md"
        try:
            path.resolve(strict=False).relative_to(
                self._items_root.resolve(strict=False)
            )
            path.resolve(strict=False).relative_to(state_root.resolve(strict=False))
        except ValueError as exc:
            raise ValueError(f"Agent job path escapes items mount: {job_id!r}") from exc
        return path

    def _lock_path(self) -> Path:
        path = self._root / ".queue-state"
        lock_file = path.with_suffix(path.suffix + ".lock")
        if _path_is_linklike(lock_file):
            raise ValueError(f"Agent job queue lock cannot be a link: {lock_file}")
        try:
            lock_file.resolve(strict=False).relative_to(
                self._items_root.resolve(strict=False)
            )
        except ValueError as exc:
            raise ValueError("Agent job queue lock escapes items mount") from exc
        return path

    def _assert_queue_directory(self, path: Path) -> None:
        if _path_is_linklike(path):
            raise ValueError(f"Agent job queue directory cannot be a link: {path}")
        try:
            path.resolve(strict=False).relative_to(
                self._items_root.resolve(strict=False)
            )
        except ValueError as exc:
            raise ValueError(
                f"Agent job queue directory escapes items mount: {path}"
            ) from exc

    def ensure(self) -> None:
        """Create the queue hierarchy with the same durable marker contract."""
        self._ensure_durable()

    def _ensure_durable(self) -> None:
        """Create and persist the hierarchy used by a durable publication."""
        # ``context_root`` itself may not exist yet (the queue historically
        # created the complete mount hierarchy). Anchor durability at its
        # existing parent so first use can still create and sync the context
        # directory instead of turning an empty workspace into an unavailable
        # queue.
        durable_base = self._root.parent.parent
        while not durable_base.is_dir():
            parent = durable_base.parent
            if parent == durable_base:
                raise FileNotFoundError(
                    f"no existing durable base for agent jobs {self._root}"
                )
            durable_base = parent
        self._assert_queue_directory(self._root)
        _mkdir_durable(self._root, durable_base=durable_base)
        self._assert_queue_directory(self._root)
        for state in JOB_STATES:
            state_root = self._state_root(state)
            self._assert_queue_directory(state_root)
            _mkdir_durable(state_root, durable_base=durable_base)
            self._assert_queue_directory(state_root)

    def _write(self, job: AgentJob, *, durable_publish: bool = False) -> Path:
        if durable_publish:
            self._ensure_durable()
            job.durable_publish = True
        else:
            self.ensure()
        path = self._path(job.status, job.id)
        lines = ["---"]
        for key, value in job.metadata().items():
            lines.append(f"{key}: {_clean_scalar(str(value))}")
        lines.extend(["---", "", job.prompt.rstrip(), ""])
        content = "\n".join(lines)
        if durable_publish:
            try:
                _atomic_write_text(path, content)
            except AgentJobPublishError as exc:
                exc.job_id = job.id
                raise
        else:
            # Retained for private compatibility; public create/move paths use
            # the durable publication contract.
            path.write_text(content, encoding="utf-8")
        return path

    def _parse(self, path: Path) -> AgentJob | None:
        status = path.parent.name
        job_id = path.stem
        if status not in JOB_STATES:
            return None
        try:
            expected_path = self._path(status, job_id)
        except ValueError:
            return None
        if path.resolve(strict=False) != expected_path.resolve(strict=False):
            return None
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
            durable_publish=_parse_bool(metadata.get("durable_publish", "")),
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
            durable_publish=True,
            created_at=now,
            updated_at=now,
        )
        self._write(job, durable_publish=True)
        return job

    def get(self, job_id: str) -> AgentJob | None:
        if not self._root.exists():
            return None
        self._assert_queue_directory(self._root)
        with _file_lock(self._lock_path()):
            return self._get_unlocked(job_id)

    def _get_unlocked(self, job_id: str) -> AgentJob | None:
        for state in JOB_STATES:
            path = self._path(state, job_id)
            if path.exists():
                return self._parse(path)
        return None

    def is_durably_adoptable(self, job: AgentJob) -> bool:
        """Return whether this build can safely consume a receipt for ``job``."""
        return job.durable_publish or not _WINDOWS_DURABILITY

    def confirm_durable(self, job_id: str) -> AgentJob | None:
        """Flush a visible job record before its reactor receipt is consumed.

        State transitions may move the record while confirmation runs. Retry
        the bounded state scan when a candidate vanishes so the caller never
        acknowledges an inode that is no longer a live queue record.
        """
        self._ensure_durable()
        # CPython's Windows CRT handles do not normally share deletion. Keep
        # confirmation and state moves under one cross-process queue lock so
        # MoveFileEx cannot fail merely because the receipt handshake is
        # flushing or parsing the same live record.
        with _file_lock(self._lock_path()):
            for _attempt in range(3):
                retry = False
                for state in JOB_STATES:
                    path = self._path(state, job_id)
                    try:
                        mode = "r+b" if _WINDOWS_DURABILITY else "rb"
                        with path.open(mode) as handle:
                            os.fsync(handle.fileno())
                        _fsync_directory(path.parent)
                        job = self._parse(path)
                    except FileNotFoundError:
                        retry = True
                        continue
                    if (
                        job is not None
                        and job.id == job_id
                        and self.is_durably_adoptable(job)
                    ):
                        return job
                if not retry:
                    break
            return None

    def list(self, *, status: str | None = None) -> list[AgentJob]:
        if not self._root.exists():
            return []
        self._assert_queue_directory(self._root)
        with _file_lock(self._lock_path()):
            return self._list_unlocked(status=status)

    def _list_unlocked(self, *, status: str | None = None) -> list[AgentJob]:
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
        self._ensure_durable()
        with _file_lock(self._lock_path()):
            return self._move_unlocked(
                job_id,
                status,
                assigned_to=assigned_to,
                result=result,
                run_id=run_id,
            )

    def _move_unlocked(
        self,
        job_id: str,
        status: str,
        *,
        assigned_to: str = "",
        result: str = "",
        run_id: str = "",
    ) -> AgentJob:
        job = self._get_unlocked(job_id)
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
        new_path = self._write(job, durable_publish=True)
        if old_path.exists() and old_path.resolve(strict=False) != new_path.resolve(
            strict=False
        ):
            # Durably hide the old state name before best-effort cleanup. If
            # a crash restores the final unlink, only the hidden tombstone can
            # reappear; queue readers never mistake it for a second live job.
            tombstone = old_path.with_name(
                f".{old_path.name}.{uuid.uuid4().hex}.moved"
            )
            try:
                _replace_durable(old_path, tombstone, replace=False)
            except FileNotFoundError:
                pass
            else:
                try:
                    _fsync_directory(old_path.parent)
                    tombstone.unlink()
                    _fsync_directory(old_path.parent)
                except OSError:
                    # The new state is already durable and tombstones are not
                    # part of any queue glob, so cleanup failure cannot lose or
                    # duplicate executable work.
                    pass
        return job

    def claim(self, job_id: str, agent_name: str) -> AgentJob:
        self._ensure_durable()
        with _file_lock(self._lock_path()):
            job = self._get_unlocked(job_id)
            if job is None:
                raise FileNotFoundError(f"Agent job not found: {job_id}")
            if job.status != "queue":
                raise ValueError(f"Cannot claim job in state: {job.status}")
            return self._move_unlocked(
                job_id,
                "running",
                assigned_to=agent_name,
            )


def _parse_bool(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}
