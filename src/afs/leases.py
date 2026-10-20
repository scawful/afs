"""Machine-wide resource leases stored as directory locks.

The on-disk format matches the shared heavy-lane lock used by Oracle of
Secrets and yaze build helpers (``/tmp/oos-heavy.lock``):

* the lock is a directory created with ``mkdir`` (atomic);
* ``owner`` holds the owner label and is written first;
* ``metadata.json`` holds ``{"schema": 1, "token", "pid", "owner", "started_at"}``
  and is published second via rename.

AFS adds optional metadata keys (``client``, ``task``, ``source``,
``expires_at``, ``child_pgid``) that older clients ignore. A lease is
reclaimable only when its owner process is gone, no recorded worker process
group or process working under ``source`` remains, and any ``expires_at`` has
passed. Locks without ``metadata.json`` are never reclaimed.
"""

from __future__ import annotations

import json
import math
import os
import re
import signal
import subprocess
import time
import uuid
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

LEASE_SCHEMA = 1
DEFAULT_LEASE_DIR = Path("/tmp")
DEFAULT_WAIT_SECONDS = 600.0
DEFAULT_POLL_SECONDS = 15.0
DEFAULT_ACQUIRE_TTL_MINUTES = 60
LEASE_FILES = frozenset({"owner", "metadata.json"})
_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


class LeaseBusy(RuntimeError):
    def __init__(self, state: dict[str, Any]) -> None:
        self.state = state
        super().__init__(f"lease busy: {describe(state)}")


def lease_path(name_or_path: str) -> Path:
    """``oos-heavy`` -> ``/tmp/oos-heavy.lock``; anything with a ``/`` is a path."""
    if "/" in name_or_path:
        return Path(name_or_path).expanduser()
    if not _NAME.match(name_or_path):
        raise ValueError("lease names use letters, digits, '.', '_' or '-' (max 64)")
    return DEFAULT_LEASE_DIR / f"{name_or_path}.lock"


def _pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _group_alive(pgid: int) -> bool:
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _processes_under(source: Path) -> list[int]:
    """PIDs whose working directory is inside *source* (best effort, via lsof)."""
    try:
        result = subprocess.run(
            ["lsof", "-d", "cwd", "-Fpn"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return []
    prefix = str(source.resolve())
    pids: list[int] = []
    current: int | None = None
    for line in result.stdout.splitlines():
        if line.startswith("p"):
            current = int(line[1:]) if line[1:].isdigit() else None
        elif line.startswith("n") and current is not None:
            cwd = line[1:]
            if cwd == prefix or cwd.startswith(prefix + os.sep):
                pids.append(current)
    return sorted(set(pids))


def _command_names(pids: list[int]) -> dict[int, str]:
    if not pids:
        return {}
    try:
        result = subprocess.run(
            ["ps", "-o", "pid=,comm=", "-p", ",".join(str(pid) for pid in pids)],
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return {}
    names: dict[int, str] = {}
    for line in result.stdout.splitlines():
        pid_text, _, command = line.strip().partition(" ")
        if pid_text.isdigit():
            names[int(pid_text)] = os.path.basename(command.strip())
    return names


def _read_metadata(path: Path) -> dict[str, Any] | None:
    try:
        data = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    except FileNotFoundError:
        return None
    except (OSError, json.JSONDecodeError):
        return {"malformed": True}
    return data if isinstance(data, dict) else {"malformed": True}


def _number(value: Any) -> float | None:
    """A finite int/float (bools excluded), else ``None``."""
    if type(value) not in (int, float) or not math.isfinite(value):
        return None
    return float(value)


def _valid_metadata(meta: dict[str, Any] | None, owner: str) -> bool:
    if not meta or meta.get("malformed"):
        return False
    started = _number(meta.get("started_at"))
    return (
        meta.get("schema") == LEASE_SCHEMA
        and isinstance(meta.get("token"), str)
        and re.fullmatch(r"[a-f0-9]{32}", meta["token"]) is not None
        and type(meta.get("pid")) is int
        and meta["pid"] > 0
        and meta.get("owner") == owner
        and started is not None
        and started > 0
    )


def inspect(path: Path, *, scan_workers: bool = True, now: float | None = None) -> dict[str, Any]:
    """Describe a lease: ``free``, ``held``, ``owner-exited`` (workers remain), ``stale`` or ``legacy``."""
    current = time.time() if now is None else now
    if not path.exists():
        return {"path": str(path), "status": "free"}
    try:
        owner = (path / "owner").read_text(encoding="utf-8").strip() or "?"
    except OSError:
        owner = "?"
    meta = _read_metadata(path)
    try:
        age = int(current - path.stat().st_mtime)
    except OSError:
        age = None
    state: dict[str, Any] = {"path": str(path), "owner": owner, "age_seconds": age}
    if not _valid_metadata(meta, owner):
        state.update(status="legacy", metadata=meta)
        return state
    assert meta is not None
    pid = int(meta["pid"])
    state.update(metadata=meta, pid=pid, pid_alive=_pid_alive(pid))
    state["age_seconds"] = int(current - float(meta["started_at"]))
    if state["pid_alive"]:
        state["status"] = "held"
        return state
    workers: list[dict[str, Any]] = []
    child_pgid = meta.get("child_pgid")
    if type(child_pgid) is int and child_pgid > 0 and _group_alive(child_pgid):
        workers.append({"pgid": child_pgid, "reason": "recorded worker process group"})
    source = meta.get("source")
    if scan_workers and isinstance(source, str) and source and Path(source).is_dir():
        pids = [pid for pid in _processes_under(Path(source)) if pid != os.getpid()]
        names = _command_names(pids)
        for worker_pid in pids:
            workers.append(
                {
                    "pid": worker_pid,
                    "command": names.get(worker_pid, ""),
                    "reason": f"working directory under {source}",
                }
            )
    state["workers"] = workers
    expires_at = _number(meta.get("expires_at"))
    if workers:
        state["status"] = "owner-exited"
    elif expires_at is not None and current < expires_at:
        state["status"] = "held"
        state["held_until"] = expires_at
    else:
        state["status"] = "stale"
    return state


def describe(state: dict[str, Any]) -> str:
    status = state.get("status", "?")
    if status == "free":
        return f"{state['path']}: free"
    parts = [f"{state['path']}: {status}", f"owner={state.get('owner', '?')}"]
    if "pid" in state:
        parts.append(f"pid={state['pid']}{'' if state.get('pid_alive') else ' (exited)'}")
    if state.get("age_seconds") is not None:
        parts.append(f"age={state['age_seconds']}s")
    if state.get("workers"):
        parts.append(f"workers={len(state['workers'])}")
    return ", ".join(parts)


def _publish(path: Path, owner: str, metadata: dict[str, Any]) -> None:
    with (path / "owner").open("x", encoding="utf-8") as stream:
        stream.write(owner + "\n")
    _write_metadata(path, metadata)


def _write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    pending = path / f".metadata-{metadata['token']}.tmp"
    with pending.open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream)
    pending.replace(path / "metadata.json")


def acquire(
    path: Path,
    owner: str,
    *,
    pid: int | None = None,
    task: str = "",
    source: str = "",
    ttl_minutes: float | None = None,
    wait_seconds: float = DEFAULT_WAIT_SECONDS,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    on_wait: Callable[[dict[str, Any]], None] | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> dict[str, Any]:
    """Create the lease or wait up to *wait_seconds*; returns the published metadata."""
    if not owner.strip() or owner != owner.strip() or len(owner) > 512 or "\n" in owner:
        raise ValueError("lease owner must be a non-empty single line without outer spaces")
    deadline = time.monotonic() + max(0.0, wait_seconds)
    while True:
        try:
            path.mkdir(mode=0o755)
        except FileExistsError:
            state = inspect(path)
            if time.monotonic() >= deadline:
                raise LeaseBusy(state) from None
            if on_wait is not None:
                on_wait(state)
            sleep(max(0.1, min(poll_seconds, deadline - time.monotonic())))
            continue
        started = time.time()
        metadata: dict[str, Any] = {
            "schema": LEASE_SCHEMA,
            "token": uuid.uuid4().hex,
            "pid": pid if pid is not None else os.getpid(),
            "owner": owner,
            "started_at": started,
            "client": "afs",
        }
        if task:
            metadata["task"] = task
        if source:
            metadata["source"] = str(Path(source).expanduser().resolve())
        if ttl_minutes is not None:
            metadata["expires_at"] = started + float(ttl_minutes) * 60.0
        try:
            _publish(path, owner, metadata)
        except BaseException:
            for name in ("owner", "metadata.json"):
                (path / name).unlink(missing_ok=True)
            path.rmdir()
            raise
        return metadata


def _remove(path: Path) -> None:
    entries = set(os.listdir(path))
    if entries - LEASE_FILES:
        raise RuntimeError(
            f"lease directory has unexpected entries: {sorted(entries - LEASE_FILES)}"
        )
    (path / "metadata.json").unlink(missing_ok=True)
    (path / "owner").unlink(missing_ok=True)
    path.rmdir()


def release(path: Path, *, token: str = "", owner: str = "") -> dict[str, Any]:
    """Remove the lease only when *token* (or *owner*) matches its metadata."""
    if not token and not owner:
        raise ValueError("release needs the lease token or owner")
    state = inspect(path, scan_workers=False)
    if state["status"] == "free":
        return {"released": False, "reason": "lease is free", "state": state}
    meta = state.get("metadata") or {}
    if state["status"] == "legacy":
        return {"released": False, "reason": "lease has no AFS-readable metadata", "state": state}
    if token and meta.get("token") != token:
        return {"released": False, "reason": "token does not match", "state": state}
    if owner and meta.get("owner") != owner:
        return {"released": False, "reason": "owner does not match", "state": state}
    _remove(path)
    return {"released": True, "state": state}


def reclaim(path: Path, *, apply: bool = False) -> dict[str, Any]:
    """Remove a ``stale`` lease (owner gone, no workers, not unexpired). Dry-run by default."""
    state = inspect(path)
    if state["status"] != "stale":
        return {
            "reclaimed": False,
            "applied": apply,
            "reason": f"lease is {state['status']}",
            "state": state,
        }
    if apply:
        _remove(path)
    return {"reclaimed": apply, "applied": apply, "state": state}


def run(
    path: Path,
    owner: str,
    command: Sequence[str],
    *,
    task: str = "",
    source: str = "",
    wait_seconds: float = DEFAULT_WAIT_SECONDS,
    poll_seconds: float = DEFAULT_POLL_SECONDS,
    drain_seconds: float = 120.0,
    on_wait: Callable[[dict[str, Any]], None] | None = None,
) -> int:
    """Hold the lease while *command* and its process group run, then release it."""
    if not command:
        raise ValueError("lease run needs a command")
    metadata = acquire(
        path,
        owner,
        task=task or " ".join(command)[:200],
        source=source,
        wait_seconds=wait_seconds,
        poll_seconds=poll_seconds,
        on_wait=on_wait,
    )
    child: subprocess.Popen[bytes] | None = None
    previous: dict[int, Any] = {}

    def forward(signum: int, _frame: Any) -> None:
        if child is not None:
            try:
                os.killpg(child.pid, signum)
            except ProcessLookupError:
                pass

    try:
        for signum in (signal.SIGINT, signal.SIGTERM):
            previous[signum] = signal.signal(signum, forward)
        child = subprocess.Popen(list(command), start_new_session=True)
        metadata = {**metadata, "child_pgid": child.pid}
        _write_metadata(path, metadata)
        returncode = child.wait()
        deadline = time.monotonic() + drain_seconds
        while _group_alive(child.pid) and time.monotonic() < deadline:
            time.sleep(0.5)
        if _group_alive(child.pid):
            return returncode
        release(path, token=metadata["token"])
        return returncode
    except BaseException:
        if child is None:
            release(path, token=metadata["token"])
        raise
    finally:
        for restored, handler in previous.items():
            signal.signal(restored, handler)


__all__ = [
    "DEFAULT_WAIT_SECONDS",
    "LeaseBusy",
    "acquire",
    "describe",
    "inspect",
    "lease_path",
    "reclaim",
    "release",
    "run",
]
