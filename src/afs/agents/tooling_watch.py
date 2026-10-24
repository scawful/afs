"""Check opted-in tooling deliveries and alert agents only when findings change."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import stat
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..atomic_io import atomic_write_text, secure_mkdir
from ..config import load_config_model
from ..context_paths import resolve_mount_root
from ..messages import MessageBus
from ..models import MountType
from ..path_safety import assert_no_linklike_components
from ..profiles import resolve_active_profile
from ..tasks import Task, TaskQueue
from .base import AgentResult, emit_result, now_iso
from .guardrails import _file_lock

AGENT_NAME = "tooling-watch"
MAX_TASKS = 256
MAX_FILE_BYTES = 16 * 1024 * 1024


def _absolute(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        raise ValueError(f"absolute path required: {value}")
    return assert_no_linklike_components(path)


def _read_json(path: Path, default: dict[str, Any]) -> dict[str, Any]:
    try:
        if path.stat().st_size > 2 * 1024 * 1024:
            raise ValueError(f"JSON exceeds 2 MiB: {path}")
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return default
    value = json.loads(text)
    if not isinstance(value, dict):
        raise ValueError(f"JSON object required: {path}")
    return value


def _digest(value: str) -> str:
    path = _absolute(value)
    try:
        info = path.stat()
    except FileNotFoundError:
        return "missing"
    if not stat.S_ISREG(info.st_mode) or info.st_size > MAX_FILE_BYTES:
        raise ValueError(f"artifact must be a regular file <=16 MiB: {path}")
    with path.open("rb") as stream:
        data = stream.read(MAX_FILE_BYTES + 1)
    if len(data) > MAX_FILE_BYTES:
        raise ValueError(f"artifact grew beyond 16 MiB: {path}")
    return hashlib.sha256(data).hexdigest()


def _observe(task: Task, now: datetime) -> dict[str, Any]:
    delivery = task.context["delivery"]
    if not isinstance(delivery, dict) or delivery.get("state") not in {"ready", "adopted", "deferred"}:
        raise ValueError(f"invalid delivery state: {task.id}")
    if delivery["state"] == "deferred":
        return {"state": "deferred", "issues": []}
    for key in ("source_path", "sha256", "verification", "evidence", "next_action", "consumer"):
        if not isinstance(delivery.get(key), str) or not delivery[key].strip():
            raise ValueError(f"delivery {task.id} needs {key}")
    expected = delivery["sha256"]
    if not re.fullmatch(r"[0-9a-f]{64}", expected):
        raise ValueError(f"invalid sha256: {task.id}")
    row = {key: delivery[key] for key in ("state", "consumer", "verification", "evidence", "next_action")}
    row.update(title=task.title, owner=task.assigned_to, source_path=delivery["source_path"], expected=expected)
    row["source_hash"] = _digest(delivery["source_path"])
    issues = []
    if row["source_hash"] != expected:
        issues.append("source missing" if row["source_hash"] == "missing" else "source changed")
    if delivery.get("installed_path"):
        row["installed_path"] = delivery["installed_path"]
        row["installed_hash"] = _digest(delivery["installed_path"])
        if row["installed_hash"] != expected:
            issues.append("installation missing" if row["installed_hash"] == "missing" else "installation changed")
    if delivery["state"] == "ready":
        issues.append("ready; adoption pending")
    if not task.assigned_to:
        issues.append("owner missing")
    if "updated_at" in delivery and "stale_after_hours" in delivery:
        stamp = datetime.fromisoformat(delivery["updated_at"].replace("Z", "+00:00"))
        hours = delivery["stale_after_hours"]
        if stamp.tzinfo is None or isinstance(hours, bool) or not isinstance(hours, (int, float)) or (isinstance(hours, float) and not math.isfinite(hours)) or hours <= 0:
            raise ValueError(f"invalid explicit progress expiry: {task.id}")
        if delivery["state"] == "ready" and (now - stamp).total_seconds() > hours * 3600:
            issues.append("progress stale")
    return {**row, "issues": issues}


def _collect(context: Path, now: datetime) -> dict[str, Any]:
    root = _absolute(resolve_mount_root(context, MountType.ITEMS))
    files = sorted(root.glob("task-*.json"))
    if len(files) > MAX_TASKS:
        raise ValueError(f"task inventory exceeds {MAX_TASKS}; scan not complete")
    for path in files:
        raw = _read_json(_absolute(path), {})
        if not isinstance(raw.get("context", {}), dict):
            raise ValueError(f"task context must be an object: {path}")
    tasks = TaskQueue(context).list()
    if {p.stem[5:] for p in files} != {task.id for task in tasks}:
        raise ValueError("task inventory contains unreadable or malformed records")
    return {task.id: _observe(task, now) for task in tasks
            if task.context.get("kind") == "tooling" and "delivery" in task.context}


def _send(bus: MessageBus, payload: dict[str, Any], consumer: str = "") -> int:
    recipients = {name.lower(): name for name in ("codex", consumer) if name}
    for recipient in recipients.values():
        bus.send(AGENT_NAME, "finding", payload, to=recipient, topic="tooling:ready")
    return len(recipients)


def run_watch(context: Path, state: Path, *, dry_run: bool = False) -> dict[str, Any]:
    context = context.resolve(strict=True)
    scratchpad = _absolute(resolve_mount_root(context, MountType.SCRATCHPAD))
    state = assert_no_linklike_components(_absolute(state), boundary=scratchpad)
    error_path = _absolute(state.with_suffix(state.suffix + ".error.json"))
    _absolute(state.with_suffix(state.suffix + ".lock"))
    if dry_run:
        return {"status": "ok", "observations": _collect(context, datetime.now(timezone.utc)), "alerts": 0}
    secure_mkdir(state.parent)
    bus = MessageBus(context, scope_id="common")
    with _file_lock(state):
        prior_error = ""
        try:
            prior_error = _read_json(error_path, {}).get("error", "")
            old = _read_json(state, {"version": 1, "observations": {}})
            if old.get("version") != 1 or not isinstance(old.get("observations"), dict) or any(not isinstance(row, dict) for row in old["observations"].values()):
                raise ValueError("invalid tooling watch state")
            observations = _collect(context, datetime.now(timezone.utc))
            missing = old["observations"].keys() - observations.keys()
            if missing:
                raise ValueError(f"previously watched tooling records missing: {sorted(missing)}")
            alerts = 0
            for task_id, row in observations.items():
                if row["issues"] and row != old["observations"].get(task_id):
                    summary = f"{row['title'][:64]}: {', '.join(row['issues'])}. Next: {row['next_action'][:96]}"
                    alerts += _send(bus, {"summary": summary, "next": row["next_action"], "task": task_id, **row}, row.get("consumer", ""))
            if prior_error:
                alerts += _send(bus, {"summary": "Tooling scan recovered", "state": str(state)})
            atomic_write_text(state, json.dumps({"version": 1, "observations": observations}, sort_keys=True), mode=0o600)
            if prior_error:
                atomic_write_text(error_path, json.dumps({"error": ""}), mode=0o600)
            return {"status": "ok", "observations": observations, "alerts": alerts}
        except (OSError, ValueError, TypeError, KeyError, RuntimeError) as exc:
            error = f"{type(exc).__name__}: {exc}"
            if error != prior_error:
                _send(bus, {"summary": "Tooling scan failed", "error": error, "state": str(state)})
                atomic_write_text(error_path, json.dumps({"error": error}), mode=0o600)
            return {"status": "error", "error": error, "alerts": int(error != prior_error)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("path", "context-root", "state", "config"):
        parser.add_argument(f"--{name}")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    started, clock = now_iso(), time.monotonic()
    try:
        config = load_config_model(config_path=_absolute(args.config) if args.config else None, merge_user=True)
        agent = os.environ.get("AFS_AGENT_NAME", AGENT_NAME)
        entry = next((item for item in resolve_active_profile(config).agent_configs if item.name == agent), None)
        extra = entry.extra if entry else {}
        project = args.path or extra.get("project_path")
        root = args.context_root or extra.get("context_root")
        if not root and not project:
            raise ValueError("provide --path/--context-root or a configured project_path")
        context = _absolute(root) if root else (_absolute(project or "") / ".context").resolve(strict=True)
        state = args.state or extra.get("state_path") or resolve_mount_root(context, MountType.SCRATCHPAD) / "afs_agents/tooling_watch_state.json"
        payload = run_watch(context, _absolute(state), dry_run=args.dry_run)
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        payload = {"status": "error", "error": f"{type(exc).__name__}: {exc}"}
    emit_result(AgentResult(AGENT_NAME, payload["status"], started, now_iso(), time.monotonic() - clock,
                           metrics={"alerts": payload.get("alerts", 0)}, payload=payload),
                output_path=None, force_stdout=True, pretty=False)
    return int(payload["status"] != "ok")


if __name__ == "__main__":
    raise SystemExit(main())
