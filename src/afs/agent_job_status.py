"""Status and watchdog summary for markdown-backed agent jobs."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .agent_hooks import DEFAULT_WORKER_LABEL, default_launchd_plist_path
from .agent_job_worker import job_needs_destructive_opt_in
from .agent_jobs import AgentJob, AgentJobQueue, JOB_STATES
from .agent_runs import AgentRun, AgentRunStore

LaunchdProbe = Callable[[str], dict[str, Any]]


def probe_worker_launchd(label: str = DEFAULT_WORKER_LABEL) -> dict[str, Any]:
    """Return best-effort launchd state for the agent-jobs worker."""

    plist_path = default_launchd_plist_path(label)
    payload: dict[str, Any] = {
        "label": label,
        "plist": str(plist_path),
        "installed": plist_path.exists(),
        "available": sys.platform == "darwin",
        "loaded": False,
        "running": False,
        "state": "unavailable" if sys.platform != "darwin" else "not_loaded",
        "detail": "",
    }
    if sys.platform != "darwin":
        payload["detail"] = "launchd is only available on macOS"
        return payload

    domain_label = f"gui/{os.getuid()}/{label}"
    try:
        result = subprocess.run(
            ["launchctl", "print", domain_label],
            capture_output=True,
            text=True,
            check=False,
        )
    except FileNotFoundError:
        payload["available"] = False
        payload["state"] = "unavailable"
        payload["detail"] = "launchctl not found"
        return payload

    output = result.stdout or result.stderr or ""
    if result.returncode != 0:
        payload["detail"] = _trim(output)
        return payload

    state = _match_value(output, r"\bstate = ([^\n]+)") or "loaded"
    active_count = _match_value(output, r"\bactive count = ([0-9]+)")
    payload.update(
        {
            "loaded": True,
            "running": state == "running" or _to_int(active_count) > 0,
            "state": state,
            "active_count": _to_int(active_count),
            "runs": _to_int(_match_value(output, r"\bruns = ([0-9]+)")),
            "last_exit_status": _match_value(output, r"\blast exit code = ([^\n]+)") or "",
            "last_signal": _match_value(output, r"\blast terminating signal = ([^\n]+)") or "",
        }
    )
    return payload


def build_agent_job_status(
    context_path: Path,
    *,
    label: str = DEFAULT_WORKER_LABEL,
    stale_after_seconds: float = 3600.0,
    recent_runs_limit: int = 5,
    launchd_probe: LaunchdProbe | None = None,
) -> dict[str, Any]:
    queue = AgentJobQueue(context_path)
    jobs = queue.list()
    recent_runs = (
        AgentRunStore(context_path).list(limit=recent_runs_limit)
        if recent_runs_limit > 0
        else []
    )
    now = datetime.now(timezone.utc)
    counts = {state: 0 for state in JOB_STATES}
    for job in jobs:
        counts[job.status] = counts.get(job.status, 0) + 1

    queued = [job for job in jobs if job.status == "queue"]
    running = [job for job in jobs if job.status == "running"]
    failed = [job for job in jobs if job.status == "failed"]
    blocked_destructive = [
        job
        for job in queued
        if job_needs_destructive_opt_in(job) and not job.allow_destructive
    ]
    runnable = [job for job in queued if job not in blocked_destructive]
    stale_running = [
        job
        for job in running
        if _age_seconds(job.updated_at or job.created_at, now) is not None
        and _age_seconds(job.updated_at or job.created_at, now) >= stale_after_seconds
    ]
    recent_failed_runs = [
        run for run in recent_runs if run.status in {"failed", "abandoned"}
    ]
    worker = (launchd_probe or probe_worker_launchd)(label)
    checks = _build_watchdog_checks(
        worker=worker,
        blocked_destructive=blocked_destructive,
        runnable=runnable,
        stale_running=stale_running,
        failed=failed,
        recent_failed_runs=recent_failed_runs,
    )
    return {
        "context_path": str(context_path),
        "counts": counts,
        "queue": {
            "total": len(queued),
            "runnable": len(runnable),
            "blocked_destructive": len(blocked_destructive),
            "blocked_items": [_job_brief(job, now=now) for job in blocked_destructive],
        },
        "running": {
            "total": len(running),
            "stale": len(stale_running),
            "stale_after_seconds": stale_after_seconds,
            "stale_items": [_job_brief(job, now=now) for job in stale_running],
        },
        "failed": {
            "total": len(failed),
            "items": [_job_brief(job, now=now) for job in failed[:10]],
        },
        "recent_runs": {
            "total": len(recent_runs),
            "failed": len(recent_failed_runs),
            "items": [_run_brief(run) for run in recent_runs],
        },
        "worker": worker,
        "watchdog": {
            "ok": not any(check["status"] == "error" for check in checks),
            "healthy": not any(check["status"] in {"error", "warning"} for check in checks),
            "warning_count": sum(1 for check in checks if check["status"] == "warning"),
            "error_count": sum(1 for check in checks if check["status"] == "error"),
            "checks": checks,
        },
    }


def format_agent_job_status(payload: dict[str, Any]) -> str:
    counts = payload.get("counts") or {}
    queue = payload.get("queue") or {}
    running = payload.get("running") or {}
    recent_runs = payload.get("recent_runs") or {}
    worker = payload.get("worker") or {}
    watchdog = payload.get("watchdog") or {}
    lines = [
        f"context: {payload.get('context_path', '')}",
        (
            "worker: "
            f"{worker.get('state', 'unknown')} "
            f"(installed={_bool_text(worker.get('installed'))}, "
            f"loaded={_bool_text(worker.get('loaded'))}, "
            f"running={_bool_text(worker.get('running'))})"
        ),
        "jobs: " + " ".join(f"{state}={counts.get(state, 0)}" for state in JOB_STATES),
        (
            "queue: "
            f"runnable={queue.get('runnable', 0)} "
            f"blocked_destructive={queue.get('blocked_destructive', 0)}"
        ),
        (
            "running: "
            f"total={running.get('total', 0)} "
            f"stale={running.get('stale', 0)} "
            f"stale_after={running.get('stale_after_seconds', 0)}s"
        ),
        (
            "recent_runs: "
            f"total={recent_runs.get('total', 0)} "
            f"failed={recent_runs.get('failed', 0)}"
        ),
        (
            "watchdog: "
            f"{'healthy' if watchdog.get('healthy') else 'attention'} "
            f"(warnings={watchdog.get('warning_count', 0)}, "
            f"errors={watchdog.get('error_count', 0)})"
        ),
    ]
    checks = watchdog.get("checks") or []
    if checks:
        lines.append("watchdog_checks:")
        for check in checks:
            lines.append(
                f"  {check.get('status', 'unknown')}\t{check.get('name', '')}\t{check.get('summary', '')}"
            )
    blocked_items = queue.get("blocked_items") or []
    if blocked_items:
        lines.append("blocked_destructive_jobs:")
        for item in blocked_items[:10]:
            lines.append(f"  {item['id']}\tp{item['priority']}\t{item['title']}")
    stale_items = running.get("stale_items") or []
    if stale_items:
        lines.append("stale_running_jobs:")
        for item in stale_items[:10]:
            lines.append(f"  {item['id']}\t{item.get('assigned_to') or '-'}\t{item['title']}")
    run_items = recent_runs.get("items") or []
    if run_items:
        lines.append("recent_runs:")
        for item in run_items[:5]:
            lines.append(
                f"  {item['id']}\t[{item['status']}]\t{item.get('harness') or '-'}\t{item['task']}"
            )
    return "\n".join(lines)


def _build_watchdog_checks(
    *,
    worker: dict[str, Any],
    blocked_destructive: list[AgentJob],
    runnable: list[AgentJob],
    stale_running: list[AgentJob],
    failed: list[AgentJob],
    recent_failed_runs: list[AgentRun],
) -> list[dict[str, Any]]:
    checks: list[dict[str, Any]] = []
    if not worker.get("available", True):
        checks.append(_check("worker_launchd", "warning", str(worker.get("detail") or "worker state unavailable")))
    elif not worker.get("installed"):
        checks.append(_check("worker_launchd", "warning", "worker LaunchAgent is not installed"))
    elif not worker.get("loaded"):
        checks.append(_check("worker_launchd", "warning", "worker LaunchAgent is installed but not loaded"))
    elif not worker.get("running"):
        checks.append(_check("worker_launchd", "warning", "worker LaunchAgent is loaded but not running"))
    else:
        checks.append(_check("worker_launchd", "ok", "worker LaunchAgent is running"))

    if blocked_destructive:
        suffix = "safe queued jobs can still run" if runnable else "queue has no runnable safe jobs"
        checks.append(
            _check(
                "destructive_opt_in",
                "warning",
                f"{len(blocked_destructive)} queued job(s) need --allow-destructive; {suffix}",
            )
        )
    else:
        checks.append(_check("destructive_opt_in", "ok", "no queued jobs need destructive opt-in"))

    if stale_running:
        checks.append(
            _check("stale_running_jobs", "warning", f"{len(stale_running)} running job(s) look stale")
        )
    else:
        checks.append(_check("stale_running_jobs", "ok", "no stale running jobs"))

    if failed:
        checks.append(_check("failed_jobs", "warning", f"{len(failed)} job(s) are in failed state"))
    else:
        checks.append(_check("failed_jobs", "ok", "no failed jobs"))

    if recent_failed_runs:
        checks.append(
            _check("recent_run_failures", "warning", f"{len(recent_failed_runs)} recent run(s) failed")
        )
    else:
        checks.append(_check("recent_run_failures", "ok", "no recent run failures"))
    return checks


def _job_brief(job: AgentJob, *, now: datetime) -> dict[str, Any]:
    return {
        "id": job.id,
        "title": job.title,
        "status": job.status,
        "priority": job.priority,
        "assigned_to": job.assigned_to,
        "scope": job.scope,
        "allow_destructive": job.allow_destructive,
        "needs_destructive_opt_in": job_needs_destructive_opt_in(job),
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "age_seconds": _age_seconds(job.updated_at or job.created_at, now),
    }


def _run_brief(run: AgentRun) -> dict[str, Any]:
    return {
        "id": run.id,
        "task": run.task,
        "harness": run.harness,
        "status": run.status,
        "workspace": run.workspace,
        "updated_at": run.updated_at,
        "finished_at": run.finished_at,
        "summary": run.summary,
    }


def _check(name: str, status: str, summary: str) -> dict[str, str]:
    return {"name": name, "status": status, "summary": summary}


def _age_seconds(value: str, now: datetime) -> float | None:
    parsed = _parse_iso(value)
    if parsed is None:
        return None
    return max(0.0, (now - parsed).total_seconds())


def _parse_iso(value: str) -> datetime | None:
    if not value:
        return None
    normalized = value.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _match_value(text: str, pattern: str) -> str:
    match = re.search(pattern, text)
    return match.group(1).strip() if match else ""


def _to_int(value: str | None) -> int:
    try:
        return int(value or 0)
    except ValueError:
        return 0


def _trim(value: str, limit: int = 500) -> str:
    stripped = value.strip()
    if len(stripped) <= limit:
        return stripped
    return stripped[-limit:]


def _bool_text(value: object) -> str:
    return "true" if bool(value) else "false"
