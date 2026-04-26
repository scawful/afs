"""Review inbox for markdown-backed background agent jobs."""

from __future__ import annotations

import re
import shlex
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .agent_job_worker import job_needs_destructive_opt_in
from .agent_jobs import JOB_STATES, AgentJob, AgentJobQueue
from .agent_runs import AgentRunStore
from .context_paths import resolve_mount_root
from .models import MountType


def build_agent_job_inbox(
    context_path: Path,
    *,
    stale_after_seconds: float = 3600.0,
    limit: int = 20,
) -> dict[str, Any]:
    """Build a human review queue for background agent output."""

    queue = AgentJobQueue(context_path)
    jobs = queue.list()
    now = datetime.now(timezone.utc)
    counts = dict.fromkeys(JOB_STATES, 0)
    for job in jobs:
        counts[job.status] = counts.get(job.status, 0) + 1

    queued = [job for job in jobs if job.status == "queue"]
    running = [job for job in jobs if job.status == "running"]
    reviewable = _sort_attention([job for job in jobs if job.status == "done"])
    failed = _sort_attention([job for job in jobs if job.status == "failed"])
    stale_running = _sort_attention(
        [
            job
            for job in running
            if _age_seconds(job.updated_at or job.created_at, now) is not None
            and _age_seconds(job.updated_at or job.created_at, now) >= stale_after_seconds
        ]
    )
    blocked_destructive = _sort_attention(
        [
            job
            for job in queued
            if job_needs_destructive_opt_in(job) and not job.allow_destructive
        ]
    )
    runnable = [
        job
        for job in queued
        if not (job_needs_destructive_opt_in(job) and not job.allow_destructive)
    ]

    attention_total = (
        len(reviewable)
        + len(failed)
        + len(stale_running)
        + len(blocked_destructive)
    )
    return {
        "context_path": str(context_path.expanduser().resolve()),
        "command": _base_command(context_path, "inbox"),
        "stale_after_seconds": stale_after_seconds,
        "attention_count": attention_total,
        "counts": counts,
        "reviewable": {
            "total": len(reviewable),
            "items": [_job_brief(job, now=now) for job in reviewable[:limit]],
        },
        "failed": {
            "total": len(failed),
            "items": [_job_brief(job, now=now) for job in failed[:limit]],
        },
        "stale_running": {
            "total": len(stale_running),
            "items": [_job_brief(job, now=now) for job in stale_running[:limit]],
        },
        "blocked_destructive": {
            "total": len(blocked_destructive),
            "items": [_job_brief(job, now=now) for job in blocked_destructive[:limit]],
        },
        "queued": {
            "total": len(queued),
            "runnable": len(runnable),
        },
    }


def review_agent_job(context_path: Path, job_id: str) -> dict[str, Any]:
    """Return job details plus a linked run record when one is available."""

    job = _require_job(context_path, job_id)
    run = AgentRunStore(context_path).get(job.run_id) if job.run_id else None
    return {
        "context_path": str(context_path.expanduser().resolve()),
        "job": job.to_dict(),
        "run": run.to_dict() if run else None,
        "commands": {
            "review": _base_command(context_path, "review", job.id),
            "archive": _base_command(context_path, "archive", job.id),
            "promote_to_handoff": _base_command(context_path, "promote", job.id)
            + " --to-handoff",
        },
    }


def archive_agent_job(context_path: Path, job_id: str) -> AgentJob:
    """Move a job to the archived state without deleting its markdown record."""

    _require_job(context_path, job_id)
    return AgentJobQueue(context_path).move(job_id, "archived")


def promote_agent_job_to_handoff(
    context_path: Path,
    job_id: str,
    *,
    handoff_name: str = "",
) -> dict[str, Any]:
    """Write a job review packet to scratchpad/handoffs for future agents."""

    review = review_agent_job(context_path, job_id)
    job = review["job"]
    run = review.get("run")
    scratchpad_root = resolve_mount_root(context_path, MountType.SCRATCHPAD)
    handoff_root = scratchpad_root / "handoffs"
    handoff_root.mkdir(parents=True, exist_ok=True)
    filename = _safe_handoff_name(handoff_name or f"agent-job-{job['id']}.md")
    target = _next_available_path(handoff_root / filename)
    target.write_text(_render_handoff(review), encoding="utf-8")
    return {
        "context_path": review["context_path"],
        "path": str(target),
        "job": job,
        "run": run,
        "commands": review["commands"],
    }


def format_agent_job_inbox(payload: dict[str, Any]) -> str:
    counts = payload.get("counts") or {}
    lines = [
        f"context: {payload.get('context_path', '')}",
        f"command: {payload.get('command', '')}",
        (
            "jobs: "
            + " ".join(f"{state}={counts.get(state, 0)}" for state in JOB_STATES)
        ),
        f"attention: {payload.get('attention_count', 0)}",
    ]
    _append_inbox_section(lines, payload, "reviewable", "review_ready")
    _append_inbox_section(lines, payload, "failed", "failed")
    _append_inbox_section(lines, payload, "stale_running", "stale_running")
    _append_inbox_section(lines, payload, "blocked_destructive", "blocked_destructive")
    if payload.get("attention_count", 0) == 0:
        lines.append("empty: no completed, failed, stale, or blocked jobs need review")
    return "\n".join(lines)


def format_agent_job_review(payload: dict[str, Any]) -> str:
    job = payload["job"]
    run = payload.get("run")
    commands = payload.get("commands") or {}
    lines = [
        f"job: {job.get('id', '')}",
        f"title: {job.get('title', '')}",
        f"status: {job.get('status', '')}",
        f"priority: {job.get('priority', '')}",
        f"created_by: {job.get('created_by', '') or '-'}",
        f"assigned_to: {job.get('assigned_to', '') or '-'}",
        f"scope: {job.get('scope', '') or '-'}",
        f"expected_output: {job.get('expected_output', '') or '-'}",
        f"run_id: {job.get('run_id', '') or '-'}",
        f"archive: {commands.get('archive', '')}",
        f"promote_to_handoff: {commands.get('promote_to_handoff', '')}",
    ]
    if job.get("result"):
        lines.extend(["result:", *_indent_block(str(job["result"]))])
    if run:
        lines.extend(
            [
                "run:",
                f"  status: {run.get('status', '')}",
                f"  harness: {run.get('harness', '') or '-'}",
                f"  workspace: {run.get('workspace', '') or '-'}",
                f"  updated_at: {run.get('updated_at', '') or '-'}",
            ]
        )
        if run.get("summary"):
            lines.extend(["  summary:", *_indent_block(str(run["summary"]), prefix="    ")])
        commands_ran = run.get("commands") or []
        if commands_ran:
            lines.append("  commands:")
            for command in commands_ran:
                lines.append(f"    - {command}")
        verification = run.get("verification") or []
        if verification:
            lines.append("  verification:")
            for item in verification:
                if isinstance(item, dict):
                    lines.append(
                        f"    - {item.get('status', 'recorded')}: {item.get('command', '')}"
                    )
    else:
        lines.append("run: no linked run record")
    return "\n".join(lines)


def _append_inbox_section(
    lines: list[str],
    payload: dict[str, Any],
    key: str,
    heading: str,
) -> None:
    section = payload.get(key) or {}
    items = section.get("items") or []
    if not items:
        return
    lines.append(f"{heading}: {section.get('total', len(items))}")
    context_path = str(payload.get("context_path", ""))
    for item in items:
        assigned = f" -> {item['assigned_to']}" if item.get("assigned_to") else ""
        lines.append(
            f"  {item['id']}\t[{item['status']}]\tp{item['priority']}\t{item['title']}{assigned}"
        )
        lines.append(
            f"    review: {_base_command(Path(context_path), 'review', item['id'])}"
        )


def _render_handoff(review: dict[str, Any]) -> str:
    job = review["job"]
    run = review.get("run")
    commands = review.get("commands") or {}
    lines = [
        f"# Agent Job Handoff: {job.get('title', job.get('id', ''))}",
        "",
        "## Job",
        f"- id: {job.get('id', '')}",
        f"- status: {job.get('status', '')}",
        f"- priority: {job.get('priority', '')}",
        f"- created_by: {job.get('created_by', '') or '-'}",
        f"- assigned_to: {job.get('assigned_to', '') or '-'}",
        f"- scope: {job.get('scope', '') or '-'}",
        f"- expected_output: {job.get('expected_output', '') or '-'}",
        f"- run_id: {job.get('run_id', '') or '-'}",
        "",
        "## Commands",
        f"- review: `{commands.get('review', '')}`",
        f"- archive: `{commands.get('archive', '')}`",
        f"- promote_to_handoff: `{commands.get('promote_to_handoff', '')}`",
        "",
        "## Prompt",
        str(job.get("prompt", "")).rstrip() or "(empty)",
        "",
        "## Result",
        str(job.get("result", "")).rstrip() or "(empty)",
    ]
    if run:
        lines.extend(
            [
                "",
                "## Run",
                f"- status: {run.get('status', '')}",
                f"- harness: {run.get('harness', '') or '-'}",
                f"- workspace: {run.get('workspace', '') or '-'}",
                f"- updated_at: {run.get('updated_at', '') or '-'}",
                "",
                "### Summary",
                str(run.get("summary", "")).rstrip() or "(empty)",
            ]
        )
        commands_ran = run.get("commands") or []
        if commands_ran:
            lines.extend(["", "### Commands"])
            for command in commands_ran:
                lines.append(f"- `{command}`")
        verification = run.get("verification") or []
        if verification:
            lines.extend(["", "### Verification"])
            for item in verification:
                if isinstance(item, dict):
                    lines.append(
                        f"- {item.get('status', 'recorded')}: `{item.get('command', '')}`"
                    )
    lines.append("")
    return "\n".join(lines)


def _require_job(context_path: Path, job_id: str) -> AgentJob:
    job_id = job_id.strip()
    if not job_id:
        raise ValueError("job_id is required")
    job = AgentJobQueue(context_path).get(job_id)
    if job is None:
        raise FileNotFoundError(f"Agent job not found: {job_id}")
    return job


def _job_brief(job: AgentJob, *, now: datetime) -> dict[str, Any]:
    age = _age_seconds(job.updated_at or job.created_at, now)
    return {
        "id": job.id,
        "title": job.title,
        "status": job.status,
        "priority": job.priority,
        "assigned_to": job.assigned_to,
        "scope": job.scope,
        "expected_output": job.expected_output,
        "allow_destructive": job.allow_destructive,
        "needs_destructive_opt_in": job_needs_destructive_opt_in(job),
        "run_id": job.run_id,
        "created_at": job.created_at,
        "updated_at": job.updated_at,
        "age_seconds": age,
    }


def _sort_attention(jobs: list[AgentJob]) -> list[AgentJob]:
    return sorted(jobs, key=lambda item: (item.priority, item.updated_at or item.created_at))


def _base_command(context_path: Path, subcommand: str, job_id: str = "") -> str:
    command = f"afs agent-jobs {subcommand}"
    if job_id:
        command += f" {shlex.quote(job_id)}"
    if str(context_path):
        command += f" --context-root {shlex.quote(str(context_path.expanduser().resolve()))}"
    return command


def _safe_handoff_name(value: str) -> str:
    name = value.strip() or "agent-job-handoff.md"
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", name).strip("-._")
    if not name:
        name = "agent-job-handoff"
    if not name.endswith(".md"):
        name += ".md"
    return name


def _next_available_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for index in range(2, 1000):
        candidate = path.with_name(f"{stem}-{index}{suffix}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"Could not choose a unique handoff path under {path.parent}")


def _indent_block(text: str, *, prefix: str = "  ") -> list[str]:
    return [f"{prefix}{line}" if line else prefix.rstrip() for line in text.splitlines()]


def _age_seconds(value: str, now: datetime) -> float | None:
    parsed = _parse_iso(value)
    if parsed is None:
        return None
    return max(0.0, (now - parsed).total_seconds())


def _parse_iso(value: str) -> datetime | None:
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)
