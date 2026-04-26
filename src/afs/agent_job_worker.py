"""Local worker for markdown-backed agent jobs."""

from __future__ import annotations

import os
import shlex
import subprocess
from dataclasses import dataclass
from pathlib import Path

from .agent_jobs import AgentJob, AgentJobQueue
from .agent_runs import AgentRunStore
from .context_paths import resolve_mount_root
from .models import MountType

DESTRUCTIVE_MARKERS = (
    "rm -rf",
    "rm -fr",
    "delete everything",
    "delete all",
    "wipe",
    "erase",
    "git reset --hard",
    "git clean -fd",
    "force push",
    "force-push",
    "drop database",
    "truncate table",
    "destroy",
)


@dataclass(frozen=True)
class AgentJobWorkerResult:
    job_id: str
    title: str
    status: str
    command: str = ""
    run_id: str = ""
    exit_code: int | None = None
    result: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "job_id": self.job_id,
            "title": self.title,
            "status": self.status,
            "command": self.command,
            "run_id": self.run_id,
            "exit_code": self.exit_code,
            "result": self.result,
        }


def _tail(text: str, limit: int = 4000) -> str:
    stripped = text.strip()
    if len(stripped) <= limit:
        return stripped
    return stripped[-limit:]


def _render_command(command: str, job: AgentJob, prompt_file: Path) -> str:
    rendered = command
    replacements = {
        "{job_id}": shlex.quote(job.id),
        "{title}": shlex.quote(job.title),
        "{prompt}": shlex.quote(job.prompt),
        "{prompt_file}": shlex.quote(str(prompt_file)),
    }
    for token, value in replacements.items():
        rendered = rendered.replace(token, value)
    return rendered


def job_needs_destructive_opt_in(job: AgentJob) -> bool:
    text = f"{job.title}\n{job.prompt}\n{job.scope}".lower()
    return any(marker in text for marker in DESTRUCTIVE_MARKERS)


def worker_policy_prompt() -> str:
    return "\n".join(
        [
            "AFS background worker policy:",
            "- You may inspect, edit, build, test, and document within the requested scope.",
            "- Prefer the smallest useful change and verify it.",
            "- Do not delete broad directories, rewrite history, force-push, wipe data, or run destructive cleanup unless the job explicitly says destructive actions are allowed.",
            "- If a requested destructive action is necessary but not explicitly allowed, stop and record what approval or explicit opt-in is needed.",
            "",
        ]
    )


def run_agent_job_worker(
    context_path: Path,
    *,
    agent_name: str,
    command: str,
    workspace: Path | None = None,
    limit: int = 1,
    timeout: int | None = None,
    dry_run: bool = False,
    allow_destructive: bool = False,
) -> list[AgentJobWorkerResult]:
    if not agent_name.strip():
        raise ValueError("agent_name is required")
    if not command.strip() and not dry_run:
        raise ValueError("command is required unless dry_run is set")
    if limit < 1:
        return []

    queue = AgentJobQueue(context_path)
    run_store = AgentRunStore(context_path)
    prompt_root = resolve_mount_root(context_path, MountType.SCRATCHPAD) / "agent_job_prompts"
    workspace_path = (workspace or Path.cwd()).expanduser().resolve()
    results: list[AgentJobWorkerResult] = []
    runnable_processed = 0

    for job in queue.list(status="queue"):
        if runnable_processed >= limit:
            break
        prompt_file = prompt_root / f"{job.id}.md"
        rendered = _render_command(command, job, prompt_file) if command.strip() else ""
        if (
            job_needs_destructive_opt_in(job)
            and not job.allow_destructive
            and not allow_destructive
        ):
            results.append(
                AgentJobWorkerResult(
                    job_id=job.id,
                    title=job.title,
                    status="skipped_destructive",
                    command=rendered,
                    result="background worker skipped obvious destructive request; rerun with --allow-destructive or create the job with --allow-destructive",
                )
            )
            continue
        if dry_run:
            results.append(
                AgentJobWorkerResult(
                    job_id=job.id,
                    title=job.title,
                    status="would_run",
                    command=rendered,
                    result="queued job would be claimed and executed",
                )
            )
            runnable_processed += 1
            continue

        prompt_file.parent.mkdir(parents=True, exist_ok=True)
        prompt_file.write_text(worker_policy_prompt() + job.prompt.rstrip() + "\n", encoding="utf-8")
        claimed = queue.claim(job.id, agent_name)
        run = run_store.start(
            claimed.title,
            harness=agent_name,
            workspace=str(workspace_path),
            prompt=claimed.prompt,
        )
        env = os.environ.copy()
        env.update(
            {
                "AFS_AGENT_JOB_ID": claimed.id,
                "AFS_AGENT_JOB_TITLE": claimed.title,
                "AFS_AGENT_JOB_PROMPT": claimed.prompt,
                "AFS_AGENT_JOB_PROMPT_FILE": str(prompt_file),
                "AFS_AGENT_RUN_ID": run.id,
            }
        )
        try:
            completed = subprocess.run(
                rendered,
                shell=True,
                cwd=str(workspace_path),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
            )
            status = "done" if completed.returncode == 0 else "failed"
            result_text = _tail(completed.stdout or completed.stderr or f"exit code {completed.returncode}")
            queue.move(claimed.id, status, result=result_text)
            run_store.finish(
                run.id,
                status=status,
                summary=result_text,
                commands=[rendered],
                verification=[
                    {
                        "command": rendered,
                        "status": "passed" if completed.returncode == 0 else "failed",
                        "exit_code": completed.returncode,
                    }
                ],
            )
            results.append(
                AgentJobWorkerResult(
                    job_id=claimed.id,
                    title=claimed.title,
                    status=status,
                    command=rendered,
                    run_id=run.id,
                    exit_code=completed.returncode,
                    result=result_text,
                )
            )
            runnable_processed += 1
        except Exception as exc:
            result_text = str(exc)
            queue.move(claimed.id, "failed", result=result_text)
            run_store.finish(
                run.id,
                status="failed",
                summary=result_text,
                commands=[rendered],
                verification=[{"command": rendered, "status": "failed", "error": result_text}],
            )
            results.append(
                AgentJobWorkerResult(
                    job_id=claimed.id,
                    title=claimed.title,
                    status="failed",
                    command=rendered,
                    run_id=run.id,
                    result=result_text,
                )
            )
            runnable_processed += 1

    return results
