from __future__ import annotations

import shlex
import sys
from pathlib import Path

from afs.agent_job_worker import run_agent_job_worker
from afs.agent_jobs import AgentJobQueue
from afs.agent_runs import AgentRunStore


def test_agent_job_worker_runs_command_and_records_run(tmp_path: Path) -> None:
    context_path = tmp_path / ".context"
    (context_path / "items").mkdir(parents=True)
    (context_path / "scratchpad").mkdir(parents=True)
    job = AgentJobQueue(context_path).create(
        "Summarize docs",
        "Write a concise summary.",
        created_by="tester",
    )

    output_path = tmp_path / "worker-output.txt"
    script = (
        "import os, pathlib; "
        f"pathlib.Path({str(output_path)!r}).write_text(os.environ['AFS_AGENT_JOB_PROMPT'], encoding='utf-8'); "
        "print('ok')"
    )
    command = f"{shlex.quote(sys.executable)} -c {shlex.quote(script)}"

    results = run_agent_job_worker(
        context_path,
        agent_name="codex-worker",
        command=command,
        workspace=tmp_path,
    )

    assert len(results) == 1
    assert results[0].status == "done"
    assert results[0].job_id == job.id
    assert output_path.read_text(encoding="utf-8") == "Write a concise summary."
    assert AgentJobQueue(context_path).get(job.id).status == "done"  # type: ignore[union-attr]
    runs = AgentRunStore(context_path).list()
    assert len(runs) == 1
    assert runs[0].harness == "codex-worker"
    assert runs[0].status == "done"


def test_agent_job_worker_dry_run_does_not_claim(tmp_path: Path) -> None:
    context_path = tmp_path / ".context"
    (context_path / "items").mkdir(parents=True)
    (context_path / "scratchpad").mkdir(parents=True)
    job = AgentJobQueue(context_path).create("Review", "Check state.")

    results = run_agent_job_worker(
        context_path,
        agent_name="reviewer",
        command="echo {job_id} {prompt_file}",
        dry_run=True,
    )

    assert results[0].status == "would_run"
    assert job.id in results[0].command
    assert AgentJobQueue(context_path).get(job.id).status == "queue"  # type: ignore[union-attr]


def test_agent_job_worker_skips_obvious_destructive_jobs_without_opt_in(tmp_path: Path) -> None:
    context_path = tmp_path / ".context"
    (context_path / "items").mkdir(parents=True)
    (context_path / "scratchpad").mkdir(parents=True)
    job = AgentJobQueue(context_path).create("Delete all temp files", "run rm -rf /tmp/project")

    results = run_agent_job_worker(
        context_path,
        agent_name="worker",
        command="echo should-not-run",
        workspace=tmp_path,
    )

    assert results[0].status == "skipped_destructive"
    assert AgentJobQueue(context_path).get(job.id).status == "queue"  # type: ignore[union-attr]
    assert AgentRunStore(context_path).list() == []


def test_agent_job_worker_runs_destructive_job_with_explicit_opt_in(tmp_path: Path) -> None:
    context_path = tmp_path / ".context"
    (context_path / "items").mkdir(parents=True)
    (context_path / "scratchpad").mkdir(parents=True)
    job = AgentJobQueue(context_path).create(
        "Delete generated fixture",
        "delete all generated fixture files",
        allow_destructive=True,
    )

    results = run_agent_job_worker(
        context_path,
        agent_name="worker",
        command="printf ok",
        workspace=tmp_path,
    )

    assert results[0].status == "done"
    assert AgentJobQueue(context_path).get(job.id).status == "done"  # type: ignore[union-attr]


def test_agent_job_worker_skips_destructive_jobs_without_blocking_safe_jobs(tmp_path: Path) -> None:
    context_path = tmp_path / ".context"
    (context_path / "items").mkdir(parents=True)
    (context_path / "scratchpad").mkdir(parents=True)
    queue = AgentJobQueue(context_path)
    destructive = queue.create("Delete all fixtures", "delete all generated fixture files")
    safe = queue.create("Review docs", "summarize docs")

    results = run_agent_job_worker(
        context_path,
        agent_name="worker",
        command="printf safe-ok",
        workspace=tmp_path,
        limit=1,
    )

    assert [result.status for result in results] == ["skipped_destructive", "done"]
    assert AgentJobQueue(context_path).get(destructive.id).status == "queue"  # type: ignore[union-attr]
    assert AgentJobQueue(context_path).get(safe.id).status == "done"  # type: ignore[union-attr]
