from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from afs.agent_job_status import build_agent_job_status, format_agent_job_status
from afs.agent_jobs import AgentJobQueue
from afs.agent_runs import AgentRunStore


def test_agent_job_status_reports_worker_queue_and_watchdog(tmp_path: Path) -> None:
    context_path = tmp_path / ".context"
    (context_path / "items").mkdir(parents=True)
    (context_path / "scratchpad").mkdir(parents=True)
    queue = AgentJobQueue(context_path)
    queue.create("Delete all generated files", "delete all generated fixture files")
    queue.create("Review docs", "summarize docs")
    running = queue.claim(queue.create("Long running task", "inspect state").id, "worker")
    running.updated_at = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    queue._write(running)
    run = AgentRunStore(context_path).start("Recent failure", harness="worker")
    AgentRunStore(context_path).finish(run.id, status="failed", summary="failed smoke")

    payload = build_agent_job_status(
        context_path,
        stale_after_seconds=60,
        launchd_probe=lambda label: {
            "label": label,
            "installed": True,
            "available": True,
            "loaded": True,
            "running": True,
            "state": "running",
        },
    )

    assert payload["counts"]["queue"] == 2
    assert payload["counts"]["running"] == 1
    assert payload["queue"]["runnable"] == 1
    assert payload["queue"]["blocked_destructive"] == 1
    assert payload["running"]["stale"] == 1
    assert payload["recent_runs"]["failed"] == 1
    assert payload["watchdog"]["healthy"] is False
    check_names = {check["name"] for check in payload["watchdog"]["checks"]}
    assert "worker_launchd" in check_names
    assert "destructive_opt_in" in check_names
    assert "stale_running_jobs" in check_names
    assert "recent_run_failures" in check_names

    text = format_agent_job_status(payload)
    assert "blocked_destructive=1" in text
    assert "stale=1" in text
    assert "Recent failure" in text

    no_runs_payload = build_agent_job_status(
        context_path,
        recent_runs_limit=0,
        launchd_probe=lambda label: {
            "label": label,
            "installed": True,
            "available": True,
            "loaded": True,
            "running": True,
            "state": "running",
        },
    )
    assert no_runs_payload["recent_runs"]["total"] == 0


def test_agent_job_status_is_healthy_for_running_worker_and_empty_queue(tmp_path: Path) -> None:
    context_path = tmp_path / ".context"
    (context_path / "items").mkdir(parents=True)
    (context_path / "scratchpad").mkdir(parents=True)

    payload = build_agent_job_status(
        context_path,
        launchd_probe=lambda label: {
            "label": label,
            "installed": True,
            "available": True,
            "loaded": True,
            "running": True,
            "state": "running",
        },
    )

    assert payload["watchdog"]["healthy"] is True
    assert payload["watchdog"]["warning_count"] == 0
