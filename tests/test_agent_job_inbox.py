from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

from afs.agent_job_inbox import (
    archive_agent_job,
    build_agent_job_inbox,
    format_agent_job_inbox,
    format_agent_job_review,
    promote_agent_job_to_handoff,
    review_agent_job,
)
from afs.agent_jobs import AgentJobQueue
from afs.agent_runs import AgentRunStore


def test_agent_job_inbox_collects_reviewable_failed_stale_and_blocked_jobs(
    tmp_path: Path,
) -> None:
    context_path = tmp_path / ".context"
    (context_path / "items").mkdir(parents=True)
    (context_path / "scratchpad").mkdir(parents=True)
    queue = AgentJobQueue(context_path)

    done = queue.create("Completed report", "summarize findings", priority=1)
    run = AgentRunStore(context_path).start("Completed report", harness="codex")
    AgentRunStore(context_path).finish(
        run.id,
        status="done",
        summary="all good",
        commands=["pytest -q"],
        verification=[{"command": "pytest -q", "status": "passed"}],
    )
    queue.move(done.id, "done", result="ready to review", run_id=run.id)

    failed = queue.create("Failed report", "run smoke", priority=2)
    queue.move(failed.id, "failed", result="smoke failed")

    stale = queue.claim(queue.create("Long runner", "inspect state").id, "worker")
    stale.updated_at = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
    queue._write(stale)

    queue.create("Delete all fixtures", "run rm -rf ./fixtures")
    archived = queue.create("Old report", "already handled")
    archive_agent_job(context_path, archived.id)

    payload = build_agent_job_inbox(context_path, stale_after_seconds=60)

    assert payload["attention_count"] == 4
    assert payload["counts"]["archived"] == 1
    assert payload["reviewable"]["items"][0]["id"] == done.id
    assert payload["failed"]["items"][0]["id"] == failed.id
    assert payload["stale_running"]["items"][0]["id"] == stale.id
    assert payload["blocked_destructive"]["total"] == 1
    assert payload["command"].startswith("afs agent-jobs inbox ")
    text = format_agent_job_inbox(payload)
    assert "review_ready: 1" in text
    assert f"review: afs agent-jobs review {done.id}" in text


def test_agent_job_review_archive_and_promote_to_handoff(tmp_path: Path) -> None:
    context_path = tmp_path / ".context"
    (context_path / "items").mkdir(parents=True)
    (context_path / "scratchpad").mkdir(parents=True)
    queue = AgentJobQueue(context_path)
    job = queue.create("Completed report", "summarize findings", priority=1)
    run = AgentRunStore(context_path).start("Completed report", harness="codex")
    AgentRunStore(context_path).finish(run.id, status="done", summary="summary text")
    queue.move(job.id, "done", result="ready to review", run_id=run.id)

    review = review_agent_job(context_path, job.id)

    assert review["job"]["run_id"] == run.id
    assert review["run"]["summary"] == "summary text"
    assert "archive" in review["commands"]
    assert "summary text" in format_agent_job_review(review)

    promoted = promote_agent_job_to_handoff(
        context_path,
        job.id,
        handoff_name="completed-report.md",
    )
    handoff = Path(promoted["path"])
    assert handoff.exists()
    handoff_text = handoff.read_text(encoding="utf-8")
    assert "Completed report" in handoff_text
    assert "ready to review" in handoff_text

    archived = archive_agent_job(context_path, job.id)
    assert archived.status == "archived"
    assert (context_path / "items" / "agent_jobs" / "archived" / f"{job.id}.md").exists()
    assert build_agent_job_inbox(context_path)["attention_count"] == 0
