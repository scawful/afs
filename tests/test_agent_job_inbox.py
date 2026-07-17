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
from afs.config import AFSConfig
from afs.context_layout import scaffold_v2
from afs.handoff import HandoffStore
from afs.manager import AFSManager
from afs.schema import GeneralConfig
from afs.session_bootstrap import build_session_bootstrap


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


def test_v2_agent_job_promotion_creates_visible_common_handoff(tmp_path: Path) -> None:
    context_path = tmp_path / ".context"
    scaffold_v2(context_path)
    queue = AgentJobQueue(context_path)
    job = queue.create("Completed index report", "summarize findings", priority=1)
    run = AgentRunStore(context_path).start("Completed index report", harness="codex")
    AgentRunStore(context_path).finish(run.id, status="done", summary="index is healthy")
    queue.move(job.id, "done", result="ready to review", run_id=run.id)

    promoted = promote_agent_job_to_handoff(
        context_path,
        job.id,
        handoff_name="completed-index-report.md",
    )

    artifact = Path(promoted["path"])
    assert artifact.is_file()
    assert artifact.is_relative_to(context_path / "memory" / "common" / "handoffs")
    assert "completed-index-report" in artifact.name
    assert promoted["handoff"]["title"] == "completed index report"
    packets = HandoffStore(context_path, scope_id="common").list()
    assert [packet.revision_id for packet in packets] == [
        promoted["handoff"]["revision_id"]
    ]
    assert packets[0].metadata == {
        "source": "afs.agent_job_inbox",
        "job_id": job.id,
    }

    manager = AFSManager(
        config=AFSConfig(general=GeneralConfig(context_root=context_path))
    )
    bootstrap = build_session_bootstrap(
        manager,
        context_path,
        record_event=False,
    )
    assert bootstrap["handoff"]["revision_id"] == packets[0].revision_id
    assert any(
        "ready to review" in item
        for item in bootstrap["handoff"]["accomplished"]
    )
