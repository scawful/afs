from __future__ import annotations

from pathlib import Path

import pytest

from afs.agent_jobs import AgentJobQueue


def test_agent_job_markdown_queue_lifecycle(tmp_path: Path) -> None:
    ctx = tmp_path / ".context"
    (ctx / "items").mkdir(parents=True)
    queue = AgentJobQueue(ctx)

    job = queue.create(
        "Review stale docs",
        "Find stale model aliases.",
        created_by="codex",
        scope="docs/",
        expected_output="findings with paths",
        allow_destructive=True,
        dedupe_key="seed:repo-maintenance:docs:once",
        priority=2,
    )
    assert job.status == "queue"
    assert job.allow_destructive is True
    assert job.dedupe_key == "seed:repo-maintenance:docs:once"
    assert (ctx / "items" / "agent_jobs" / "queue" / f"{job.id}.md").exists()

    claimed = queue.claim(job.id, "reviewer")
    assert claimed.status == "running"
    assert claimed.assigned_to == "reviewer"
    assert not (ctx / "items" / "agent_jobs" / "queue" / f"{job.id}.md").exists()
    assert (ctx / "items" / "agent_jobs" / "running" / f"{job.id}.md").exists()

    done = queue.move(job.id, "done", result="no findings", run_id="run-1")
    assert done.status == "done"
    assert done.allow_destructive is True
    assert done.dedupe_key == "seed:repo-maintenance:docs:once"
    assert done.run_id == "run-1"
    assert done.result == "no findings"
    assert queue.get(job.id).status == "done"  # type: ignore[union-attr]

    archived = queue.move(job.id, "archived")
    assert archived.status == "archived"
    assert queue.get(job.id).status == "archived"  # type: ignore[union-attr]


def test_agent_job_cannot_claim_nonqueued(tmp_path: Path) -> None:
    ctx = tmp_path / ".context"
    (ctx / "items").mkdir(parents=True)
    queue = AgentJobQueue(ctx)
    job = queue.create("Run tests", "pytest")
    queue.claim(job.id, "worker")

    with pytest.raises(ValueError, match="Cannot claim"):
        queue.claim(job.id, "other")
