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
        priority=2,
    )
    assert job.status == "queue"
    assert (ctx / "items" / "agent_jobs" / "queue" / f"{job.id}.md").exists()

    claimed = queue.claim(job.id, "reviewer")
    assert claimed.status == "running"
    assert claimed.assigned_to == "reviewer"
    assert not (ctx / "items" / "agent_jobs" / "queue" / f"{job.id}.md").exists()
    assert (ctx / "items" / "agent_jobs" / "running" / f"{job.id}.md").exists()

    done = queue.move(job.id, "done", result="no findings")
    assert done.status == "done"
    assert done.result == "no findings"
    assert queue.get(job.id).status == "done"  # type: ignore[union-attr]


def test_agent_job_cannot_claim_nonqueued(tmp_path: Path) -> None:
    ctx = tmp_path / ".context"
    (ctx / "items").mkdir(parents=True)
    queue = AgentJobQueue(ctx)
    job = queue.create("Run tests", "pytest")
    queue.claim(job.id, "worker")

    with pytest.raises(ValueError, match="Cannot claim"):
        queue.claim(job.id, "other")
