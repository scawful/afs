from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from afs.agent_job_seeds import seed_agent_jobs
from afs.agent_jobs import AgentJobQueue


def test_seed_agent_jobs_creates_report_only_repo_maintenance_jobs(tmp_path: Path) -> None:
    context_path = tmp_path / ".context"
    (context_path / "items").mkdir(parents=True)

    payload = seed_agent_jobs(
        context_path,
        profile="repo-maintenance",
        cadence="daily",
        now=datetime(2026, 4, 26, tzinfo=timezone.utc),
    )

    assert payload["created"] == 6
    assert payload["skipped"] == 0
    jobs = AgentJobQueue(context_path).list(status="queue")
    assert len(jobs) == 6
    assert all(job.allow_destructive is False for job in jobs)
    assert all(job.dedupe_key.startswith("seed:repo-maintenance:") for job in jobs)
    assert any("report-only AFS maintenance job" in job.prompt for job in jobs)


def test_seed_agent_jobs_is_idempotent_by_dedupe_key(tmp_path: Path) -> None:
    context_path = tmp_path / ".context"
    (context_path / "items").mkdir(parents=True)
    now = datetime(2026, 4, 26, tzinfo=timezone.utc)

    first = seed_agent_jobs(context_path, now=now)
    second = seed_agent_jobs(context_path, now=now)

    assert first["created"] == 6
    assert second["created"] == 0
    assert second["skipped"] == 6
    assert len(AgentJobQueue(context_path).list(status="queue")) == 6


def test_seed_agent_jobs_does_not_reseed_when_open_job_exists_across_cadence_window(
    tmp_path: Path,
) -> None:
    context_path = tmp_path / ".context"
    (context_path / "items").mkdir(parents=True)

    seed_agent_jobs(context_path, now=datetime(2026, 4, 26, tzinfo=timezone.utc))
    next_day = seed_agent_jobs(context_path, now=datetime(2026, 4, 27, tzinfo=timezone.utc))

    assert next_day["created"] == 0
    assert next_day["skipped"] == 6
    assert all("open job already exists" in result["reason"] for result in next_day["results"])


def test_seed_agent_jobs_can_force_reseed(tmp_path: Path) -> None:
    context_path = tmp_path / ".context"
    (context_path / "items").mkdir(parents=True)

    seed_agent_jobs(context_path, now=datetime(2026, 4, 26, tzinfo=timezone.utc))
    forced = seed_agent_jobs(
        context_path,
        force=True,
        now=datetime(2026, 4, 26, tzinfo=timezone.utc),
    )

    assert forced["created"] == 6
    assert len(AgentJobQueue(context_path).list(status="queue")) == 12


def test_seed_agent_jobs_dry_run_does_not_create_jobs(tmp_path: Path) -> None:
    context_path = tmp_path / ".context"
    (context_path / "items").mkdir(parents=True)

    payload = seed_agent_jobs(context_path, dry_run=True)

    assert payload["created"] == 0
    assert payload["would_create"] == 6
    assert AgentJobQueue(context_path).list(status="queue") == []


def test_seed_agent_jobs_rejects_unknown_profile(tmp_path: Path) -> None:
    context_path = tmp_path / ".context"
    (context_path / "items").mkdir(parents=True)

    with pytest.raises(ValueError, match="Unknown seed profile"):
        seed_agent_jobs(context_path, profile="not-real")
