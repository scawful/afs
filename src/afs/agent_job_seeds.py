"""Safe, idempotent background job seeding profiles."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .agent_jobs import AgentJob, AgentJobQueue

SEED_CADENCES = ("daily", "weekly", "once")


@dataclass(frozen=True)
class AgentJobSeed:
    key: str
    title: str
    prompt: str
    scope: str
    expected_output: str
    priority: int = 7


@dataclass(frozen=True)
class AgentJobSeedResult:
    key: str
    dedupe_key: str
    title: str
    status: str
    job_id: str = ""
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "dedupe_key": self.dedupe_key,
            "title": self.title,
            "status": self.status,
            "job_id": self.job_id,
            "reason": self.reason,
        }


REPORT_ONLY_POLICY = "\n".join(
    [
        "This is a report-only AFS maintenance job.",
        "Do not edit files, commit, push, rewrite history, or run cleanup commands.",
        "Inspect the repository state and produce a concise report with paths, evidence, and suggested next steps.",
    ]
)


REPO_MAINTENANCE_SEEDS = (
    AgentJobSeed(
        key="stale-docs-reference-scan",
        title="Scan stale docs and references",
        scope="README.md docs/ .context/ AGENTS.md",
        expected_output="Findings with file paths, stale text, suggested replacement, and confidence.",
        prompt=(
            "Scan repository docs, AFS handoffs, and agent instructions for stale references "
            "to commands, model names, MCP tool names, harness behavior, or workflow claims."
        ),
    ),
    AgentJobSeed(
        key="skill-drift-scan",
        title="Scan skill and harness drift",
        scope="skills/ configs/ scripts/ docs/",
        expected_output="Drift report with paths, missing copies, obsolete instructions, and suggested fixes.",
        prompt=(
            "Compare shared skill references, harness exports, and local harness documentation "
            "for drift, missing paths, or obsolete behavior claims."
        ),
    ),
    AgentJobSeed(
        key="mcp-tool-name-drift-scan",
        title="Scan MCP and tool name drift",
        scope="src/afs/mcp_server.py docs/ tests/",
        expected_output="Mismatch report covering CLI/MCP parity, stale names, and missing docs.",
        prompt=(
            "Scan MCP tools, CLI docs, tests, and agent-facing references for stale tool names "
            "or missing parity between CLI and MCP surfaces."
        ),
    ),
    AgentJobSeed(
        key="todo-fixme-summary",
        title="Summarize TODO and FIXME items",
        scope="src/ tests/ scripts/ docs/",
        expected_output="Grouped TODO/FIXME/HACK summary by path, risk, and likely owner area.",
        prompt=(
            "Find actionable TODO, FIXME, HACK, or follow-up comments in source, tests, scripts, "
            "and docs. Group them by risk and maintenance area."
        ),
    ),
    AgentJobSeed(
        key="verification-suggestion-scan",
        title="Suggest focused verification commands",
        scope="pyproject.toml tests/ scripts/ src/",
        expected_output="Short list of fastest useful verification commands and when to run each.",
        prompt=(
            "Inspect repository metadata, test layout, scripts, and recent maintenance surfaces. "
            "Suggest the fastest useful verification commands for common AFS changes."
        ),
    ),
    AgentJobSeed(
        key="uncommitted-change-review",
        title="Review uncommitted changes",
        scope=".",
        expected_output="Findings-first summary of uncommitted changes, risk, and missing verification.",
        prompt=(
            "Review current uncommitted changes for unrelated files, missing tests, risky edits, "
            "or documentation gaps. If the tree is clean, report that directly."
        ),
        priority=6,
    ),
)

SEED_PROFILES: dict[str, tuple[AgentJobSeed, ...]] = {
    "repo-maintenance": REPO_MAINTENANCE_SEEDS,
}


def seed_agent_jobs(
    context_path: Path,
    *,
    profile: str = "repo-maintenance",
    cadence: str = "daily",
    created_by: str = "agent-job-seed",
    dry_run: bool = False,
    force: bool = False,
    now: datetime | None = None,
) -> dict[str, Any]:
    if profile not in SEED_PROFILES:
        raise ValueError(f"Unknown seed profile: {profile}")
    if cadence not in SEED_CADENCES:
        raise ValueError(f"Invalid seed cadence: {cadence}")

    queue = AgentJobQueue(context_path)
    existing_jobs = queue.list()
    current = now or datetime.now(timezone.utc)
    results: list[AgentJobSeedResult] = []
    for seed in SEED_PROFILES[profile]:
        dedupe_key = _dedupe_key(profile, seed.key, cadence, current)
        open_match = _find_open_match(existing_jobs, profile=profile, seed_key=seed.key)
        exact_match = _find_exact_match(existing_jobs, dedupe_key)
        if not force and open_match is not None:
            results.append(
                AgentJobSeedResult(
                    key=seed.key,
                    dedupe_key=dedupe_key,
                    title=seed.title,
                    status="skipped",
                    job_id=open_match.id,
                    reason=f"open job already exists: {open_match.status}",
                )
            )
            continue
        if not force and exact_match is not None:
            results.append(
                AgentJobSeedResult(
                    key=seed.key,
                    dedupe_key=dedupe_key,
                    title=seed.title,
                    status="skipped",
                    job_id=exact_match.id,
                    reason="dedupe key already exists",
                )
            )
            continue
        if dry_run:
            results.append(
                AgentJobSeedResult(
                    key=seed.key,
                    dedupe_key=dedupe_key,
                    title=seed.title,
                    status="would_create",
                    reason="dry run",
                )
            )
            continue
        job = queue.create(
            seed.title,
            _render_prompt(seed),
            priority=seed.priority,
            created_by=created_by,
            scope=seed.scope,
            expected_output=seed.expected_output,
            allow_destructive=False,
            dedupe_key=dedupe_key,
        )
        existing_jobs.append(job)
        results.append(
            AgentJobSeedResult(
                key=seed.key,
                dedupe_key=dedupe_key,
                title=seed.title,
                status="created",
                job_id=job.id,
            )
        )
    return {
        "profile": profile,
        "cadence": cadence,
        "dry_run": dry_run,
        "force": force,
        "created": sum(1 for result in results if result.status == "created"),
        "skipped": sum(1 for result in results if result.status == "skipped"),
        "would_create": sum(1 for result in results if result.status == "would_create"),
        "results": [result.to_dict() for result in results],
    }


def _render_prompt(seed: AgentJobSeed) -> str:
    return "\n\n".join([REPORT_ONLY_POLICY, f"Task: {seed.prompt}", f"Expected output: {seed.expected_output}"])


def _dedupe_key(profile: str, seed_key: str, cadence: str, now: datetime) -> str:
    if cadence == "once":
        window = "once"
    elif cadence == "weekly":
        iso = now.isocalendar()
        window = f"{iso.year}-W{iso.week:02d}"
    else:
        window = now.date().isoformat()
    return f"seed:{profile}:{seed_key}:{window}"


def _find_open_match(
    jobs: list[AgentJob],
    *,
    profile: str,
    seed_key: str,
) -> AgentJob | None:
    prefix = f"seed:{profile}:{seed_key}:"
    for job in jobs:
        if job.status in {"queue", "running"} and job.dedupe_key.startswith(prefix):
            return job
    return None


def _find_exact_match(jobs: list[AgentJob], dedupe_key: str) -> AgentJob | None:
    for job in jobs:
        if job.dedupe_key == dedupe_key:
            return job
    return None
