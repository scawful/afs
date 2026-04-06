"""Workspace analyst agent — periodic codebase health, git drift, and dependency analysis."""

from __future__ import annotations

import json
import logging
import subprocess
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..agent_context import ContextAwareAgent
from .base import (
    AgentResult,
    build_base_parser,
    configure_logging,
    emit_progress,
    emit_result,
    now_iso,
)
from .guardrails import GuardrailConfig, GuardrailedAgent

logger = logging.getLogger(__name__)

AGENT_NAME = "workspace-analyst"
AGENT_DESCRIPTION = (
    "Periodic codebase health analysis across configured workspace roots. "
    "Checks git drift, stale branches, uncommitted changes, build status, "
    "and dependency freshness."
)

AGENT_CAPABILITIES = {
    "tools": ["context.read", "context.write", "context.query", "git_status", "git_log"],
    "topics": ["workspace", "health", "git", "dependencies"],
    "mount_types": ["scratchpad", "knowledge"],
    "description": "Read-only workspace health analyzer with scratchpad write access",
}


def _resolve_default_scan_roots() -> list[Path]:
    """Resolve scan roots from AFS config or env, with no hardcoded fallback.

    Order of precedence:
        1. ``AFS_WORKSPACE_ROOTS`` env var (colon-separated paths)
        2. ``general.workspace_directories`` from the loaded AFS config
        3. Empty list — caller must pass ``--scan-roots``
    """
    import os as _os

    env_roots = _os.environ.get("AFS_WORKSPACE_ROOTS")
    if env_roots:
        return [Path(p).expanduser() for p in env_roots.split(":") if p.strip()]

    try:
        from ..config import load_config_model

        model, _ = load_config_model()
        ws_dirs = getattr(model.general, "workspace_directories", []) or []
        roots = [getattr(ws, "path", None) for ws in ws_dirs]
        return [Path(p).expanduser() for p in roots if p]
    except Exception:  # noqa: BLE001 — config is best-effort
        return []


# Backwards-compatible alias; resolved once at import time. Prefer
# ``_resolve_default_scan_roots()`` if you need fresh env/config lookup.
DEFAULT_SCAN_ROOTS = _resolve_default_scan_roots()


@dataclass
class RepoHealth:
    path: str
    name: str
    status: str = "ok"  # ok, dirty, stale, error
    branch: str = ""
    uncommitted_files: int = 0
    untracked_files: int = 0
    ahead: int = 0
    behind: int = 0
    last_commit_age_hours: float = 0.0
    stale_branches: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "name": self.name,
            "status": self.status,
            "branch": self.branch,
            "uncommitted_files": self.uncommitted_files,
            "untracked_files": self.untracked_files,
            "ahead": self.ahead,
            "behind": self.behind,
            "last_commit_age_hours": round(self.last_commit_age_hours, 1),
            "stale_branches": self.stale_branches,
            "notes": self.notes,
        }


def _run_git(repo: Path, *args: str, timeout: int = 10) -> str | None:
    """Run a git command in a repo, return stdout or None on failure."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo)] + list(args),
            capture_output=True, text=True, timeout=timeout,
        )
        return result.stdout.strip() if result.returncode == 0 else None
    except (subprocess.TimeoutExpired, OSError):
        return None


def _analyze_repo(repo_path: Path) -> RepoHealth | None:
    """Analyze a single git repository."""
    health = RepoHealth(path=str(repo_path), name=repo_path.name)

    # Check if it's actually a git repo
    if _run_git(repo_path, "rev-parse", "--git-dir") is None:
        return None

    # Current branch
    branch = _run_git(repo_path, "branch", "--show-current")
    health.branch = branch or "detached"

    # Status (porcelain for parsing)
    status_output = _run_git(repo_path, "status", "--porcelain")
    if status_output is not None:
        lines = [line for line in status_output.splitlines() if line.strip()]
        health.uncommitted_files = sum(1 for line in lines if not line.startswith("??"))
        health.untracked_files = sum(1 for line in lines if line.startswith("??"))
        if health.uncommitted_files > 0:
            health.status = "dirty"
            health.notes.append(f"{health.uncommitted_files} uncommitted files")

    # Ahead/behind (if tracking remote)
    ab_output = _run_git(repo_path, "rev-list", "--left-right", "--count", f"{health.branch}...@{{u}}")
    if ab_output:
        parts = ab_output.split()
        if len(parts) == 2:
            health.ahead = int(parts[0])
            health.behind = int(parts[1])
            if health.behind > 0:
                health.notes.append(f"{health.behind} commits behind remote")

    # Last commit age
    last_commit_ts = _run_git(repo_path, "log", "-1", "--format=%ct")
    if last_commit_ts:
        try:
            age_seconds = time.time() - float(last_commit_ts)
            health.last_commit_age_hours = age_seconds / 3600
            if health.last_commit_age_hours > 720:  # 30 days
                health.notes.append("Last commit > 30 days ago")
                if health.status == "ok":
                    health.status = "stale"
        except ValueError:
            pass

    # Stale branches (merged branches that aren't main/master/current)
    merged_output = _run_git(repo_path, "branch", "--merged")
    if merged_output:
        protected = {"main", "master", health.branch, "develop", "dev"}
        for line in merged_output.splitlines():
            branch_name = line.strip().lstrip("* ")
            if branch_name and branch_name not in protected:
                health.stale_branches.append(branch_name)
        if health.stale_branches:
            health.notes.append(f"{len(health.stale_branches)} merged branches could be cleaned")

    return health


def _find_repos(scan_roots: list[Path], max_depth: int = 3) -> list[Path]:
    """Find git repositories under scan roots."""
    repos = []
    seen = set()
    for root in scan_roots:
        if not root.exists():
            continue
        # Check if root itself is a repo
        if (root / ".git").exists():
            resolved = root.resolve()
            if resolved not in seen:
                repos.append(root)
                seen.add(resolved)
            continue
        # Scan children up to max_depth
        for depth in range(1, max_depth + 1):
            pattern = "/".join(["*"] * depth) + "/.git"
            for git_dir in root.glob(pattern):
                repo = git_dir.parent
                resolved = repo.resolve()
                if resolved not in seen:
                    repos.append(repo)
                    seen.add(resolved)
    return sorted(repos)


def _write_report(
    results: list[RepoHealth],
    output_dir: Path,
) -> Path:
    """Write health report to scratchpad."""
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "workspace_health.json"

    report = {
        "generated_at": now_iso(),
        "agent": AGENT_NAME,
        "summary": {
            "total_repos": len(results),
            "dirty": sum(1 for r in results if r.status == "dirty"),
            "stale": sum(1 for r in results if r.status == "stale"),
            "ok": sum(1 for r in results if r.status == "ok"),
            "total_uncommitted": sum(r.uncommitted_files for r in results),
            "total_behind": sum(r.behind for r in results),
            "repos_with_stale_branches": sum(1 for r in results if r.stale_branches),
        },
        "repos": [r.to_dict() for r in results],
        "attention_needed": [
            r.to_dict() for r in results
            if r.status != "ok" or r.behind > 0 or r.stale_branches
        ],
    }

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report_path


def build_parser():
    parser = build_base_parser(
        "Analyze workspace health across configured workspace roots."
    )
    parser.add_argument(
        "--scan-roots",
        nargs="*",
        help=(
            "Directories to scan for git repos. Defaults to "
            "$AFS_WORKSPACE_ROOTS (colon-separated) or "
            "general.workspace_directories from afs.toml."
        ),
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=3,
        help="Max directory depth to search for repos.",
    )
    parser.add_argument(
        "--context-root",
        default=None,
        help=(
            "Context root for writing reports. Defaults to "
            "general.context_root from afs.toml or ~/.context."
        ),
    )
    return parser


class _AnalystAgent(ContextAwareAgent):
    """Workspace analyst with context-aware decision making."""

    def query_prior_health(self) -> dict[str, Any] | None:
        """Check if a prior workspace health report exists in the context index."""
        results = self.query_context(
            "workspace health repos dirty stale uncommitted",
            mount_types=["scratchpad"],
            limit=1,
            include_content=True,
        )
        if not results:
            return None
        content = results[0].get("content_text", "") or results[0].get("content_excerpt", "")
        if not content:
            return None
        try:
            import json as _json
            return _json.loads(content)
        except (ValueError, TypeError):
            return None

    def detect_drift(self, current: list[RepoHealth], prior: dict[str, Any] | None) -> list[str]:
        """Compare current scan against prior results to detect meaningful drift."""
        if prior is None:
            return []
        drift_notes: list[str] = []
        prior_summary = prior.get("summary", {})
        cur_dirty = sum(1 for r in current if r.status == "dirty")
        cur_stale = sum(1 for r in current if r.status == "stale")
        prev_dirty = prior_summary.get("dirty", 0)
        prev_stale = prior_summary.get("stale", 0)
        if cur_dirty > prev_dirty:
            drift_notes.append(f"Drift: dirty repos {prev_dirty} → {cur_dirty}")
        if cur_stale > prev_stale:
            drift_notes.append(f"Drift: stale repos {prev_stale} → {cur_stale}")
        cur_uncommitted = sum(r.uncommitted_files for r in current)
        prev_uncommitted = prior_summary.get("total_uncommitted", 0)
        if cur_uncommitted > prev_uncommitted + 5:
            drift_notes.append(f"Drift: uncommitted files {prev_uncommitted} → {cur_uncommitted}")
        return drift_notes


def run(args) -> int:
    configure_logging(args.quiet)
    started_at = now_iso()
    start = time.time()

    # Initialize context-aware agent
    agent = _AnalystAgent()
    ctx = agent.load_context()
    if ctx:
        logger.info(
            "Context loaded: %d indexed, %d memory topics, %d active agents",
            ctx.index_total, len(ctx.memory_topics), len(ctx.active_agents),
        )

    guard = GuardrailedAgent(AGENT_NAME, config=GuardrailConfig(task_tier="background"))

    # Resolve scan roots — re-resolve from env/config each invocation so the
    # CLI honours updated config without a process restart.
    if args.scan_roots:
        scan_roots = [Path(p).expanduser() for p in args.scan_roots]
    else:
        scan_roots = _resolve_default_scan_roots()

    if not scan_roots:
        logger.warning(
            "No scan roots configured. Pass --scan-roots, set "
            "AFS_WORKSPACE_ROOTS, or add general.workspace_directories to "
            "afs.toml."
        )
        result = AgentResult(
            name=AGENT_NAME,
            status="warn",
            started_at=started_at,
            finished_at=now_iso(),
            duration_seconds=time.time() - start,
            metrics={},
            notes=["no scan roots configured"],
            payload={},
        )
        emit_result(result, output_path=None, force_stdout=False, pretty=False)
        return 0

    emit_progress(AGENT_NAME, "scan_start", f"Scanning {len(scan_roots)} roots")

    # Query prior health report to detect drift after scan
    prior_health = agent.query_prior_health()
    if prior_health:
        logger.info("Found prior health report for drift comparison")

    # Check mount freshness — report stale mounts alongside repo health
    mount_freshness = agent.get_mount_freshness()
    stale_mounts = [name for name, mf in mount_freshness.items() if mf.get("stale")]
    if stale_mounts:
        logger.info("Stale context mounts: %s", ", ".join(stale_mounts))

    # Find and analyze repos
    repos = _find_repos(scan_roots, max_depth=args.max_depth)
    logger.info("Found %d repositories to analyze", len(repos))

    results: list[RepoHealth] = []
    for repo in repos:
        if not guard.should_continue():
            logger.info("Hit iteration cap, stopping analysis")
            break
        health = _analyze_repo(repo)
        if health:
            results.append(health)

    # Track model usage for the analysis
    model = guard.resolve_model()
    guard.record_call(model.provider)

    # Detect drift against prior report
    drift_notes = agent.detect_drift(results, prior_health)
    if drift_notes:
        logger.info("Drift detected: %s", "; ".join(drift_notes))

    # Write report — resolve context_root from CLI, then config, then default.
    if args.context_root:
        context_root = Path(args.context_root).expanduser()
    else:
        try:
            from ..config import load_config_model

            model, _ = load_config_model()
            context_root = Path(model.general.context_root).expanduser()
        except Exception:  # noqa: BLE001
            context_root = Path.home() / ".context"
    report_path = _write_report(results, context_root / "scratchpad" / "afs_agents")

    emit_progress(AGENT_NAME, "scan_complete", f"Analyzed {len(results)} repos")

    dirty_count = sum(1 for r in results if r.status == "dirty")
    stale_count = sum(1 for r in results if r.status == "stale")

    notes = [
        f"Scanned {len(results)} repos across {len(scan_roots)} roots",
        f"Report written to {report_path}",
    ]
    notes.extend(drift_notes)

    result = AgentResult(
        name=AGENT_NAME,
        status="ok",
        started_at=started_at,
        finished_at=now_iso(),
        duration_seconds=time.time() - start,
        metrics={
            "repos_scanned": len(results),
            "repos_dirty": dirty_count,
            "repos_stale": stale_count,
            "repos_ok": len(results) - dirty_count - stale_count,
            "total_uncommitted_files": sum(r.uncommitted_files for r in results),
        },
        notes=notes,
        payload={
            "report_path": str(report_path),
            "attention_needed": [
                r.to_dict() for r in results
                if r.status != "ok" or r.behind > 0
            ],
            "quota_usage": guard.usage_summary(),
            "context_state": {
                "index_total": ctx.index_total if ctx else 0,
                "stale_mounts": stale_mounts,
                "active_agents": ctx.active_agents if ctx else [],
                "mount_freshness": mount_freshness,
            },
            "drift": drift_notes,
        },
    )
    emit_result(
        result,
        output_path=Path(args.output).expanduser() if args.output else None,
        force_stdout=bool(args.stdout),
        pretty=bool(args.pretty),
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
