"""Dashboard data exporter for barista (SketchyBar) and Cortex (macOS app).

Aggregates supervisor audit, workspace health, quota usage, pending approvals,
and mission state into a unified JSON dashboard and a one-line status string
consumable by external tools.
"""

from __future__ import annotations

import json
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from .base import (
    AgentResult,
    build_base_parser,
    configure_logging,
    emit_result,
    load_agent_config,
    now_iso,
)

AGENT_NAME = "dashboard-export"
AGENT_DESCRIPTION = (
    "Export unified dashboard JSON and status line for barista and Cortex consumption."
)

AGENT_CAPABILITIES = {
    "mount_types": ["scratchpad"],
    "topics": ["dashboard", "status"],
    "tools": ["context.read"],
    "description": "Aggregates agent, workspace, quota, and mission data into dashboard exports.",
}

# Default output locations
DEFAULT_DASHBOARD_JSON = Path("~/.config/afs/agents/dashboard.json")
DEFAULT_STATUS_TXT = Path("~/.config/afs/agents/status.txt")

# Default data source locations
DEFAULT_AGENT_OUTPUT_ROOT = Path("~/.config/afs/agents")
DEFAULT_SCRATCHPAD_ROOT = Path("~/.context/scratchpad/afs_agents")
DEFAULT_MISSIONS_ROOT = Path("~/.context/scratchpad/missions")


def _read_json(path: Path) -> Any:
    """Read and parse a JSON file, returning None on any error."""
    resolved = path.expanduser().resolve()
    if not resolved.exists():
        return None
    try:
        return json.loads(resolved.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _read_supervisor_audit(
    agent_output_root: Path,
    scratchpad_root: Path,
) -> dict[str, Any]:
    """Read the latest supervisor audit data."""
    # Try scratchpad agent_supervisor.json first (most recent full run)
    data = _read_json(scratchpad_root / "agent_supervisor.json")
    if isinstance(data, dict):
        audit = data.get("payload", {}).get("audit", {})
        if audit:
            return audit
        # Might be a flat audit dict
        if "counts" in data:
            return data

    # Try supervisor_last_run.json
    data = _read_json(scratchpad_root / "supervisor_last_run.json")
    if isinstance(data, dict):
        audit = data.get("payload", {}).get("audit", {})
        if audit:
            return audit
        if "counts" in data:
            return data

    return {}


def _read_workspace_health(scratchpad_root: Path) -> dict[str, Any]:
    """Read workspace health report summary."""
    data = _read_json(scratchpad_root / "workspace_health.json")
    if not isinstance(data, dict):
        return {}
    return data.get("summary", data)


def _read_quota(agent_output_root: Path) -> dict[str, Any]:
    """Read quota usage from the agents directory."""
    data = _read_json(agent_output_root / "quota.json")
    if not isinstance(data, dict):
        return {}
    return data


def _read_approvals(agent_output_root: Path) -> list[dict[str, Any]]:
    """Read pending approvals."""
    data = _read_json(agent_output_root / "approvals.json")
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    return []


def _scan_missions(missions_root: Path) -> dict[str, int]:
    """Count missions by status from TOML files in the missions directory."""
    resolved = missions_root.expanduser().resolve()
    counts: dict[str, int] = {"active": 0, "completed": 0, "blocked": 0}
    if not resolved.is_dir():
        return counts

    try:
        import tomllib
    except ModuleNotFoundError:
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ModuleNotFoundError:
            # Fall back: count .toml files as active if we cannot parse
            toml_files = list(resolved.glob("*.toml"))
            counts["active"] = len(toml_files)
            return counts

    for toml_path in sorted(resolved.glob("*.toml")):
        try:
            data = tomllib.loads(toml_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        mission = data.get("mission", data)
        status = str(mission.get("status", "pending")).lower()
        if status in ("completed", "done", "finished"):
            counts["completed"] += 1
        elif status in ("blocked", "failed", "error"):
            counts["blocked"] += 1
        else:
            counts["active"] += 1

    return counts


def _quota_status(quota: dict[str, Any]) -> str:
    """Derive a short quota status string.

    Returns 'ok', 'warn', or 'over' based on usage ratios.
    If quota data includes limit information, checks against it;
    otherwise returns 'ok' as a safe default.
    """
    for _provider, info in quota.items():
        if not isinstance(info, dict):
            continue
        limit = info.get("limit")
        used = info.get("used") or info.get("calls_today", 0)
        if limit and isinstance(limit, (int, float)) and limit > 0:
            ratio = used / limit
            if ratio >= 1.0:
                return "over"
            if ratio >= 0.8:
                return "warn"
    return "ok"


def _build_alerts(
    audit: dict[str, Any],
    workspace: dict[str, Any],
    quota: dict[str, Any],
    approvals: list[dict[str, Any]],
    missions: dict[str, int],
) -> list[str]:
    """Build human-readable alert strings for anything needing attention."""
    alerts: list[str] = []

    # Agent failures
    counts = audit.get("counts", {})
    failed = int(counts.get("failed", 0))
    if failed > 0:
        alerts.append(f"{failed} agent(s) in failed state")

    # Stale PIDs
    stale = audit.get("stale_pid_files", [])
    if stale:
        alerts.append(f"Stale PID files: {', '.join(stale)}")

    # Dirty repos
    dirty = int(workspace.get("dirty", 0))
    if dirty > 5:
        alerts.append(f"{dirty} repos have uncommitted changes")

    uncommitted = int(workspace.get("total_uncommitted", 0))
    if uncommitted > 50:
        alerts.append(f"{uncommitted} total uncommitted files across workspace")

    # Quota warnings
    qs = _quota_status(quota)
    if qs == "over":
        alerts.append("API quota exceeded")
    elif qs == "warn":
        alerts.append("API quota approaching limit")

    # Pending approvals
    pending = len([a for a in approvals if a.get("status") == "pending"])
    if pending > 0:
        alerts.append(f"{pending} approval(s) waiting for review")

    # Blocked missions
    blocked = missions.get("blocked", 0)
    if blocked > 0:
        alerts.append(f"{blocked} mission(s) blocked")

    return alerts


def build_dashboard(
    *,
    agent_output_root: Path | None = None,
    scratchpad_root: Path | None = None,
    missions_root: Path | None = None,
) -> dict[str, Any]:
    """Build the unified dashboard payload."""
    aor = (agent_output_root or DEFAULT_AGENT_OUTPUT_ROOT).expanduser().resolve()
    spr = (scratchpad_root or DEFAULT_SCRATCHPAD_ROOT).expanduser().resolve()
    msr = (missions_root or DEFAULT_MISSIONS_ROOT).expanduser().resolve()

    audit = _read_supervisor_audit(aor, spr)
    workspace_raw = _read_workspace_health(spr)
    quota = _read_quota(aor)
    approvals = _read_approvals(aor)
    missions = _scan_missions(msr)

    counts = audit.get("counts", {})
    running = int(counts.get("running", 0))
    failed = int(counts.get("failed", 0))
    configured = int(counts.get("configured", 0))

    total_repos = int(workspace_raw.get("total_repos", 0))
    dirty_repos = int(workspace_raw.get("dirty", 0))
    stale_repos = int(workspace_raw.get("stale", 0))
    uncommitted_files = int(workspace_raw.get("total_uncommitted", 0))

    # Build quota section — normalise to {provider: {used, limit}} shape
    quota_section: dict[str, dict[str, Any]] = {}
    for provider, info in quota.items():
        if not isinstance(info, dict):
            continue
        quota_section[provider] = {
            "used": info.get("used") or info.get("calls_today", 0),
            "limit": info.get("limit", 0),
        }

    pending_count = len([a for a in approvals if a.get("status") == "pending"])

    alerts = _build_alerts(audit, workspace_raw, quota, approvals, missions)

    return {
        "timestamp": now_iso(),
        "agents": {
            "running": running,
            "failed": failed,
            "total": configured,
        },
        "workspace": {
            "repos": total_repos,
            "dirty": dirty_repos,
            "stale": stale_repos,
            "uncommitted_files": uncommitted_files,
        },
        "quota": quota_section,
        "approvals_pending": pending_count,
        "missions": missions,
        "alerts": alerts,
    }


def format_status_line(dashboard: dict[str, Any]) -> str:
    """Format a one-line status string for barista shell consumption.

    Format: ``agents:8/8 | repos:4dirty | quota:ok | approvals:1``
    """
    agents = dashboard.get("agents", {})
    running = agents.get("running", 0)
    total = agents.get("total", 0)

    workspace = dashboard.get("workspace", {})
    dirty = workspace.get("dirty", 0)

    quota = dashboard.get("quota", {})
    qs = _quota_status(quota)

    pending = dashboard.get("approvals_pending", 0)

    return f"agents:{running}/{total} | repos:{dirty}dirty | quota:{qs} | approvals:{pending}"


def export_dashboard(
    dashboard: dict[str, Any],
    *,
    dashboard_path: Path | None = None,
    status_path: Path | None = None,
) -> tuple[Path, Path]:
    """Write dashboard JSON and status line to disk."""
    dp = (dashboard_path or DEFAULT_DASHBOARD_JSON).expanduser().resolve()
    sp = (status_path or DEFAULT_STATUS_TXT).expanduser().resolve()

    dp.parent.mkdir(parents=True, exist_ok=True)
    sp.parent.mkdir(parents=True, exist_ok=True)

    dp.write_text(json.dumps(dashboard, indent=2) + "\n", encoding="utf-8")
    sp.write_text(format_status_line(dashboard) + "\n", encoding="utf-8")

    return dp, sp


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def build_parser():
    parser = build_base_parser(AGENT_DESCRIPTION)
    parser.add_argument(
        "--agent-output-root",
        help="Override agent output root (default: ~/.config/afs/agents).",
    )
    parser.add_argument(
        "--scratchpad-root",
        help="Override scratchpad root (default: ~/.context/scratchpad/afs_agents).",
    )
    parser.add_argument(
        "--missions-root",
        help="Override missions root (default: ~/.context/scratchpad/missions).",
    )
    parser.add_argument(
        "--dashboard-path",
        help="Override dashboard JSON output path.",
    )
    parser.add_argument(
        "--status-path",
        help="Override status line output path.",
    )
    return parser


def run(args) -> int:
    configure_logging(args.quiet)
    _config = load_agent_config(args.config)

    started_at = now_iso()
    start = time.monotonic()

    dashboard = build_dashboard(
        agent_output_root=Path(args.agent_output_root) if args.agent_output_root else None,
        scratchpad_root=Path(args.scratchpad_root) if args.scratchpad_root else None,
        missions_root=Path(args.missions_root) if args.missions_root else None,
    )

    dp, sp = export_dashboard(
        dashboard,
        dashboard_path=Path(args.dashboard_path) if args.dashboard_path else None,
        status_path=Path(args.status_path) if args.status_path else None,
    )

    duration = time.monotonic() - start

    result = AgentResult(
        name=AGENT_NAME,
        status="ok",
        started_at=started_at,
        finished_at=now_iso(),
        duration_seconds=duration,
        metrics={
            "agents_running": dashboard["agents"]["running"],
            "agents_total": dashboard["agents"]["total"],
            "repos_dirty": dashboard["workspace"]["dirty"],
            "approvals_pending": dashboard["approvals_pending"],
            "alert_count": len(dashboard["alerts"]),
        },
        notes=dashboard["alerts"][:5],
        payload={
            "dashboard_path": str(dp),
            "status_path": str(sp),
            "dashboard": dashboard,
        },
    )

    output_path = Path(args.output) if args.output else None
    emit_result(
        result,
        output_path=output_path,
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
