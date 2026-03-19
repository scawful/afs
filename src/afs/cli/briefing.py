"""AFS morning briefing — synthesize git velocity, task state, and project health.

Usage:
    afs briefing              # full morning briefing
    afs briefing --short      # compact one-screen summary
    afs briefing --json       # machine-readable for IDE integration
    afs briefing --no-gws     # skip Google Workspace integration
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Project registry — repos to track
# ---------------------------------------------------------------------------

PROJECTS: dict[str, dict[str, Any]] = {
    # Hobby
    "yaze": {"path": "~/src/hobby/yaze", "category": "hobby"},
    "oracle-of-secrets": {"path": "~/src/hobby/oracle-of-secrets", "category": "hobby"},
    "mesen2-oos": {"path": "~/src/hobby/mesen2-oos", "category": "hobby"},
    "z3dk": {"path": "~/src/hobby/z3dk", "category": "hobby"},
    # Lab
    "afs": {"path": "~/src/lab/afs", "category": "lab"},
    "afs-scawful": {"path": "~/src/lab/afs-scawful", "category": "lab"},
    "echoflow": {"path": "~/src/lab/echoflow", "category": "lab"},
    "barista": {"path": "~/src/lab/barista", "category": "lab"},
    "halext-org": {"path": "~/src/lab/halext-org", "category": "lab"},
    # Tools
    "org-sync": {"path": "~/src/tools/org-sync", "category": "tools"},
    "dotfiles": {"path": "~/src/config/dotfiles", "category": "config"},
}

STALE_THRESHOLD_DAYS = 14
CLOUD_NEXT_DATE = datetime(2026, 4, 22)


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def _git_commits_since(repo_path: Path, days: int = 7) -> list[dict[str, str]]:
    """Return recent commits as [{hash, subject, date}]."""
    since = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d")
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_path), "log", f"--after={since}",
             "--format=%H|%s|%aI", "--no-merges"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return []
        commits = []
        for line in result.stdout.strip().splitlines():
            parts = line.split("|", 2)
            if len(parts) == 3:
                commits.append({"hash": parts[0][:8], "subject": parts[1], "date": parts[2]})
        return commits
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return []


def _git_last_commit_date(repo_path: Path) -> datetime | None:
    """Return the date of the most recent commit."""
    try:
        result = subprocess.run(
            ["git", "-C", str(repo_path), "log", "-1", "--format=%aI"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0 or not result.stdout.strip():
            return None
        dt = datetime.fromisoformat(result.stdout.strip())
        # Strip timezone info for naive comparison
        return dt.replace(tzinfo=None)
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        return None


# ---------------------------------------------------------------------------
# Halext-org task pull (optional, fails gracefully)
# ---------------------------------------------------------------------------

def _fetch_halext_tasks() -> list[dict[str, Any]]:
    """Pull open tasks from halext-org API. Returns [] on failure."""
    try:
        import urllib.request
        req = urllib.request.Request(
            "http://localhost:8000/tasks/?status=todo&limit=10",
            headers={"Accept": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=3) as resp:
            return json.loads(resp.read())
    except Exception:
        return []


# ---------------------------------------------------------------------------
# Weekly review carry-over
# ---------------------------------------------------------------------------

def _latest_weekly_carryover() -> list[str]:
    """Parse carry-over items from the most recent weekly review."""
    weekly_dir = Path.home() / "Journal" / "weekly"
    if not weekly_dir.is_dir():
        return []
    files = sorted(weekly_dir.glob("2026-W*.org"), reverse=True)
    if not files:
        return []
    items = []
    in_carry = False
    for line in files[0].read_text(errors="replace").splitlines():
        if line.startswith("* Carry Over"):
            in_carry = True
            continue
        if in_carry and line.startswith("* "):
            break
        if in_carry and line.strip().startswith("- ["):
            items.append(line.strip())
    return items


# ---------------------------------------------------------------------------
# Google Workspace CLI integration (optional — requires `gws` binary)
# Uses afs.gws.GWSClient for all GWS operations.
# ---------------------------------------------------------------------------

def _get_gws_client():
    """Lazy import to avoid circular deps."""
    from ..gws import get_client
    return get_client()


# ---------------------------------------------------------------------------
# Agent registry (Phase 2 — reads if file exists)
# ---------------------------------------------------------------------------

def _read_agent_registry() -> list[dict[str, Any]]:
    """Read agent task registry if it exists."""
    registry_path = Path.home() / ".afs" / "agent_registry.json"
    if not registry_path.exists():
        return []
    try:
        data = json.loads(registry_path.read_text())
        return data if isinstance(data, list) else []
    except (json.JSONDecodeError, OSError):
        return []


# ---------------------------------------------------------------------------
# Briefing assembly
# ---------------------------------------------------------------------------

def _build_briefing(days: int = 7, include_gws: bool = True) -> dict[str, Any]:
    """Assemble the full briefing data structure."""
    now = datetime.now()

    # Git velocity
    velocity: dict[str, Any] = {}
    stale: list[str] = []
    total_commits = 0

    for name, meta in PROJECTS.items():
        repo = Path(meta["path"]).expanduser()
        if not repo.is_dir():
            continue
        commits = _git_commits_since(repo, days=days)
        count = len(commits)
        total_commits += count
        last_date = _git_last_commit_date(repo)
        days_since = (now - last_date).days if last_date else None

        velocity[name] = {
            "commits": count,
            "category": meta["category"],
            "last_commit": last_date.isoformat() if last_date else None,
            "days_since": days_since,
            "top_subjects": [c["subject"] for c in commits[:3]],
        }

        if days_since is not None and days_since >= STALE_THRESHOLD_DAYS:
            stale.append(f"{name} ({days_since}d)")

    # Sort by commit count descending
    velocity = dict(sorted(velocity.items(), key=lambda x: x[1]["commits"], reverse=True))

    # Deadlines
    days_to_next = (CLOUD_NEXT_DATE - now).days

    # Tasks
    tasks = _fetch_halext_tasks()
    carry = _latest_weekly_carryover()
    agents = _read_agent_registry()

    # Google Workspace (optional)
    calendar_agenda: list[dict[str, Any]] = []
    gmail_unread: list[dict[str, Any]] = []
    gws_available = False
    if include_gws:
        gws = _get_gws_client()
        if gws.available and gws.authenticated:
            gws_available = True
            calendar_agenda = gws.calendar_agenda()
            gmail_unread = gws.gmail_unread()

    return {
        "date": now.strftime("%Y-%m-%d %A"),
        "cloud_next_days": days_to_next,
        "total_commits_7d": total_commits,
        "velocity": velocity,
        "stale_projects": stale,
        "open_tasks": tasks,
        "carry_over": carry,
        "active_agents": agents,
        "gws_available": gws_available,
        "calendar_agenda": calendar_agenda,
        "gmail_unread": gmail_unread,
    }


# ---------------------------------------------------------------------------
# Renderers
# ---------------------------------------------------------------------------

def _render_text(briefing: dict[str, Any], short: bool = False) -> str:
    """Render briefing as readable text."""
    lines: list[str] = []
    lines.append(f"=== AFS Morning Briefing — {briefing['date']} ===")
    lines.append("")

    # Countdown
    d = briefing["cloud_next_days"]
    if d > 0:
        lines.append(f"  Cloud Next 2026: {d} days away")
    elif d == 0:
        lines.append("  Cloud Next 2026: TODAY")
    lines.append(f"  Total commits (7d): {briefing['total_commits_7d']}")
    if briefing.get("gws_available"):
        lines.append("  Google Workspace: connected")
    lines.append("")

    # Calendar agenda
    if briefing.get("calendar_agenda"):
        lines.append("--- Today's Calendar ---")
        for event in briefing["calendar_agenda"]:
            summary = event.get("summary", event.get("title", "untitled"))
            start = event.get("start", {})
            time_str = start.get("dateTime", start.get("date", "")) if isinstance(start, dict) else str(start)
            # Extract just the time portion if it's a datetime
            if "T" in str(time_str):
                try:
                    t = datetime.fromisoformat(str(time_str).replace("Z", "+00:00"))
                    time_str = t.strftime("%I:%M %p")
                except ValueError:
                    pass
            lines.append(f"  {time_str:<10} {summary}")
        lines.append("")

    # Gmail unread
    if briefing.get("gmail_unread") and not short:
        lines.append(f"--- Gmail ({len(briefing['gmail_unread'])} unread in primary) ---")
        for msg in briefing["gmail_unread"]:
            thread_id = msg.get("threadId", msg.get("id", ""))[:8]
            snippet = msg.get("snippet", "")[:60]
            if snippet:
                lines.append(f"  {thread_id}  {snippet}")
            else:
                lines.append(f"  {thread_id}")
        lines.append("")

    # Velocity
    lines.append("--- Project Velocity (7d) ---")
    for name, v in briefing["velocity"].items():
        if v["commits"] == 0 and short:
            continue
        marker = ""
        if v["days_since"] is not None and v["days_since"] >= STALE_THRESHOLD_DAYS:
            marker = " ⚠ STALE"
        commits_str = f"{v['commits']:>3} commits"
        days_str = f"(last: {v['days_since']}d ago)" if v["days_since"] is not None else "(no commits)"
        lines.append(f"  {name:<22} {commits_str}  {days_str}{marker}")
        if not short and v["top_subjects"]:
            for subj in v["top_subjects"]:
                lines.append(f"    · {subj[:72]}")
    lines.append("")

    # Stale alerts
    if briefing["stale_projects"]:
        lines.append("--- Stale Projects ---")
        for s in briefing["stale_projects"]:
            lines.append(f"  ⚠ {s}")
        lines.append("")

    # Carry-over
    if briefing["carry_over"]:
        lines.append("--- Carry Over (from latest weekly) ---")
        for item in briefing["carry_over"]:
            lines.append(f"  {item}")
        lines.append("")

    # Halext tasks
    if briefing["open_tasks"]:
        lines.append("--- Open Tasks (halext-org) ---")
        for t in briefing["open_tasks"][:5]:
            title = t.get("title", "untitled")
            priority = t.get("priority", "")
            status = t.get("status", "")
            lines.append(f"  [{priority or '-'}] {title} ({status})")
        lines.append("")

    # Active agents
    if briefing["active_agents"]:
        lines.append("--- Active Agents ---")
        for a in briefing["active_agents"]:
            name = a.get("name", "unknown")
            task = a.get("task", "")
            status = a.get("status", "unknown")
            lines.append(f"  {name}: {task} [{status}]")
        lines.append("")

    return "\n".join(lines)


def _render_org(briefing: dict[str, Any]) -> str:
    """Render briefing as org-mode for Emacs buffer."""
    lines: list[str] = []
    lines.append(f"#+TITLE: Morning Briefing — {briefing['date']}")
    lines.append("")

    d = briefing["cloud_next_days"]
    if d > 0:
        lines.append(f"*Cloud Next 2026: {d} days*")
    lines.append(f"Total commits (7d): {briefing['total_commits_7d']}")
    lines.append("")

    if briefing.get("calendar_agenda"):
        lines.append("* Today's Calendar")
        for event in briefing["calendar_agenda"]:
            summary = event.get("summary", event.get("title", "untitled"))
            start = event.get("start", {})
            time_str = start.get("dateTime", start.get("date", "")) if isinstance(start, dict) else str(start)
            if "T" in str(time_str):
                try:
                    t = datetime.fromisoformat(str(time_str).replace("Z", "+00:00"))
                    time_str = t.strftime("%I:%M %p")
                except ValueError:
                    pass
            lines.append(f"- {time_str} — {summary}")
        lines.append("")

    if briefing.get("gmail_unread"):
        lines.append(f"* Gmail ({len(briefing['gmail_unread'])} unread)")
        for msg in briefing["gmail_unread"]:
            snippet = msg.get("snippet", msg.get("id", ""))[:80]
            lines.append(f"- {snippet}")
        lines.append("")

    lines.append("* Project Velocity")
    for name, v in briefing["velocity"].items():
        if v["commits"] == 0:
            continue
        lines.append(f"** {name} — {v['commits']} commits")
        if v["top_subjects"]:
            for subj in v["top_subjects"]:
                lines.append(f"- {subj[:80]}")

    if briefing["stale_projects"]:
        lines.append("")
        lines.append("* Stale Projects")
        for s in briefing["stale_projects"]:
            lines.append(f"- {s}")

    if briefing["carry_over"]:
        lines.append("")
        lines.append("* Carry Over")
        for item in briefing["carry_over"]:
            lines.append(item)

    if briefing["active_agents"]:
        lines.append("")
        lines.append("* Active Agents")
        for a in briefing["active_agents"]:
            lines.append(f"- {a.get('name', '?')}: {a.get('task', '')} [{a.get('status', '')}]")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# CLI registration
# ---------------------------------------------------------------------------

def _briefing_command(args: argparse.Namespace) -> int:
    days = getattr(args, "days", 7)
    include_gws = not getattr(args, "no_gws", False)
    briefing = _build_briefing(days=days, include_gws=include_gws)

    if getattr(args, "json", False):
        print(json.dumps(briefing, indent=2, default=str))
    elif getattr(args, "org", False):
        print(_render_org(briefing))
    else:
        short = getattr(args, "short", False)
        print(_render_text(briefing, short=short))
    return 0


def register_parsers(subparsers: argparse._SubParsersAction) -> None:
    """Register the briefing command."""
    parser = subparsers.add_parser(
        "briefing",
        help="Morning briefing — git velocity, tasks, project health.",
    )
    parser.add_argument("--short", "-s", action="store_true", help="Compact single-screen output.")
    parser.add_argument("--json", "-j", action="store_true", help="JSON output for IDE integration.")
    parser.add_argument("--org", action="store_true", help="Org-mode output for Emacs.")
    parser.add_argument("--days", "-d", type=int, default=7, help="Lookback window in days (default: 7).")
    parser.add_argument("--no-gws", action="store_true", help="Skip Google Workspace integration.")
    parser.set_defaults(func=_briefing_command)
