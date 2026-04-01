"""Bridge agent findings into Tether ADHD capture items.

Converts workspace-analyst findings, mission results, and pending approvals
into structured capture items compatible with Tether's Entry model. Supports
file-based JSON handoff, tether:// deep links, and optional halext-org API push.
"""

from __future__ import annotations

import argparse
import json
import logging
import time
import urllib.parse
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path
from typing import Any

from .base import (
    AgentResult,
    build_base_parser,
    configure_logging,
    emit_result,
    now_iso,
)

logger = logging.getLogger(__name__)

AGENT_NAME = "tether-bridge"
AGENT_DESCRIPTION = (
    "Convert agent findings into Tether-compatible capture items "
    "for ADHD-friendly task triage."
)

# Default output location for file-based handoff
DEFAULT_CAPTURES_PATH = Path("~/.config/afs/agents/tether_captures.json")

# Tether URL scheme (iOS deep links)
TETHER_SCHEME = "tether"

# halext-org API default
HALEXT_API_DEFAULT = "https://org.halext.org/api"


# ---------------------------------------------------------------------------
# Priority mapping
# ---------------------------------------------------------------------------

def _classify_priority(finding: dict[str, Any]) -> str:
    """Map a finding to a Tether priority level.

    Rules:
      - dirty repos with >10 files changed = high
      - pending approvals = medium
      - stale repos = low
      - everything else = medium (safe default)
    """
    finding_type = finding.get("type", "")
    severity = finding.get("severity", "")

    # Dirty repo with many files
    dirty_files = finding.get("dirty_file_count", 0)
    if not isinstance(dirty_files, (int, float)):
        dirty_files = 0
    if finding_type == "dirty_repo" and dirty_files > 10:
        return "high"

    # Stale repos
    if finding_type == "stale_repo":
        return "low"

    # Pending approvals
    if finding_type == "pending_approval":
        return "medium"

    # Severity-based fallback
    if severity in ("critical", "high"):
        return "high"
    if severity == "low":
        return "low"

    return "medium"


# ---------------------------------------------------------------------------
# Finding -> Capture conversion
# ---------------------------------------------------------------------------

def findings_to_captures(findings: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Convert workspace-analyst findings into Tether-compatible capture items.

    Each capture item mirrors Tether's Entry model fields:
      - title: str
      - body: str | None
      - priority: "low" | "medium" | "high"
      - source: str (agent name that produced the finding)
      - tags: list[str]
      - entry_type: str (maps to Tether EntryType: task/note/idea)
      - status: str (always "inbox" for new captures)
      - created_at: str (ISO timestamp)
      - deep_link: str | None (tether:// URL)
    """
    captures: list[dict[str, Any]] = []
    now = datetime.now().isoformat()

    for finding in findings:
        title = _build_title(finding)
        body = _build_body(finding)
        priority = _classify_priority(finding)
        source = finding.get("agent", finding.get("source", "unknown"))
        tags = _build_tags(finding)
        entry_type = _infer_entry_type(finding)

        capture: dict[str, Any] = {
            "title": title,
            "body": body,
            "priority": priority,
            "source": str(source),
            "tags": tags,
            "entry_type": entry_type,
            "status": "inbox",
            "created_at": now,
            "deep_link": generate_deep_link(title),
        }
        captures.append(capture)

    return captures


def _build_title(finding: dict[str, Any]) -> str:
    """Build a concise capture title from a finding."""
    finding_type = finding.get("type", "")
    name = finding.get("name", finding.get("repo", finding.get("project", "")))

    if finding_type == "dirty_repo":
        count = finding.get("dirty_file_count", "?")
        return f"Dirty repo: {name} ({count} files)"
    if finding_type == "stale_repo":
        days = finding.get("days_stale", "?")
        return f"Stale repo: {name} ({days}d inactive)"
    if finding_type == "pending_approval":
        return f"Pending approval: {name}"
    if finding_type == "missing_context":
        return f"Missing .context: {name}"

    # Generic fallback
    title = finding.get("title", finding.get("message", ""))
    if title:
        return str(title)
    if name:
        return f"Finding: {name}"
    return f"Agent finding ({finding_type or 'unknown'})"


def _build_body(finding: dict[str, Any]) -> str | None:
    """Build a descriptive body from a finding."""
    parts: list[str] = []

    description = finding.get("description", finding.get("detail", ""))
    if description:
        parts.append(str(description))

    path = finding.get("path", finding.get("repo_path", ""))
    if path:
        parts.append(f"Path: {path}")

    finding_type = finding.get("type", "")
    if finding_type:
        parts.append(f"Type: {finding_type}")

    agent = finding.get("agent", finding.get("source", ""))
    if agent:
        parts.append(f"Source agent: {agent}")

    return "\n".join(parts) if parts else None


def _build_tags(finding: dict[str, Any]) -> list[str]:
    """Build tags list from a finding."""
    tags: list[str] = ["afs-agent"]

    finding_type = finding.get("type", "")
    if finding_type:
        tags.append(finding_type.replace("_", "-"))

    agent = finding.get("agent", finding.get("source", ""))
    if agent:
        tags.append(f"from:{agent}")

    existing_tags = finding.get("tags", [])
    if isinstance(existing_tags, list):
        tags.extend(str(t) for t in existing_tags if str(t) not in tags)

    return tags


def _infer_entry_type(finding: dict[str, Any]) -> str:
    """Infer Tether EntryType from a finding.

    Maps to Tether's EntryType enum: task, note, idea, decision, journal.
    """
    finding_type = finding.get("type", "")

    # Actionable findings become tasks
    if finding_type in ("dirty_repo", "pending_approval", "missing_context"):
        return "task"

    # Informational findings become notes
    if finding_type in ("stale_repo", "health_report", "summary"):
        return "note"

    return "task"


# ---------------------------------------------------------------------------
# Deep link generation
# ---------------------------------------------------------------------------

def generate_deep_link(title: str) -> str:
    """Generate a tether://capture deep link for a capture item.

    Uses tether://capture?title=<encoded_title> format matching
    the iOS app's onOpenURL handler.
    """
    encoded_title = urllib.parse.quote(title, safe="")
    return f"{TETHER_SCHEME}://capture?title={encoded_title}"


# ---------------------------------------------------------------------------
# Output: file-based handoff
# ---------------------------------------------------------------------------

def write_captures_file(
    captures: list[dict[str, Any]],
    output_path: Path | None = None,
) -> Path:
    """Write captures to a JSON file for Tether to pick up.

    Returns the path written to.
    """
    path = (output_path or DEFAULT_CAPTURES_PATH).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "version": 1,
        "generated_at": datetime.now().isoformat(),
        "agent": AGENT_NAME,
        "count": len(captures),
        "captures": captures,
    }

    path.write_text(
        json.dumps(payload, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    logger.info("Wrote %d captures to %s", len(captures), path)
    return path


# ---------------------------------------------------------------------------
# Output: halext-org API push (optional, best-effort)
# ---------------------------------------------------------------------------

def push_to_halext(
    captures: list[dict[str, Any]],
    *,
    api_url: str = HALEXT_API_DEFAULT,
    token: str | None = None,
) -> list[dict[str, Any]]:
    """Push captures as todos to halext-org REST API.

    Returns list of created task dicts (empty on failure).
    This is best-effort; network errors are logged but not raised.
    """
    if not token:
        logger.debug("No halext-org token; skipping API push")
        return []

    import urllib.request

    created: list[dict[str, Any]] = []
    for capture in captures:
        body = {
            "title": capture["title"],
            "description": capture.get("body") or "",
            "labels": capture.get("tags", []),
        }
        data = json.dumps(body).encode("utf-8")
        req = urllib.request.Request(
            f"{api_url}/tasks/",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {token}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=10) as resp:
                result = json.loads(resp.read())
                created.append(result)
        except Exception as exc:
            logger.warning("Failed to push capture '%s' to halext-org: %s", capture["title"], exc)

    logger.info("Pushed %d/%d captures to halext-org", len(created), len(captures))
    return created


# ---------------------------------------------------------------------------
# Input: read latest agent reports
# ---------------------------------------------------------------------------

def _read_latest_findings(
    reports_dir: Path | None = None,
) -> list[dict[str, Any]]:
    """Read findings from latest workspace health reports and supervisor state.

    Scans the AFS agent output directory for recent reports and extracts
    actionable findings.
    """
    search_dir = (reports_dir or Path("~/.config/afs/agents")).expanduser().resolve()
    findings: list[dict[str, Any]] = []

    if not search_dir.exists():
        logger.debug("Agent reports directory does not exist: %s", search_dir)
        return findings

    # Look for JSON report files from workspace-analyst and other agents
    for report_path in sorted(search_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True):
        if report_path.name == "tether_captures.json":
            continue  # Skip our own output
        try:
            data = json.loads(report_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as exc:
            logger.debug("Skipping unreadable report %s: %s", report_path, exc)
            continue

        # Extract findings from payload
        payload = data.get("payload", {})
        if not isinstance(payload, dict):
            continue

        # Common finding formats from various agents
        for key in ("invalid", "findings", "issues", "items", "results"):
            items = payload.get(key, [])
            if isinstance(items, list):
                for item in items:
                    if isinstance(item, dict):
                        # Tag with source agent
                        if "agent" not in item:
                            item["agent"] = data.get("name", report_path.stem)
                        findings.append(item)

    # Also check supervisor state for pending reviews
    state_dir = search_dir / "state"
    if state_dir.is_dir():
        for state_file in state_dir.glob("*.json"):
            try:
                state = json.loads(state_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue
            if state.get("state") == "awaiting_review":
                findings.append({
                    "type": "pending_approval",
                    "name": state.get("name", state_file.stem),
                    "agent": "agent-supervisor",
                    "detail": f"Agent '{state.get('name', '')}' is awaiting review",
                })

    return findings


# ---------------------------------------------------------------------------
# CLI / Agent entry point
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = build_base_parser(
        "Convert agent findings into Tether-compatible capture items."
    )
    parser.add_argument(
        "--captures-output",
        default=str(DEFAULT_CAPTURES_PATH),
        help="Path for the captures JSON file (default: %(default)s).",
    )
    parser.add_argument(
        "--reports-dir",
        help="Directory containing agent report JSON files.",
    )
    parser.add_argument(
        "--halext-token",
        help="Bearer token for halext-org API (optional).",
    )
    parser.add_argument(
        "--halext-url",
        default=HALEXT_API_DEFAULT,
        help="halext-org API base URL (default: %(default)s).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate captures but do not write files or push to API.",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    configure_logging(args.quiet)
    started_at = now_iso()
    start = time.monotonic()

    # 1. Read latest findings
    reports_dir = Path(args.reports_dir) if args.reports_dir else None
    findings = _read_latest_findings(reports_dir)
    logger.info("Found %d findings from agent reports", len(findings))

    # 2. Convert to captures
    captures = findings_to_captures(findings)
    logger.info("Generated %d capture items", len(captures))

    # 3. Write captures file
    captures_written = False
    captures_path = Path(args.captures_output)
    if not args.dry_run and captures:
        write_captures_file(captures, captures_path)
        captures_written = True

    # 4. Push to halext-org (best-effort)
    halext_pushed = 0
    if not args.dry_run and args.halext_token and captures:
        pushed = push_to_halext(
            captures,
            api_url=args.halext_url,
            token=args.halext_token,
        )
        halext_pushed = len(pushed)

    finished_at = now_iso()
    duration = time.monotonic() - start

    # Priority breakdown
    priority_counts = {"low": 0, "medium": 0, "high": 0}
    for capture in captures:
        p = capture.get("priority", "medium")
        priority_counts[p] = priority_counts.get(p, 0) + 1

    result = AgentResult(
        name=AGENT_NAME,
        status="ok" if captures or not findings else "warn",
        started_at=started_at,
        finished_at=finished_at,
        duration_seconds=duration,
        task="Convert agent findings to Tether captures",
        metrics={
            "findings_read": len(findings),
            "captures_generated": len(captures),
            "halext_pushed": halext_pushed,
            "priority_high": priority_counts["high"],
            "priority_medium": priority_counts["medium"],
            "priority_low": priority_counts["low"],
        },
        notes=[
            f"captures written to {captures_path}" if captures_written else "no captures written",
        ],
        payload={
            "captures": captures,
            "dry_run": args.dry_run,
            "captures_path": str(captures_path) if captures_written else None,
        },
    )

    output_path = Path(args.output) if args.output else None
    emit_result(
        result,
        output_path=output_path,
        force_stdout=args.stdout,
        pretty=args.pretty,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
