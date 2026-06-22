"""Weekly review drafting for an org-mode thoughts file + a markdown task list.

This agent drafts the AI portion of a hybrid weekly review. It expects:

- a thoughts file (org-mode) with dated headlines like ``* 4 April 2026``
- a tasks file (markdown) with ``- [ ]`` / ``- [x]`` items
- a weekly directory where ``YYYY-WNN.org`` files live

Path resolution (highest precedence first):
    1. CLI flags ``--thoughts`` / ``--active-tasks`` / ``--weekly-dir``
    2. Per-field env vars ``AFS_JOURNAL_THOUGHTS``, ``AFS_JOURNAL_ACTIVE_TASKS``,
       ``AFS_JOURNAL_WEEKLY_DIR``
    3. Sub-paths derived from ``AFS_JOURNAL_ROOT`` (``thoughts.org``,
       ``tasks/active.md``, ``weekly/``)
    4. ``~/.local/share/afs/journal`` as a generic fallback

The default fallback intentionally lives under ``~/.local/share/`` so AFS
does not assume any particular workspace layout. Users who keep a writing
folder elsewhere should export ``AFS_JOURNAL_ROOT``.
"""

from __future__ import annotations

import argparse
import os
import re
import time
from collections.abc import Sequence
from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from ..agent_context import ContextAwareAgent
from .base import (
    AgentResult,
    build_base_parser,
    configure_logging,
    emit_result,
    now_iso,
)

AGENT_NAME = "journal-agent"
AGENT_DESCRIPTION = (
    "Drafts the AI portion of a hybrid weekly review by scanning a "
    "thoughts.org file and a tasks/active.md file."
)

AGENT_CAPABILITIES = {
    "mount_types": [],
    "topics": ["journal:weekly"],
    "tools": [],
    "description": (
        "Reads dated entries from a thoughts.org file and items from a "
        "tasks/active.md file for the current ISO week, then writes an AI "
        "draft below the divider in <weekly_dir>/YYYY-WNN.org. The human "
        "section above the divider is preserved untouched. Paths are "
        "configured via AFS_JOURNAL_ROOT or per-field env vars."
    ),
}

_FALLBACK_JOURNAL_ROOT = Path("~/.local/share/afs/journal").expanduser()


def _resolve_journal_root() -> Path:
    """Resolve the journal root from env, with a generic fallback."""
    env_value = os.environ.get("AFS_JOURNAL_ROOT")
    if env_value:
        return Path(env_value).expanduser()
    return _FALLBACK_JOURNAL_ROOT


def default_thoughts_path() -> Path:
    """Resolve the default thoughts.org path."""
    explicit = os.environ.get("AFS_JOURNAL_THOUGHTS")
    if explicit:
        return Path(explicit).expanduser()
    return _resolve_journal_root() / "thoughts.org"


def default_active_tasks_path() -> Path:
    """Resolve the default tasks/active.md path."""
    explicit = os.environ.get("AFS_JOURNAL_ACTIVE_TASKS")
    if explicit:
        return Path(explicit).expanduser()
    return _resolve_journal_root() / "tasks" / "active.md"


def default_weekly_dir() -> Path:
    """Resolve the default weekly review directory."""
    explicit = os.environ.get("AFS_JOURNAL_WEEKLY_DIR")
    if explicit:
        return Path(explicit).expanduser()
    return _resolve_journal_root() / "weekly"

_DIVIDER = "---"
_DIVIDER_HINT = "(AI draft below"

# Org-mode regex patterns retained for thoughts.org scanning
_CHECKBOX_UNCHECKED_RE = re.compile(r"^\s*-\s+\[ \]\s+.+$")
_CHECKBOX_CHECKED_RE = re.compile(r"^\s*-\s+\[[xX]\]\s+.+$")
_TODO_HEADLINE_RE = re.compile(r"^\*+\s+TODO\s+.+$")
_DONE_HEADLINE_RE = re.compile(r"^\*+\s+DONE\s+.+$")
_TOP_HEADING_RE = re.compile(r"^\*\s+(.+)$")

# Markdown task patterns for tasks/active.md
_MD_UNCHECKED_RE = re.compile(r"^\s*-\s+\[ \]\s+(.+)$")
_MD_CHECKED_RE = re.compile(r"^\s*-\s+\[[xX]\]\s+(.+)$")

# Date headlines in thoughts.org look like: * 4 April 2026
_THOUGHT_DATE_RE = re.compile(
    r"^\*\s+(\d{1,2})\s+([A-Za-z]+)\s+(\d{4})\s*$"
)
_MONTHS = {
    m.lower(): i + 1
    for i, m in enumerate(
        [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        ]
    )
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _iso_week_bounds(d: date) -> tuple[date, date]:
    """Return (monday, sunday) for the ISO week containing d."""
    monday = d - timedelta(days=d.weekday())
    return monday, monday + timedelta(days=6)


def _parse_thought_date(line: str) -> date | None:
    m = _THOUGHT_DATE_RE.match(line.strip())
    if not m:
        return None
    day_str, month_str, year_str = m.groups()
    month = _MONTHS.get(month_str.lower())
    if not month:
        return None
    try:
        return date(int(year_str), month, int(day_str))
    except ValueError:
        return None


def extract_thought_entries(
    text: str,
    monday: date,
    sunday: date,
) -> list[tuple[date, str]]:
    """Return (date, body) tuples for thoughts.org entries within [monday, sunday]."""
    lines = text.splitlines()
    entries: list[tuple[date, list[str]]] = []
    current_date: date | None = None
    current_body: list[str] = []

    for line in lines:
        d = _parse_thought_date(line)
        if d is not None:
            # flush previous
            if current_date is not None and monday <= current_date <= sunday:
                entries.append((current_date, current_body))
            current_date = d
            current_body = []
            continue
        if current_date is not None:
            current_body.append(line)

    # flush trailing
    if current_date is not None and monday <= current_date <= sunday:
        entries.append((current_date, current_body))

    return [(d, "\n".join(body).strip()) for d, body in entries]


def extract_active_tasks(text: str) -> tuple[list[str], list[str]]:
    """Return (open_items, completed_items) from a tasks/active.md style file."""
    open_items: list[str] = []
    done_items: list[str] = []
    for line in text.splitlines():
        m_open = _MD_UNCHECKED_RE.match(line)
        if m_open:
            open_items.append(m_open.group(1).strip())
            continue
        m_done = _MD_CHECKED_RE.match(line)
        if m_done:
            done_items.append(m_done.group(1).strip())
    return open_items, done_items


def split_weekly_human_ai(text: str) -> tuple[str, str]:
    """Split an existing weekly file at the divider line.

    Returns (human_section, _existing_ai_section). Either may be empty.
    The divider line itself is included with the human section so it stays
    where the user put it.
    """
    lines = text.splitlines()
    divider_idx: int | None = None
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped == _DIVIDER or stripped.startswith(_DIVIDER_HINT):
            divider_idx = idx
            break
    if divider_idx is None:
        return text.rstrip(), ""
    human = "\n".join(lines[: divider_idx + 1]).rstrip()
    ai = "\n".join(lines[divider_idx + 1 :]).strip()
    return human, ai


# ---------------------------------------------------------------------------
# Weekly review drafting
# ---------------------------------------------------------------------------


def _hybrid_skeleton(week_label: str, month_range: str) -> str:
    author = os.environ.get("AFS_JOURNAL_AUTHOR", "").strip()
    author_line = f"#+AUTHOR: {author}\n" if author else ""
    return (
        f"#+TITLE: Weekly Review — {week_label} ({month_range})\n"
        f"{author_line}"
        f"\n"
        f"* What happened\n"
        f"-\n"
        f"-\n"
        f"-\n"
        f"\n"
        f"* What I want next week\n"
        f"-\n"
        f"\n"
        f"---\n"
        f"(AI draft below — edit or ignore)"
    )


def _format_ai_draft(
    week_label: str,
    thought_entries: list[tuple[date, str]],
    open_tasks: list[str],
    done_tasks: list[str],
    agent_activity: list[str],
) -> str:
    lines = [
        "",
        f"* AI draft — {week_label}",
        "",
    ]

    # Thoughts summary
    lines.append("** Thoughts.org entries this week")
    if thought_entries:
        for d, body in thought_entries:
            day_label = d.strftime("%a %b %-d")
            preview = (body[:240] + "…") if len(body) > 240 else body
            preview = preview.replace("\n", " ").strip() or "(empty)"
            lines.append(f"- *{day_label}* — {preview}")
    else:
        lines.append("- (no dated entries this week)")
    lines.append("")

    # Tasks
    lines.append("** Tasks")
    if done_tasks:
        lines.append("- Completed:")
        for item in done_tasks[:10]:
            lines.append(f"  - [x] {item}")
    if open_tasks:
        lines.append("- Still open:")
        for item in open_tasks[:10]:
            lines.append(f"  - [ ] {item}")
    if not done_tasks and not open_tasks:
        lines.append("- (no tasks tracked in tasks/active.md)")
    lines.append("")

    # Agent activity
    if agent_activity:
        lines.append("** Agent activity")
        lines.extend(agent_activity)
        lines.append("")

    return "\n".join(lines).rstrip() + "\n"


def run_weekly_review(
    week_date: date,
    thoughts_path: Path,
    active_tasks_path: Path,
    weekly_dir: Path,
    agent_activity: list[str] | None = None,
    overwrite: bool = False,
) -> tuple[str, dict[str, Any]]:
    """Draft (or refresh) the AI portion of the hybrid weekly review.

    Behaviour:
      - If the weekly file does not exist, scaffold a hybrid skeleton and
        append the AI draft below the divider.
      - If it exists with an AI section, replace only the AI section
        unless overwrite=False AND there is no AI section yet.
      - The human section above the divider is preserved verbatim.
    """
    monday, sunday = _iso_week_bounds(week_date)
    iso_year, iso_week, _ = monday.isocalendar()
    week_label = f"{iso_year}-W{iso_week:02d}"

    if monday.month == sunday.month:
        month_range = f"{monday.strftime('%b %-d')}–{sunday.strftime('%-d')}"
    else:
        month_range = f"{monday.strftime('%b %-d')}–{sunday.strftime('%b %-d')}"

    # Gather inputs
    thought_entries: list[tuple[date, str]] = []
    if thoughts_path.exists():
        thought_entries = extract_thought_entries(
            thoughts_path.read_text(encoding="utf-8"),
            monday,
            sunday,
        )

    open_tasks: list[str] = []
    done_tasks: list[str] = []
    if active_tasks_path.exists():
        open_tasks, done_tasks = extract_active_tasks(
            active_tasks_path.read_text(encoding="utf-8")
        )

    ai_draft = _format_ai_draft(
        week_label,
        thought_entries,
        open_tasks,
        done_tasks,
        agent_activity or [],
    )

    weekly_dir.mkdir(parents=True, exist_ok=True)
    out_path = weekly_dir / f"{week_label}.org"

    if out_path.exists():
        existing = out_path.read_text(encoding="utf-8")
        human, existing_ai = split_weekly_human_ai(existing)
        if existing_ai and not overwrite:
            return str(out_path), {
                "action": "skipped",
                "reason": "AI draft already present (pass --overwrite to refresh)",
                "path": str(out_path),
                "week": week_label,
            }
        new_content = human + "\n" + ai_draft
        out_path.write_text(new_content, encoding="utf-8")
        return str(out_path), {
            "action": "refreshed" if existing_ai else "drafted",
            "path": str(out_path),
            "week": week_label,
            "thoughts_count": len(thought_entries),
            "open_tasks": len(open_tasks),
            "done_tasks": len(done_tasks),
        }

    # New file: scaffold hybrid skeleton + draft
    skeleton = _hybrid_skeleton(week_label, month_range)
    new_content = skeleton + "\n" + ai_draft
    out_path.write_text(new_content, encoding="utf-8")
    return str(out_path), {
        "action": "created",
        "path": str(out_path),
        "week": week_label,
        "thoughts_count": len(thought_entries),
        "open_tasks": len(open_tasks),
        "done_tasks": len(done_tasks),
    }


# ---------------------------------------------------------------------------
# Context-aware agent
# ---------------------------------------------------------------------------


class _JournalAgent(ContextAwareAgent):
    """Journal agent with context-aware enrichment."""

    @staticmethod
    def _event_is_within_days(event: dict[str, Any], *, days: int) -> bool:
        raw_timestamp = str(event.get("timestamp") or event.get("ts") or "").strip()
        if not raw_timestamp:
            return False
        normalized = raw_timestamp[:-1] + "+00:00" if raw_timestamp.endswith("Z") else raw_timestamp
        try:
            parsed = datetime.fromisoformat(normalized)
        except ValueError:
            return False
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        cutoff = datetime.now(timezone.utc) - timedelta(days=max(days, 1))
        return parsed >= cutoff

    def get_agent_activity_summary(self, days: int = 7) -> list[str]:
        """Get recent agent activity as bullet points for weekly reviews."""
        events = self.get_recent_events(
            event_types={"agent_progress"},
            limit=50,
        )
        if not events:
            return []
        lines: list[str] = []
        seen: set[str] = set()
        for ev in events:
            if not self._event_is_within_days(ev, days=days):
                continue
            agent = ev.get("metadata", {}).get("agent", "") or ev.get("source", "")
            detail = ev.get("metadata", {}).get("detail", "") or ev.get("op", "")
            key = f"{agent}:{detail}"
            if key in seen or not agent:
                continue
            seen.add(key)
            lines.append(f"- {agent}: {detail}" if detail else f"- {agent}")
        return lines[:10]


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = build_base_parser(AGENT_DESCRIPTION)
    parser.add_argument(
        "--week",
        metavar="YYYY-WNN",
        help="ISO week to draft (default: current week).",
    )
    # Defaults are resolved at parse time so env vars are honoured for each
    # invocation rather than frozen at import.
    parser.add_argument(
        "--thoughts",
        default=None,
        help=(
            "Path to thoughts.org. Defaults to $AFS_JOURNAL_THOUGHTS, "
            "$AFS_JOURNAL_ROOT/thoughts.org, or "
            "~/.local/share/afs/journal/thoughts.org."
        ),
    )
    parser.add_argument(
        "--active-tasks",
        default=None,
        help=(
            "Path to tasks/active.md. Defaults to $AFS_JOURNAL_ACTIVE_TASKS, "
            "$AFS_JOURNAL_ROOT/tasks/active.md, or "
            "~/.local/share/afs/journal/tasks/active.md."
        ),
    )
    parser.add_argument(
        "--weekly-dir",
        default=None,
        help=(
            "Weekly review directory. Defaults to $AFS_JOURNAL_WEEKLY_DIR, "
            "$AFS_JOURNAL_ROOT/weekly, or ~/.local/share/afs/journal/weekly."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing AI draft section instead of skipping.",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    configure_logging(args.quiet)

    agent = _JournalAgent()
    ctx = agent.load_context()
    if ctx:
        import logging as _log
        _log.getLogger(__name__).info(
            "Journal context: %d recent events, %d memory topics",
            len(ctx.recent_events), len(ctx.memory_topics),
        )

    thoughts_path = (
        Path(args.thoughts).expanduser() if args.thoughts else default_thoughts_path()
    )
    active_tasks_path = (
        Path(args.active_tasks).expanduser()
        if args.active_tasks
        else default_active_tasks_path()
    )
    weekly_dir = (
        Path(args.weekly_dir).expanduser() if args.weekly_dir else default_weekly_dir()
    )

    if args.week:
        parts = args.week.split("-W")
        iso_year, iso_wk = int(parts[0]), int(parts[1])
        week_date = date.fromisocalendar(iso_year, iso_wk, 1)
    else:
        week_date = date.today()

    started_at = now_iso()
    start = time.monotonic()

    agent_activity = agent.get_agent_activity_summary(days=7)

    out_path, meta = run_weekly_review(
        week_date,
        thoughts_path,
        active_tasks_path,
        weekly_dir,
        agent_activity=agent_activity,
        overwrite=args.overwrite,
    )

    notes = [f"weekly-review: {meta['action']} → {out_path}"]
    if agent_activity:
        notes.append(f"included {len(agent_activity)} agent activity entries")

    metrics = {
        "thoughts_count": meta.get("thoughts_count", 0),
        "open_tasks": meta.get("open_tasks", 0),
        "done_tasks": meta.get("done_tasks", 0),
        "agent_activity": len(agent_activity),
    }

    finished_at = now_iso()
    duration = time.monotonic() - start

    result = AgentResult(
        name=AGENT_NAME,
        status="ok",
        started_at=started_at,
        finished_at=finished_at,
        duration_seconds=duration,
        task="journal:weekly-review",
        metrics=metrics,
        notes=notes,
        payload={"weekly_review": meta},
    )

    output_path = Path(args.output) if args.output else None
    emit_result(result, output_path=output_path, force_stdout=args.stdout, pretty=args.pretty)
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
