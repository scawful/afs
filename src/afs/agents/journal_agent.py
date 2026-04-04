"""Journal carry-forward, daily template generation, stale-TODO alerting, and weekly review."""

from __future__ import annotations

import argparse
import re
import time
from collections.abc import Sequence
from datetime import date, timedelta
from pathlib import Path
from typing import Any

from .base import (
    AgentResult,
    build_base_parser,
    configure_logging,
    emit_result,
    now_iso,
)

AGENT_NAME = "journal-agent"
AGENT_DESCRIPTION = (
    "Journal carry-forward, daily template generation, "
    "stale-TODO alerting, and weekly review drafting."
)

AGENT_CAPABILITIES = {
    "mount_types": [],
    "topics": ["journal:daily", "journal:weekly"],
    "tools": [],
    "description": (
        "Scans ~/Journal/daily/ org files for unchecked TODOs, carries them forward, "
        "generates daily templates, flags items stale for 3+ consecutive days, "
        "and drafts weekly reviews."
    ),
}

_DEFAULT_DAILY_DIR = Path("~/Journal/daily").expanduser()
_DEFAULT_WEEKLY_DIR = Path("~/Journal/weekly").expanduser()

# Org-mode regex patterns
_CHECKBOX_UNCHECKED_RE = re.compile(r"^\s*-\s+\[ \]\s+.+$")
_CHECKBOX_CHECKED_RE = re.compile(r"^\s*-\s+\[[xX]\]\s+.+$")
_TODO_HEADLINE_RE = re.compile(r"^\*+\s+TODO\s+.+$")
_DONE_HEADLINE_RE = re.compile(r"^\*+\s+DONE\s+.+$")
_TOP_HEADING_RE = re.compile(r"^\*\s+(.+)$")


# ---------------------------------------------------------------------------
# Org-mode parsing helpers
# ---------------------------------------------------------------------------


def _daily_path(d: date, daily_dir: Path) -> Path:
    return daily_dir / f"{d.isoformat()}.org"


def _extract_unchecked(text: str) -> list[str]:
    """Return unchecked TODO items: `- [ ] …` checkboxes and `* TODO …` headlines."""
    items: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if _CHECKBOX_UNCHECKED_RE.match(line):
            items.append(stripped)
        elif _TODO_HEADLINE_RE.match(line):
            items.append(stripped)
    return items


def _extract_done(text: str) -> list[str]:
    """Return completed items: `- [x] …` checkboxes and `* DONE …` headlines."""
    items: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if _CHECKBOX_CHECKED_RE.match(line):
            items.append(stripped)
        elif _DONE_HEADLINE_RE.match(line):
            items.append(stripped)
    return items


def _extract_section_bullets(text: str, section_name: str) -> list[str]:
    """Return non-empty bullet lines from a named top-level (* …) section."""
    lines = text.splitlines()
    in_section = False
    bullets: list[str] = []
    for line in lines:
        m = _TOP_HEADING_RE.match(line)
        if m:
            if m.group(1).strip().lower() == section_name.lower():
                in_section = True
                continue
            elif in_section:
                break  # next top-level heading ends this section
        if in_section and line.strip().startswith("-"):
            bullets.append(line.strip())
    return bullets


def _todo_to_checkbox(item: str) -> str:
    """Normalize a TODO item (headline or checkbox) to `- [ ] text` format."""
    if _TODO_HEADLINE_RE.match(item):
        text = re.sub(r"^\*+\s+TODO\s+", "", item).strip()
        return f"- [ ] {text}"
    # Already a checkbox — ensure it's unchecked
    return re.sub(r"^-\s+\[[xX]\]", "- [ ]", item)


def _iso_week_bounds(d: date) -> tuple[date, date]:
    """Return (monday, sunday) for the ISO week containing d."""
    monday = d - timedelta(days=d.weekday())
    return monday, monday + timedelta(days=6)


# ---------------------------------------------------------------------------
# Task: carry-forward
# ---------------------------------------------------------------------------


def run_carry_forward(
    source_date: date,
    daily_dir: Path,
) -> tuple[list[str], dict[str, Any]]:
    """Extract unchecked TODO items from source_date's daily file."""
    src_path = _daily_path(source_date, daily_dir)
    if not src_path.exists():
        return [], {"source": str(src_path), "found": False, "todo_count": 0}

    text = src_path.read_text(encoding="utf-8")
    unchecked = _extract_unchecked(text)
    return unchecked, {
        "source": str(src_path),
        "found": True,
        "todo_count": len(unchecked),
        "items": unchecked,
    }


# ---------------------------------------------------------------------------
# Task: template-gen
# ---------------------------------------------------------------------------


def _build_carry_section(items: list[str]) -> str:
    lines = ["* Carry Over"]
    for item in items:
        lines.append(_todo_to_checkbox(item))
    return "\n".join(lines)


def run_template_gen(
    target_date: date,
    carry_items: list[str],
    daily_dir: Path,
    overwrite: bool = False,
) -> tuple[str, dict[str, Any]]:
    """Create (or update) tomorrow's daily org file with carry-over items."""
    target_path = _daily_path(target_date, daily_dir)

    if target_path.exists() and not overwrite:
        existing = target_path.read_text(encoding="utf-8")
        if "* Carry Over" in existing:
            return str(target_path), {
                "action": "skipped",
                "reason": "carry-over section already present",
                "path": str(target_path),
                "items_added": 0,
            }
        if carry_items:
            carry_section = _build_carry_section(carry_items)
            updated = existing.rstrip() + "\n\n" + carry_section + "\n"
            target_path.write_text(updated, encoding="utf-8")
            return str(target_path), {
                "action": "appended",
                "path": str(target_path),
                "items_added": len(carry_items),
            }
        return str(target_path), {
            "action": "skipped",
            "reason": "file exists and no carry items",
            "path": str(target_path),
            "items_added": 0,
        }

    day_name = target_date.strftime("%A")
    sections = [
        f"#+TITLE: Daily — {target_date.isoformat()} {day_name}",
        "#+AUTHOR: AFS",
        "",
        "* Morning",
        "- ",
        "",
        "* Work",
        "- ",
        "",
        "* AFS / Side Projects",
        "- ",
        "",
        "* Personal",
        "- ",
        "",
        "* Mood",
        "- ",
    ]
    if carry_items:
        sections.append("")
        sections.append(_build_carry_section(carry_items))
    content = "\n".join(sections) + "\n"

    daily_dir.mkdir(parents=True, exist_ok=True)
    target_path.write_text(content, encoding="utf-8")
    return str(target_path), {
        "action": "created",
        "path": str(target_path),
        "items_added": len(carry_items),
    }


# ---------------------------------------------------------------------------
# Task: stale-check
# ---------------------------------------------------------------------------


def run_stale_check(
    daily_dir: Path,
    scan_days: int = 14,
    stale_threshold: int = 3,
    today: date | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Flag TODO items that have been unchecked for stale_threshold+ consecutive days."""
    if today is None:
        today = date.today()

    # Build per-day item sets, oldest first
    days_data: list[tuple[date, set[str]]] = []
    for i in range(scan_days - 1, -1, -1):
        d = today - timedelta(days=i)
        path = _daily_path(d, daily_dir)
        items: set[str] = set()
        if path.exists():
            items = set(_extract_unchecked(path.read_text(encoding="utf-8")))
        days_data.append((d, items))

    # Track current consecutive streak per item (oldest → newest)
    streaks: dict[str, list[date]] = {}
    for d, items in days_data:
        for item in items:
            if item not in streaks:
                streaks[item] = [d]
            else:
                last = streaks[item][-1]
                if (d - last).days == 1:
                    streaks[item].append(d)
                else:
                    streaks[item] = [d]  # gap: reset streak

    # Report items whose streak is both long enough and still current (ended ≤1 day ago)
    recent_cutoff = today - timedelta(days=1)
    stale: list[dict[str, Any]] = []
    for item, streak_dates in streaks.items():
        if len(streak_dates) >= stale_threshold and streak_dates[-1] >= recent_cutoff:
            stale.append(
                {
                    "todo": item,
                    "streak_days": len(streak_dates),
                    "first_seen": streak_dates[0].isoformat(),
                    "last_seen": streak_dates[-1].isoformat(),
                }
            )

    stale.sort(key=lambda x: x["streak_days"], reverse=True)
    files_scanned = sum(1 for _, items in days_data if items)
    return stale, {
        "scan_days": scan_days,
        "threshold": stale_threshold,
        "files_scanned": files_scanned,
        "stale_count": len(stale),
    }


# ---------------------------------------------------------------------------
# Task: weekly-review
# ---------------------------------------------------------------------------


def run_weekly_review(
    week_date: date,
    daily_dir: Path,
    weekly_dir: Path,
    overwrite: bool = False,
) -> tuple[str, dict[str, Any]]:
    """Draft a weekly review org file for the ISO week containing week_date."""
    monday, sunday = _iso_week_bounds(week_date)
    iso_cal = monday.isocalendar()
    iso_year, iso_week = iso_cal[0], iso_cal[1]
    week_label = f"{iso_year}-W{iso_week:02d}"

    # Month range label, e.g. "Mar 16–22"
    if monday.month == sunday.month:
        month_range = f"{monday.strftime('%b %-d')}–{sunday.strftime('%-d')}"
    else:
        month_range = f"{monday.strftime('%b %-d')}–{sunday.strftime('%b %-d')}"

    today = date.today()
    wins: list[str] = []
    all_unchecked: dict[str, list[date]] = {}

    for i in range(7):
        d = monday + timedelta(days=i)
        if d > today:
            break
        path = _daily_path(d, daily_dir)
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")

        # Wins: completed checkboxes + DONE headlines
        wins.extend(_extract_done(text))

        # Wins: narrative bullets from Work and AFS sections (non-checkbox)
        for section in ("Work", "AFS / Side Projects"):
            for bullet in _extract_section_bullets(text, section):
                if not _CHECKBOX_UNCHECKED_RE.match(bullet) and not _CHECKBOX_CHECKED_RE.match(bullet):
                    wins.append(f"{bullet} ({d.strftime('%a')})")

        # Track unchecked items with dates
        for item in _extract_unchecked(text):
            if item not in all_unchecked:
                all_unchecked[item] = []
            all_unchecked[item].append(d)

    # Carry-over: unchecked items from the last scanned day
    last_day = min(sunday, today)
    last_path = _daily_path(last_day, daily_dir)
    if last_path.exists():
        carry_items = _extract_unchecked(last_path.read_text(encoding="utf-8"))
    else:
        # Fall back to items seen on the latest day with a file
        if all_unchecked:
            latest = max(d for dates in all_unchecked.values() for d in dates)
            carry_items = [item for item, dates in all_unchecked.items() if latest in dates]
        else:
            carry_items = []

    # Misses: items that appeared during the week but aren't in carry-over (dropped silently)
    carry_set = set(carry_items)
    misses = [item for item in all_unchecked if item not in carry_set]

    # Build content
    def _fmt(items: list[str], prefix: str = "- ") -> str:
        return "".join(f"{prefix}{item}\n" for item in items) if items else f"{prefix}(none)\n"

    carry_lines = "\n".join(_todo_to_checkbox(item) for item in carry_items) + "\n" if carry_items else "- (none)\n"

    content = (
        f"#+TITLE: Weekly Review — {week_label} ({month_range})\n"
        f"#+AUTHOR: AFS\n"
        f"\n"
        f"* Wins\n"
        f"{_fmt(wins)}"
        f"\n"
        f"* Misses\n"
        f"{_fmt(misses)}"
        f"\n"
        f"* Carry Over\n"
        f"{carry_lines}"
        f"\n"
        f"* Top 3 for Next Week\n"
        f"1. \n"
        f"2. \n"
        f"3. \n"
    )

    weekly_dir.mkdir(parents=True, exist_ok=True)
    out_path = weekly_dir / f"{week_label}.org"
    existed = out_path.exists()

    if existed and not overwrite:
        return str(out_path), {
            "action": "skipped",
            "reason": "file exists (pass --overwrite to replace)",
            "path": str(out_path),
        }

    out_path.write_text(content, encoding="utf-8")
    return str(out_path), {
        "action": "updated" if existed else "created",
        "path": str(out_path),
        "week": week_label,
        "wins_count": len(wins),
        "misses_count": len(misses),
        "carry_count": len(carry_items),
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    parser = build_base_parser(AGENT_DESCRIPTION)
    parser.add_argument(
        "--task",
        choices=["carry-forward", "template-gen", "stale-check", "weekly-review", "all"],
        default="all",
        help=(
            "Task to execute. "
            "carry-forward: extract unchecked TODOs from source date. "
            "template-gen: create/update tomorrow's daily entry with carry-over. "
            "stale-check: flag TODOs unchecked for 3+ consecutive days. "
            "weekly-review: draft this week's review file. "
            "all: carry-forward + template-gen + stale-check (default)."
        ),
    )
    parser.add_argument(
        "--date",
        metavar="YYYY-MM-DD",
        help="Source date for carry-forward (default: yesterday).",
    )
    parser.add_argument(
        "--week",
        metavar="YYYY-WNN",
        help="ISO week for weekly-review (default: current week).",
    )
    parser.add_argument(
        "--daily-dir",
        default=str(_DEFAULT_DAILY_DIR),
        help=f"Daily journal directory (default: {_DEFAULT_DAILY_DIR}).",
    )
    parser.add_argument(
        "--weekly-dir",
        default=str(_DEFAULT_WEEKLY_DIR),
        help=f"Weekly journal directory (default: {_DEFAULT_WEEKLY_DIR}).",
    )
    parser.add_argument(
        "--scan-days",
        type=int,
        default=14,
        help="Days of history to scan for stale-check (default: 14).",
    )
    parser.add_argument(
        "--stale-threshold",
        type=int,
        default=3,
        help="Consecutive-day streak before a TODO is considered stale (default: 3).",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files instead of skipping or appending.",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    configure_logging(args.quiet)

    # Load context snapshot for awareness of recent events and memory
    try:
        from ..agent_context import load_agent_context_snapshot
        ctx = load_agent_context_snapshot()
        if ctx and ctx.recent_events:
            import logging as _log
            _log.getLogger(__name__).info(
                "Journal agent context: %d recent events, %d memory topics",
                len(ctx.recent_events), len(ctx.memory_topics),
            )
    except Exception:
        pass

    daily_dir = Path(args.daily_dir).expanduser()
    weekly_dir = Path(args.weekly_dir).expanduser()
    today = date.today()

    source_date = date.fromisoformat(args.date) if args.date else today - timedelta(days=1)
    target_date = today + timedelta(days=1)

    started_at = now_iso()
    start = time.monotonic()

    payload: dict[str, Any] = {}
    notes: list[str] = []
    metrics: dict[str, Any] = {}
    status = "ok"
    task = args.task

    # carry-forward (also used by template-gen and all)
    carry_items: list[str] = []
    if task in ("carry-forward", "template-gen", "all"):
        carry_items, cf_meta = run_carry_forward(source_date, daily_dir)
        payload["carry_forward"] = cf_meta
        metrics["carry_forward_count"] = len(carry_items)
        if not cf_meta["found"]:
            notes.append(f"no daily file for {source_date} — carry-over will be empty")
            if task == "carry-forward":
                status = "warn"

    # template-gen
    if task in ("template-gen", "all"):
        tg_path, tg_meta = run_template_gen(target_date, carry_items, daily_dir, args.overwrite)
        payload["template_gen"] = tg_meta
        metrics["template_items_added"] = tg_meta.get("items_added", 0)
        notes.append(f"template: {tg_meta['action']} → {tg_path}")

    # stale-check
    if task in ("stale-check", "all"):
        stale, sc_meta = run_stale_check(
            daily_dir,
            scan_days=args.scan_days,
            stale_threshold=args.stale_threshold,
            today=today,
        )
        payload["stale_check"] = {**sc_meta, "stale_todos": stale}
        metrics["stale_count"] = len(stale)
        if stale:
            notes.append(
                f"STALE: {len(stale)} TODO(s) unchecked for {args.stale_threshold}+ days — "
                + ", ".join(f'"{s["todo"][:40]}"' for s in stale[:3])
                + ("…" if len(stale) > 3 else "")
            )

    # weekly-review
    if task == "weekly-review":
        if args.week:
            parts = args.week.split("-W")
            iso_year, iso_wk = int(parts[0]), int(parts[1])
            week_date = date.fromisocalendar(iso_year, iso_wk, 1)
        else:
            week_date = today
        wr_path, wr_meta = run_weekly_review(week_date, daily_dir, weekly_dir, args.overwrite)
        payload["weekly_review"] = wr_meta
        metrics.update(
            {
                "wins_count": wr_meta.get("wins_count", 0),
                "misses_count": wr_meta.get("misses_count", 0),
                "carry_count": wr_meta.get("carry_count", 0),
            }
        )
        notes.append(f"weekly-review: {wr_meta['action']} → {wr_path}")

    finished_at = now_iso()
    duration = time.monotonic() - start

    result = AgentResult(
        name=AGENT_NAME,
        status=status,
        started_at=started_at,
        finished_at=finished_at,
        duration_seconds=duration,
        task=f"journal:{task}",
        metrics=metrics,
        notes=notes,
        payload=payload,
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
