from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from afs.coordination import (
    append_inbox_line,
    format_inbox_line,
    mark_inbox_done,
    parse_board,
    parse_inbox,
    read_inbox,
    set_board_section,
)
from afs.file_io import FileWriteConflict, read_text_snapshot, write_text_checked

NOW = datetime(2026, 10, 20, 2, 37, tzinfo=timezone.utc)

INBOX = """Current goal: Ship 1.0 by Christmas (handoffs/plan/PLAN.md)
FORMAT: YYYY-MM-DD HH:MMZ from -> to: ask (path). One line.
DONE 2026-10-18 12:50Z claude -> codex/cursor: Claim: dream scene. Stay off. (handoffs/dream/WORK.md)
2026-10-18 codex -> claude/cursor: Claim Bombchu crawl fix (handoffs/bombchu/WORK.md).
2026-10-20 02:04Z cursor -> codex-lyra/all: Repin the dev project. (handoffs/docks/PLAN.md)
2026-10-20 02:32Z cursor-yaze (080-client-update) -> all: client update needs a native check
## stray heading
"""

BOARD = """# Agent board — each owner overwrites only its own section.
Heavy lock: unknown — truth is /tmp/oos-heavy.lock/owner
Current goal: Ship 1.0

## Codex — 2026-10-20 01:20Z
Result: Docks preview complete.
Ran: Build 14/14.
Next: integrate.
Holds: No heavy lock.
Handoff: handoffs/runway/PLAN.md

## Claude-Kydrog — 2026-10-19 20:02Z
Result: PR 214 art.
Ran: build.
Next: paused.
Holds: kydrog_boss.asm
Handoff: handoffs/kydrog/WORK.md
"""


def test_parse_inbox_reads_goal_entries_and_paths() -> None:
    inbox = parse_inbox(INBOX)
    assert inbox.goal == "Ship 1.0 by Christmas (handoffs/plan/PLAN.md)"
    assert len(inbox.entries) == 4
    assert inbox.unparsed_lines == 1
    first, second, third, fourth = inbox.entries
    assert first.done and first.time == "12:50" and first.path == "handoffs/dream/WORK.md"
    assert not second.done and second.time == "" and second.to == "claude/cursor"
    assert third.sender == "cursor" and third.to == "codex-lyra/all"
    assert fourth.sender == "cursor-yaze (080-client-update)" and fourth.path == ""


def test_open_for_matches_recipients_and_skips_own_and_done_lines() -> None:
    inbox = parse_inbox(INBOX)
    assert [entry.line_no for entry in inbox.open_for("cursor")] == [4, 6]
    assert [entry.line_no for entry in inbox.open_for("codex")] == [5, 6]
    assert [entry.line_no for entry in inbox.open_for("claude")] == [4, 5, 6]


def test_format_inbox_line_stamps_utc_and_enforces_limits() -> None:
    line = format_inbox_line("cursor", "codex", "Release the lane", path="handoffs/x.md", now=NOW)
    assert line == "2026-10-20 02:37Z cursor -> codex: Release the lane (handoffs/x.md)"
    with pytest.raises(ValueError, match="limit is 240"):
        format_inbox_line("cursor", "codex", "x" * 240, now=NOW)
    with pytest.raises(ValueError, match="single line"):
        format_inbox_line("cursor", "codex", "a\nb", now=NOW)
    with pytest.raises(ValueError):
        format_inbox_line("cursor", "codex: x", "ask", now=NOW)


def test_append_and_done_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "inbox.md"
    path.write_text(INBOX.rstrip("\n"), encoding="utf-8")
    append_inbox_line(path, format_inbox_line("codex", "cursor", "Review #300", now=NOW))
    inbox, _digest = read_inbox(path)
    assert inbox.entries[-1].text == "Review #300"
    assert path.read_text(encoding="utf-8").endswith("cursor: Review #300\n")

    entry = mark_inbox_done(path, match="Review #300")
    assert entry.text == "Review #300"
    assert parse_inbox(path.read_text(encoding="utf-8")).entries[-1].done
    with pytest.raises(LookupError):
        mark_inbox_done(path, match="Review #300")
    with pytest.raises(LookupError, match="3 open lines"):
        mark_inbox_done(path, match="cursor")


def test_append_creates_inbox_with_header(tmp_path: Path) -> None:
    path = tmp_path / "scratch" / "inbox.md"
    append_inbox_line(path, format_inbox_line("codex", "all", "Lane free", now=NOW))
    inbox, _digest = read_inbox(path)
    assert inbox.goal == "(set by the user)"
    assert [entry.text for entry in inbox.entries] == ["Lane free"]


def test_writes_fail_on_concurrent_change(tmp_path: Path) -> None:
    path = tmp_path / "inbox.md"
    path.write_text(INBOX, encoding="utf-8")
    _text, digest = read_text_snapshot(path)
    path.write_text(INBOX + "2026-10-20 03:00Z claude -> codex: new ask\n", encoding="utf-8")
    with pytest.raises(FileWriteConflict):
        write_text_checked(path, "replacement", if_match=digest)


def test_parse_board_sections_and_pinned_lines() -> None:
    board = parse_board(BOARD)
    assert board.pinned()["Heavy lock"].startswith("unknown")
    assert [section.owner for section in board.sections] == ["Codex", "Claude-Kydrog"]
    assert board.section("claude-kydrog").fields["Holds"] == "kydrog_boss.asm"
    assert board.section("claude") is None


def test_set_board_section_replaces_only_that_owner(tmp_path: Path) -> None:
    path = tmp_path / "agent_board.md"
    path.write_text(BOARD, encoding="utf-8")
    set_board_section(
        path,
        "Codex",
        {"Result": "Merged", "Ran": "tests", "Next": "", "Holds": "", "Handoff": "h.md"},
        now=NOW,
    )
    text = path.read_text(encoding="utf-8")
    board = parse_board(text)
    assert [section.owner for section in board.sections] == ["Codex", "Claude-Kydrog"]
    assert board.section("Codex").stamp == "2026-10-20 02:37Z"
    assert board.section("Codex").fields == {
        "Result": "Merged",
        "Ran": "tests",
        "Next": "none",
        "Holds": "none",
        "Handoff": "h.md",
    }
    assert "## Claude-Kydrog — 2026-10-19 20:02Z\nResult: PR 214 art." in text
    assert text.startswith("# Agent board")

    set_board_section(path, "Cursor", {"Result": "new owner"}, now=NOW)
    assert [s.owner for s in parse_board(path.read_text(encoding="utf-8")).sections] == [
        "Codex",
        "Claude-Kydrog",
        "Cursor",
    ]
    with pytest.raises(ValueError, match="limit is 160"):
        set_board_section(path, "Cursor", {"Result": "x" * 161}, now=NOW)
