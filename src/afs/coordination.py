"""Read and write the plain-text coordination files agents keep in a scratchpad.

Two files, one line or section per entry, designed to stay small enough to
read at every session start:

``inbox.md``
    Line 1 ``Current goal: ...`` (or ``GOAL: ...``), line 2 a ``FORMAT:``
    reminder, then one line per open ask, claim or release::

        YYYY-MM-DD HH:MMZ from -> to: ask (path)

    Answered lines are prefixed ``DONE``; a compaction script archives them.

``agent_board.md``
    A title and pinned header lines (heavy lock, goal), then one section per
    owner that only that owner overwrites::

        ## <Owner> — YYYY-MM-DD HH:MMZ
        Result: ...
        Ran: ...
        Next: ...
        Holds: ...
        Handoff: ...

Every write re-reads the file under the AFS writer lock and fails with
``FileWriteConflict`` if another writer changed it in between.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .file_io import read_text_snapshot, write_text_checked
from .hivemind import normalize_agent_name, recipient_matches

INBOX_FILE = "inbox.md"
BOARD_FILE = "agent_board.md"
INBOX_MAX_CHARS = 240
BOARD_FIELD_MAX_CHARS = 160
BOARD_FIELDS = ("Result", "Ran", "Next", "Holds", "Handoff")
GOAL_PREFIXES = ("Current goal:", "GOAL:")

_INBOX_LINE = re.compile(
    r"^(?P<done>DONE\s+)?"
    r"(?P<date>\d{4}-\d{2}-\d{2})(?:\s+(?P<time>\d{2}:\d{2})Z)?\s+"
    r"(?P<sender>[^\s].*?)\s*->\s*(?P<to>[^:]+?)\s*:\s*(?P<text>.*)$"
)
_TRAILING_PATH = re.compile(r"\(([^()\s][^()]*)\)\s*\.?\s*$")
_BOARD_HEADING = re.compile(r"^##\s+(?P<owner>.+?)\s+[—-]\s+(?P<stamp>\d{4}-\d{2}-\d{2}.*)$")


@dataclass(frozen=True)
class InboxEntry:
    line_no: int
    raw: str
    done: bool
    date: str
    time: str
    sender: str
    to: str
    text: str
    path: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class Inbox:
    goal: str
    entries: tuple[InboxEntry, ...]
    unparsed_lines: int

    def open_for(self, recipient: str) -> list[InboxEntry]:
        """Open lines addressed to *recipient*, excluding its own lines."""
        me = normalize_agent_name(recipient)
        return [
            entry
            for entry in self.entries
            if not entry.done
            and normalize_agent_name(entry.sender) != me
            and recipient_matches(entry.to, recipient)
        ]


@dataclass(frozen=True)
class BoardSection:
    owner: str
    stamp: str
    start: int
    end: int
    fields: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {"owner": self.owner, "stamp": self.stamp, "fields": dict(self.fields)}


@dataclass(frozen=True)
class Board:
    header: tuple[str, ...]
    sections: tuple[BoardSection, ...]

    def section(self, owner: str) -> BoardSection | None:
        wanted = normalize_agent_name(owner)
        for section in self.sections:
            if normalize_agent_name(section.owner) == wanted:
                return section
        return None

    def pinned(self) -> dict[str, str]:
        """Header lines of the form ``Label: value`` (heavy lock, goal)."""
        result: dict[str, str] = {}
        for line in self.header:
            label, sep, value = line.partition(":")
            if sep and label and not label.startswith("#") and len(label) <= 40:
                result[label.strip()] = value.strip()
        return result


def utc_stamp(now: datetime | None = None) -> tuple[str, str]:
    current = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    return current.strftime("%Y-%m-%d"), current.strftime("%H:%M")


def parse_inbox(text: str) -> Inbox:
    goal = ""
    entries: list[InboxEntry] = []
    unparsed = 0
    for index, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        if not goal and stripped.startswith(GOAL_PREFIXES):
            goal = stripped.split(":", 1)[1].strip()
            continue
        if stripped.startswith("FORMAT:"):
            continue
        match = _INBOX_LINE.match(stripped)
        if match is None:
            unparsed += 1
            continue
        body = match.group("text").strip()
        path_match = _TRAILING_PATH.search(body)
        entries.append(
            InboxEntry(
                line_no=index,
                raw=line,
                done=bool(match.group("done")),
                date=match.group("date"),
                time=match.group("time") or "",
                sender=match.group("sender").strip(),
                to=match.group("to").strip(),
                text=body,
                path=path_match.group(1).strip() if path_match else "",
            )
        )
    return Inbox(goal=goal, entries=tuple(entries), unparsed_lines=unparsed)


def format_inbox_line(
    sender: str, to: str, ask: str, *, path: str = "", now: datetime | None = None
) -> str:
    for label, value in (("from", sender), ("to", to), ("ask", ask), ("path", path)):
        if "\n" in value or "\r" in value:
            raise ValueError(f"inbox {label} must be a single line")
    if not sender.strip() or not to.strip() or not ask.strip():
        raise ValueError("inbox lines need from, to and ask")
    if "->" in sender or ":" in to:
        raise ValueError("'from' cannot contain '->' and 'to' cannot contain ':'")
    date, clock = utc_stamp(now)
    line = f"{date} {clock}Z {sender.strip()} -> {to.strip()}: {ask.strip()}"
    if path.strip():
        line += f" ({path.strip()})"
    if len(line) > INBOX_MAX_CHARS:
        raise ValueError(
            f"inbox line is {len(line)} chars; the limit is {INBOX_MAX_CHARS}. "
            "Put detail in the handoff file and point to it."
        )
    return line


def parse_board(text: str) -> Board:
    lines = text.splitlines()
    header: list[str] = []
    sections: list[BoardSection] = []
    current: dict[str, Any] | None = None

    def close(end: int) -> None:
        if current is not None:
            sections.append(
                BoardSection(
                    owner=current["owner"],
                    stamp=current["stamp"],
                    start=current["start"],
                    end=end,
                    fields=current["fields"],
                )
            )

    for index, line in enumerate(lines):
        if line.startswith("## "):
            close(index)
            match = _BOARD_HEADING.match(line.strip())
            current = {
                "owner": match.group("owner").strip() if match else line[3:].strip(),
                "stamp": match.group("stamp").strip() if match else "",
                "start": index,
                "fields": {},
            }
            continue
        if current is None:
            header.append(line)
            continue
        label, sep, value = line.partition(":")
        if sep and label.strip() in BOARD_FIELDS:
            current["fields"][label.strip()] = value.strip()
    close(len(lines))
    return Board(header=tuple(header), sections=tuple(sections))


def render_board_section(
    owner: str, fields: dict[str, str], *, now: datetime | None = None
) -> list[str]:
    if not owner.strip() or "\n" in owner or "—" in owner:
        raise ValueError("board owner must be a single line without '—'")
    date, clock = utc_stamp(now)
    lines = [f"## {owner.strip()} — {date} {clock}Z"]
    for label in BOARD_FIELDS:
        value = " ".join(str(fields.get(label, "")).split())
        if len(value) > BOARD_FIELD_MAX_CHARS:
            raise ValueError(
                f"board {label} is {len(value)} chars; the limit is {BOARD_FIELD_MAX_CHARS}"
            )
        lines.append(f"{label}: {value or 'none'}")
    return lines


def resolve_coordination_file(roots: list[Path] | tuple[Path, ...], name: str) -> Path:
    """Pick the most specific scratchpad root that already has *name*, else the first root."""
    if not roots:
        raise FileNotFoundError("no scratchpad root is visible for this context")
    for root in roots:
        candidate = root / name
        if candidate.exists():
            return candidate
    return roots[0] / name


def read_inbox(path: Path) -> tuple[Inbox, str]:
    if not path.is_file():
        return Inbox(goal="", entries=(), unparsed_lines=0), "missing"
    text, digest = read_text_snapshot(path, preserve_newlines=True)
    return parse_inbox(text), digest


def append_inbox_line(path: Path, line: str) -> str:
    """Append one formatted line; fails on a concurrent change."""
    if path.is_file():
        text, digest = read_text_snapshot(path, preserve_newlines=True)
        separator = "" if not text or text.endswith("\n") else "\n"
        return write_text_checked(path, f"{text}{separator}{line}\n", if_match=digest)
    path.parent.mkdir(parents=True, exist_ok=True)
    header = (
        "Current goal: (set by the user)\n"
        f"FORMAT: YYYY-MM-DD HH:MMZ from -> to: ask (path). One line, <={INBOX_MAX_CHARS} chars. "
        "Asks, claims and releases only; status goes on agent_board.md. Prefix DONE when answered.\n"
    )
    return write_text_checked(path, f"{header}{line}\n", if_match="missing")


def mark_inbox_done(path: Path, *, line_no: int | None = None, match: str = "") -> InboxEntry:
    """Prefix ``DONE`` on one open line chosen by line number or unique substring."""
    text, digest = read_text_snapshot(path, preserve_newlines=True)
    inbox = parse_inbox(text)
    candidates = [entry for entry in inbox.entries if not entry.done]
    if line_no is not None:
        candidates = [entry for entry in candidates if entry.line_no == line_no]
    if match:
        candidates = [entry for entry in candidates if match in entry.raw]
    if not candidates:
        raise LookupError("no open inbox line matches")
    if len(candidates) > 1:
        numbers = ", ".join(str(entry.line_no) for entry in candidates[:10])
        raise LookupError(f"{len(candidates)} open lines match (lines {numbers}); be more specific")
    target = candidates[0]
    lines = text.splitlines(keepends=True)
    original = lines[target.line_no - 1]
    indent = original[: len(original) - len(original.lstrip())]
    lines[target.line_no - 1] = f"{indent}DONE {original.lstrip()}"
    write_text_checked(path, "".join(lines), if_match=digest)
    return target


def read_board(path: Path) -> tuple[Board, str]:
    if not path.is_file():
        return Board(header=(), sections=()), "missing"
    text, digest = read_text_snapshot(path, preserve_newlines=True)
    return parse_board(text), digest


def set_board_section(
    path: Path, owner: str, fields: dict[str, str], *, now: datetime | None = None
) -> list[str]:
    """Replace *owner*'s section (or append one), leaving every other byte alone."""
    rendered = render_board_section(owner, fields, now=now)
    if path.is_file():
        text, digest = read_text_snapshot(path, preserve_newlines=True)
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        text, digest = (
            "# Agent board — each owner overwrites only its own section "
            f"(5 lines, <={BOARD_FIELD_MAX_CHARS} chars each).\n",
            "missing",
        )
    board = parse_board(text)
    lines = text.splitlines()
    existing = board.section(owner)
    block = [*rendered, ""]
    if existing is None:
        while lines and not lines[-1].strip():
            lines.pop()
        lines = [*lines, "", *block]
    else:
        lines = [*lines[: existing.start], *block, *lines[existing.end :]]
    while lines and not lines[-1].strip():
        lines.pop()
    write_text_checked(path, "\n".join(lines) + "\n", if_match=digest)
    return rendered


__all__ = [
    "BOARD_FIELDS",
    "BOARD_FILE",
    "Board",
    "BoardSection",
    "INBOX_FILE",
    "Inbox",
    "InboxEntry",
    "append_inbox_line",
    "format_inbox_line",
    "mark_inbox_done",
    "parse_board",
    "parse_inbox",
    "read_board",
    "read_inbox",
    "render_board_section",
    "resolve_coordination_file",
    "set_board_section",
]
