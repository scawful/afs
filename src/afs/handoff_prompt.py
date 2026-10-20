"""Paste-ready next-agent prompts and supersede markers for handoffs.

Two handoff shapes exist in practice: AFS handoff revisions (``afs handoff
create``) and plain handoff folders (``handoffs/<topic>_<date>/`` with
``NEXT_AGENT.md``/``HANDOFF.md``/``README.md``). Both render into the same
prompt so any harness (Codex, Claude, Cursor, Antigravity) starts the same way.
Folder files are often referenced by SHA, so supersede never edits them; it
writes a separate ``SUPERSEDED.json`` beside them.
"""

from __future__ import annotations

import json
import shlex
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from .file_io import write_text_checked
from .handoff import HandoffPacket

SUPERSEDED_FILE = "SUPERSEDED.json"
FOLDER_ENTRY_FILES = (
    "NEXT_AGENT.md",
    "HANDOFF.md",
    "README.md",
    "WORK.md",
    "PLAN.md",
    "NOTES.md",
)
MAX_ENTRY_CHARS = 6000
MAX_LISTED_FILES = 20


@dataclass(frozen=True)
class SupersedeMarker:
    folder: str
    superseded_by: str
    by: str
    reason: str
    at: str

    def to_dict(self) -> dict[str, str]:
        return {
            "folder": self.folder,
            "superseded_by": self.superseded_by,
            "by": self.by,
            "reason": self.reason,
            "at": self.at,
        }


def read_supersede_marker(folder: Path) -> SupersedeMarker | None:
    path = folder / SUPERSEDED_FILE
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or not isinstance(data.get("superseded_by"), str):
        return None
    return SupersedeMarker(
        folder=str(folder),
        superseded_by=data["superseded_by"],
        by=str(data.get("by", "")),
        reason=str(data.get("reason", "")),
        at=str(data.get("at", "")),
    )


def supersede_folder(
    folder: Path, *, superseded_by: str, by: str, reason: str = ""
) -> SupersedeMarker:
    """Mark ``folder`` superseded without touching its existing files."""
    folder = folder.expanduser().resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"handoff folder not found: {folder}")
    if not superseded_by.strip() or not by.strip():
        raise ValueError("superseded_by and by are required")
    marker = SupersedeMarker(
        folder=str(folder),
        superseded_by=superseded_by.strip(),
        by=by.strip(),
        reason=reason.strip(),
        at=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    write_text_checked(
        folder / SUPERSEDED_FILE,
        json.dumps(marker.to_dict(), indent=2) + "\n",
        if_match="missing",
    )
    return marker


def folder_entry_file(folder: Path) -> Path | None:
    for name in FOLDER_ENTRY_FILES:
        candidate = folder / name
        if candidate.is_file():
            return candidate
    return None


def _startup_lines(harness: str, project_path: Path | None) -> list[str]:
    path_arg = f" --path {shlex.quote(str(project_path))}" if project_path else ""
    return [
        "Start by loading shared context:",
        f"  afs session bootstrap --short --agent-name {harness}{path_arg}",
        f"  (MCP: prompt afs.session.bootstrap with agent_name={harness})",
        f"Check what is waiting for you: afs messages list --to {harness}; afs inbox list --to {harness}",
    ]


def _bullets(label: str, values: list[str]) -> list[str]:
    if not values:
        return []
    return [f"{label}:", *(f"- {value}" for value in values)]


def render_revision_prompt(
    packet: HandoffPacket,
    *,
    harness: str,
    latest: HandoffPacket | None = None,
    project_path: Path | None = None,
) -> str:
    lines = [
        f'You are {harness}, continuing handoff "{packet.title or packet.revision_id}"'
        f" from {packet.agent_name}.",
        "",
        *_startup_lines(harness, project_path),
        "",
        f"Handoff revision: {packet.revision_id} (thread {packet.stream_id})",
        f"Priority: {packet.priority}",
    ]
    if packet.closed:
        lines.append(
            "WARNING: this handoff thread is closed; confirm with the user before resuming."
        )
    if latest is not None and latest.revision_id != packet.revision_id:
        lines.append(
            f"WARNING: superseded by newer revision {latest.revision_id}"
            f' ("{latest.title}"); prefer that one.'
        )
    for label, values in (
        ("Accomplished", packet.accomplished),
        ("Blocked", packet.blocked),
        ("Next steps", packet.next_steps),
    ):
        block = _bullets(label, values)
        if block:
            lines.extend(["", *block])
    lines.extend(
        [
            "",
            f"When you pick this up: afs handoff ack {packet.revision_id} --by {harness}",
            "When you stop: afs handoff revise "
            f"{packet.revision_id} --title ... --agent-name {harness} --next ...",
        ]
    )
    return "\n".join(lines) + "\n"


def render_folder_prompt(folder: Path, *, harness: str, project_path: Path | None = None) -> str:
    folder = folder.expanduser().resolve()
    if not folder.is_dir():
        raise FileNotFoundError(f"handoff folder not found: {folder}")
    entry = folder_entry_file(folder)
    lines = [
        f"You are {harness}, continuing the handoff in {folder}.",
        "",
        *_startup_lines(harness, project_path),
        "",
    ]
    marker = read_supersede_marker(folder)
    if marker is not None:
        reason = f": {marker.reason}" if marker.reason else ""
        lines.extend(
            [
                f"WARNING: superseded by {marker.superseded_by} ({marker.by}, {marker.at}){reason}.",
                "Read the newer handoff first; use this one only for history.",
                "",
            ]
        )
    files = sorted(p.name for p in folder.iterdir() if p.is_file() and p.name != SUPERSEDED_FILE)
    listed = files[:MAX_LISTED_FILES]
    more = f" (+{len(files) - len(listed)} more)" if len(files) > len(listed) else ""
    lines.append(f"Files: {', '.join(listed) or '(none)'}{more}")
    if entry is not None:
        text = entry.read_text(encoding="utf-8", errors="replace")
        truncated = len(text) > MAX_ENTRY_CHARS
        lines.extend(["", f"--- {entry.name} ---", text[:MAX_ENTRY_CHARS].rstrip()])
        if truncated:
            lines.append(f"[truncated; read {entry} for the rest]")
        lines.append(f"--- end {entry.name} ---")
    lines.extend(
        [
            "",
            "Coordinate through the shared inbox/board, not new ad hoc files:",
            f'  afs inbox add --from {harness} --to <agent> --ref {folder.name} "<ask>"',
            f"  afs board set --owner {harness.capitalize()} --result ... --next ... --holds ...",
        ]
    )
    return "\n".join(lines) + "\n"
