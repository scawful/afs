"""Bounded startup context for agents with native reasoning and skill loading."""

from __future__ import annotations

import logging
import re
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .context_layout import LAYOUT_VERSION
from .manager import AFSManager
from .models import MountType
from .profiles import resolve_active_profile
from .scopes import ResolvedScope, resolve_scope, visible_mount_roots
from .skills import resolve_skill_roots

logger = logging.getLogger(__name__)

MAX_PENDING_MESSAGES = 5
MAX_PENDING_HANDOFFS = 3
MAX_INBOX_LINES = 6
MAX_BOARD_HOLDS = 6
INBOX_RECENT_HOURS = 72
LINE_CHARS = 240
_LOCK_PATH = re.compile(r"(/\S+?\.lock)\b")
_NO_HOLDS = re.compile(r"^(none|no\b|-|all claims released)", re.IGNORECASE)


def _clip(text: str, width: int = LINE_CHARS) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= width else text[: width - 3] + "..."


def _pending_messages(
    manager: AFSManager, context_path: Path, scope: ResolvedScope, agent: str
) -> dict[str, Any]:
    from .hivemind import message_summary
    from .messages import MessageBus

    bus = MessageBus(
        context_path,
        scope_id=scope.scope_id,
        config=manager.config,
        include_legacy=scope.layout_version != LAYOUT_VERSION,
    )
    entries = bus.inbox(agent, limit=0)
    return {
        "total": len(entries),
        "items": [
            {
                "id": entry["id"],
                "from": entry["from"],
                "to": entry.get("to"),
                "state": entry["state"],
                "summary": message_summary(entry.get("payload"), width=160),
            }
            for entry in entries[-MAX_PENDING_MESSAGES:]
        ],
        "ack_command": f"afs messages ack <id> --by {agent} --done",
    }


def _pending_handoffs(
    manager: AFSManager, context_path: Path, scope: ResolvedScope, agent: str
) -> dict[str, Any]:
    from .handoff import HandoffStore

    store = HandoffStore(context_path, scope_id=scope.scope_id, config=manager.config)
    pending = store.pending_for_agent(agent)
    return {
        "total": len(pending),
        "items": [
            {
                "revision_id": packet.revision_id,
                "title": packet.title,
                "from": packet.agent_name,
                "priority": packet.priority,
                "next": [_clip(step, 160) for step in packet.next_steps[:2]],
            }
            for packet in pending[:MAX_PENDING_HANDOFFS]
        ],
        "ack_command": f"afs handoff ack <revision> --by {agent}",
    }


def _coordination_files(roots: tuple[Path, ...], agent: str, now: datetime) -> dict[str, Any]:
    from . import leases
    from .coordination import (
        BOARD_FILE,
        INBOX_FILE,
        read_board,
        read_inbox,
        resolve_coordination_file,
    )

    result: dict[str, Any] = {}
    inbox_path = resolve_coordination_file(roots, INBOX_FILE)
    if inbox_path.is_file():
        inbox, _digest = read_inbox(inbox_path)
        cutoff = (now - timedelta(hours=INBOX_RECENT_HOURS)).strftime("%Y-%m-%d")
        recent = [entry for entry in inbox.open_for(agent) if entry.date >= cutoff] if agent else []
        result["inbox"] = {
            "path": str(inbox_path),
            "goal": _clip(inbox.goal, 400),
            "open_for_agent": len(recent),
            "lines": [f"{entry.line_no}: {_clip(entry.raw)}" for entry in recent[-MAX_INBOX_LINES:]],
            "done_command": f"afs inbox done --line N --path {inbox_path.parent}",
        }
    board_path = resolve_coordination_file(roots, BOARD_FILE)
    if board_path.is_file():
        board, _digest = read_board(board_path)
        sections = sorted(board.sections, key=lambda section: section.stamp, reverse=True)
        holds = [
            {"owner": section.owner, "stamp": section.stamp, "holds": _clip(section.fields["Holds"], 160)}
            for section in sections
            if section.fields.get("Holds") and not _NO_HOLDS.match(section.fields["Holds"])
        ]
        pinned = board.pinned()
        result["board"] = {
            "path": str(board_path),
            "pinned": {label: _clip(value, 240) for label, value in pinned.items()},
            "holds": holds[:MAX_BOARD_HOLDS],
            "own_section": next(
                (s.to_dict() for s in board.sections if agent and s.owner.lower() == agent.lower()),
                None,
            ),
        }
        lock_paths = sorted(
            {match.group(1) for value in pinned.values() for match in _LOCK_PATH.finditer(value)}
        )
        if lock_paths:
            result["leases"] = [leases.describe(leases.inspect(Path(path))) for path in lock_paths[:3]]
    return result


def build_session_brief(
    manager: AFSManager,
    context_path: Path,
    *,
    project_path: Path | None = None,
    agent_name: str = "",
) -> dict[str, Any]:
    """Read named notes and what is waiting for *agent_name*; never walk skill trees."""
    from .session_bootstrap import _safe_scoped_candidate

    context_path = context_path.expanduser().resolve()
    scope = resolve_scope(context_path, requester_path=project_path)
    scratchpad_root = manager.resolve_mount_root(context_path, MountType.SCRATCHPAD)
    roots = visible_mount_roots(
        scratchpad_root, mount_type=MountType.SCRATCHPAD, scoped=scope
    )
    notes: list[dict[str, Any]] = []
    for root in roots:
        for name in ("state.md", "deferred.md"):
            path = _safe_scoped_candidate(root / name, root=root, scoped=scope)
            if path is None or not path.is_file():
                continue
            with path.open(encoding="utf-8", errors="replace") as stream:
                text = stream.read(1201)
            notes.append({"path": str(path), "text": text[:1200], "truncated": len(text) > 1200})
    agent = agent_name.strip() if agent_name and agent_name != "cli" else ""
    pending: dict[str, Any] = {"agent": agent}
    collectors: list[tuple[str, Any]] = [
        ("coordination", lambda: _coordination_files(roots, agent, datetime.now(timezone.utc))),
    ]
    if agent:
        collectors.extend(
            [
                ("messages", lambda: _pending_messages(manager, context_path, scope, agent)),
                ("handoffs", lambda: _pending_handoffs(manager, context_path, scope, agent)),
            ]
        )
    for key, collect in collectors:
        try:
            value = collect()
        except (OSError, ValueError, PermissionError, RuntimeError) as exc:
            logger.warning("session brief: %s unavailable: %s", key, exc)
            pending.setdefault("errors", []).append(f"{key}: {exc}")
            continue
        if key == "coordination":
            pending.update(value)
        else:
            pending[key] = value
    profile = resolve_active_profile(manager.config)
    next_commands = [
        "afs context query --help",
        "afs session bootstrap --json --no-write-artifacts",
    ]
    if agent:
        next_commands.insert(0, f"afs messages list --to {agent}")
    return {
        "compact": True,
        "context_path": str(context_path),
        "scope_id": scope.scope_id,
        "layout_version": scope.layout_version,
        "profile": profile.name,
        "scratchpad_roots": [str(root) for root in roots],
        "notes": notes,
        "pending": pending,
        "skills": {
            "mode": "native",
            "roots": [str(root) for root in resolve_skill_roots(list(profile.skill_roots))],
            "matches": [],
        },
        "next_commands": next_commands,
        "guidance": (
            "Use the host's skill loader. Act on `pending` first: ack messages and handoffs "
            "you take, prefix DONE on inbox lines you answer, update your board section. "
            "Query context when the task needs prior decisions. Use the full bootstrap for "
            "diagnostics."
        ),
    }


def render_session_brief(summary: dict[str, Any]) -> str:
    lines = [f"AFS: {summary['context_path']}", f"Scope: {summary['scope_id']}"]
    pending = summary.get("pending") or {}
    inbox = pending.get("inbox") or {}
    if inbox.get("goal"):
        lines.append(f"Goal: {inbox['goal']}")
    for lease in pending.get("leases", []):
        lines.append(f"Lease: {lease}")
    agent = pending.get("agent") or "this agent"
    messages = pending.get("messages") or {}
    if messages.get("total"):
        lines.extend(["", f"Messages for {agent} ({messages['total']}):"])
        for item in messages["items"]:
            lines.append(f"- {item['id']} [{item['state']}] {item['from']}: {item['summary']}")
    handoffs = pending.get("handoffs") or {}
    if handoffs.get("total"):
        lines.extend(["", f"Handoffs for {agent} ({handoffs['total']}):"])
        for item in handoffs["items"]:
            lines.append(f"- {item['revision_id']} {item['title']} (from {item['from']})")
    if inbox.get("lines"):
        lines.extend(["", f"Inbox lines for {agent} ({inbox['open_for_agent']} open, {inbox['path']}):"])
        lines.extend(f"- {line}" for line in inbox["lines"])
    board = pending.get("board") or {}
    if board.get("holds"):
        lines.extend(["", f"Board holds ({board['path']}):"])
        lines.extend(f"- {h['owner']} {h['stamp']}: {h['holds']}" for h in board["holds"])
    for note in summary["notes"]:
        lines.extend(["", note["path"], note["text"]])
        if note["truncated"]:
            lines.append("[truncated; read the source when relevant]")
    lines.extend(["", "Skill roots:", *summary["skills"]["roots"], "", summary["guidance"]])
    return "\n".join(lines)
