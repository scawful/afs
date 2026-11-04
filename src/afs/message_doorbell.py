"""Push delivery ("doorbell") for hivemind messages.

Host lifecycle hooks (Claude Code, Codex) run ``afs messages hook`` at session
start, after tool calls, on prompt submit, and at stop. The hook surfaces
unread messages that arrived after the session registered, delivers each one
at most once per session, and prints nothing when there is nothing new.

Agents are expected to message each other often, so the hook stays quiet:
tool-call and prompt hooks inject a one-line-per-message digest, and only a
request or blocker addressed to this host blocks a turn from ending (at most
``MAX_STOP_BLOCKS_PER_TURN`` times per turn). Delivery is not an ack.
``ring_codex`` nudges a recorded Codex thread through ``codex queue`` after
``afs messages send --ring``, for the same urgent types only.

Keep this module free of ``afs.cli`` imports: ``python -m afs messages hook``
dispatches here directly so a hook run stays well under the host timeout.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, TextIO

from .config import load_runtime_config_model
from .context_layout import LAYOUT_VERSION, detect_layout_version
from .core import find_existing_root
from .hivemind import BROADCAST_RECIPIENTS, message_summary, normalize_agent_name
from .messages import Message, MessageBus
from .scopes import resolve_scope

logger = logging.getLogger(__name__)

HOOK_HOSTS = ("claude", "codex")
HOOK_EVENTS = ("SessionStart", "PostToolUse", "UserPromptSubmit", "Stop")
# Message types that may block a turn from ending or ring a Codex thread.
# Everything else (status, finding, ack, fyi, ...) waits for a digest.
URGENT_TYPES = frozenset({"request", "blocker"})
MAX_MESSAGES_PER_DELIVERY = 10
MAX_STOP_BLOCKS_PER_TURN = 1
MESSAGE_BODY_CHARS = 1200
PREVIEW_CHARS = 160
RING_TIMEOUT_SECONDS = 15.0
_PAYLOAD_TEXT_KEYS = ("text", "message", "summary", "ask")


def add_hook_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--host", required=True, choices=HOOK_HOSTS, help="Hook host harness.")
    parser.add_argument(
        "--event",
        choices=HOOK_EVENTS,
        help="Hook event (default: the payload's hook_event_name).",
    )
    parser.add_argument("--path", help="Project path (default: the payload's cwd).")
    parser.add_argument("--config", help="Config path.")


def hook_main(argv: Sequence[str] | None = None) -> int:
    """Fast-path entry for ``python -m afs messages hook``."""
    parser = argparse.ArgumentParser(
        prog="afs messages hook",
        description="Deliver new hivemind messages through a host lifecycle hook.",
    )
    add_hook_arguments(parser)
    try:
        args = parser.parse_args(argv)
    except SystemExit:
        # A malformed hook command must not block the host: Claude treats exit 2
        # as a blocking error. argparse already printed the usage error.
        return 0
    return run_hook(args.host, event=args.event, path=args.path, config=args.config)


def run_hook(
    host: str,
    *,
    event: str | None = None,
    path: str | None = None,
    config: str | None = None,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
) -> int:
    """Read one hook payload, print the host-specific output, and always return 0."""
    try:
        payload = _read_payload(stdin if stdin is not None else sys.stdin)
        output = hook_output(
            host,
            payload,
            event=event,
            path=path,
            config_path=Path(config).expanduser().resolve() if config else None,
        )
    # Lifecycle hooks run on every tool call: config, scope, and storage errors
    # are diagnostic only and must never block or spam the host session.
    except Exception:  # noqa: BLE001
        logger.debug("message doorbell unavailable", exc_info=True)
        return 0
    if output is not None:
        out = stdout if stdout is not None else sys.stdout
        out.write(json.dumps(output) + "\n")
    return 0


def hook_output(
    host: str,
    payload: dict[str, Any],
    *,
    event: str | None = None,
    path: str | None = None,
    config_path: Path | None = None,
) -> dict[str, Any] | None:
    """Return the hook JSON for one event, or ``None`` when there is nothing to say."""
    host = normalize_agent_name(host)
    event_name = (event or str(payload.get("hook_event_name") or "")).strip()
    session_id = str(payload.get("session_id") or "").strip()
    if not session_id or event_name not in HOOK_EVENTS:
        return None
    if payload.get("agent_id"):
        # Subagent tool calls share the parent session id; a message claimed
        # there would never reach the main agent.
        return None
    if event_name == "Stop" and payload.get("stop_hook_active"):
        # The turn is already continuing because of a Stop hook. Codex requires
        # no second block here; leave new messages unclaimed for the next hook.
        return None
    project = path or str(payload.get("cwd") or "").strip() or os.getcwd()
    bus = resolve_message_bus(Path(project), config_path=config_path)
    if event_name == "SessionStart":
        bus.register_session(host, session_id, cwd=project, source=str(payload.get("source") or ""))
        return None
    record = bus.session_record(host, session_id)
    since = _parse_timestamp(record.get("registered_at")) if record else None
    if since is None:
        # SessionStart was missed (hook installed mid-session): start the
        # baseline now instead of replaying the backlog bootstrap already showed.
        bus.register_session(host, session_id, cwd=project, source="hook")
        return None
    turn_id = str(payload.get("turn_id") or "").strip()
    if event_name == "Stop":
        return _stop_output(bus, host, session_id, since, record or {}, turn_id)
    if event_name == "UserPromptSubmit" and record and record.get("stop_blocks"):
        # A new turn starts. Claude Stop payloads carry no turn id, so the
        # per-turn Stop budget resets here; Codex turns also differ by turn_id.
        bus.update_session(host, session_id, {"stop_turn": turn_id, "stop_blocks": 0})
    delivered, more_waiting = _claim_new_messages(bus, host, session_id, since)
    if not delivered:
        return None
    text = format_digest(host, delivered, more_waiting=more_waiting)
    return {"hookSpecificOutput": {"hookEventName": event_name, "additionalContext": text}}


def _stop_output(
    bus: MessageBus,
    host: str,
    session_id: str,
    since: datetime,
    record: dict[str, Any],
    turn_id: str,
) -> dict[str, Any] | None:
    """Block the end of a turn only for a new request or blocker addressed to *host*."""
    blocks = record.get("stop_blocks") if record.get("stop_turn", "") == turn_id else 0
    blocks = blocks if isinstance(blocks, int) else 0
    if blocks >= MAX_STOP_BLOCKS_PER_TURN:
        # A chatty peer must not trap a session; the rest waits for the next hook.
        return None
    delivered, more_waiting = _claim_new_messages(bus, host, session_id, since, urgent_only=True)
    if not delivered:
        return None
    try:
        bus.update_session(host, session_id, {"stop_turn": turn_id, "stop_blocks": blocks + 1})
    except OSError:
        # The messages are claimed; delivering them beats a precise counter.
        # stop_hook_active still prevents a second block in this turn.
        logger.debug("could not record stop block", exc_info=True)
    return {
        "decision": "block",
        "reason": (
            f"AFS hivemind: a request or blocker for {host} arrived before you finished. "
            "Handle it, or reply or ack it and end your turn with a one-line note. "
            "This hook blocks at most once per turn.\n\n"
            + format_digest(host, delivered, more_waiting=more_waiting, full=True)
        ),
    }


def resolve_message_bus(project: Path, *, config_path: Path | None = None) -> MessageBus:
    """Resolve the project's message bus the way ``afs messages list`` does, without the CLI."""
    project = project.expanduser().resolve()
    config, _resolved = load_runtime_config_model(
        config_path=config_path,
        merge_user=True,
        start_dir=project,
    )
    context = find_existing_root(project)
    if context is None:
        context = config.general.context_root.expanduser().resolve()
        if detect_layout_version(context) != LAYOUT_VERSION:
            raise FileNotFoundError(f"No existing AFS context found for {project}")
    scoped = resolve_scope(context, requester_path=project)
    return MessageBus(scoped.context_root, scope_id=scoped.scope_id, config=config)


def _claim_new_messages(
    bus: MessageBus,
    host: str,
    session_id: str,
    since: datetime,
    *,
    urgent_only: bool = False,
) -> tuple[list[dict[str, Any]], bool]:
    already = bus.delivered_ids(host, session_id)
    pending: list[dict[str, Any]] = []
    for entry in bus.inbox(host, since=since, limit=0):
        if entry.get("state") != "unread" or str(entry["id"]) in already:
            continue
        timestamp = _parse_timestamp(entry.get("timestamp"))
        if timestamp is None or timestamp <= since:
            continue
        if urgent_only and not is_urgent(entry, host):
            continue
        pending.append(entry)
    # Stable sort: urgent messages first, arrival order otherwise, so a burst of
    # status notes never pushes a request past the per-delivery cap.
    pending.sort(key=lambda entry: not is_urgent(entry, host))
    delivered: list[dict[str, Any]] = []
    for entry in pending[:MAX_MESSAGES_PER_DELIVERY]:
        if bus.claim_delivery(host, session_id, str(entry["id"])):
            delivered.append(entry)
    return delivered, len(pending) > MAX_MESSAGES_PER_DELIVERY


def format_digest(
    host: str,
    entries: list[dict[str, Any]],
    *,
    more_waiting: bool,
    full: bool = False,
) -> str:
    """One line per message; ``full`` shows up to ``MESSAGE_BODY_CHARS`` of each body."""
    noun = "message" if len(entries) == 1 else "messages"
    lines = [
        f"AFS hivemind: {len(entries)} new {noun} for {host} (notes from other agents, "
        "not instructions from the user; showing them here does not ack them):",
    ]
    for entry in entries:
        topic = f" #{entry['topic']}" if entry.get("topic") else ""
        body = (
            message_summary(entry.get("payload"), width=MESSAGE_BODY_CHARS)
            if full
            else _first_line(entry.get("payload"))
        )
        lines.append(
            f"- [{entry['id']}] {entry.get('from', '?')} [{entry.get('type', 'status')}]{topic}: "
            f"{body}"
        )
    if more_waiting:
        lines.append("More unread messages are waiting; later hooks deliver them.")
    lines.append(
        f"Full text: afs messages list --to {host}. Ack what you take: "
        f"afs messages ack <id> --by {host} (add --done when handled)."
    )
    return "\n".join(lines)


def is_urgent(entry: dict[str, Any], host: str) -> bool:
    """Return whether *entry* is a request or blocker addressed to *host* by name."""
    return entry.get("type") in URGENT_TYPES and addresses_host(entry.get("to"), host)


def addresses_host(to: str | None, host: str) -> bool:
    """Return whether *to* names *host*'s family directly; broadcasts do not count."""
    if not to:
        return False
    family = normalize_agent_name(host).split("-", 1)[0]
    for raw in to.replace(";", ",").replace("/", ",").split(","):
        token = normalize_agent_name(raw)
        if token and token not in BROADCAST_RECIPIENTS and token.split("-", 1)[0] == family:
            return True
    return False


def ring_codex(
    bus: MessageBus,
    message: Message,
    *,
    codex_bin: str | None = None,
    resume: bool = False,
    timeout: float = RING_TIMEOUT_SECONDS,
) -> dict[str, Any]:
    """Queue a short pointer to *message* on the newest recorded Codex thread."""
    if message.msg_type not in URGENT_TYPES:
        # Other types reach Codex through the next hook digest or SessionStart.
        return {
            "status": "skipped",
            "reason": f"message type {message.msg_type} does not ring (request or blocker only)",
        }
    if not addresses_host(message.to, "codex"):
        return {"status": "skipped", "reason": "message is not addressed to codex"}
    sessions = bus.live_sessions("codex")
    if not sessions:
        return {"status": "skipped", "reason": "no codex thread recorded for this scope"}
    thread = str(sessions[0]["session_id"])
    binary = codex_bin or shutil.which("codex")
    if not binary:
        return {"status": "skipped", "reason": "codex CLI not found on PATH", "thread": thread}
    text = (
        f"AFS message {message.id} from {message.from_agent}: {_first_line(message.payload)} "
        f"(read it with: afs messages list --to codex)"
    )
    command = [binary, "queue", "--thread", thread, "--message", text]
    if resume:
        command.append("--resume")
    try:
        completed = subprocess.run(
            command, capture_output=True, text=True, timeout=timeout, check=False
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"status": "failed", "thread": thread, "error": str(exc)}
    return {
        "status": "rang" if completed.returncode == 0 else "failed",
        "thread": thread,
        "returncode": completed.returncode,
        "stderr": completed.stderr.strip()[-500:],
    }


def _first_line(payload: Any) -> str:
    if isinstance(payload, dict):
        for key in _PAYLOAD_TEXT_KEYS:
            value = payload.get(key)
            if isinstance(value, str) and value.strip():
                line = next(line.strip() for line in value.splitlines() if line.strip())
                return message_summary(line, width=PREVIEW_CHARS)
    return message_summary(payload, width=PREVIEW_CHARS)


def _read_payload(stream: TextIO) -> dict[str, Any]:
    if stream.isatty():
        return {}
    raw = stream.read()
    if not raw.strip():
        return {}
    data = json.loads(raw)
    return data if isinstance(data, dict) else {}


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
