"""Deterministic next-action routing for AFS-aware agents."""

from __future__ import annotations

import shlex
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import load_runtime_config_model
from .core import find_root, resolve_context_root
from .history import query_events
from .manager import AFSManager
from .models import MountType
from .session_bootstrap import build_agent_discovery_path


@dataclass(frozen=True)
class NextCommand:
    """Concrete command or slash route an agent can follow."""

    label: str
    command: str
    when: str = ""

    def to_dict(self) -> dict[str, str]:
        return {"label": self.label, "command": self.command, "when": self.when}


@dataclass(frozen=True)
class NextAction:
    """Small read model for the next deterministic AFS step."""

    intent: str
    canonical_intent: str
    workspace: Path
    context_path: Path
    summary: str
    first_step: str
    mcp_sequence: list[str]
    commands: list[NextCommand]
    slash_command: str = ""
    stop_when: str = ""
    avoid: list[str] = field(default_factory=list)
    discovery_path: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "intent": self.intent,
            "canonical_intent": self.canonical_intent,
            "workspace": str(self.workspace),
            "context_path": str(self.context_path),
            "summary": self.summary,
            "first_step": self.first_step,
            "mcp_sequence": list(self.mcp_sequence),
            "commands": [command.to_dict() for command in self.commands],
            "stop_when": self.stop_when,
            "avoid": list(self.avoid),
            "discovery_path": dict(self.discovery_path),
        }
        if self.slash_command:
            payload["slash_command"] = self.slash_command
        return payload


_INTENT_ALIASES: dict[str, set[str]] = {
    "continue": {"", "continue", "resume", "catch-up", "catchup", "next", "start"},
    "context": {"context", "query", "search", "history", "scratchpad", "memory", "prior", "lookup"},
    "review": {"review", "audit", "findings", "critic", "diff", "inspect"},
    "ship": {"ship", "commit", "push", "pr", "publish", "release"},
    "work-writing": {"work", "work-writing", "writing", "slack", "email", "reply", "post", "comment"},
    "verify": {"verify", "test", "check", "validation", "smoke"},
    "handoff": {"handoff", "transfer", "pause", "handover", "summary"},
    "setup": {"setup", "install", "onboard", "manager", "gui", "client"},
    "refresh": {"refresh", "repair", "stale", "index", "doctor"},
    "pack": {"pack", "session-pack", "export"},
}

_AVOID = [
    "Do not browse the full AFS/MCP catalog before the routed command answers.",
    "Do not write memory or knowledge unless the user explicitly asked for durable context.",
    "Do not call session.pack unless the intent is an explicit export or handoff pack.",
]


def canonicalize_intent(intent: str | None) -> str:
    """Normalize a fuzzy next-action intent into a stable route key."""
    normalized = (intent or "continue").strip().lower().replace("_", "-")
    for canonical, aliases in _INTENT_ALIASES.items():
        if normalized in aliases:
            return canonical
    for canonical, aliases in _INTENT_ALIASES.items():
        if any(alias and alias in normalized for alias in aliases):
            return canonical
    return "continue"


def resolve_workspace_and_context(workspace: Path | str = Path(".")) -> tuple[Path, Path]:
    """Resolve the workspace and best-known context path for next-action output."""
    root = Path(workspace).expanduser().resolve()
    config, _config_path = load_runtime_config_model(merge_user=True, start_dir=root)
    linked_root = find_root(root) or root
    try:
        context_path = resolve_context_root(config, linked_root)
    except Exception:
        context_path = root / ".context"
        if not context_path.exists():
            context_path = config.general.context_root
    return root, context_path.expanduser().resolve()


def _cmd(path: Path, suffix: str) -> str:
    return f"afs {suffix} --path {shlex.quote(str(path))}"


def _status_cmd(path: Path) -> str:
    return f"afs status --start-dir {shlex.quote(str(path))}"


def _verify_cmd(path: Path, suffix: str) -> str:
    return f"afs verify {suffix} --cwd {shlex.quote(str(path))}"


def build_next_action(intent: str | None = None, *, workspace: Path | str = Path(".")) -> NextAction:
    """Build the deterministic next action for a common agent intent."""
    root, context_path = resolve_workspace_and_context(workspace)
    original_intent = (intent or "continue").strip() or "continue"
    canonical = canonicalize_intent(original_intent)
    query_command = f"afs query {shlex.quote('recent handoff deferred current work')} --path {shlex.quote(str(root))} --mount memory --mount scratchpad --limit 8 --json"

    routes: dict[str, dict[str, Any]] = {
        "continue": {
            "summary": "Catch up with the smallest useful AFS read, then continue the actual task.",
            "first_step": "context.status",
            "mcp_sequence": ["context.status", "context.query: recent handoff deferred current work", "context.read: exact hits only"],
            "slash_command": "/afs-brief",
            "commands": [
                NextCommand("status", _status_cmd(root), "check health first"),
                NextCommand("recent-context", query_command, "only if prior state is needed"),
            ],
            "stop_when": "You know the current task, dirty state, and any handoff/deferred blocker.",
        },
        "context": {
            "summary": "Find prior context without expanding the tool surface.",
            "first_step": "context.query",
            "mcp_sequence": ["context.status", "context.query", "context.read/context.list: exact hits only"],
            "slash_command": "/afs-query <query>",
            "commands": [NextCommand("query", f"afs query <query> --path {shlex.quote(str(root))} --limit 8 --json", "replace <query> with the user's topic")],
            "stop_when": "Query results point to specific sources or no matching context exists.",
        },
        "review": {
            "summary": "Review with AFS context only where it changes findings.",
            "first_step": "context.status, then git diff",
            "mcp_sequence": ["context.status", "context.query: relevant prior decisions", "context.read: exact hits only"],
            "slash_command": "/afs-review-context",
            "commands": [
                NextCommand("review-context", f"afs query {shlex.quote('review risks prior decisions current diff')} --path {shlex.quote(str(root))} --mount scratchpad --mount knowledge --limit 8 --json", "before findings if history matters"),
                NextCommand("diff-check", "git diff --check", "cheap hygiene check"),
                NextCommand("verify-plan", _verify_cmd(root, "plan --json"), "if changed files need tests"),
            ],
            "stop_when": "Findings have exact files/symbols and verification gaps are explicit.",
        },
        "ship": {
            "summary": "Prepare commit/push by checking scope, verification, and residual risk.",
            "first_step": "git status --short",
            "mcp_sequence": ["context.status", "context.query: handoff or deferred blockers only if needed"],
            "slash_command": "/afs-verify",
            "commands": [
                NextCommand("status", "git status --short", "confirm staged and unstaged scope"),
                NextCommand("verify-plan", _verify_cmd(root, "plan --json"), "choose the smallest trustworthy checks"),
                NextCommand("diff-check", "git diff --check", "always cheap before commit"),
            ],
            "stop_when": "Changed scope is intentional, checks ran or blockers are named, and commit message is obvious.",
        },
        "work-writing": {
            "summary": "Ground work-facing writing in evidence and keep external posting approval-gated.",
            "first_step": "work communication preflight",
            "mcp_sequence": ["context.status"],
            "slash_command": "/afs-work-preflight",
            "commands": [NextCommand("work-preflight", _cmd(root, "work communication preflight --json"), "before drafting work-facing text")],
            "stop_when": "Style evidence, pending approvals, ready_to_post, and approval guardrail are known.",
        },
        "verify": {
            "summary": "Select and run the narrowest useful verification.",
            "first_step": "afs verify plan",
            "mcp_sequence": ["context.status"],
            "slash_command": "/afs-verify",
            "commands": [
                NextCommand("verify-plan", _verify_cmd(root, "plan --json"), "inspect proposed checks"),
                NextCommand("verify-run", _verify_cmd(root, "run --json"), "run after confirming the plan is appropriate"),
            ],
            "stop_when": "One relevant check passed, failed narrowly, or could not run for a concrete reason.",
        },
        "handoff": {
            "summary": "Create or read a concise operational handoff without packing the whole session by default.",
            "first_step": "list canonical handoff threads or create one immutable revision",
            "mcp_sequence": ["handoff.list", "handoff.read: exact revision only", "handoff.create or handoff.revise: when writing"],
            "slash_command": "/afs-handoff",
            "commands": [
                NextCommand("list-handoffs", _cmd(root, "session handoff list --json"), "if looking for existing handoff packets"),
            ],
            "stop_when": "The handoff names current state, changed files, verification, blockers, and the next narrow step.",
        },
        "setup": {
            "summary": "Use the human-friendly setup path before asking agents to mutate client config.",
            "first_step": "afs manager snapshot or setup dry-run",
            "mcp_sequence": ["context.status"],
            "slash_command": "/afs-help setup",
            "commands": [
                NextCommand("manager", _cmd(root, "manager snapshot --json"), "inspect client config and extensions without opening a GUI"),
                NextCommand("setup-preview", f"afs setup --workspace {shlex.quote(str(root))} --dry-run", "preview before writing config"),
            ],
            "stop_when": "The user has a previewed setup command or a GUI manager action to approve.",
        },
        "refresh": {
            "summary": "Refresh only when health or search freshness makes it necessary.",
            "first_step": "context.status health/freshness check",
            "mcp_sequence": ["context.status"],
            "slash_command": "/afs-refresh",
            "commands": [NextCommand("repair-preview", _cmd(root, "context repair --dry-run --json"), "dry-run before remapping or rebuilding")],
            "stop_when": "A concrete missing/stale/broken surface is identified, or refresh is unnecessary.",
        },
        "pack": {
            "summary": "Pack only for an explicit handoff/export request.",
            "first_step": "confirm export is needed",
            "mcp_sequence": ["context.status"],
            "slash_command": "/afs-pack",
            "commands": [NextCommand("session-pack", _cmd(root, "session pack --json"), "heavy export path")],
            "stop_when": "The pack artifact path is known or the user no longer needs an export.",
        },
    }

    route = routes[canonical]
    return NextAction(
        intent=original_intent,
        canonical_intent=canonical,
        workspace=root,
        context_path=context_path,
        summary=route["summary"],
        first_step=route["first_step"],
        mcp_sequence=list(route["mcp_sequence"]),
        commands=list(route["commands"]),
        slash_command=str(route.get("slash_command", "")),
        stop_when=route["stop_when"],
        avoid=_AVOID,
        discovery_path=build_agent_discovery_path(context_path),
    )


def summarize_next_usage(context_path: Path, *, limit: int = 200) -> dict[str, Any]:
    """Summarize recent AFS funnel usage from history events."""
    manager = AFSManager()
    history_root = manager.resolve_mount_root(context_path, MountType.HISTORY)
    events = query_events(history_root, limit=limit) if history_root.exists() else []
    cli_counter: Counter[str] = Counter()
    mcp_counter: Counter[str] = Counter()
    route_counter: Counter[str] = Counter()
    for event in events:
        metadata = event.get("metadata") if isinstance(event.get("metadata"), dict) else {}
        if event.get("source") == "afs.next":
            route_counter[str(metadata.get("intent", "unknown"))] += 1
        if event.get("type") == "cli":
            argv = metadata.get("argv")
            if isinstance(argv, list) and argv:
                head = [str(item) for item in argv[:3]]
                if head and head[0] == "afs":
                    head = head[1:]
                cli_counter[" ".join(head[:2]) or "afs"] += 1
        if event.get("type") == "mcp_tool":
            tool = str(metadata.get("tool_name", "unknown"))
            mcp_counter[tool] += 1
    default_tools = {"context.status", "context.query", "context.read", "context.list", "context.write"}
    heavy_mcp_calls = {
        tool: count
        for tool, count in sorted(mcp_counter.items())
        if tool not in default_tools and not tool.startswith("fs.")
    }
    return {
        "context_path": str(context_path),
        "events_scanned": len(events),
        "next_routes": dict(sorted(route_counter.items())),
        "cli_commands": dict(cli_counter.most_common(20)),
        "mcp_tools": dict(mcp_counter.most_common(20)),
        "heavy_mcp_calls": heavy_mcp_calls,
        "signal": "healthy" if route_counter or mcp_counter or cli_counter else "no recent AFS usage recorded",
    }
