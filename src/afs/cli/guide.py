"""Human-oriented guide pages for common AFS workflows."""

from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class GuideTopic:
    name: str
    title: str
    summary: str
    commands: list[tuple[str, str]]
    notes: list[str]


GUIDES: dict[str, GuideTopic] = {
    "start": GuideTopic(
        name="start",
        title="Getting Started",
        summary="Open the manager, set up AFS, and learn where context lives.",
        commands=[
            ("afs manager", "open the friendly Python GUI manager"),
            ("afs next --intent continue", "ask AFS for the next deterministic action"),
            ("afs setup", "guided setup wizard"),
            ("afs status", "show the active context, mounts, and index state"),
            ("afs guide context", "context root and mount workflow"),
            ("afs doctor", "diagnose common setup issues"),
        ],
        notes=[
            "Use the setup wizard first on a new machine.",
            "AFS is safe to inspect in dry-run mode before writing shell or config files.",
        ],
    ),
    "context": GuideTopic(
        name="context",
        title="Context Management",
        summary="Create, inspect, search, and repair .context roots.",
        commands=[
            ("afs context discover --path .", "find contexts below the current directory"),
            ("afs status --start-dir .", "show the resolved context and health"),
            ("afs context repair --path . --dry-run", "preview mount/provenance repairs"),
            ("afs context repair --path . --rebuild-index", "repair and refresh the index"),
            ("afs query \"handoff\" --path . --mount scratchpad", "search context"),
            ("afs fs list scratchpad --path .", "browse writable notes"),
        ],
        notes=[
            "Write routine working notes to scratchpad.",
            "Treat memory and knowledge as durable, deliberate updates.",
            "Use a shared context root when a managed workspace cannot contain .context.",
        ],
    ),
    "manager": GuideTopic(
        name="manager",
        title="AFS Manager",
        summary="Use the Python GUI for setup, tasks, clients, and extension hooks.",
        commands=[
            ("afs manager", "open the GUI for the current workspace"),
            ("afs manager snapshot --json", "print the same read model without a GUI"),
            ("afs manager open --path ~/src/project", "open the manager for a project"),
            ("afs-manager", "launcher shortcut installed from the repo scripts directory"),
        ],
        notes=[
            "The manager is the normie-friendly setup surface; advanced harnesses can stay opinionated.",
            "Project .gemini/.claude/.codex/.opencode files are inspected without requiring agents to know every config format.",
            "Extensions can expose manager actions with a [manager] actions list in extension.toml.",
        ],
    ),
    "next": GuideTopic(
        name="next",
        title="Next Action Router",
        summary="Route common agent intents without exposing the full AFS catalog.",
        commands=[
            ("afs next --intent continue --json", "catch up from status/query/read"),
            ("afs next --intent work-writing --json", "find the work preflight path"),
            ("afs next --intent verify --json", "find the verification path"),
            ("afs next report --json", "measure recent AFS funnel usage"),
        ],
        notes=[
            "Use this when an agent might otherwise browse tools or docs to decide what AFS surface to touch.",
            "The router returns the first MCP step, exact CLI/slash route, stop condition, and avoided heavy surfaces.",
            "History records afs.next route events, so usage can be inspected later with `afs next report`.",
        ],
    ),
    "shell": GuideTopic(
        name="shell",
        title="Shell Integration",
        summary="Install approachable aliases, colors, and zsh completion.",
        commands=[
            (
                "afs agent-hooks install-shell --helpers-only --apply",
                "add helpers/completion without routing AI harness commands",
            ),
            (
                "afs agent-hooks install-shell --apply",
                "also route generic harnesses through AFS wrappers",
            ),
            ("source scripts/afs-shell-init.sh", "enable helpers for the current shell"),
            ("AFS_COMPLETE=zsh_source afs > ~/.zfunc/_afs", "write completion source manually"),
        ],
        notes=[
            "The helpers-only mode is the least surprising default for work machines.",
            "Full agent hooks can be disabled in a shell with afs-agent-hooks-off.",
        ],
    ),
    "mcp": GuideTopic(
        name="mcp",
        title="MCP Setup",
        summary="Expose AFS context tools to MCP-aware clients.",
        commands=[
            ("afs manager", "inspect .gemini/.claude/.codex project setup"),
            ("afs mcp serve", "run the AFS stdio MCP server"),
            ("afs gemini setup --scope project", "register AFS for Gemini CLI"),
            ("afs claude setup --path .", "register AFS for Claude-compatible settings"),
            ("afs guide google-workspace", "optional Google Workspace helper setup"),
        ],
        notes=[
            "Keep the default MCP surface small: status, query, read, write, and list.",
            "Add organization-specific MCP servers in the client config that owns those credentials.",
        ],
    ),
    "google-workspace": GuideTopic(
        name="google-workspace",
        title="Google Workspace",
        summary="Connect optional Google Workspace helpers without making it core AFS.",
        commands=[
            ("scripts/setup_gws.sh --dry-run", "preview install/auth steps"),
            ("scripts/setup_gws.sh", "install/check gws and authenticate"),
            ("afs gws status", "show authentication status"),
            ("afs gws agenda", "show today's calendar agenda"),
            ("afs gws unread", "show unread primary inbox snippets"),
        ],
        notes=[
            "Keep credentials in the Google Workspace tool's config directory.",
            "Use work-approved OAuth/client setup and scopes.",
        ],
    ),
    "agents": GuideTopic(
        name="agents",
        title="Agents and Background Work",
        summary="Use background jobs when the output is independently reviewable.",
        commands=[
            ("afs manager", "browse tasks, client config, and extension hooks"),
            ("afs agent-hooks status --path .", "show hooks, worker status, and next commands"),
            ("afs agent-jobs status --path .", "queue and watchdog summary"),
            ("afs agent-jobs inbox --path .", "review completed, failed, and blocked jobs"),
            ("afs agent-jobs create \"task\" --prompt \"...\"", "queue a background job"),
        ],
        notes=[
            "Background agents are optional and should stay report-oriented by default.",
            "Destructive jobs require explicit opt-in.",
        ],
    ),
}

ALIASES = {
    "getting-started": "start",
    "workspace": "context",
    "contexts": "context",
    "gui": "manager",
    "router": "next",
    "completion": "shell",
    "autocomplete": "shell",
    "google": "google-workspace",
    "gws": "google-workspace",
    "background": "agents",
}


def _style(text: str, code: str) -> str:
    if not sys.stdout.isatty() or os.getenv("NO_COLOR") is not None:
        return text
    return f"\033[{code}m{text}\033[0m"


def _section(title: str) -> str:
    return _style(title, "1;36")


def _cmd(text: str) -> str:
    return _style(text, "1;34")


def _topic_for(raw: str | None) -> GuideTopic | None:
    if not raw:
        return None
    key = raw.strip().lower()
    key = ALIASES.get(key, key)
    return GUIDES.get(key)


def _render_topic(topic: GuideTopic) -> str:
    lines = [
        _section(topic.title),
        topic.summary,
        "",
        _section("Commands"),
    ]
    width = max(len(command) for command, _description in topic.commands)
    for command, description in topic.commands:
        lines.append(f"  {_cmd(command.ljust(width))}  {description}")
    if topic.notes:
        lines.extend(["", _section("Notes")])
        lines.extend(f"  - {note}" for note in topic.notes)
    return "\n".join(lines) + "\n"


def _render_menu() -> str:
    lines = [
        _section("AFS Guide"),
        "Pick a topic, then run the shown commands.",
        "",
        _section("Topics"),
    ]
    width = max(len(name) for name in GUIDES)
    for name, topic in GUIDES.items():
        lines.append(f"  {_cmd(name.ljust(width))}  {topic.summary}")
    lines.extend(
        [
            "",
            _section("Examples"),
            f"  {_cmd('afs guide context')}",
            f"  {_cmd('afs next --intent continue')}",
            f"  {_cmd('afs manager')}",
            f"  {_cmd('afs guide shell')}",
            f"  {_cmd('afs setup')}",
        ]
    )
    return "\n".join(lines) + "\n"


def guide_command(args: argparse.Namespace) -> int:
    topic = _topic_for(args.topic)
    if topic is None:
        if args.topic:
            print(f"Unknown guide topic: {args.topic}")
            print("Available topics: " + ", ".join(GUIDES))
            return 1
        print(_render_menu(), end="")
        return 0
    print(_render_topic(topic), end="")
    return 0


def register_parsers(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("guide", help="Show friendly workflow guides.")
    parser.add_argument(
        "topic",
        nargs="?",
        help="Topic: start, next, manager, context, shell, mcp, google-workspace, agents.",
    )
    parser.set_defaults(func=guide_command)
