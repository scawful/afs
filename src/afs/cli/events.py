"""Event log CLI commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..context_paths import resolve_mount_root
from ..history import query_events
from ..models import MountType
from ._utils import load_manager, resolve_context_paths


def _resolve_history_root(args: argparse.Namespace) -> Path:
    config_path = (
        Path(args.config).expanduser().resolve()
        if getattr(args, "config", None)
        else None
    )
    manager = load_manager(config_path)
    _project_path, context_path, _context_root, _context_dir = resolve_context_paths(
        args,
        manager,
    )
    return resolve_mount_root(context_path, MountType.HISTORY, config=manager.config)


def events_list_command(args: argparse.Namespace) -> int:
    """List history events with optional filters."""
    history_root = _resolve_history_root(args)
    event_types = {args.type} if args.type else None
    events = query_events(
        history_root,
        event_types=event_types,
        since=args.since,
        limit=args.limit,
        source=args.source,
    )

    if args.json:
        print(json.dumps(events, indent=2))
        return 0

    for event in events:
        ts = str(event.get("timestamp", ""))[:19]
        etype = event.get("type", "")
        source = event.get("source", "")
        op = event.get("op", "")
        meta = event.get("metadata", {})
        label = f"{ts} [{etype}] {source}"
        if op:
            label += f" op={op}"
        # Show key metadata inline
        for key in ("tool_name", "agent_name", "session_id", "msg_id"):
            if key in meta:
                label += f" {key}={meta[key]}"
        print(label)
    return 0


def events_tail_command(args: argparse.Namespace) -> int:
    """Show the most recent events."""
    history_root = _resolve_history_root(args)
    events = query_events(history_root, limit=args.limit)

    if args.json:
        print(json.dumps(events, indent=2))
        return 0

    for event in events:
        ts = str(event.get("timestamp", ""))[:19]
        etype = event.get("type", "")
        source = event.get("source", "")
        op = event.get("op", "")
        line = f"{ts} [{etype}] {source}"
        if op:
            line += f" op={op}"
        print(line)
    return 0


def register_parsers(subparsers: argparse._SubParsersAction) -> None:
    """Register events command parsers."""
    events_parser = subparsers.add_parser("events", help="Query the AFS event log.")
    events_parser.add_argument("--config", help="Config path.")
    events_parser.add_argument("--path", help="Project path.")
    events_parser.add_argument("--context-root", help="Context root override.")
    events_parser.add_argument("--context-dir", help="Context directory name.")
    events_sub = events_parser.add_subparsers(dest="events_command")

    list_parser = events_sub.add_parser("list", help="List events with filters.")
    list_parser.add_argument("--type", help="Filter by event type.")
    list_parser.add_argument("--since", help="ISO 8601 datetime cutoff.")
    list_parser.add_argument("--limit", type=int, default=50, help="Max events.")
    list_parser.add_argument("--source", help="Filter by source.")
    list_parser.add_argument("--json", action="store_true", help="Output JSON.")
    list_parser.set_defaults(func=events_list_command)

    tail_parser = events_sub.add_parser("tail", help="Show most recent events.")
    tail_parser.add_argument("--limit", type=int, default=20, help="Max events.")
    tail_parser.add_argument("--json", action="store_true", help="Output JSON.")
    tail_parser.set_defaults(func=events_tail_command)
