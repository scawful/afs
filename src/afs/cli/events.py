"""Event log CLI commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..context_paths import resolve_mount_root
from ..event_log import build_session_replay, summarize_event_analytics
from ..history import query_events
from ..models import MountType
from ._utils import load_manager, resolve_context_paths


def _resolve_manager_context_history(
    args: argparse.Namespace,
) -> tuple[object, Path, Path]:
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
    history_root = resolve_mount_root(context_path, MountType.HISTORY, config=manager.config)
    return manager, context_path, history_root


def events_list_command(args: argparse.Namespace) -> int:
    """List history events with optional filters."""
    _manager, _context_path, history_root = _resolve_manager_context_history(args)
    event_types = {args.type} if args.type else None
    events = query_events(
        history_root,
        event_types=event_types,
        since=args.since,
        limit=args.limit,
        source=args.source,
        session_id=getattr(args, "session_id", None),
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
    _manager, _context_path, history_root = _resolve_manager_context_history(args)
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


def events_analytics_command(args: argparse.Namespace) -> int:
    """Summarize recent event usage and MCP tool behavior."""
    manager, context_path, _history_root = _resolve_manager_context_history(args)
    types = [args.type] if args.type else None
    summary = summarize_event_analytics(
        context_path,
        lookback_hours=args.hours,
        event_types=types,
        config=manager.config,
    )

    if args.json:
        print(json.dumps(summary, indent=2))
        return 0

    print(f"lookback_hours: {summary['lookback_hours']}")
    print(f"total_events: {summary['total_events']}")
    time_range = summary["time_range"]
    if time_range.get("oldest") or time_range.get("newest"):
        print(f"time_range: {time_range.get('oldest', '')} -> {time_range.get('newest', '')}")
    if summary["event_types"]:
        print()
        print("event_types:")
        for name, count in summary["event_types"].items():
            print(f"  {name}: {count}")
    if summary["mcp_tools"]:
        print()
        print("mcp_tools:")
        for name, metrics in summary["mcp_tools"].items():
            avg = metrics["avg_duration_ms"]
            avg_text = f"{avg}ms" if avg is not None else "n/a"
            print(
                f"  {name}: count={metrics['count']} errors={metrics['errors']} "
                f"error_rate={metrics['error_rate']} avg={avg_text}"
            )
    return 0


def events_replay_command(args: argparse.Namespace) -> int:
    """Replay a recorded AFS session timeline."""
    manager, context_path, _history_root = _resolve_manager_context_history(args)
    replay = build_session_replay(
        context_path,
        session_id=args.session_id,
        limit=args.limit,
        include_payloads=args.include_payloads,
        config=manager.config,
    )

    if args.json:
        print(json.dumps(replay, indent=2))
        return 0

    print(f"session_id: {replay['session_id']}")
    print(f"events: {replay['count']}")
    if replay["truncated"]:
        print(f"truncated: {replay['truncated']} earlier events omitted")
    if replay.get("started_at") or replay.get("ended_at"):
        print(f"time_range: {replay.get('started_at', '')} -> {replay.get('ended_at', '')}")
    if replay["event_types"]:
        print("types: " + ", ".join(f"{name}={count}" for name, count in replay["event_types"].items()))
    print()
    for event in replay["events"]:
        ts = str(event.get("timestamp", ""))[:19]
        etype = event.get("type", "")
        source = event.get("source", "")
        op = event.get("op", "")
        line = f"{ts} [{etype}] {source}"
        if op:
            line += f" op={op}"
        meta = event.get("metadata")
        if isinstance(meta, dict):
            for key in ("tool_name", "agent_name", "msg_id"):
                if key in meta:
                    line += f" {key}={meta[key]}"
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
    list_parser.add_argument("--session-id", help="Filter by recorded AFS session ID.")
    list_parser.add_argument("--json", action="store_true", help="Output JSON.")
    list_parser.set_defaults(func=events_list_command)

    tail_parser = events_sub.add_parser("tail", help="Show most recent events.")
    tail_parser.add_argument("--limit", type=int, default=20, help="Max events.")
    tail_parser.add_argument("--json", action="store_true", help="Output JSON.")
    tail_parser.set_defaults(func=events_tail_command)

    analytics_parser = events_sub.add_parser(
        "analytics",
        help="Summarize event volume, MCP tool usage, and error rates.",
    )
    analytics_parser.add_argument("--hours", type=int, default=24, help="Lookback window in hours.")
    analytics_parser.add_argument("--type", help="Filter analytics to a single event type.")
    analytics_parser.add_argument("--json", action="store_true", help="Output JSON.")
    analytics_parser.set_defaults(func=events_analytics_command)

    replay_parser = events_sub.add_parser(
        "replay",
        help="Replay a recorded AFS session timeline by session ID.",
    )
    replay_parser.add_argument("--session-id", required=True, help="Recorded AFS session ID.")
    replay_parser.add_argument("--limit", type=int, default=200, help="Max events to show (0 for all).")
    replay_parser.add_argument("--include-payloads", action="store_true", help="Include event payloads in JSON output.")
    replay_parser.add_argument("--json", action="store_true", help="Output JSON.")
    replay_parser.set_defaults(func=events_replay_command)
