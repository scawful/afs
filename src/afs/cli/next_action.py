"""CLI entry point for deterministic AFS next-action routing."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..history import log_event
from ..next_action import build_next_action, resolve_workspace_and_context, summarize_next_usage


def _path(args: argparse.Namespace) -> Path:
    return Path(getattr(args, "path", ".") or ".").expanduser().resolve()


def _print_action(action, *, as_json: bool) -> None:
    payload = action.to_dict()
    if as_json:
        print(json.dumps(payload, indent=2))
        return
    print(f"intent: {payload['canonical_intent']} ({payload['intent']})")
    print(f"workspace: {payload['workspace']}")
    print(f"summary: {payload['summary']}")
    print(f"first: {payload['first_step']}")
    if payload.get("slash_command"):
        print(f"slash: {payload['slash_command']}")
    print("mcp:")
    for item in payload["mcp_sequence"]:
        print(f"  - {item}")
    print("commands:")
    for command in payload["commands"]:
        suffix = f"  # {command['when']}" if command.get("when") else ""
        print(f"  - {command['command']}{suffix}")
    print(f"stop_when: {payload['stop_when']}")


def next_command(args: argparse.Namespace) -> int:
    action = build_next_action(getattr(args, "intent", None), workspace=_path(args))
    log_event(
        "agent_route",
        "afs.next",
        op="route",
        metadata={
            "intent": action.canonical_intent,
            "requested_intent": action.intent,
            "workspace": str(action.workspace),
            "first_step": action.first_step,
            "slash_command": action.slash_command,
        },
        context_root=action.context_path,
        include_payloads=False,
    )
    _print_action(action, as_json=bool(getattr(args, "json", False)))
    return 0


def next_report_command(args: argparse.Namespace) -> int:
    _workspace, context_path = resolve_workspace_and_context(_path(args))
    report = summarize_next_usage(context_path, limit=max(1, int(getattr(args, "limit", 200))))
    if getattr(args, "json", False):
        print(json.dumps(report, indent=2))
        return 0
    print(f"context: {report['context_path']}")
    print(f"events_scanned: {report['events_scanned']}")
    print(f"signal: {report['signal']}")
    if report["next_routes"]:
        print("next_routes:")
        for name, count in report["next_routes"].items():
            print(f"  {name}: {count}")
    if report["heavy_mcp_calls"]:
        print("heavy_mcp_calls:")
        for name, count in report["heavy_mcp_calls"].items():
            print(f"  {name}: {count}")
    return 0


def register_parsers(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("next", help="Route an agent intent to the next deterministic AFS action.")
    parser.add_argument("--intent", default="continue", help="Intent: continue, context, review, ship, work-writing, verify, handoff, setup, refresh, or pack.")
    parser.add_argument("--path", default=".", help="Workspace/project path.")
    parser.add_argument("--json", action="store_true", help="Output JSON.")
    parser.set_defaults(func=next_command, next_command="route", _allow_missing_subcommand=True)

    sub = parser.add_subparsers(dest="next_command")
    route = sub.add_parser("route", help="Route an intent to the next action.")
    route.add_argument("--intent", default="continue", help="Intent to route.")
    route.add_argument("--path", default=".", help="Workspace/project path.")
    route.add_argument("--json", action="store_true", help="Output JSON.")
    route.set_defaults(func=next_command)

    report = sub.add_parser("report", help="Summarize recent AFS funnel usage from history.")
    report.add_argument("--path", default=".", help="Workspace/project path.")
    report.add_argument("--limit", type=int, default=200, help="Recent history events to scan.")
    report.add_argument("--json", action="store_true", help="Output JSON.")
    report.set_defaults(func=next_report_command)
