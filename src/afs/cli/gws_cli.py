"""AFS CLI wrapper for Google Workspace operations.

Usage:
    afs gws status              # check auth status
    afs gws agenda              # today's calendar
    afs gws unread              # unread primary inbox
    afs gws raw <args...>       # pass-through to gws binary
"""

from __future__ import annotations

import argparse
import json

from ..gws import get_client


def _gws_status(args: argparse.Namespace) -> int:
    gws = get_client()
    if not gws.available:
        print("gws: not installed (brew install googleworkspace-cli)")
        return 1
    status = gws.auth_status()
    print(json.dumps(status, indent=2))
    return 0


def _gws_agenda(args: argparse.Namespace) -> int:
    gws = get_client()
    if not gws.available:
        print("gws: not installed")
        return 1
    events = gws.calendar_agenda()
    if not events:
        print("No events today (or not authenticated — run `gws auth setup`).")
        return 0
    for event in events:
        summary = event.get("summary", event.get("title", "untitled"))
        start = event.get("start", {})
        time_str = start.get("dateTime", start.get("date", "")) if isinstance(start, dict) else str(start)
        print(f"  {time_str}  {summary}")
    return 0


def _gws_unread(args: argparse.Namespace) -> int:
    gws = get_client()
    if not gws.available:
        print("gws: not installed")
        return 1
    messages = gws.gmail_unread()
    if not messages:
        print("No unread messages (or not authenticated — run `gws auth setup`).")
        return 0
    for msg in messages:
        snippet = msg.get("snippet", msg.get("id", ""))[:80]
        print(f"  {snippet}")
    return 0


def _gws_raw(args: argparse.Namespace) -> int:
    gws = get_client()
    if not gws.available:
        print("gws: not installed")
        return 1
    passthrough = getattr(args, "gws_args", [])
    if not passthrough:
        print("Usage: afs gws raw <gws-command> [args...]")
        return 1
    result = gws._run(passthrough, timeout=30)
    if result.stdout:
        print(result.stdout, end="")
    if result.stderr:
        import sys
        print(result.stderr, end="", file=sys.stderr)
    return result.returncode


def register_parsers(subparsers: argparse._SubParsersAction) -> None:
    """Register gws commands under `afs gws`."""
    parser = subparsers.add_parser("gws", help="Google Workspace integration.")
    sub = parser.add_subparsers(dest="gws_command")

    sub.add_parser("status", help="Show gws auth status.").set_defaults(func=_gws_status)
    sub.add_parser("agenda", help="Today's calendar agenda.").set_defaults(func=_gws_agenda)
    sub.add_parser("unread", help="Unread primary inbox.").set_defaults(func=_gws_unread)

    raw_parser = sub.add_parser("raw", help="Pass-through to gws binary.")
    raw_parser.add_argument("gws_args", nargs=argparse.REMAINDER, help="Arguments to pass to gws.")
    raw_parser.set_defaults(func=_gws_raw)
