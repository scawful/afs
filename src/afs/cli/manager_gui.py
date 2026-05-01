"""CLI entry points for the friendly AFS Manager GUI."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..manager_gui import collect_manager_snapshot, launch_manager


def _path(args: argparse.Namespace) -> Path:
    return Path(getattr(args, "path", ".") or ".").expanduser().resolve()


def manager_open_command(args: argparse.Namespace) -> int:
    if getattr(args, "json", False):
        snapshot = collect_manager_snapshot(_path(args))
        print(json.dumps(snapshot.to_dict(), indent=2))
        return 0
    return launch_manager(_path(args))


def manager_snapshot_command(args: argparse.Namespace) -> int:
    snapshot = collect_manager_snapshot(_path(args))
    if getattr(args, "json", False):
        print(json.dumps(snapshot.to_dict(), indent=2))
        return 0
    print(f"workspace: {snapshot.workspace}")
    print(f"context: {snapshot.context_path}")
    print(f"healthy: {snapshot.context_healthy}")
    print(f"tasks: {len(snapshot.tasks)}")
    print(f"extensions: {len(snapshot.extensions)}")
    for client in snapshot.clients:
        state = "registered" if client.registered else client.note
        print(f"{client.name}: {state} ({client.path})")
    return 0


def register_parsers(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("manager", help="Open the friendly AFS Manager GUI.")
    parser.add_argument("--path", default=".", help="Workspace/project path.")
    parser.add_argument("--json", action="store_true", help="Print a snapshot instead of opening the GUI.")
    parser.set_defaults(
        func=manager_open_command,
        manager_command="open",
        _allow_missing_subcommand=True,
    )

    sub = parser.add_subparsers(dest="manager_command")

    open_parser = sub.add_parser("open", help="Open the AFS Manager GUI.")
    open_parser.add_argument("--path", default=".", help="Workspace/project path.")
    open_parser.add_argument("--json", action="store_true", help="Print a snapshot instead of opening the GUI.")
    open_parser.set_defaults(func=manager_open_command)

    snapshot_parser = sub.add_parser("snapshot", help="Print the manager read model.")
    snapshot_parser.add_argument("--path", default=".", help="Workspace/project path.")
    snapshot_parser.add_argument("--json", action="store_true", help="Output JSON.")
    snapshot_parser.set_defaults(func=manager_snapshot_command)
