"""Opt-in agreements on the existing handoff CLI and derived board surface."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from ..commitments import CommitmentStore
from ._utils import load_manager, resolve_context_paths


def _store(args: argparse.Namespace) -> CommitmentStore:
    config = Path(args.config).expanduser().resolve() if args.config else None
    manager = load_manager(config)
    project, context, _override, _directory = resolve_context_paths(args, manager)
    return CommitmentStore(context, requester_path=project, config=manager.config)


def agreement_command(args: argparse.Namespace) -> int:
    store = _store(args)
    request = json.loads(Path(args.event_file).read_text(encoding="utf-8"))
    result = store.apply(request)
    print(json.dumps(result, indent=2))
    return 0


def inspect_command(args: argparse.Namespace) -> int:
    print(json.dumps(_store(args).inspect(args.task_id, revision=args.revision), indent=2))
    return 0


def board_command(args: argparse.Namespace) -> int:
    result = _store(args).board()
    if args.json:
        print(json.dumps(result, indent=2))
    else:
        print("Accepted task agreements (derived from SQLite events)")
        for row in result["accepted"]:
            print(f"{row['task_id']} r{row['revision']} v{row['version']} | "
                  f"{row['owner']} | {row['status']} | {row['title']}")
        for row in result["proposals"]:
            print(f"pending: {row['task_id']} r{row['revision']} -> {row['owner']}")
    return 0


def _context_args(parser: argparse.ArgumentParser) -> None:
    for name, help_text in (("config", "Config path."), ("path", "Project or working path."),
                            ("context-root", "Context root override."),
                            ("context-dir", "Context directory name.")):
        parser.add_argument(f"--{name}", help=help_text)


def register_handoff_parser(commands: Any) -> None:
    parser = commands.add_parser(
        "agree", help="Apply an explicit versioned task agreement event; return its derived board.")
    _context_args(parser)
    parser.add_argument("--event-file", required=True, help="JSON event (max 32 KiB).")
    parser.set_defaults(func=agreement_command)
    inspect = commands.add_parser("agreement", help="Inspect agreement versions and exact evidence bindings.")
    _context_args(inspect)
    inspect.add_argument("task_id")
    inspect.add_argument("--revision", type=int)
    inspect.set_defaults(func=inspect_command)


def register_board_parser(commands: Any) -> None:
    parser = commands.add_parser(
        "accepted", help="Replay the opt-in accepted task board from SQLite (no Markdown write).")
    _context_args(parser)
    parser.add_argument("--json", action="store_true")
    parser.set_defaults(func=board_command)
