"""History / scratchpad retention CLI (`afs history reap`)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..context_paths import resolve_mount_root
from ..history import (
    DEFAULT_HISTORY_KEEP_DAYS,
    DEFAULT_SCRATCHPAD_KEEP_DAYS,
    reap_storage,
)
from ..models import MountType
from ._utils import load_manager, resolve_context_paths


def history_reap_command(args: argparse.Namespace) -> int:
    """Preview or apply history gzip + scratchpad age-out."""
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
    scratchpad_root = resolve_mount_root(
        context_path, MountType.SCRATCHPAD, config=manager.config
    )

    history_keep = (
        args.keep_days
        if args.keep_days is not None
        else getattr(manager.config.history, "retention_days", DEFAULT_HISTORY_KEEP_DAYS)
    )
    scratchpad_keep = (
        args.scratchpad_days
        if args.scratchpad_days is not None
        else getattr(manager.config.scratchpad, "retention_days", DEFAULT_SCRATCHPAD_KEEP_DAYS)
    )
    protect = list(getattr(manager.config.scratchpad, "protect", []))
    gzip_older = bool(getattr(manager.config.history, "gzip_older", True))
    dry_run = not bool(args.apply)

    try:
        report = reap_storage(
            history_root=history_root,
            scratchpad_root=scratchpad_root,
            history_keep_days=history_keep,
            scratchpad_keep_days=scratchpad_keep,
            scratchpad_protect=protect,
            gzip_history=gzip_older,
            dry_run=dry_run,
        )
    except ValueError as exc:
        print(str(exc))
        return 1

    payload = report.to_dict()
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    verb = "would reap" if dry_run else "reaped"
    hist_n = payload["history_candidates"]
    pad_n = payload["scratchpad_candidates"]
    print(
        f"{verb}: history_files={hist_n} scratchpad_paths={pad_n} "
        f"(history_keep_days={history_keep}, scratchpad_keep_days={scratchpad_keep}"
        f"{', dry-run' if dry_run else ''})"
    )
    if dry_run and (hist_n or pad_n):
        print("Re-run with --apply to perform the cleanup.")
    if args.verbose:
        if report.history is not None:
            for entry in report.history.entries:
                label = "skip" if entry.skipped else "history"
                detail = f" ({entry.skipped})" if entry.skipped else ""
                print(f"  [{label}] {entry.path}{detail}")
        for entry in report.scratchpad:
            label = "skip" if entry.skipped else entry.kind
            detail = f" ({entry.skipped})" if entry.skipped else ""
            print(f"  [{label}] {entry.path}{detail}")
    skipped = False
    if report.history is not None:
        skipped = any(entry.skipped for entry in report.history.entries)
    skipped = skipped or any(entry.skipped for entry in report.scratchpad)
    return 1 if skipped and not dry_run else 0


def register_parsers(subparsers: argparse._SubParsersAction) -> None:
    """Register ``afs history`` retention commands."""

    def _add_context_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--config", help="Config path.")
        parser.add_argument("--path", help="Project path.")
        parser.add_argument("--context-root", help="Context root override.")
        parser.add_argument("--context-dir", help="Context directory name.")

    history_parser = subparsers.add_parser(
        "history",
        help="History and scratchpad retention hygiene.",
    )
    history_sub = history_parser.add_subparsers(dest="history_command")

    reap_parser = history_sub.add_parser(
        "reap",
        help="Gzip aged history logs and age out scratchpad session clutter.",
    )
    _add_context_args(reap_parser)
    reap_parser.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Preview candidates without writing (default).",
    )
    reap_parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply gzip/archive and delete eligible scratchpad paths.",
    )
    reap_parser.add_argument(
        "--keep-days",
        type=int,
        default=None,
        help=f"Uncompressed history days (default from config, {DEFAULT_HISTORY_KEEP_DAYS}).",
    )
    reap_parser.add_argument(
        "--scratchpad-days",
        type=int,
        default=None,
        help=f"Scratchpad age-out days (default from config, {DEFAULT_SCRATCHPAD_KEEP_DAYS}).",
    )
    reap_parser.add_argument("--verbose", action="store_true", help="List each path.")
    reap_parser.add_argument("--json", action="store_true", help="Output JSON.")
    reap_parser.set_defaults(func=history_reap_command, _skip_cli_history=True)
