"""Inventory AFS contexts and mount counts."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Sequence

from ..discovery import get_project_stats
from ..models import MountType
from .base import (
    AgentResult,
    build_base_parser,
    configure_logging,
    emit_result,
    load_agent_config,
    now_iso,
    resolve_contexts,
)

AGENT_NAME = "context-inventory"
AGENT_DESCRIPTION = "Summarize AFS contexts and mount counts."


def build_parser() -> argparse.ArgumentParser:
    parser = build_base_parser("Inventory AFS contexts and mount counts.")
    parser.add_argument(
        "--path",
        action="append",
        help="Root path to scan for contexts (repeatable).",
    )
    parser.add_argument(
        "--context-path",
        action="append",
        help="Explicit .context path to include (repeatable).",
    )
    parser.add_argument(
        "--max-depth",
        type=int,
        default=3,
        help="Maximum directory depth to scan (default: 3).",
    )
    parser.add_argument(
        "--ignore",
        action="append",
        help="Directory name to ignore (repeatable).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Limit number of contexts in output (0 = no limit).",
    )
    return parser


def run(args: argparse.Namespace) -> int:
    configure_logging(args.quiet)
    config = load_agent_config(args.config)
    contexts = resolve_contexts(
        config,
        context_paths=args.context_path,
        search_paths=args.path,
        max_depth=args.max_depth,
        ignore_names=args.ignore,
    )

    started_at = now_iso()
    start = time.monotonic()

    notes: list[str] = []
    if args.limit and args.limit > 0 and len(contexts) > args.limit:
        contexts = contexts[: args.limit]
        notes.append("context list truncated by --limit")

    items: list[dict[str, object]] = []
    for context in contexts:
        mount_counts = {
            mount_type.value: len(context.mounts.get(mount_type, []))
            for mount_type in MountType
        }
        items.append(
            {
                "name": context.project_name,
                "path": str(context.path),
                "valid": context.is_valid,
                "total_mounts": context.total_mounts,
                "mounts": mount_counts,
            }
        )

    stats = get_project_stats(contexts)
    finished_at = now_iso()
    duration = time.monotonic() - start

    if not contexts:
        notes.append("no contexts found")

    result = AgentResult(
        name=AGENT_NAME,
        status="ok" if contexts else "warn",
        started_at=started_at,
        finished_at=finished_at,
        duration_seconds=duration,
        metrics={str(key): value for key, value in stats.items()},
        notes=notes,
        payload={
            "contexts": items,
        },
    )

    output_path = None
    if args.output:
        output_path = Path(args.output)

    emit_result(
        result,
        output_path=output_path,
        force_stdout=args.stdout,
        pretty=args.pretty,
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
