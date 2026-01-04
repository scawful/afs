"""Audit AFS contexts for missing required directories."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Sequence

from ..validator import AFSValidator
from .base import (
    AgentResult,
    build_base_parser,
    configure_logging,
    emit_result,
    load_agent_config,
    now_iso,
    resolve_contexts,
)

AGENT_NAME = "context-audit"
AGENT_DESCRIPTION = "Audit AFS contexts for missing required directories."


def build_parser() -> argparse.ArgumentParser:
    parser = build_base_parser("Audit AFS contexts for missing required directories.")
    parser.add_argument(
        "--path",
        action="append",
        help="Root path to scan for contexts (repeatable).",
    )
    parser.add_argument(
        "--context-path",
        action="append",
        help="Explicit .context path to audit (repeatable).",
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

    invalid: list[dict[str, object]] = []
    missing_counts: dict[str, int] = {}

    for context in contexts:
        validator = AFSValidator(context.path, config.directories)
        status = validator.check_integrity()
        if not status.get("valid", False):
            missing = [str(item) for item in status.get("missing", [])]
            invalid.append(
                {
                    "name": context.project_name,
                    "path": str(context.path),
                    "missing": missing,
                    "errors": status.get("errors", []),
                }
            )
            for item in missing:
                missing_counts[item] = missing_counts.get(item, 0) + 1

    finished_at = now_iso()
    duration = time.monotonic() - start

    total = len(contexts)
    invalid_count = len(invalid)
    valid_count = total - invalid_count
    status = "ok" if invalid_count == 0 else "warn"
    notes: list[str] = []
    if total == 0:
        status = "warn"
        notes.append("no contexts found")

    result = AgentResult(
        name=AGENT_NAME,
        status=status,
        started_at=started_at,
        finished_at=finished_at,
        duration_seconds=duration,
        metrics={
            "total_contexts": total,
            "valid_contexts": valid_count,
            "invalid_contexts": invalid_count,
        },
        notes=notes,
        payload={
            "invalid": invalid,
            "missing_counts": missing_counts,
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
