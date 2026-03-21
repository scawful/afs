"""Background agent for consolidating history into durable memory."""

from __future__ import annotations

import argparse
import time
from collections.abc import Sequence
from pathlib import Path

from ..memory_consolidation import consolidate_history_to_memory
from .base import (
    AgentResult,
    build_base_parser,
    configure_logging,
    emit_result,
    load_agent_config,
    now_iso,
)

AGENT_NAME = "history-memory"
AGENT_DESCRIPTION = "Consolidate recent history events into durable memory summaries."

AGENT_CAPABILITIES = {
    "mount_types": ["history", "memory", "scratchpad"],
    "topics": ["agent:lifecycle"],
    "tools": ["memory.search"],
    "description": "Consolidates history events into durable memory summaries.",
}


def build_parser() -> argparse.ArgumentParser:
    parser = build_base_parser(AGENT_DESCRIPTION)
    parser.add_argument("--context-root", help="Context root override.")
    parser.add_argument("--history-root", help="History directory override.")
    parser.add_argument("--memory-root", help="Memory directory override.")
    parser.add_argument("--checkpoint", help="Checkpoint file override.")
    parser.add_argument("--interval", type=int, help="Seconds between runs (0 = run once).")
    parser.add_argument(
        "--max-events",
        type=int,
        help="Maximum history events to consolidate per run.",
    )
    parser.add_argument(
        "--max-events-per-entry",
        type=int,
        help="Maximum history events per memory entry.",
    )
    parser.add_argument(
        "--event-type",
        action="append",
        dest="event_types",
        help="Limit consolidation to specific history event types.",
    )
    parser.add_argument(
        "--no-markdown",
        action="store_true",
        help="Skip writing markdown summaries alongside JSONL entries.",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=0,
        help="Maximum runs when interval > 0 (0 = unlimited).",
    )
    parser.add_argument(
        "--sleep-first",
        action="store_true",
        help="Sleep for interval before the first run.",
    )
    return parser


def _resolve_interval(args: argparse.Namespace, config) -> int:
    if args.interval is not None:
        return args.interval
    return int(config.memory_consolidation.interval_seconds)


def _run_consolidation(args: argparse.Namespace, config) -> AgentResult:
    context_root = (
        Path(args.context_root).expanduser().resolve()
        if args.context_root
        else config.general.context_root
    )
    history_root = (
        Path(args.history_root).expanduser().resolve()
        if args.history_root
        else None
    )
    memory_root = (
        Path(args.memory_root).expanduser().resolve()
        if args.memory_root
        else None
    )
    checkpoint_path = (
        Path(args.checkpoint).expanduser().resolve()
        if args.checkpoint
        else None
    )

    started_at = now_iso()
    start = time.monotonic()
    consolidation = consolidate_history_to_memory(
        context_root,
        config=config,
        history_root=history_root,
        memory_root=memory_root,
        checkpoint_path=checkpoint_path,
        max_events_per_run=args.max_events,
        max_events_per_entry=args.max_events_per_entry,
        include_event_types=args.event_types,
        write_markdown=not args.no_markdown,
    )
    duration = time.monotonic() - start

    status = "ok"
    if not consolidation.history_root.exists():
        status = "warn"

    return AgentResult(
        name=AGENT_NAME,
        status=status,
        started_at=started_at,
        finished_at=now_iso(),
        duration_seconds=duration,
        metrics={
            "scanned_events": consolidation.scanned_events,
            "consolidated_events": consolidation.consolidated_events,
            "entries_written": consolidation.entries_written,
            "markdown_written": consolidation.markdown_written,
        },
        notes=list(consolidation.notes),
        payload=consolidation.to_dict(),
    )


def run(args: argparse.Namespace) -> int:
    configure_logging(args.quiet)
    config = load_agent_config(args.config)
    interval = _resolve_interval(args, config)
    runs = 0

    if interval > 0 and args.sleep_first:
        time.sleep(interval)

    while True:
        runs += 1
        result = _run_consolidation(args, config)
        result.payload["run_index"] = runs
        output_path = None
        if args.output:
            output_path = Path(args.output)
        elif config.memory_consolidation.report_output:
            output_path = Path(config.memory_consolidation.report_output)
        emit_result(
            result,
            output_path=output_path,
            force_stdout=args.stdout,
            pretty=args.pretty,
        )

        if interval <= 0:
            break
        if args.max_runs and runs >= args.max_runs:
            break
        time.sleep(interval)

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
