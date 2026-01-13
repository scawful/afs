"""Export memory entries into training data."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Sequence

from ..training import export_memory_to_dataset
from .base import (
    AgentResult,
    build_base_parser,
    configure_logging,
    emit_result,
    load_agent_config,
    now_iso,
)

AGENT_NAME = "memory-export"
AGENT_DESCRIPTION = "Export memory entries into TrainingSample JSONL."


def build_parser() -> argparse.ArgumentParser:
    parser = build_base_parser("Export memory entries into TrainingSample JSONL.")
    parser.add_argument("--context-root", help="Context root override.")
    parser.add_argument("--memory-root", help="Memory directory override.")
    parser.add_argument("--output", help="Write JSON report to this path.")
    parser.add_argument("--dataset-output", help="Training JSONL output path.")
    parser.add_argument("--domain", default="memory", help="Default domain.")
    parser.add_argument(
        "--allow-raw",
        action="store_true",
        help="Allow raw content entries without explicit instruction/output.",
    )
    parser.add_argument(
        "--default-instruction",
        help="Instruction to use when allow-raw is set.",
    )
    parser.add_argument("--limit", type=int, help="Maximum exported samples.")
    parser.add_argument(
        "--interval",
        type=int,
        help="Seconds between runs (0 = run once).",
    )
    parser.add_argument(
        "--max-runs",
        type=int,
        default=0,
        help="Maximum runs (0 = unlimited when interval > 0).",
    )
    parser.add_argument(
        "--sleep-first",
        action="store_true",
        help="Sleep for interval before first run.",
    )
    return parser


def _resolve_interval(args: argparse.Namespace, config) -> int:
    if args.interval is not None:
        return args.interval
    return int(config.memory_export.interval_seconds)


def _resolve_limit(args: argparse.Namespace, config) -> int | None:
    if args.limit is not None:
        return args.limit
    limit = int(config.memory_export.limit)
    return limit if limit > 0 else None


def _run_export(args: argparse.Namespace, config) -> AgentResult:
    context_root = (
        Path(args.context_root).expanduser().resolve()
        if args.context_root
        else config.general.context_root
    )
    memory_root = (
        Path(args.memory_root).expanduser().resolve()
        if args.memory_root
        else (Path(context_root) / "memory")
    )
    dataset_output = (
        Path(args.dataset_output).expanduser().resolve()
        if args.dataset_output
        else Path(config.memory_export.dataset_output)
    )

    started_at = now_iso()
    start = time.monotonic()
    export_result = export_memory_to_dataset(
        memory_root,
        dataset_output,
        default_domain=args.domain,
        allow_raw=args.allow_raw or config.memory_export.allow_raw,
        allow_raw_tags=config.memory_export.allow_raw_tags,
        default_instruction=args.default_instruction or config.memory_export.default_instruction,
        limit=_resolve_limit(args, config),
        require_quality=config.memory_export.require_quality,
        min_quality_score=config.memory_export.min_quality_score,
        score_profile=config.memory_export.score_profile,
        enable_asar=config.memory_export.enable_asar,
    )
    route_results = []
    for route in config.memory_export.routes:
        if not route.tags:
            continue
        route_output = Path(route.output)
        route_result = export_memory_to_dataset(
            memory_root,
            route_output,
            default_domain=route.domain or args.domain,
            allow_raw=args.allow_raw or config.memory_export.allow_raw,
            allow_raw_tags=config.memory_export.allow_raw_tags,
            default_instruction=args.default_instruction or config.memory_export.default_instruction,
            include_tags=route.tags,
            limit=_resolve_limit(args, config),
            require_quality=config.memory_export.require_quality,
            min_quality_score=config.memory_export.min_quality_score,
            score_profile=config.memory_export.score_profile,
            enable_asar=config.memory_export.enable_asar,
        )
        route_results.append({
            "tags": list(route.tags),
            "output": str(route_output),
            "exported": route_result.exported,
            "skipped": route_result.skipped,
            "filtered": route_result.filtered,
        })
    duration = time.monotonic() - start

    result = AgentResult(
        name=AGENT_NAME,
        status="ok" if export_result.exported else "warn",
        started_at=started_at,
        finished_at=now_iso(),
        duration_seconds=duration,
        metrics={
            "total_entries": export_result.total_entries,
            "exported": export_result.exported,
            "skipped": export_result.skipped,
            "filtered": export_result.filtered,
            "errors": len(export_result.errors),
        },
        notes=export_result.errors[:5],
        payload={
            "memory_root": str(memory_root),
            "dataset_output": str(dataset_output),
            "routes": route_results,
        },
    )
    return result


def run(args: argparse.Namespace) -> int:
    configure_logging(args.quiet)
    config = load_agent_config(args.config)
    interval = _resolve_interval(args, config)
    runs = 0

    if interval > 0 and args.sleep_first:
        time.sleep(interval)

    while True:
        runs += 1
        result = _run_export(args, config)
        result.payload["run_index"] = runs
        output_path = None
        if args.output:
            output_path = Path(args.output)
        elif config.memory_export.report_output:
            output_path = Path(config.memory_export.report_output)
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
        time.sleep(args.interval)

    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
