"""Mine repeated successful session traces into reviewable skill candidates."""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from pathlib import Path

from ..manager import AFSManager
from ..skill_mining import mine_skill_candidates, write_skill_candidate_artifacts
from .base import (
    AgentResult,
    build_base_parser,
    configure_logging,
    emit_result,
    load_agent_config,
    now_iso,
)

logger = logging.getLogger(__name__)

AGENT_NAME = "skills-mine"
AGENT_DESCRIPTION = (
    "Mine repeated successful session traces into reviewable skill candidates."
)
MAX_ERROR_CHARS = 2000

AGENT_CAPABILITIES = {
    "mount_types": ["scratchpad", "history"],
    "topics": ["skills:mine"],
    "tools": ["skills.mine"],
    "description": "Weekly skill mining; candidates land in scratchpad for review.",
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_base_parser(AGENT_DESCRIPTION)
    parser.add_argument(
        "--context-path",
        help="Context root to mine (default: the configured global context root).",
    )
    parser.add_argument(
        "--lookback-hours",
        type=int,
        default=7 * 24,
        help="Only analyze sessions ending within this many hours (default: 168).",
    )
    parser.add_argument(
        "--no-write-artifacts",
        action="store_true",
        help="Do not write candidate artifacts to scratchpad.",
    )
    args = parser.parse_args(argv)
    configure_logging(args.quiet)

    started_at = now_iso()
    start = time.time()
    config = load_agent_config(args.config)
    manager = AFSManager(config=config)
    context_path = (
        Path(args.context_path).expanduser().resolve()
        if args.context_path
        else config.general.context_root.expanduser().resolve()
    )

    status = "ok"
    notes: list[str] = []
    payload: dict[str, object] = {"context_path": str(context_path)}
    metrics: dict[str, float | int] = {"candidates": 0, "successful_sessions": 0}

    if not context_path.is_dir():
        status = "skipped"
        notes.append(f"context path does not exist: {context_path}")
    else:
        try:
            mined = mine_skill_candidates(
                context_path,
                lookback_hours=args.lookback_hours,
                config=config,
            )
            metrics["candidates"] = int(mined.get("candidate_count", 0))
            metrics["successful_sessions"] = int(
                mined.get("successful_sessions", 0)
            )
            if metrics["candidates"] and not args.no_write_artifacts:
                mined["artifact_paths"] = write_skill_candidate_artifacts(
                    manager,
                    context_path,
                    mined,
                )
            payload["mined"] = mined
            if not metrics["candidates"]:
                notes.append("no repeated successful traces found this window")
        except Exception as exc:  # noqa: BLE001 - report background failures
            error = f"{type(exc).__name__}: {exc}"[:MAX_ERROR_CHARS]
            status = "error"
            payload["error"] = error
            notes.append("skill mining failed")

    result = AgentResult(
        name=AGENT_NAME,
        status=status,
        started_at=started_at,
        finished_at=now_iso(),
        duration_seconds=time.time() - start,
        metrics=metrics,
        notes=notes,
        payload=payload,
    )
    emit_result(
        result,
        output_path=Path(args.output).expanduser() if args.output else None,
        force_stdout=args.stdout,
        pretty=args.pretty,
    )
    return 1 if status == "error" else 0


if __name__ == "__main__":
    raise SystemExit(main())
