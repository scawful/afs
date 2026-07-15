"""Write a daily briefing digest into the scratchpad briefings directory."""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from ..context_paths import resolve_mount_root
from ..models import MountType
from .base import (
    AgentResult,
    build_base_parser,
    configure_logging,
    emit_result,
    load_agent_config,
    now_iso,
)

logger = logging.getLogger(__name__)

AGENT_NAME = "morning-briefing"
AGENT_DESCRIPTION = (
    "Write a daily briefing digest to the configured scratchpad directory."
)
MAX_ERROR_CHARS = 2000

AGENT_CAPABILITIES = {
    "mount_types": ["scratchpad"],
    "topics": ["briefing:daily"],
    "tools": [],
    "description": "Daily one-shot briefing snapshot; idempotent per calendar day.",
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_base_parser(AGENT_DESCRIPTION)
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Lookback window in days (default: 7).",
    )
    parser.add_argument(
        "--gws",
        action="store_true",
        help="Include Google Workspace data (off by default for background runs).",
    )
    parser.add_argument(
        "--local-tasks",
        action="store_true",
        help="Query the local task HTTP service (off by default for background runs).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rewrite today's briefing even if it already exists.",
    )
    args = parser.parse_args(argv)
    configure_logging(args.quiet)

    started_at = now_iso()
    start = time.time()
    config = load_agent_config(args.config)

    status = "ok"
    notes: list[str] = []
    payload: dict[str, object] = {}
    metrics: dict[str, float | int] = {"written": 0}

    context_path = config.general.context_root.expanduser().resolve()
    briefing_dir = resolve_mount_root(
        context_path,
        MountType.SCRATCHPAD,
        config=config,
    ) / "briefings"
    target = briefing_dir / f"briefing-{datetime.now():%Y-%m-%d}.md"
    payload.update(
        {
            "context_path": str(context_path),
            "path": str(target),
            "gws": bool(args.gws),
            "local_tasks": bool(args.local_tasks),
        }
    )

    if not context_path.is_dir():
        status = "skipped"
        notes.append(f"context path does not exist: {context_path}")
    else:
        try:
            # The briefing builder lives with its CLI; reuse it rather than
            # duplicating git/task collection here.
            from ..cli.briefing import _build_briefing, _render_text

            briefing = _build_briefing(
                days=args.days,
                include_gws=args.gws,
                include_tasks=args.local_tasks,
            )
            briefing_dir.mkdir(parents=True, exist_ok=True)
            mode = "w" if args.force else "x"
            try:
                with target.open(mode, encoding="utf-8") as stream:
                    stream.write(_render_text(briefing) + "\n")
                metrics["written"] = 1
            except FileExistsError:
                notes.append("today's briefing already exists; skipping")
        except Exception as exc:  # noqa: BLE001 - background agent must not crash silently
            error = f"{type(exc).__name__}: {exc}"[:MAX_ERROR_CHARS]
            status = "error"
            payload["error"] = error
            notes.append("briefing build failed")

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
