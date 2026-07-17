"""Write a daily briefing digest into the scratchpad briefings directory."""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

from ..context_layout import LAYOUT_VERSION, _atomic_write_text, detect_layout_version
from ..context_paths import resolve_mount_root
from ..models import MountType
from ..path_safety import assert_no_linklike_components
from ..schema import AFSConfig
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


def _briefing_output_root(context_path: Path, *, config: AFSConfig) -> Path:
    """Return the shared briefing root without bypassing v2 scope boundaries."""

    scratchpad_root = resolve_mount_root(
        context_path,
        MountType.SCRATCHPAD,
        config=config,
    )
    if detect_layout_version(context_path) == LAYOUT_VERSION:
        output_root = scratchpad_root / "common" / "briefings"
        output_root = assert_no_linklike_components(
            output_root,
            boundary=context_path,
        )
    else:
        output_root = scratchpad_root / "briefings"
    output_root.mkdir(parents=True, exist_ok=True)
    if detect_layout_version(context_path) == LAYOUT_VERSION:
        output_root = assert_no_linklike_components(
            output_root,
            boundary=context_path,
            allow_missing=False,
        )
    return output_root


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
    payload.update(
        {
            "context_path": str(context_path),
            "gws": bool(args.gws),
            "local_tasks": bool(args.local_tasks),
        }
    )

    if not context_path.is_dir():
        status = "skipped"
        notes.append(f"context path does not exist: {context_path}")
    else:
        try:
            briefing_dir = _briefing_output_root(context_path, config=config)
            target = briefing_dir / f"briefing-{datetime.now():%Y-%m-%d}.md"
            if detect_layout_version(context_path) == LAYOUT_VERSION:
                target = assert_no_linklike_components(
                    target,
                    boundary=briefing_dir,
                )
            payload["path"] = str(target)

            # The briefing builder lives with its CLI; reuse it rather than
            # duplicating git/task collection here.
            from ..cli.briefing import _build_briefing, _render_text

            briefing = _build_briefing(
                days=args.days,
                include_gws=args.gws,
                include_tasks=args.local_tasks,
            )
            try:
                rendered = _render_text(briefing) + "\n"
                if args.force:
                    _atomic_write_text(target, rendered)
                else:
                    with target.open("x", encoding="utf-8") as stream:
                        stream.write(rendered)
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
