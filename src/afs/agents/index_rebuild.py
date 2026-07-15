"""Rebuild the context SQLite index for a context root."""

from __future__ import annotations

import logging
import time
from collections.abc import Sequence
from pathlib import Path

from ..context_index import ContextSQLiteIndex
from ..manager import AFSManager
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

AGENT_NAME = "index-rebuild"
AGENT_DESCRIPTION = "Rebuild the context SQLite index when it is missing or stale."
INDEX_MOUNT_TYPES = [MountType.KNOWLEDGE, MountType.MEMORY]
MAX_ERROR_CHARS = 2000

AGENT_CAPABILITIES = {
    "mount_types": ["knowledge", "memory"],
    "topics": ["context:refresh"],
    "tools": ["context.index.rebuild"],
    "description": "One-shot index refresh; skips work when the index is already fresh.",
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_base_parser(AGENT_DESCRIPTION)
    parser.add_argument(
        "--context-path",
        help="Context root to index (default: the configured global context root).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Rebuild even when the index reports itself fresh.",
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
    metrics: dict[str, float | int] = {"rebuilt": 0}

    if not context_path.is_dir():
        status = "skipped"
        notes.append(f"context path does not exist: {context_path}")
    else:
        try:
            index = ContextSQLiteIndex(manager, context_path)
            needs_rebuild = (
                args.force
                or not index.has_entries(mount_types=INDEX_MOUNT_TYPES)
                or index.needs_refresh(mount_types=INDEX_MOUNT_TYPES)
            )
            if not needs_rebuild:
                notes.append("index already fresh; nothing to do")
            else:
                summary = index.rebuild(
                    mount_types=INDEX_MOUNT_TYPES,
                    include_content=config.context_index.include_content,
                    max_file_size_bytes=config.context_index.max_file_size_bytes,
                    max_content_chars=config.context_index.max_content_chars,
                )
                payload["rebuild"] = summary.to_dict()
                if summary.errors:
                    error = "; ".join(summary.errors)[:MAX_ERROR_CHARS]
                    status = "error"
                    payload["error"] = error
                    notes.append("index rebuild completed with errors")
                else:
                    metrics["rebuilt"] = 1
        except Exception as exc:  # noqa: BLE001 - report background failures
            error = f"{type(exc).__name__}: {exc}"[:MAX_ERROR_CHARS]
            status = "error"
            payload["error"] = error
            notes.append("index rebuild failed")

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
