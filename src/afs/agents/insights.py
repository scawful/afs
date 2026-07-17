"""Opt-in one-shot agent for scoped, deterministic history reflection."""

from __future__ import annotations

import os
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..insights import (
    DEFAULT_SCHEDULED_INSIGHT_HISTORY_WINDOW,
    InsightStore,
    reflect_evidence,
)
from ..profiles import resolve_active_profile
from ..schema import AFSConfig, AgentConfig
from ..scopes import resolve_scope
from .base import (
    AgentResult,
    build_base_parser,
    configure_logging,
    emit_result,
    load_agent_config,
    now_iso,
)

AGENT_NAME = "insights-reflect"
AGENT_DESCRIPTION = (
    "Create a reviewable insight candidate from explicitly attributed local history."
)
MAX_ERROR_CHARS = 2000

AGENT_CAPABILITIES = {
    "mount_types": ["history", "scratchpad"],
    "topics": ["insights:reflect"],
    "tools": [],
    "description": (
        "Opt-in deterministic history reflection; no model, network, or promotion."
    ),
}


def _configured_agent(config: AFSConfig) -> AgentConfig | None:
    runtime_name = os.getenv("AFS_AGENT_NAME", AGENT_NAME).strip() or AGENT_NAME
    profile = resolve_active_profile(config)
    return next(
        (agent for agent in profile.agent_configs if agent.name == runtime_name),
        None,
    )


def _positive_int(value: Any, *, default: int) -> int:
    if isinstance(value, bool):
        return default
    if isinstance(value, int) and value > 0:
        return value
    if isinstance(value, str) and value.strip().isdigit():
        parsed = int(value.strip())
        return parsed if parsed > 0 else default
    return default


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_base_parser(AGENT_DESCRIPTION)
    parser.add_argument(
        "--project-path",
        help=(
            "Registered project to reflect on. Scheduled profiles may instead set "
            "project_path on this agent config."
        ),
    )
    parser.add_argument(
        "--common",
        action="store_true",
        help="Reflect on common-scope events instead of a project.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        help="Maximum attributed events to inspect (default: 100, maximum: 200).",
    )
    parser.add_argument(
        "--no-write",
        action="store_true",
        help="Build and reflect on evidence without writing a candidate.",
    )
    args = parser.parse_args(argv)
    configure_logging(args.quiet)

    started_at = now_iso()
    started = time.time()
    config = load_agent_config(args.config)
    configured = _configured_agent(config)
    extra = configured.extra if configured is not None else {}

    configured_project = extra.get("project_path", "")
    raw_project = args.project_path or (
        configured_project if isinstance(configured_project, str) else ""
    )
    configured_common = extra.get("common") is True
    common = bool(args.common or configured_common)
    limit = args.limit or _positive_int(extra.get("limit"), default=100)
    context_path = config.general.context_root.expanduser().resolve()

    status = "ok"
    notes: list[str] = []
    metrics: dict[str, int | float] = {"evidence_events": 0, "candidates": 0}
    payload: dict[str, Any] = {
        "context_path": str(context_path),
        "common": common,
        "history_window": DEFAULT_SCHEDULED_INSIGHT_HISTORY_WINDOW,
        "network_used": False,
        "model_used": False,
    }

    project_path: Path | None = None
    if raw_project:
        project_path = Path(raw_project).expanduser().resolve()
    elif common:
        project_path = context_path

    if project_path is None:
        status = "skipped"
        notes.append(
            "project_path is required; set it on the agent config or pass "
            "--project-path (use --common explicitly for common scope)"
        )
    elif not context_path.is_dir():
        status = "skipped"
        notes.append(f"context path does not exist: {context_path}")
    elif not project_path.is_dir():
        status = "skipped"
        notes.append(f"project path does not exist: {project_path}")
    else:
        payload["project_path"] = str(project_path)
        try:
            scoped = resolve_scope(
                context_path,
                requester_path=project_path,
                common=common,
            )
            payload["scope_id"] = scoped.scope_id
            store = InsightStore(
                context_path,
                scope_id=scoped.scope_id,
                requester_path=project_path,
                config=config,
            )
            packet = store.build_evidence_packet(
                limit=limit,
                recent_history_limit=DEFAULT_SCHEDULED_INSIGHT_HISTORY_WINDOW,
            )
            metrics["evidence_events"] = len(packet.events)
            payload["evidence_digest"] = packet.evidence_digest
            candidate = reflect_evidence(packet)
            if candidate is None:
                notes.append("no repeated attributed pattern met the reflection threshold")
            elif args.no_write:
                payload["candidate_preview"] = candidate
                notes.append("candidate preview built without writing")
            else:
                creation = store.create_candidate_result(
                    candidate,
                    evidence=packet,
                    agent_name=AGENT_NAME,
                    recent_history_limit=DEFAULT_SCHEDULED_INSIGHT_HISTORY_WINDOW,
                )
                record = store.show(creation.artifact.metadata.artifact_id)
                if record is None:  # pragma: no cover - store creation contract
                    raise RuntimeError("created insight candidate could not be read back")
                metrics["candidates"] = int(creation.created)
                payload["candidate"] = record.to_dict()
                payload["candidate_status"] = record.status
                payload["candidate_created"] = creation.created
                payload["inspected_evidence_digest"] = packet.evidence_digest
                payload["bound_evidence_digest"] = creation.bound_evidence_digest
                if not creation.created:
                    notes.append("matching insight candidate already exists")
        except Exception as exc:  # noqa: BLE001 - report bounded background failures
            status = "error"
            payload["error"] = f"{type(exc).__name__}: {exc}"[:MAX_ERROR_CHARS]
            notes.append("insight reflection failed")

    result = AgentResult(
        name=AGENT_NAME,
        status=status,
        started_at=started_at,
        finished_at=now_iso(),
        duration_seconds=time.time() - started,
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
