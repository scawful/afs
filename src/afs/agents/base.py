"""Shared helpers for AFS background agents."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from ..config import load_config_model
from ..discovery import discover_contexts
from ..manager import AFSManager
from ..models import ContextRoot
from ..schema import AFSConfig

VALID_STATUSES: frozenset[str] = frozenset({"ok", "warning", "error", "skipped", "timeout"})


@dataclass
class AgentResult:
    name: str
    status: str
    started_at: str
    finished_at: str
    duration_seconds: float
    task: str = ""
    metrics: dict[str, float | int] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": self.duration_seconds,
            "task": self.task,
            "metrics": dict(self.metrics),
            "notes": list(self.notes),
            "payload": dict(self.payload),
        }

    def validate(self) -> list[str]:
        """Validate the result fields, returning a list of error strings (empty = valid)."""
        errors: list[str] = []

        if self.status not in VALID_STATUSES:
            errors.append(
                f"invalid status {self.status!r}; expected one of {sorted(VALID_STATUSES)}"
            )

        if not isinstance(self.name, str) or not self.name.strip():
            errors.append("name must be a non-empty string")

        started_dt = _parse_iso_timestamp(self.started_at)
        if started_dt is None:
            errors.append(f"started_at is not a valid ISO timestamp: {self.started_at!r}")

        finished_dt = _parse_iso_timestamp(self.finished_at)
        if finished_dt is None:
            errors.append(f"finished_at is not a valid ISO timestamp: {self.finished_at!r}")

        if started_dt is not None and finished_dt is not None and finished_dt < started_dt:
            errors.append(
                f"finished_at ({self.finished_at}) is before started_at ({self.started_at})"
            )

        if self.duration_seconds < 0:
            errors.append(f"duration_seconds must be >= 0, got {self.duration_seconds}")

        for key in self.metrics:
            if not isinstance(key, str):
                errors.append(f"metrics key {key!r} is not a string")

        return errors


def _parse_iso_timestamp(value: str) -> datetime | None:
    """Parse an ISO-format timestamp string, returning None on failure."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return None


def validate_result_dict(data: dict[str, Any]) -> tuple[AgentResult | None, list[str]]:
    """Parse a dict into an AgentResult with validation.

    Returns (result, errors).  If parsing fails outright, result is None and
    errors contains the reason.  If parsing succeeds, result is populated and
    errors comes from ``result.validate()``.
    """
    errors: list[str] = []

    # --- required fields ---
    name = data.get("name")
    if not isinstance(name, str) or not name.strip():
        errors.append("missing or empty 'name'")
        name = ""

    status = data.get("status")
    if not isinstance(status, str):
        errors.append("missing or non-string 'status'")
        status = ""

    started_at = data.get("started_at")
    if not isinstance(started_at, str):
        errors.append("missing or non-string 'started_at'")
        started_at = ""

    finished_at = data.get("finished_at")
    if not isinstance(finished_at, str):
        errors.append("missing or non-string 'finished_at'")
        finished_at = ""

    duration_seconds = data.get("duration_seconds")
    if not isinstance(duration_seconds, (int, float)):
        errors.append("missing or non-numeric 'duration_seconds'")
        duration_seconds = 0.0

    # If we had structural parse errors, bail early.
    if errors:
        return None, errors

    # --- optional fields ---
    task = str(data.get("task", ""))
    raw_metrics = data.get("metrics", {})
    metrics: dict[str, float | int] = {}
    if isinstance(raw_metrics, dict):
        for k, v in raw_metrics.items():
            if isinstance(k, str) and isinstance(v, (int, float)):
                metrics[k] = v
            else:
                errors.append(f"invalid metrics entry: {k!r}={v!r}")
    else:
        errors.append("metrics is not a dict")

    raw_notes = data.get("notes", [])
    notes = [str(n) for n in raw_notes] if isinstance(raw_notes, list) else []

    raw_payload = data.get("payload", {})
    payload = dict(raw_payload) if isinstance(raw_payload, dict) else {}

    result = AgentResult(
        name=name,
        status=status,
        started_at=started_at,
        finished_at=finished_at,
        duration_seconds=float(duration_seconds),
        task=task,
        metrics=metrics,
        notes=notes,
        payload=payload,
    )

    # Run semantic validation on the constructed result.
    errors.extend(result.validate())
    return result, errors


def build_base_parser(description: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", help="Path to AFS config.")
    parser.add_argument("--output", help="Write JSON report to this path.")
    parser.add_argument(
        "--stdout",
        action="store_true",
        help="Force JSON output to stdout (default: only when interactive).",
    )
    parser.add_argument(
        "--pretty",
        action="store_true",
        help="Pretty-print JSON output.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce logging to errors only.",
    )
    return parser


def configure_logging(quiet: bool) -> None:
    level = logging.ERROR if quiet else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")


def load_agent_config(config_path: str | None) -> AFSConfig:
    path = Path(config_path).expanduser() if config_path else None
    return load_config_model(config_path=path, merge_user=True)


def resolve_contexts(
    config: AFSConfig,
    *,
    context_paths: Iterable[str] | None,
    search_paths: Iterable[str] | None,
    max_depth: int,
    ignore_names: Iterable[str] | None,
) -> list[ContextRoot]:
    manager = AFSManager(config=config)

    contexts: list[ContextRoot] = []
    if context_paths:
        for raw_path in context_paths:
            candidate = Path(raw_path).expanduser()
            if candidate.name != ".context" and (candidate / ".context").is_dir():
                candidate = candidate / ".context"
            contexts.append(manager.list_context(context_path=candidate))
        return contexts

    paths = [Path(path).expanduser() for path in search_paths] if search_paths else None
    return discover_contexts(
        search_paths=paths,
        max_depth=max_depth,
        ignore_names=ignore_names,
        config=config,
    )


def emit_result(
    result: AgentResult,
    *,
    output_path: Path | None,
    force_stdout: bool,
    pretty: bool,
) -> None:
    payload = result.to_dict()
    text = json.dumps(payload, indent=2 if pretty else None) + "\n"

    if output_path:
        output_path = output_path.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(text, encoding="utf-8")

    try:
        from ..agent_registry import AgentRegistry

        AgentRegistry().mark_result(
            name=result.name,
            status=result.status,
            task=result.task,
            started_at=result.started_at,
            finished_at=result.finished_at,
            output_path=str(output_path) if output_path else "",
            last_error=str(result.payload.get("error", "") or ""),
            metadata={"duration_seconds": result.duration_seconds},
        )
    except Exception:
        logging.getLogger(__name__).debug("failed to update agent registry", exc_info=True)

    # Auto-index agent output into the context index so other agents can discover it
    if output_path:
        try:
            from ..agent_context import index_agent_output

            index_agent_output(output_path, result.name, payload)
        except Exception:
            logging.getLogger(__name__).debug(
                "failed to index agent output for %s", result.name, exc_info=True,
            )

    if force_stdout or sys.stdout.isatty():
        print(text, end="")


def now_iso() -> str:
    return datetime.now().isoformat()


def emit_progress(
    agent_name: str,
    event: str,
    detail: str = "",
    *,
    context_root: Path | None = None,
) -> str | None:
    """Emit an agent progress event to the history log.

    Returns the event ID if logged, otherwise None.
    """
    from ..history import log_event as _log_event

    return _log_event(
        "agent_progress",
        f"agent.{agent_name}",
        op=event,
        context_root=context_root,
        metadata={"detail": detail, "agent": agent_name},
    )
