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


@dataclass
class AgentResult:
    name: str
    status: str
    started_at: str
    finished_at: str
    duration_seconds: float
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
            "metrics": dict(self.metrics),
            "notes": list(self.notes),
            "payload": dict(self.payload),
        }


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

    if force_stdout or sys.stdout.isatty():
        print(text, end="")


def now_iso() -> str:
    return datetime.now().isoformat()
