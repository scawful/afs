"""History logging helpers for AFS."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable

from .config import load_config_model
from .core import find_root, resolve_context_root


SENSITIVE_MARKERS = ("key", "token", "secret", "password")


@dataclass(frozen=True)
class HistoryEvent:
    """Immutable history event record."""

    timestamp: str
    command: list[str]
    exit_code: int
    cwd: str
    context_root: str | None

    def to_dict(self) -> dict:
        return {
            "timestamp": self.timestamp,
            "command": list(self.command),
            "exit_code": self.exit_code,
            "cwd": self.cwd,
            "context_root": self.context_root,
        }


def _should_redact(arg: str) -> bool:
    lowered = arg.lower()
    return any(marker in lowered for marker in SENSITIVE_MARKERS)


def sanitize_argv(argv: Iterable[str]) -> list[str]:
    """Redact sensitive arguments from argv."""
    sanitized: list[str] = []
    redact_next = False

    for arg in argv:
        if redact_next:
            sanitized.append("[redacted]")
            redact_next = False
            continue

        if "=" in arg:
            key, value = arg.split("=", 1)
            if _should_redact(key):
                sanitized.append(f"{key}=[redacted]")
                continue
        if arg.startswith("--") and _should_redact(arg):
            sanitized.append(arg)
            redact_next = True
            continue
        sanitized.append(arg)

    return sanitized


def _resolve_context_root(start_dir: Path | None = None) -> Path | None:
    config = load_config_model()
    root = find_root(start_dir)
    try:
        return resolve_context_root(config, root)
    except Exception:
        return None


def _history_log_path(context_root: Path | None) -> Path | None:
    if context_root is None:
        return None
    history_dir = context_root / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d")
    return history_dir / f"commands_{stamp}.jsonl"


def log_cli_invocation(argv: Iterable[str], exit_code: int) -> None:
    """Append a CLI invocation to the history log."""
    if os.getenv("AFS_HISTORY_DISABLED") == "1":
        return

    sanitized = sanitize_argv(argv)
    context_root = _resolve_context_root(Path.cwd())
    log_path = _history_log_path(context_root)
    if log_path is None:
        return

    event = HistoryEvent(
        timestamp=datetime.now(timezone.utc).isoformat(),
        command=sanitized,
        exit_code=exit_code,
        cwd=str(Path.cwd()),
        context_root=str(context_root) if context_root else None,
    )

    with log_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(event.to_dict()) + "\n")
