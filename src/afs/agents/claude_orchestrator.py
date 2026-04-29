"""Compatibility shim for the extension-owned Claude orchestration agent."""

from __future__ import annotations

import importlib
from collections.abc import Sequence
from typing import Any

AGENT_NAME = "claude-orchestrator"
AGENT_DESCRIPTION = "Extension-owned Claude orchestration agent."
_EXTENSION_MODULE = "afs_scawful.claude_orchestrator"
_ERROR = (
    "claude-orchestrator moved out of core AFS. Install/enable the "
    "afs_scawful extension repo and run its registered agent instead."
)


def _load_extension_module() -> Any:
    try:
        return importlib.import_module(_EXTENSION_MODULE)
    except Exception as exc:  # pragma: no cover - compatibility path
        raise RuntimeError(_ERROR) from exc


def main(argv: Sequence[str] | None = None) -> int:
    return int(_load_extension_module().main(argv))


def __getattr__(name: str) -> Any:
    return getattr(_load_extension_module(), name)
