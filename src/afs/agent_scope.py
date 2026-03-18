"""Runtime helpers for env-scoped agent permissions."""

from __future__ import annotations

import fnmatch
import os

from .models import MountType


def _env_list(name: str) -> list[str]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return []
    return [item.strip() for item in raw.split(",") if item.strip()]


def allowed_mounts() -> set[str] | None:
    values = _env_list("AFS_ALLOWED_MOUNTS")
    return set(values) if values else None


def assert_mount_allowed(mount_type: MountType, *, operation: str) -> None:
    allowed = allowed_mounts()
    if allowed is None or mount_type.value in allowed:
        return
    agent_name = os.environ.get("AFS_AGENT_NAME", "unknown")
    raise PermissionError(
        f"agent {agent_name} not allowed to {operation} {mount_type.value}"
    )


def allowed_tools() -> list[str] | None:
    values = _env_list("AFS_ALLOWED_TOOLS")
    return values if values else None


def assert_tool_allowed(tool_name: str) -> None:
    patterns = allowed_tools()
    if patterns is None or any(fnmatch.fnmatch(tool_name, pattern) for pattern in patterns):
        return
    agent_name = os.environ.get("AFS_AGENT_NAME", "unknown")
    raise PermissionError(f"agent {agent_name} not allowed to use tool {tool_name}")


def workspace_isolated() -> bool:
    raw = os.environ.get("AFS_WORKSPACE_ISOLATED", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}
