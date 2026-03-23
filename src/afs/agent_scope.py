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
    """Return allowed tool patterns from AFS_ALLOWED_TOOLS or AFS_TOOL_PROFILE.

    AFS_ALLOWED_TOOLS takes precedence (explicit glob patterns).
    AFS_TOOL_PROFILE resolves to the profile's preferred_surfaces.
    Returns None when no restriction is active.
    """
    values = _env_list("AFS_ALLOWED_TOOLS")
    if values:
        return values
    profile_name = os.environ.get("AFS_TOOL_PROFILE", "").strip().lower()
    if not profile_name:
        return None
    from .session_workflows import get_tool_profile_surfaces

    surfaces = get_tool_profile_surfaces(profile_name)
    return list(surfaces) if surfaces else None


def is_tool_allowed(tool_name: str) -> bool:
    """Check whether *tool_name* is permitted by the active scope.

    Useful for filtering tool lists without raising.
    """
    patterns = allowed_tools()
    if patterns is None:
        return True
    return any(fnmatch.fnmatch(tool_name, pattern) for pattern in patterns)


def assert_tool_allowed(tool_name: str) -> None:
    if is_tool_allowed(tool_name):
        return
    agent_name = os.environ.get("AFS_AGENT_NAME", "unknown")
    raise PermissionError(f"agent {agent_name} not allowed to use tool {tool_name}")


def workspace_isolated() -> bool:
    raw = os.environ.get("AFS_WORKSPACE_ISOLATED", "").strip().lower()
    return raw in {"1", "true", "yes", "on"}
