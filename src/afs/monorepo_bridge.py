"""Monorepo bridge helpers for active workspace state."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

from .models import MountType

if TYPE_CHECKING:
    from .manager import AFSManager


ACTIVE_WORKSPACE_FILENAME = "active_workspace.toml"
DEFAULT_ACTIVE_WORKSPACE_STALE_SECONDS = 3600


@dataclass(frozen=True)
class WorkspaceBridgeStatus:
    path: Path
    exists: bool
    age_seconds: float | None
    stale: bool
    modified_at: str | None


def resolve_monorepo_root(
    context_path: Path,
    *,
    manager: AFSManager | None = None,
) -> Path:
    """Resolve the monorepo mount root for a context path."""
    resolved = context_path.expanduser().resolve()
    if manager is not None:
        try:
            return manager.resolve_mount_root(resolved, MountType.MONOREPO)
        except Exception:
            pass
    return resolved / MountType.MONOREPO.value


def active_workspace_file(
    context_path: Path,
    *,
    manager: AFSManager | None = None,
) -> Path:
    """Get the active workspace bridge file path for a context."""
    return resolve_monorepo_root(context_path, manager=manager) / ACTIVE_WORKSPACE_FILENAME


def _compute_age_seconds(path: Path, now: datetime | None = None) -> float | None:
    try:
        stat = path.stat()
    except OSError:
        return None
    current = now or datetime.now(timezone.utc)
    modified = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc)
    return max((current - modified).total_seconds(), 0.0)


def get_workspace_bridge_status(
    context_path: Path,
    *,
    manager: AFSManager | None = None,
    stale_after_seconds: int = DEFAULT_ACTIVE_WORKSPACE_STALE_SECONDS,
    now: datetime | None = None,
) -> WorkspaceBridgeStatus:
    """Return freshness details for active workspace bridge metadata."""
    bridge_path = active_workspace_file(context_path, manager=manager)
    if not bridge_path.exists():
        return WorkspaceBridgeStatus(
            path=bridge_path,
            exists=False,
            age_seconds=None,
            stale=False,
            modified_at=None,
        )

    age_seconds = _compute_age_seconds(bridge_path, now=now)
    stale = bool(age_seconds is not None and age_seconds > stale_after_seconds)
    modified_at = None
    if age_seconds is not None:
        try:
            modified_at = datetime.fromtimestamp(
                bridge_path.stat().st_mtime, tz=timezone.utc
            ).isoformat()
        except OSError:
            modified_at = None

    return WorkspaceBridgeStatus(
        path=bridge_path,
        exists=True,
        age_seconds=age_seconds,
        stale=stale,
        modified_at=modified_at,
    )
