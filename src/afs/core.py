"""Core AFS helpers."""

from __future__ import annotations

import os
from pathlib import Path

from .schema import AFSConfig


def _context_authorizes_path(context_root: Path, path: Path) -> bool:
    """Keep central v2 discovery behind the project registry boundary."""

    from .context_layout import LAYOUT_VERSION, detect_layout_version

    if detect_layout_version(context_root) != LAYOUT_VERSION:
        return True
    from .project_registry import ProjectRegistry

    try:
        return ProjectRegistry(context_root).resolve(path) is not None
    except (OSError, ValueError):
        return False


def find_root(start_dir: Path | None = None) -> Path | None:
    """Find a .context directory by walking upward."""
    if start_dir is None:
        start_dir = Path.cwd()
    current = start_dir.resolve()
    for parent in [current, *current.parents]:
        candidate = parent / ".context"
        if candidate.exists() and candidate.is_dir():
            if _context_authorizes_path(candidate, current):
                return candidate
            continue
        if (parent / "afs.toml").exists():
            return parent / ".context"
    return None


def find_existing_root(start_dir: Path | None = None) -> Path | None:
    """Find the nearest existing .context directory by walking upward."""
    if start_dir is None:
        start_dir = Path.cwd()
    current = start_dir.resolve()
    for parent in [current, *current.parents]:
        candidate = parent / ".context"
        if candidate.exists() and candidate.is_dir():
            # A central v2 root (commonly ~/.context) is not inherited merely
            # because a checkout lives below its parent directory.  Only a
            # registered project may resolve it.
            if _context_authorizes_path(candidate, current):
                return candidate
            continue
    return None


def resolve_context_root(config: AFSConfig | None, linked_root: Path | None) -> Path:
    """Resolve the active context root for this machine."""
    env_root = os.environ.get("AFS_CONTEXT_ROOT")
    if env_root:
        return Path(env_root).expanduser().resolve()
    if linked_root:
        return linked_root.resolve()
    if config:
        return config.general.context_root
    return (Path.home() / ".context").resolve()
