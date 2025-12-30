"""Core AFS helpers."""

from __future__ import annotations

import os
from pathlib import Path

from .schema import AFSConfig


def find_root(start_dir: Path | None = None) -> Path | None:
    """Find a .context directory by walking upward."""
    if start_dir is None:
        start_dir = Path.cwd()
    current = start_dir.resolve()
    for parent in [current, *current.parents]:
        candidate = parent / ".context"
        if candidate.exists() and candidate.is_dir():
            return candidate
        if (parent / "afs.toml").exists():
            return parent / ".context"
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
