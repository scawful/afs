"""Portable runtime path defaults for AFS.

Every helper accepts an explicit environment override and otherwise derives a
local default without assuming a particular source-code hierarchy.
"""

from __future__ import annotations

import os
from pathlib import Path


def _env_path(name: str) -> Path | None:
    value = os.getenv(name, "").strip()
    return Path(value).expanduser().resolve() if value else None


def default_config_root() -> Path:
    """Return the directory for user-level AFS configuration and state."""
    if configured := _env_path("AFS_CONFIG_HOME"):
        return configured
    if xdg_root := _env_path("XDG_CONFIG_HOME"):
        return xdg_root / "afs"
    if appdata := _env_path("APPDATA"):
        return appdata / "afs"
    return Path.home() / ".config" / "afs"


def default_context_root() -> Path:
    """Return the context root while preserving the established default."""
    return _env_path("AFS_CONTEXT_ROOT") or Path.home() / ".context"


def default_workspace_root(start_dir: Path | None = None) -> Path:
    """Find a workspace catalog above *start_dir*, or use that directory."""
    if configured := _env_path("AFS_WORKSPACE_ROOT"):
        return configured

    current = (start_dir or Path.cwd()).expanduser().resolve()
    if current.is_file():
        current = current.parent
    for candidate in (current, *current.parents):
        if (candidate / "WORKSPACE.toml").is_file():
            return candidate
    return current


def default_training_root() -> Path:
    """Return the generic training artifact root."""
    if configured := _env_path("AFS_TRAINING_ROOT"):
        return configured
    return default_context_root() / "scratchpad" / "common" / "training"


def default_training_dataset(filename: str) -> Path:
    """Return a dataset path below the configurable training root."""
    return default_training_root() / "datasets" / filename


def default_worktrees_root(repo_path: Path) -> Path:
    """Return an isolated worktree directory namespaced by repository."""
    repo = repo_path.expanduser().resolve()
    if configured := _env_path("AFS_WORKTREES_ROOT"):
        return configured / repo.name
    return repo.parent / ".afs-worktrees" / repo.name
