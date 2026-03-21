"""Helpers for resolving context mount roots from metadata or config."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

from .config import load_config_model
from .mapping import resolve_directory_name
from .models import MountType, ProjectMetadata
from .schema import AFSConfig, DirectoryConfig

METADATA_FILE = "metadata.json"


def load_context_metadata(context_path: Path) -> ProjectMetadata | None:
    """Load context metadata when available."""
    metadata_path = context_path.expanduser().resolve() / METADATA_FILE
    if not metadata_path.exists():
        return None
    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return ProjectMetadata.from_dict(data)


def resolve_mount_root(
    context_path: Path,
    mount_type: MountType,
    *,
    config: AFSConfig | None = None,
    directories: Iterable[DirectoryConfig] | None = None,
) -> Path:
    """Resolve a mount root using persisted metadata before config defaults."""
    resolved_context = context_path.expanduser().resolve()
    metadata = load_context_metadata(resolved_context)

    resolved_directories = directories
    if resolved_directories is None:
        resolved_directories = config.directories if config is not None else load_config_model().directories

    directory_name = resolve_directory_name(
        mount_type,
        afs_directories=resolved_directories,
        metadata=metadata,
    )
    return resolved_context / directory_name


def resolve_agent_output_root(
    context_path: Path,
    *,
    config: AFSConfig | None = None,
    directories: Iterable[DirectoryConfig] | None = None,
) -> Path:
    """Resolve the shared scratchpad output directory used by built-in agents."""
    return resolve_mount_root(
        context_path,
        MountType.SCRATCHPAD,
        config=config,
        directories=directories,
    ) / "afs_agents"


def resolve_agent_scratchpad(
    context_path: Path,
    agent_name: str,
    *,
    config: AFSConfig | None = None,
    directories: Iterable[DirectoryConfig] | None = None,
) -> Path:
    """Resolve a per-agent scratchpad directory within scratchpad/agents/<agent_name>/."""
    scratchpad_root = resolve_mount_root(
        context_path,
        MountType.SCRATCHPAD,
        config=config,
        directories=directories,
    )
    return scratchpad_root / "agents" / agent_name
