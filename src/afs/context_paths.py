"""Helpers for resolving context mount roots from metadata or config."""

from __future__ import annotations

import json
import re
from collections.abc import Iterable
from pathlib import Path

from .config import load_config_model
from .context_layout import (
    LAYOUT_VERSION,
    V2_COMPAT_MOUNT_PATHS,
    detect_layout_version,
    v2_directory_map,
)
from .mapping import resolve_directory_name
from .models import MountType, ProjectMetadata
from .path_safety import assert_no_linklike_components
from .schema import AFSConfig, DirectoryConfig

METADATA_FILE = "metadata.json"


def load_context_metadata(context_path: Path) -> ProjectMetadata | None:
    """Load context metadata when available."""
    resolved_context = context_path.expanduser().resolve()
    is_v2 = detect_layout_version(resolved_context) == LAYOUT_VERSION
    metadata_path = (
        resolved_context / ".afs" / "compat" / METADATA_FILE
        if is_v2
        else resolved_context / METADATA_FILE
    )
    if is_v2:
        metadata_path = assert_no_linklike_components(
            metadata_path,
            boundary=resolved_context,
        )
    if not metadata_path.exists():
        return ProjectMetadata(directories=v2_directory_map()) if is_v2 else None
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
    layout_version = detect_layout_version(resolved_context)
    if layout_version == LAYOUT_VERSION:
        # A v2 namespace has a fixed routing table. Compatibility metadata is
        # retained as provenance only and must never redirect a trusted mount
        # outside the central root.
        root = resolved_context / V2_COMPAT_MOUNT_PATHS[mount_type]
        return assert_no_linklike_components(root, boundary=resolved_context)

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
    scope_id: str = "common",
) -> Path:
    """Resolve the agent-output directory without crossing a v2 scope.

    Version 1 keeps the historical ``scratchpad/afs_agents`` path.  A central
    v2 context stores the same artifacts below either ``scratchpad/common`` or
    ``scratchpad/projects/<project-id>`` so two registered projects cannot
    overwrite or reuse one another's session state.
    """
    scratchpad_root = resolve_mount_root(
        context_path,
        MountType.SCRATCHPAD,
        config=config,
        directories=directories,
    )
    if detect_layout_version(context_path) != LAYOUT_VERSION:
        return scratchpad_root / "afs_agents"

    normalized_scope = str(scope_id or "common").strip()
    if normalized_scope == "common":
        scope_root = scratchpad_root / "common"
    else:
        prefix, separator, project_id = normalized_scope.partition(":")
        if (
            prefix != "project"
            or not separator
            or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", project_id)
        ):
            raise ValueError("scope_id must be 'common' or 'project:<project-id>'")
        scope_root = scratchpad_root / "projects" / project_id
    return assert_no_linklike_components(
        scope_root / "afs_agents",
        boundary=context_path.expanduser().resolve(),
    )


def resolve_agent_scratchpad(
    context_path: Path,
    agent_name: str,
    *,
    config: AFSConfig | None = None,
    directories: Iterable[DirectoryConfig] | None = None,
) -> Path:
    """Resolve a safe per-agent scratchpad directory.

    Version 1 retains ``scratchpad/agents/<name>``. Version 2 routes shared
    agent control output through ``scratchpad/common/agents/<name>``.
    """
    normalized_name = str(agent_name).strip()
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", normalized_name):
        raise ValueError("agent_name must be one safe filesystem segment")
    scratchpad_root = resolve_mount_root(
        context_path,
        MountType.SCRATCHPAD,
        config=config,
        directories=directories,
    )
    if detect_layout_version(context_path) != LAYOUT_VERSION:
        return scratchpad_root / "agents" / normalized_name
    context_root = context_path.expanduser().resolve()
    return assert_no_linklike_components(
        scratchpad_root / "common" / "agents" / normalized_name,
        boundary=context_root,
    )
