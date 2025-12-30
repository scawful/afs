"""Helpers for mapping AFS roles to on-disk directory names."""

from __future__ import annotations

from typing import Iterable

from .models import MountType, ProjectMetadata
from .schema import DirectoryConfig


def _role_to_mount_type(role_name: str) -> MountType | None:
    try:
        return MountType(role_name)
    except ValueError:
        return None


def build_directory_map_from_config(
    afs_directories: Iterable[DirectoryConfig] | None,
) -> dict[MountType, str]:
    mapping: dict[MountType, str] = {}
    if not afs_directories:
        return mapping

    for dir_config in afs_directories:
        role_name = dir_config.role.value if dir_config.role else dir_config.name
        mount_type = _role_to_mount_type(role_name)
        if not mount_type:
            continue
        mapping[mount_type] = dir_config.name
    return mapping


def build_directory_map_from_metadata(
    metadata: ProjectMetadata | None,
) -> dict[MountType, str]:
    mapping: dict[MountType, str] = {}
    if not metadata or not metadata.directories:
        return mapping

    for role_name, dir_name in metadata.directories.items():
        mount_type = _role_to_mount_type(role_name)
        if not mount_type:
            continue
        mapping[mount_type] = dir_name
    return mapping


def resolve_directory_map(
    *,
    afs_directories: Iterable[DirectoryConfig] | None = None,
    metadata: ProjectMetadata | None = None,
) -> dict[MountType, str]:
    mapping = build_directory_map_from_metadata(metadata)
    if not mapping:
        mapping = build_directory_map_from_config(afs_directories)
    return mapping


def resolve_directory_name(
    mount_type: MountType,
    *,
    afs_directories: Iterable[DirectoryConfig] | None = None,
    metadata: ProjectMetadata | None = None,
) -> str:
    mapping = resolve_directory_map(afs_directories=afs_directories, metadata=metadata)
    return mapping.get(mount_type, mount_type.value)
