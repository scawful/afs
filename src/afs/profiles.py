"""Profile resolution and context mount helpers."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from .extensions import load_extensions
from .models import MountType
from .schema import AFSConfig, ProfileConfig

if TYPE_CHECKING:
    from .manager import AFSManager


KNOWLEDGE_ALIAS_PREFIX = "profile-knowledge-"
SKILL_ALIAS_PREFIX = "profile-skill-"
MODEL_ALIAS_PREFIX = "profile-registry-"


@dataclass
class ResolvedProfile:
    name: str
    knowledge_mounts: list[Path] = field(default_factory=list)
    skill_roots: list[Path] = field(default_factory=list)
    model_registries: list[Path] = field(default_factory=list)
    enabled_extensions: list[str] = field(default_factory=list)
    policies: list[str] = field(default_factory=list)
    extension_hooks: dict[str, list[str]] = field(default_factory=dict)


@dataclass
class ProfileApplyResult:
    profile_name: str
    mounted: dict[str, int]
    skipped_missing: list[str]


@dataclass(frozen=True)
class ProfileMountSpec:
    profile_name: str
    mount_type: MountType
    alias: str
    source: Path
    source_exists: bool


def _as_env_paths(name: str) -> list[Path]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return []
    values: list[Path] = []
    for entry in raw.split(os.pathsep):
        if entry.strip():
            values.append(Path(entry).expanduser().resolve())
    return values


def _as_env_list(name: str) -> list[str]:
    raw = os.environ.get(name, "").strip()
    if not raw:
        return []
    return [entry.strip() for entry in re.split(r"[,\s]+", raw) if entry.strip()]


def _merge_unique_paths(*groups: list[Path]) -> list[Path]:
    merged: list[Path] = []
    seen: set[str] = set()
    for group in groups:
        for path in group:
            marker = str(path)
            if marker in seen:
                continue
            seen.add(marker)
            merged.append(path)
    return merged


def _merge_unique_str(*groups: list[str]) -> list[str]:
    merged: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for value in group:
            value = value.strip()
            if not value or value in seen:
                continue
            seen.add(value)
            merged.append(value)
    return merged


def _resolve_profile_graph(name: str, profiles: dict[str, ProfileConfig]) -> ProfileConfig:
    visited: set[str] = set()

    def _visit(current: str) -> ProfileConfig:
        if current in visited:
            return ProfileConfig()
        visited.add(current)
        profile = profiles.get(current)
        if profile is None:
            return ProfileConfig()

        merged = ProfileConfig()
        for parent in profile.inherits:
            parent_config = _visit(parent)
            merged = ProfileConfig(
                inherits=_merge_unique_str(merged.inherits, parent_config.inherits),
                knowledge_mounts=_merge_unique_paths(merged.knowledge_mounts, parent_config.knowledge_mounts),
                skill_roots=_merge_unique_paths(merged.skill_roots, parent_config.skill_roots),
                model_registries=_merge_unique_paths(merged.model_registries, parent_config.model_registries),
                enabled_extensions=_merge_unique_str(
                    merged.enabled_extensions,
                    parent_config.enabled_extensions,
                ),
                policies=_merge_unique_str(merged.policies, parent_config.policies),
            )

        return ProfileConfig(
            inherits=_merge_unique_str(merged.inherits, profile.inherits),
            knowledge_mounts=_merge_unique_paths(merged.knowledge_mounts, profile.knowledge_mounts),
            skill_roots=_merge_unique_paths(merged.skill_roots, profile.skill_roots),
            model_registries=_merge_unique_paths(merged.model_registries, profile.model_registries),
            enabled_extensions=_merge_unique_str(
                merged.enabled_extensions,
                profile.enabled_extensions,
            ),
            policies=_merge_unique_str(merged.policies, profile.policies),
        )

    return _visit(name)


def resolve_active_profile(
    config: AFSConfig,
    profile_name: str | None = None,
) -> ResolvedProfile:
    """Resolve active profile with inheritance, env overrides, and extensions."""
    profiles = config.profiles.profiles

    active = profile_name or os.environ.get("AFS_PROFILE") or config.profiles.active_profile
    if active not in profiles and "default" in profiles:
        active = "default"

    resolved_profile = _resolve_profile_graph(active, profiles)

    enabled_extensions = _merge_unique_str(
        resolved_profile.enabled_extensions,
        config.extensions.enabled_extensions,
        _as_env_list("AFS_ENABLED_EXTENSIONS"),
    )

    extension_manifests = load_extensions(
        config=config,
        requested=enabled_extensions,
        allow_auto_discover=False,
    )

    extension_knowledge: list[Path] = []
    extension_skills: list[Path] = []
    extension_registries: list[Path] = []
    extension_policies: list[str] = []
    extension_hooks: dict[str, list[str]] = {}

    for manifest in extension_manifests.values():
        extension_knowledge.extend(manifest.knowledge_mounts)
        extension_skills.extend(manifest.skill_roots)
        extension_registries.extend(manifest.model_registries)
        extension_policies.extend(manifest.policies)
        for event, commands in manifest.hooks.items():
            extension_hooks.setdefault(event, [])
            extension_hooks[event].extend(commands)

    return ResolvedProfile(
        name=active,
        knowledge_mounts=_merge_unique_paths(
            resolved_profile.knowledge_mounts,
            extension_knowledge,
            _as_env_paths("AFS_KNOWLEDGE_MOUNTS"),
        ),
        skill_roots=_merge_unique_paths(
            resolved_profile.skill_roots,
            extension_skills,
            _as_env_paths("AFS_SKILL_ROOTS"),
        ),
        model_registries=_merge_unique_paths(
            resolved_profile.model_registries,
            extension_registries,
            _as_env_paths("AFS_MODEL_REGISTRIES"),
        ),
        enabled_extensions=sorted(extension_manifests.keys()),
        policies=_merge_unique_str(
            resolved_profile.policies,
            extension_policies,
            _as_env_list("AFS_POLICIES"),
        ),
        extension_hooks={
            event: _merge_unique_str(commands)
            for event, commands in extension_hooks.items()
        },
    )


def _slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-") or "item"


def list_profile_mount_specs(profile: ResolvedProfile) -> list[ProfileMountSpec]:
    """Return the mount aliases AFS expects for a resolved profile."""
    specs: list[ProfileMountSpec] = []
    spec_groups = (
        (MountType.KNOWLEDGE, KNOWLEDGE_ALIAS_PREFIX, profile.knowledge_mounts),
        (MountType.TOOLS, SKILL_ALIAS_PREFIX, profile.skill_roots),
        (MountType.GLOBAL, MODEL_ALIAS_PREFIX, profile.model_registries),
    )
    for mount_type, prefix, paths in spec_groups:
        for idx, source in enumerate(paths):
            source_path = source.expanduser().resolve()
            alias = f"{prefix}{profile.name}-{idx}-{_slug(source_path.name)}"
            specs.append(
                ProfileMountSpec(
                    profile_name=profile.name,
                    mount_type=mount_type,
                    alias=alias,
                    source=source_path,
                    source_exists=source_path.exists(),
                )
            )
    return specs


def _mount_profile_paths(
    manager: AFSManager,
    context_path: Path,
    mount_type: MountType,
    prefix: str,
    specs: list[ProfileMountSpec],
) -> tuple[int, list[str]]:
    mounted = 0
    skipped: list[str] = []

    # Remove previously managed mounts for this mount role.
    context = manager.list_context(context_path=context_path)
    for mount in context.get_mounts(mount_type):
        if mount.name.startswith(prefix):
            manager.unmount(mount.name, mount_type, context_path=context_path)

    for spec in specs:
        if not spec.source_exists:
            skipped.append(str(spec.source))
            continue

        try:
            manager.mount(
                spec.source,
                mount_type,
                alias=spec.alias,
                context_path=context_path,
                managed_by="profile",
                profile_name=spec.profile_name,
            )
            mounted += 1
        except FileExistsError:
            continue

    return mounted, skipped


def apply_profile_mounts(
    manager: AFSManager,
    context_path: Path,
    profile: ResolvedProfile,
) -> ProfileApplyResult:
    """Apply profile-managed mounts into an existing context."""
    context_path = context_path.expanduser().resolve()
    specs = list_profile_mount_specs(profile)

    knowledge_mounted, missing_knowledge = _mount_profile_paths(
        manager,
        context_path,
        MountType.KNOWLEDGE,
        KNOWLEDGE_ALIAS_PREFIX,
        [spec for spec in specs if spec.mount_type == MountType.KNOWLEDGE],
    )
    skills_mounted, missing_skills = _mount_profile_paths(
        manager,
        context_path,
        MountType.TOOLS,
        SKILL_ALIAS_PREFIX,
        [spec for spec in specs if spec.mount_type == MountType.TOOLS],
    )
    registries_mounted, missing_registries = _mount_profile_paths(
        manager,
        context_path,
        MountType.GLOBAL,
        MODEL_ALIAS_PREFIX,
        [spec for spec in specs if spec.mount_type == MountType.GLOBAL],
    )

    return ProfileApplyResult(
        profile_name=profile.name,
        mounted={
            "knowledge": knowledge_mounted,
            "skills": skills_mounted,
            "model_registries": registries_mounted,
        },
        skipped_missing=[
            *missing_knowledge,
            *missing_skills,
            *missing_registries,
        ],
    )


def merge_extension_hooks(
    config: AFSConfig,
    profile: ResolvedProfile,
    event: str,
) -> list[str]:
    configured = []
    hooks = config.hooks
    if event == "before_context_read":
        configured = hooks.before_context_read
    elif event == "after_context_write":
        configured = hooks.after_context_write
    elif event == "before_agent_dispatch":
        configured = hooks.before_agent_dispatch

    extension_commands = profile.extension_hooks.get(event, [])
    return _merge_unique_str(configured, extension_commands)
