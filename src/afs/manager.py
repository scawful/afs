"""AFS manager for .context directories."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import load_config_model
from .history import log_event
from .mapping import resolve_directory_map, resolve_directory_name
from .models import (
    ContextRoot,
    MountPoint,
    MountProvenance,
    MountType,
    ProjectMetadata,
)
from .profiles import (
    apply_profile_mounts,
    list_profile_mount_specs,
    resolve_active_profile,
)
from .schema import AFSConfig, DirectoryConfig


class AFSManager:
    """Manage AFS context roots for projects."""

    CONTEXT_DIR_DEFAULT = ".context"
    METADATA_FILE = "metadata.json"
    STATE_FILE = "state.md"
    DEFERRED_FILE = "deferred.md"
    METACOGNITION_FILE = "metacognition.json"
    GOALS_FILE = "goals.json"
    EMOTIONS_FILE = "emotions.json"
    EPISTEMIC_FILE = "epistemic.json"

    DEFAULT_STATE_TEMPLATE = "# Agent State\n\n"
    DEFAULT_DEFERRED_TEMPLATE = "# Deferred\n\n"

    def __init__(
        self,
        config: AFSConfig | None = None,
        directories: list[DirectoryConfig] | None = None,
    ) -> None:
        self.config = config or load_config_model()
        self._directories = directories or list(self.config.directories)
        self._directory_map = resolve_directory_map(afs_directories=self._directories)

    def resolve_context_path(
        self,
        project_path: Path,
        context_root: Path | None = None,
        context_dir: str | None = None,
    ) -> Path:
        if context_root:
            return context_root.expanduser().resolve()
        context_dir = context_dir or self.CONTEXT_DIR_DEFAULT
        return project_path.resolve() / context_dir

    def resolve_mount_root(
        self,
        context_path: Path,
        mount_type: MountType,
    ) -> Path:
        metadata = self._load_metadata(context_path)
        directory_name = resolve_directory_name(
            mount_type,
            afs_directories=self._directories,
            metadata=metadata,
        )
        return context_path / directory_name

    def ensure(
        self,
        path: Path = Path("."),
        *,
        context_root: Path | None = None,
        context_dir: str | None = None,
        link_context: bool = False,
        profile: str | None = None,
    ) -> ContextRoot:
        project_path = path.resolve()
        context_path = self.resolve_context_path(
            project_path,
            context_root=context_root,
            context_dir=context_dir,
        )

        self._ensure_context_dirs(context_path)
        metadata = self._ensure_metadata(context_path, project_path)
        self._ensure_cognitive_scaffold(context_path)
        self._apply_profile_mounts(context_path, profile)
        if link_context and context_root:
            link_path = project_path / (context_dir or self.CONTEXT_DIR_DEFAULT)
            self._ensure_link(link_path, context_path, force=False)
        return self.list_context(context_path=context_path, metadata=metadata)

    def init(
        self,
        path: Path = Path("."),
        *,
        context_root: Path | None = None,
        context_dir: str | None = None,
        link_context: bool = False,
        force: bool = False,
        profile: str | None = None,
    ) -> ContextRoot:
        project_path = path.resolve()
        context_path = self.resolve_context_path(
            project_path,
            context_root=context_root,
            context_dir=context_dir,
        )

        if link_context and context_root:
            link_path = project_path / (context_dir or self.CONTEXT_DIR_DEFAULT)
            self._ensure_context_dirs(context_path)
            metadata = self._ensure_metadata(context_path, project_path)
            self._ensure_cognitive_scaffold(context_path)
            self._apply_profile_mounts(context_path, profile)
            self._ensure_link(link_path, context_path, force=force)
            return self.list_context(context_path=context_path, metadata=metadata)

        if context_path.exists():
            if not force:
                raise FileExistsError(f"AFS already exists at {context_path}")
            self._remove_context_path(context_path)

        self._ensure_context_dirs(context_path)
        metadata = self._ensure_metadata(context_path, project_path)
        self._ensure_cognitive_scaffold(context_path)
        self._apply_profile_mounts(context_path, profile)
        return self.list_context(context_path=context_path, metadata=metadata)

    def mount(
        self,
        source: Path,
        mount_type: MountType,
        alias: str | None = None,
        context_path: Path | None = None,
        *,
        managed_by: str = "manual",
        profile_name: str | None = None,
        remapped_from: Path | None = None,
    ) -> MountPoint:
        if context_path is None:
            context_path = Path(".") / self.CONTEXT_DIR_DEFAULT

        source = source.expanduser().resolve()
        if not source.exists():
            raise FileNotFoundError(f"Source {source} does not exist")

        context_path = context_path.resolve()
        if not context_path.exists():
            raise FileNotFoundError(f"No AFS context at {context_path}")

        metadata = self._load_metadata(context_path)
        directory_name = resolve_directory_name(
            mount_type,
            afs_directories=self._directories,
            metadata=metadata,
        )
        alias = self._normalize_mount_alias(alias or source.name)
        destination = context_path / directory_name / alias

        if destination.exists() or destination.is_symlink():
            raise FileExistsError(
                f"Mount point '{alias}' already exists in {mount_type.value}"
            )

        existing = self.list_context(context_path=context_path)
        for mount in existing.get_mounts(mount_type):
            if mount.is_symlink and mount.source == source:
                raise FileExistsError(
                    f"Source {source} is already mounted in {mount_type.value} as '{mount.name}'"
                )

        destination.symlink_to(source)
        provenance = MountProvenance(
            alias=alias,
            mount_type=mount_type,
            source=source,
            managed_by=managed_by.strip() or "manual",
            profile_name=profile_name.strip() if isinstance(profile_name, str) and profile_name.strip() else None,
            remapped_from=remapped_from.expanduser().resolve() if remapped_from else None,
            updated_at=datetime.now().isoformat(),
        )
        self._update_mount_provenance(context_path, provenance)
        log_event(
            "context",
            "afs.manager",
            op="mount",
            context_root=context_path,
            metadata={
                "mount_type": mount_type.value,
                "alias": alias,
                "context_path": str(context_path),
                "source": str(source),
                "destination": str(destination),
                "managed_by": provenance.managed_by,
                "profile_name": provenance.profile_name,
                "remapped_from": str(provenance.remapped_from) if provenance.remapped_from else None,
            },
        )
        self._sync_context_index_mount(context_path, mount_type)
        return MountPoint(
            name=alias,
            source=source,
            mount_type=mount_type,
            is_symlink=True,
            provenance=provenance,
        )

    def unmount(
        self,
        alias: str,
        mount_type: MountType,
        context_path: Path | None = None,
        *,
        keep_provenance: bool = False,
    ) -> bool:
        if context_path is None:
            context_path = Path(".") / self.CONTEXT_DIR_DEFAULT

        context_path = context_path.expanduser().resolve()
        metadata = self._load_metadata(context_path)
        directory_name = resolve_directory_name(
            mount_type,
            afs_directories=self._directories,
            metadata=metadata,
        )
        mount_path = context_path / directory_name / alias
        if mount_path.exists() or mount_path.is_symlink():
            mount_path.unlink()
            if not keep_provenance:
                self._remove_mount_provenance(context_path, mount_type, alias)
            log_event(
                "context",
                "afs.manager",
                op="unmount",
                context_root=context_path,
                metadata={
                    "mount_type": mount_type.value,
                    "alias": alias,
                    "context_path": str(context_path),
                    "mount_path": str(mount_path),
                    "kept_provenance": keep_provenance,
                },
            )
            self._sync_context_index_mount(context_path, mount_type)
            return True
        return False

    def list_context(
        self,
        context_path: Path | None = None,
        metadata: ProjectMetadata | None = None,
    ) -> ContextRoot:
        if context_path is None:
            context_path = Path(".") / self.CONTEXT_DIR_DEFAULT

        context_path = context_path.resolve()
        if not context_path.exists():
            raise FileNotFoundError("No AFS initialized")

        if metadata is None:
            metadata = self._load_metadata(context_path)

        if metadata is None:
            metadata = ProjectMetadata()

        mounts: dict[MountType, list[MountPoint]] = {}
        directory_map = resolve_directory_map(
            afs_directories=self._directories,
            metadata=metadata,
        )

        for mount_type in MountType:
            subdir = context_path / directory_map.get(mount_type, mount_type.value)
            if not subdir.exists():
                continue

            mount_list: list[MountPoint] = []
            for item in subdir.iterdir():
                if item.name in {self.METADATA_FILE}:
                    continue
                source = item.resolve(strict=False) if item.is_symlink() else item
                provenance = metadata.get_mount_provenance(mount_type, item.name)
                mount_list.append(
                    MountPoint(
                        name=item.name,
                        source=source,
                        mount_type=mount_type,
                        is_symlink=item.is_symlink(),
                        provenance=provenance,
                    )
                )

            mounts[mount_type] = mount_list

        return ContextRoot(
            path=context_path,
            project_name=context_path.parent.name,
            metadata=metadata,
            mounts=mounts,
        )

    def clean(self, context_path: Path | None = None) -> None:
        if context_path is None:
            context_path = Path(".") / self.CONTEXT_DIR_DEFAULT

        if context_path.exists():
            self._remove_context_path(context_path)

    def apply_profile(
        self,
        context_path: Path | None = None,
        *,
        profile_name: str | None = None,
    ):
        if context_path is None:
            context_path = Path(".") / self.CONTEXT_DIR_DEFAULT
        context_path = context_path.expanduser().resolve()

        resolved_profile = resolve_active_profile(self.config, profile_name=profile_name)
        result = apply_profile_mounts(self, context_path, resolved_profile)
        log_event(
            "context",
            "afs.manager",
            op="apply_profile",
            context_root=context_path,
            metadata={
                "profile": result.profile_name,
                "context_path": str(context_path),
                "mounted": result.mounted,
                "missing": result.skipped_missing,
                "policies": resolved_profile.policies,
                "extensions": resolved_profile.enabled_extensions,
            },
        )
        return result

    def context_health(
        self,
        context_path: Path | None = None,
        *,
        profile_name: str | None = None,
    ) -> dict[str, Any]:
        if context_path is None:
            context_path = Path(".") / self.CONTEXT_DIR_DEFAULT
        context_path = context_path.expanduser().resolve()

        context = self.list_context(context_path=context_path)
        directory_map = resolve_directory_map(
            afs_directories=self._directories,
            metadata=context.metadata,
        )
        missing_dirs = [
            mount_type.value
            for mount_type in MountType
            if not (context_path / directory_map.get(mount_type, mount_type.value)).exists()
        ]

        actual_symlink_mounts: dict[MountType, dict[str, MountPoint]] = {
            mount_type: {} for mount_type in MountType
        }
        broken_mounts: list[dict[str, str]] = []
        duplicate_mount_sources: list[dict[str, Any]] = []
        symlink_sources: dict[tuple[MountType, str], list[str]] = {}
        provenance_index = self._mount_provenance_index(context.metadata)
        actual_keys: set[tuple[MountType, str]] = set()
        untracked_mounts: list[dict[str, str]] = []

        for mount_type in MountType:
            for mount in context.get_mounts(mount_type):
                if not mount.is_symlink:
                    continue
                actual_symlink_mounts[mount_type][mount.name] = mount
                actual_keys.add((mount_type, mount.name))
                source_key = (mount_type, str(mount.source))
                symlink_sources.setdefault(source_key, []).append(mount.name)
                if not mount.source.exists():
                    entry = {
                        "name": mount.name,
                        "mount_type": mount_type.value,
                        "source": str(mount.source),
                    }
                    if mount.provenance is not None:
                        entry["managed_by"] = mount.provenance.managed_by
                        if mount.provenance.profile_name:
                            entry["profile_name"] = mount.provenance.profile_name
                    broken_mounts.append(entry)
                if (mount_type, mount.name) not in provenance_index:
                    untracked_mounts.append(
                        {
                            "name": mount.name,
                            "mount_type": mount_type.value,
                            "source": str(mount.source),
                        }
                    )

        for (mount_type, source), aliases in sorted(symlink_sources.items(), key=lambda item: item[0][1]):
            unique_aliases = sorted(set(aliases))
            if len(unique_aliases) < 2:
                continue
            duplicate_mount_sources.append(
                {
                    "mount_type": mount_type.value,
                    "source": source,
                    "aliases": unique_aliases,
                }
            )

        resolved_profile = resolve_active_profile(self.config, profile_name=profile_name)
        profile_missing_mounts: list[dict[str, str]] = []
        profile_missing_sources: list[dict[str, str]] = []
        profile_mismatched_mounts: list[dict[str, str]] = []
        profile_remapped_mounts: list[dict[str, str]] = []
        for spec in list_profile_mount_specs(resolved_profile):
            provenance = provenance_index.get((spec.mount_type, spec.alias))
            if not spec.source_exists:
                actual = actual_symlink_mounts.get(spec.mount_type, {}).get(spec.alias)
                if actual is not None and actual.source.exists():
                    profile_remapped_mounts.append(
                        {
                            "mount_type": spec.mount_type.value,
                            "alias": spec.alias,
                            "expected_source": str(spec.source),
                            "actual_source": str(actual.source),
                            "profile_name": resolved_profile.name,
                            "remapped_from": str(provenance.remapped_from)
                            if provenance is not None and provenance.remapped_from is not None
                            else str(spec.source),
                        }
                    )
                else:
                    profile_missing_sources.append(
                        {
                            "mount_type": spec.mount_type.value,
                            "alias": spec.alias,
                            "source": str(spec.source),
                        }
                    )
                continue
            actual = actual_symlink_mounts.get(spec.mount_type, {}).get(spec.alias)
            if actual is None:
                profile_missing_mounts.append(
                    {
                        "mount_type": spec.mount_type.value,
                        "alias": spec.alias,
                        "source": str(spec.source),
                    }
                )
                continue
            if actual.source != spec.source:
                if provenance is not None and provenance.remapped_from is not None:
                    profile_remapped_mounts.append(
                        {
                            "mount_type": spec.mount_type.value,
                            "alias": spec.alias,
                            "expected_source": str(spec.source),
                            "actual_source": str(actual.source),
                            "profile_name": resolved_profile.name,
                            "remapped_from": str(provenance.remapped_from),
                        }
                    )
                else:
                    profile_mismatched_mounts.append(
                        {
                            "mount_type": spec.mount_type.value,
                            "alias": spec.alias,
                            "expected_source": str(spec.source),
                            "actual_source": str(actual.source),
                        }
                    )

        stale_provenance: list[dict[str, str]] = []
        for key, provenance in provenance_index.items():
            if key in actual_keys:
                continue
            stale_provenance.append(
                {
                    "name": provenance.alias,
                    "mount_type": provenance.mount_type.value,
                    "source": str(provenance.source),
                    "managed_by": provenance.managed_by,
                    "profile_name": provenance.profile_name or "",
                }
            )

        actions: list[str] = []
        if broken_mounts:
            actions.append("remove or repair broken symlink mounts")
        if duplicate_mount_sources:
            actions.append("deduplicate mounts that point to the same source")
        if profile_missing_mounts or profile_mismatched_mounts:
            actions.append("reapply profile-managed mounts")
        if profile_missing_sources:
            actions.append("restore or update missing profile source paths")
        if untracked_mounts:
            actions.append("persist provenance for untracked mounts")
        if stale_provenance:
            actions.append("prune stale mount provenance records")
        if profile_remapped_mounts:
            actions.append("update profile source paths that were remapped at repair time")

        return {
            "healthy": not (
                missing_dirs
                or broken_mounts
                or duplicate_mount_sources
                or profile_missing_mounts
                or profile_missing_sources
                or profile_mismatched_mounts
            ),
            "missing_dirs": missing_dirs,
            "symlink_mounts": sum(len(mounts) for mounts in actual_symlink_mounts.values()),
            "broken_mounts": broken_mounts,
            "duplicate_mount_sources": duplicate_mount_sources,
            "provenance": {
                "tracked_mounts": len(provenance_index),
                "untracked_mounts": untracked_mounts,
                "stale_records": stale_provenance,
            },
            "profile": {
                "name": resolved_profile.name,
                "managed_mounts": len(list_profile_mount_specs(resolved_profile)),
                "missing_mounts": profile_missing_mounts,
                "missing_sources": profile_missing_sources,
                "mismatched_mounts": profile_mismatched_mounts,
                "remapped_mounts": profile_remapped_mounts,
            },
            "suggested_actions": actions,
        }

    def repair_context(
        self,
        context_path: Path | None = None,
        *,
        profile_name: str | None = None,
        dry_run: bool = False,
        reapply_profile: bool = True,
        remap_missing_sources: bool = True,
        rebuild_index: bool = False,
    ) -> dict[str, Any]:
        if context_path is None:
            context_path = Path(".") / self.CONTEXT_DIR_DEFAULT
        context_path = context_path.expanduser().resolve()

        health_before = self.context_health(context_path, profile_name=profile_name)
        applied_actions: list[str] = []
        planned_actions: list[str] = []
        remapped_mounts: list[dict[str, str]] = []
        profile_reapply: dict[str, Any] | None = None
        index_rebuild: dict[str, Any] | None = None

        seeded = self._seed_mount_provenance(
            context_path,
            profile_name=profile_name,
            dry_run=dry_run,
        )
        if seeded:
            planned_actions.append(f"seed provenance for {seeded} mount(s)")
            if not dry_run:
                applied_actions.append(f"seeded provenance for {seeded} mount(s)")

        current_health = (
            self.context_health(context_path, profile_name=profile_name)
            if not dry_run and seeded
            else health_before
        )

        if remap_missing_sources:
            remapped_mounts = self._repair_mount_sources(
                context_path,
                current_health,
                profile_name=profile_name,
                dry_run=dry_run,
            )
            if remapped_mounts:
                planned_actions.append(f"remap {len(remapped_mounts)} broken or missing mount(s)")
                if not dry_run:
                    applied_actions.append(f"remapped {len(remapped_mounts)} broken or missing mount(s)")
                    current_health = self.context_health(context_path, profile_name=profile_name)

        if reapply_profile and (
            current_health["profile"]["missing_mounts"]
            or current_health["profile"]["mismatched_mounts"]
        ):
            if current_health["profile"]["missing_sources"]:
                planned_actions.append("skip profile reapply until missing profile sources are restored or remapped")
            else:
                planned_actions.append("reapply profile-managed mounts")
                if not dry_run:
                    result = self.apply_profile(context_path, profile_name=profile_name)
                    profile_reapply = {
                        "profile": result.profile_name,
                        "mounted": result.mounted,
                        "missing": result.skipped_missing,
                    }
                    applied_actions.append("reapplied profile-managed mounts")
                    current_health = self.context_health(context_path, profile_name=profile_name)

        pruned = self._prune_stale_mount_provenance(context_path, dry_run=dry_run)
        if pruned:
            planned_actions.append(f"prune {pruned} stale provenance record(s)")
            if not dry_run:
                applied_actions.append(f"pruned {pruned} stale provenance record(s)")

        if rebuild_index and self.config.context_index.enabled:
            from .context_index import ContextSQLiteIndex

            index = ContextSQLiteIndex(self, context_path)
            if not index.has_entries() or index.needs_refresh():
                planned_actions.append("rebuild stale or empty context index")
                if not dry_run:
                    index_rebuild = self.rebuild_context_index(context_path)
                    applied_actions.append("rebuilt context index")

        health_after = (
            current_health
            if dry_run
            else self.context_health(context_path, profile_name=profile_name)
        )

        return {
            "context_path": str(context_path),
            "dry_run": dry_run,
            "changed": bool(seeded or remapped_mounts or profile_reapply or pruned or index_rebuild),
            "actions": planned_actions,
            "applied_actions": applied_actions,
            "provenance_seeded": seeded,
            "provenance_pruned": pruned,
            "remapped_mounts": remapped_mounts,
            "profile_reapply": profile_reapply,
            "index_rebuild": index_rebuild,
            "health_before": health_before,
            "health_after": health_after,
        }

    def rebuild_context_index(
        self,
        context_path: Path | None = None,
        *,
        mount_types: list[MountType] | None = None,
    ) -> dict[str, Any]:
        if context_path is None:
            context_path = Path(".") / self.CONTEXT_DIR_DEFAULT
        context_path = context_path.expanduser().resolve()
        settings = self.config.context_index
        if not settings.enabled:
            return {"enabled": False, "rebuilt": False}

        from .context_index import ContextSQLiteIndex

        index = ContextSQLiteIndex(self, context_path)
        summary = index.rebuild(
            mount_types=mount_types,
            include_content=settings.include_content,
            max_file_size_bytes=settings.max_file_size_bytes,
            max_content_chars=settings.max_content_chars,
        )
        payload = summary.to_dict()
        payload["enabled"] = True
        payload["rebuilt"] = True
        return payload

    def update_metadata(
        self,
        context_path: Path | None = None,
        *,
        description: str | None = None,
        agents: list[str] | None = None,
    ) -> ProjectMetadata:
        if context_path is None:
            context_path = Path(".") / self.CONTEXT_DIR_DEFAULT

        metadata_path = context_path / self.METADATA_FILE
        if not metadata_path.exists():
            raise FileNotFoundError("No AFS initialized")

        metadata = self._load_metadata(context_path) or ProjectMetadata()

        if description is not None:
            metadata.description = description
        if agents is not None:
            metadata.agents = agents

        self._write_metadata(metadata_path, metadata)
        return metadata

    def register_agent(
        self,
        agent_name: str,
        context_path: Path | None = None,
    ) -> ProjectMetadata:
        """Register an agent in the project metadata."""
        if context_path is None:
            context_path = Path(".") / self.CONTEXT_DIR_DEFAULT

        metadata_path = context_path / self.METADATA_FILE
        if not metadata_path.exists():
            return ProjectMetadata()

        metadata = self._load_metadata(context_path) or ProjectMetadata()
        if agent_name not in metadata.agents:
            metadata.agents.append(agent_name)
            self._write_metadata(metadata_path, metadata)
        return metadata

    def protect(
        self,
        path_str: str,
        context_path: Path | None = None,
    ) -> ProjectMetadata:
        if context_path is None:
            context_path = Path(".") / self.CONTEXT_DIR_DEFAULT

        metadata_path = context_path / self.METADATA_FILE
        if not metadata_path.exists():
            raise FileNotFoundError("No AFS initialized")

        metadata = self._load_metadata(context_path) or ProjectMetadata()
        if path_str not in metadata.manual_only:
            metadata.manual_only.append(path_str)
            self._write_metadata(metadata_path, metadata)
        return metadata

    def unprotect(
        self,
        path_str: str,
        context_path: Path | None = None,
    ) -> ProjectMetadata:
        if context_path is None:
            context_path = Path(".") / self.CONTEXT_DIR_DEFAULT

        metadata_path = context_path / self.METADATA_FILE
        if not metadata_path.exists():
            raise FileNotFoundError("No AFS initialized")

        metadata = self._load_metadata(context_path) or ProjectMetadata()
        if path_str in metadata.manual_only:
            metadata.manual_only.remove(path_str)
            self._write_metadata(metadata_path, metadata)
        return metadata

    def _ensure_context_dirs(self, context_path: Path) -> None:
        context_path.mkdir(parents=True, exist_ok=True)
        for dir_config in self._directories:
            subdir = context_path / dir_config.name
            subdir.mkdir(parents=True, exist_ok=True)

    def _sync_context_index_mount(self, context_path: Path, mount_type: MountType) -> None:
        try:
            settings = self.config.context_index
            if not settings.enabled:
                return

            from .context_index import ContextSQLiteIndex

            index = ContextSQLiteIndex(self, context_path)
            index.rebuild(
                mount_types=[mount_type],
                include_content=settings.include_content,
                max_file_size_bytes=settings.max_file_size_bytes,
                max_content_chars=settings.max_content_chars,
            )
        except Exception:
            # Mount operations should still succeed even if indexing fails.
            return

    def _ensure_metadata(self, context_path: Path, project_path: Path) -> ProjectMetadata:
        metadata_path = context_path / self.METADATA_FILE
        directory_map = {
            mount_type.value: name for mount_type, name in self._directory_map.items()
        }
        if not metadata_path.exists():
            metadata = ProjectMetadata(
                created_at=datetime.now().isoformat(),
                description=f"AFS for {project_path.name}",
                directories=directory_map,
            )
            self._write_metadata(metadata_path, metadata)
            return metadata

        metadata = self._load_metadata(context_path)
        if metadata is None:
            metadata = ProjectMetadata(
                created_at=datetime.now().isoformat(),
                description=f"AFS for {project_path.name}",
                directories=directory_map,
            )
            self._write_metadata(metadata_path, metadata)
            return metadata

        if not metadata.directories and directory_map:
            metadata.directories = directory_map
            self._write_metadata(metadata_path, metadata)
        return metadata

    def _ensure_cognitive_scaffold(self, context_path: Path) -> None:
        if not self.config.cognitive.enabled:
            return

        metadata = self._load_metadata(context_path)
        scratchpad_dir = context_path / resolve_directory_name(
            MountType.SCRATCHPAD,
            afs_directories=self._directories,
            metadata=metadata,
        )
        memory_dir = context_path / resolve_directory_name(
            MountType.MEMORY,
            afs_directories=self._directories,
            metadata=metadata,
        )
        scratchpad_dir.mkdir(parents=True, exist_ok=True)
        memory_dir.mkdir(parents=True, exist_ok=True)

        state_file = scratchpad_dir / self.STATE_FILE
        if not state_file.exists():
            state_file.write_text(self.DEFAULT_STATE_TEMPLATE, encoding="utf-8")

        deferred_file = scratchpad_dir / self.DEFERRED_FILE
        if not deferred_file.exists():
            deferred_file.write_text(self.DEFAULT_DEFERRED_TEMPLATE, encoding="utf-8")

        if self.config.cognitive.record_metacognition:
            meta_file = scratchpad_dir / self.METACOGNITION_FILE
            if not meta_file.exists():
                meta_file.write_text("{}\n", encoding="utf-8")

        if self.config.cognitive.record_goals:
            goals_file = scratchpad_dir / self.GOALS_FILE
            if not goals_file.exists():
                goals_file.write_text("[]\n", encoding="utf-8")

        if self.config.cognitive.record_emotions:
            emotions_file = scratchpad_dir / self.EMOTIONS_FILE
            if not emotions_file.exists():
                emotions_file.write_text("[]\n", encoding="utf-8")

        if self.config.cognitive.record_epistemic:
            epistemic_file = scratchpad_dir / self.EPISTEMIC_FILE
            if not epistemic_file.exists():
                epistemic_file.write_text("{}\n", encoding="utf-8")

    def _mount_provenance_index(
        self,
        metadata: ProjectMetadata | None,
    ) -> dict[tuple[MountType, str], MountProvenance]:
        index: dict[tuple[MountType, str], MountProvenance] = {}
        if metadata is None:
            return index
        for provenance in metadata.iter_mount_provenance():
            index[(provenance.mount_type, provenance.alias)] = provenance
        return index

    def _update_mount_provenance(
        self,
        context_path: Path,
        provenance: MountProvenance,
    ) -> None:
        metadata_path = context_path / self.METADATA_FILE
        metadata = self._load_metadata(context_path) or ProjectMetadata()
        metadata.set_mount_provenance(provenance)
        self._write_metadata(metadata_path, metadata)

    def _remove_mount_provenance(
        self,
        context_path: Path,
        mount_type: MountType,
        alias: str,
    ) -> None:
        metadata_path = context_path / self.METADATA_FILE
        metadata = self._load_metadata(context_path)
        if metadata is None:
            return
        metadata.remove_mount_provenance(mount_type, alias)
        self._write_metadata(metadata_path, metadata)

    def _seed_mount_provenance(
        self,
        context_path: Path,
        *,
        profile_name: str | None = None,
        dry_run: bool = False,
    ) -> int:
        context = self.list_context(context_path=context_path)
        metadata = context.metadata or ProjectMetadata()
        provenance_index = self._mount_provenance_index(metadata)
        resolved_profile = resolve_active_profile(self.config, profile_name=profile_name)
        profile_specs = {
            (spec.mount_type, spec.alias): spec
            for spec in list_profile_mount_specs(resolved_profile)
        }
        seeded = 0
        now = datetime.now().isoformat()

        for mount_type in MountType:
            for mount in context.get_mounts(mount_type):
                if not mount.is_symlink:
                    continue
                key = (mount_type, mount.name)
                if key in provenance_index:
                    continue
                spec = profile_specs.get(key)
                metadata.set_mount_provenance(
                    MountProvenance(
                        alias=mount.name,
                        mount_type=mount_type,
                        source=mount.source,
                        managed_by="profile" if spec is not None else "manual",
                        profile_name=resolved_profile.name if spec is not None else None,
                        remapped_from=spec.source
                        if spec is not None and spec.source != mount.source
                        else None,
                        updated_at=now,
                    )
                )
                seeded += 1

        if seeded and not dry_run:
            self._write_metadata(context_path / self.METADATA_FILE, metadata)
        return seeded

    def _prune_stale_mount_provenance(
        self,
        context_path: Path,
        *,
        dry_run: bool = False,
    ) -> int:
        context = self.list_context(context_path=context_path)
        metadata = context.metadata or ProjectMetadata()
        active_keys = {
            (mount.mount_type, mount.name)
            for mount_type in MountType
            for mount in context.get_mounts(mount_type)
            if mount.is_symlink
        }
        removed = 0
        for provenance in list(metadata.iter_mount_provenance()):
            if (provenance.mount_type, provenance.alias) in active_keys:
                continue
            metadata.remove_mount_provenance(provenance.mount_type, provenance.alias)
            removed += 1

        if removed and not dry_run:
            self._write_metadata(context_path / self.METADATA_FILE, metadata)
        return removed

    def _repair_mount_sources(
        self,
        context_path: Path,
        health: dict[str, Any],
        *,
        profile_name: str | None = None,
        dry_run: bool = False,
    ) -> list[dict[str, str]]:
        context = self.list_context(context_path=context_path)
        provenance_index = self._mount_provenance_index(context.metadata)
        actual_mounts = {
            (mount.mount_type, mount.name): mount
            for mount_type in MountType
            for mount in context.get_mounts(mount_type)
            if mount.is_symlink
        }
        resolved_profile = resolve_active_profile(self.config, profile_name=profile_name)
        remapped: list[dict[str, str]] = []
        processed: set[tuple[MountType, str]] = set()

        for broken in health.get("broken_mounts", []):
            try:
                mount_type = MountType(str(broken.get("mount_type", "")))
            except ValueError:
                continue
            alias = str(broken.get("name", "")).strip()
            if not alias:
                continue
            key = (mount_type, alias)
            provenance = provenance_index.get(key)
            source_value = provenance.source if provenance is not None else Path(
                str(broken.get("source", ""))
            ).expanduser().resolve(strict=False)
            candidate = self._find_mount_remap_candidate(source_value, context_path)
            if candidate is None:
                continue
            remapped.append(
                {
                    "mount_type": mount_type.value,
                    "alias": alias,
                    "previous_source": str(source_value),
                    "new_source": str(candidate),
                }
            )
            processed.add(key)
            if dry_run:
                continue
            if key in actual_mounts:
                self.unmount(alias, mount_type, context_path=context_path, keep_provenance=True)
            self.mount(
                candidate,
                mount_type,
                alias=alias,
                context_path=context_path,
                managed_by=provenance.managed_by if provenance is not None else "repair",
                profile_name=provenance.profile_name if provenance is not None else None,
                remapped_from=source_value,
            )

        for missing in health.get("profile", {}).get("missing_sources", []):
            try:
                mount_type = MountType(str(missing.get("mount_type", "")))
            except ValueError:
                continue
            alias = str(missing.get("alias", "")).strip()
            if not alias:
                continue
            key = (mount_type, alias)
            if key in processed:
                continue
            source_value = Path(str(missing.get("source", ""))).expanduser().resolve(strict=False)
            candidate = self._find_mount_remap_candidate(source_value, context_path)
            if candidate is None:
                continue
            remapped.append(
                {
                    "mount_type": mount_type.value,
                    "alias": alias,
                    "previous_source": str(source_value),
                    "new_source": str(candidate),
                }
            )
            if dry_run:
                continue
            if key in actual_mounts:
                self.unmount(alias, mount_type, context_path=context_path, keep_provenance=True)
            self.mount(
                candidate,
                mount_type,
                alias=alias,
                context_path=context_path,
                managed_by="profile",
                profile_name=resolved_profile.name,
                remapped_from=source_value,
            )
        return remapped

    def _workspace_search_roots(self, context_path: Path) -> list[Path]:
        roots: list[Path] = []
        seen: set[Path] = set()

        def _add(root: Path | None) -> None:
            if root is None:
                return
            candidate = root.expanduser().resolve()
            if candidate in seen or not candidate.exists() or not candidate.is_dir():
                return
            if candidate == context_path or candidate.is_relative_to(context_path):
                return
            seen.add(candidate)
            roots.append(candidate)

        for workspace in self.config.general.workspace_directories:
            _add(workspace.path)
        for root in self.config.general.mcp_allowed_roots:
            _add(root)
        _add(context_path.parent)
        return roots

    def _find_mount_remap_candidate(
        self,
        source: Path,
        context_path: Path,
    ) -> Path | None:
        candidate_source = source.expanduser().resolve(strict=False)
        if candidate_source.exists():
            return candidate_source

        search_roots = self._workspace_search_roots(context_path)
        seen: set[Path] = set()

        def _add(path: Path) -> Path | None:
            resolved = path.expanduser().resolve()
            if resolved in seen:
                return None
            seen.add(resolved)
            if resolved == context_path or resolved.is_relative_to(context_path):
                return None
            return resolved if resolved.exists() else None

        for anchor in search_roots:
            if candidate_source == anchor or candidate_source.is_relative_to(anchor):
                relative = candidate_source.relative_to(anchor)
                for root in search_roots:
                    found = _add(root / relative)
                    if found is not None:
                        return found

        suffixes: list[Path] = [Path(candidate_source.name)]
        if len(candidate_source.parts) >= 2:
            suffixes.append(Path(*candidate_source.parts[-2:]))
        if len(candidate_source.parts) >= 3:
            suffixes.append(Path(*candidate_source.parts[-3:]))

        for root in search_roots:
            for suffix in suffixes:
                found = _add(root / suffix)
                if found is not None:
                    return found
        return None

    def _apply_profile_mounts(self, context_path: Path, profile_name: str | None) -> None:
        if not self.config.profiles.auto_apply and not profile_name:
            return
        self.apply_profile(context_path, profile_name=profile_name)

    def _normalize_mount_alias(self, alias: str) -> str:
        value = alias.strip()
        if not value:
            raise ValueError("mount alias must be a non-empty simple name")
        candidate = Path(value)
        if candidate.name != value or value in {".", ".."}:
            raise ValueError("mount alias must be a simple name without path separators")
        if "/" in value or "\\" in value:
            raise ValueError("mount alias must be a simple name without path separators")
        return value

    def _ensure_link(self, link_path: Path, target: Path, force: bool) -> None:
        if link_path.is_symlink():
            if link_path.resolve() == target.resolve():
                return
            if not force:
                raise FileExistsError(f"Context link already exists at {link_path}")
            link_path.unlink()
        elif link_path.exists():
            if not force:
                raise FileExistsError(f"Context path already exists at {link_path}")
            self._remove_context_path(link_path)

        link_path.symlink_to(target)

    def _remove_context_path(self, context_path: Path) -> None:
        if context_path.is_symlink():
            context_path.unlink()
        elif context_path.exists():
            shutil.rmtree(context_path)

    def _load_metadata(self, context_path: Path) -> ProjectMetadata | None:
        metadata_path = context_path / self.METADATA_FILE
        if not metadata_path.exists():
            return None
        try:
            payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            return None
        return ProjectMetadata.from_dict(payload)

    def _write_metadata(self, path: Path, metadata: ProjectMetadata) -> None:
        path.write_text(
            json.dumps(metadata.to_dict(), indent=2, default=str) + "\n",
            encoding="utf-8",
        )
