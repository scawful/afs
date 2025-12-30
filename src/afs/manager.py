"""AFS manager for .context directories."""

from __future__ import annotations

import json
import shutil
from datetime import datetime
from pathlib import Path
from typing import Optional

from .config import load_config_model
from .mapping import resolve_directory_map, resolve_directory_name
from .models import ContextRoot, MountPoint, MountType, ProjectMetadata
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

    def ensure(
        self,
        path: Path = Path("."),
        *,
        context_root: Path | None = None,
        context_dir: str | None = None,
        link_context: bool = False,
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
            self._ensure_link(link_path, context_path, force=force)
            return self.list_context(context_path=context_path, metadata=metadata)

        if context_path.exists():
            if not force:
                raise FileExistsError(f"AFS already exists at {context_path}")
            self._remove_context_path(context_path)

        self._ensure_context_dirs(context_path)
        metadata = self._ensure_metadata(context_path, project_path)
        self._ensure_cognitive_scaffold(context_path)
        return self.list_context(context_path=context_path, metadata=metadata)

    def mount(
        self,
        source: Path,
        mount_type: MountType,
        alias: Optional[str] = None,
        context_path: Optional[Path] = None,
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
        alias = alias or source.name
        destination = context_path / directory_name / alias

        if destination.exists():
            raise FileExistsError(
                f"Mount point '{alias}' already exists in {mount_type.value}"
            )

        destination.symlink_to(source)
        return MountPoint(
            name=alias,
            source=source,
            mount_type=mount_type,
            is_symlink=True,
        )

    def unmount(
        self,
        alias: str,
        mount_type: MountType,
        context_path: Optional[Path] = None,
    ) -> bool:
        if context_path is None:
            context_path = Path(".") / self.CONTEXT_DIR_DEFAULT

        metadata = self._load_metadata(context_path)
        directory_name = resolve_directory_name(
            mount_type,
            afs_directories=self._directories,
            metadata=metadata,
        )
        mount_path = context_path / directory_name / alias
        if mount_path.exists() or mount_path.is_symlink():
            mount_path.unlink()
            return True
        return False

    def list_context(
        self,
        context_path: Optional[Path] = None,
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
                if item.name in {".keep", self.METADATA_FILE}:
                    continue
                source = item.resolve() if item.is_symlink() else item
                mount_list.append(
                    MountPoint(
                        name=item.name,
                        source=source,
                        mount_type=mount_type,
                        is_symlink=item.is_symlink(),
                    )
                )

            mounts[mount_type] = mount_list

        return ContextRoot(
            path=context_path,
            project_name=context_path.parent.name,
            metadata=metadata,
            mounts=mounts,
        )

    def clean(self, context_path: Optional[Path] = None) -> None:
        if context_path is None:
            context_path = Path(".") / self.CONTEXT_DIR_DEFAULT

        if context_path.exists():
            self._remove_context_path(context_path)

    def update_metadata(
        self,
        context_path: Optional[Path] = None,
        *,
        description: Optional[str] = None,
        agents: Optional[list[str]] = None,
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

    def _ensure_context_dirs(self, context_path: Path) -> None:
        context_path.mkdir(parents=True, exist_ok=True)
        for dir_config in self._directories:
            subdir = context_path / dir_config.name
            subdir.mkdir(parents=True, exist_ok=True)
            keep = subdir / ".keep"
            if not keep.exists():
                keep.touch()

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
