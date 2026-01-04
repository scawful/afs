"""Context filesystem helpers for AFS."""

from __future__ import annotations

import fnmatch
import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .manager import AFSManager
from .models import MountType
from .policy import PolicyEnforcer


@dataclass
class ContextEntry:
    path: Path
    relative_path: str
    is_dir: bool
    is_file: bool
    is_symlink: bool
    size_bytes: int
    modified_at: str | None

    @classmethod
    def from_path(cls, path: Path, relative_path: str) -> "ContextEntry":
        try:
            stat = path.stat()
            size = stat.st_size
            modified_at = datetime.fromtimestamp(stat.st_mtime).isoformat()
        except OSError:
            size = 0
            modified_at = None
        return cls(
            path=path,
            relative_path=relative_path,
            is_dir=path.is_dir(),
            is_file=path.is_file(),
            is_symlink=path.is_symlink(),
            size_bytes=size,
            modified_at=modified_at,
        )

    def to_dict(self) -> dict:
        return {
            "path": str(self.path),
            "relative_path": self.relative_path,
            "is_dir": self.is_dir,
            "is_file": self.is_file,
            "is_symlink": self.is_symlink,
            "size_bytes": self.size_bytes,
            "modified_at": self.modified_at,
        }


class ContextFileSystem:
    """Read/write and listing operations scoped to a context root."""

    def __init__(self, manager: AFSManager, context_path: Path) -> None:
        self._manager = manager
        self._context_path = context_path.resolve()
        if not self._context_path.exists():
            raise FileNotFoundError(f"No AFS context at {self._context_path}")
        self._policy = PolicyEnforcer(manager.config.directories)

    @property
    def context_path(self) -> Path:
        return self._context_path

    def resolve_mount_root(self, mount_type: MountType) -> Path:
        return self._manager.resolve_mount_root(self._context_path, mount_type)

    def resolve_path(
        self, mount_type: MountType, relative_path: str | None
    ) -> tuple[Path, Path]:
        mount_root = self.resolve_mount_root(mount_type)
        if not mount_root.exists():
            raise FileNotFoundError(f"Mount root not found: {mount_root}")
        target = mount_root if not relative_path else mount_root / relative_path
        root_resolved = mount_root.resolve()
        target_resolved = target.resolve()
        if target_resolved == root_resolved or target_resolved.is_relative_to(
            root_resolved
        ):
            return target_resolved, root_resolved
        raise ValueError("Path escapes mount root")

    def read_text(
        self,
        mount_type: MountType,
        relative_path: str,
        *,
        encoding: str = "utf-8",
        errors: str = "replace",
    ) -> str:
        allowed, message = self._policy.validate_operation(mount_type, "read")
        if not allowed:
            raise PermissionError(message)
        target, _root = self.resolve_path(mount_type, relative_path)
        if not target.exists():
            raise FileNotFoundError(f"Path not found: {target}")
        if target.is_dir():
            raise IsADirectoryError(f"Path is a directory: {target}")
        return target.read_text(encoding=encoding, errors=errors)

    def write_text(
        self,
        mount_type: MountType,
        relative_path: str,
        content: str,
        *,
        encoding: str = "utf-8",
        append: bool = False,
        mkdirs: bool = False,
    ) -> Path:
        allowed, message = self._policy.validate_operation(mount_type, "write")
        if not allowed:
            raise PermissionError(message)
        target, root = self.resolve_path(mount_type, relative_path)
        if not root.exists():
            raise FileNotFoundError(f"Mount root not found: {root}")
        if not target.parent.exists():
            if not mkdirs:
                raise FileNotFoundError(f"Parent directory missing: {target.parent}")
            target.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with target.open(mode, encoding=encoding) as handle:
            handle.write(content)
        return target

    def stat_entry(
        self,
        mount_type: MountType,
        relative_path: str,
    ) -> ContextEntry:
        target, root = self.resolve_path(mount_type, relative_path)
        if not target.exists():
            raise FileNotFoundError(f"Path not found: {target}")
        relative = target.relative_to(root).as_posix()
        return ContextEntry.from_path(target, relative)

    def list_entries(
        self,
        mount_type: MountType,
        *,
        relative_path: str | None = None,
        max_depth: int | None = 1,
        glob_patterns: list[str] | None = None,
        include_files: bool = True,
        include_dirs: bool = True,
    ) -> list[ContextEntry]:
        target, root = self.resolve_path(mount_type, relative_path)
        if not target.exists():
            raise FileNotFoundError(f"Path not found: {target}")
        if target.is_file():
            relative = target.relative_to(root).as_posix()
            return [ContextEntry.from_path(target, relative)]

        entries: list[ContextEntry] = []

        for base, dirs, files in os.walk(target):
            rel_base = Path(base).relative_to(target)
            depth = len(rel_base.parts)
            if max_depth is not None and depth >= max_depth:
                dirs[:] = []
            if include_dirs:
                for name in dirs:
                    entry_path = Path(base) / name
                    relative = entry_path.relative_to(root).as_posix()
                    if _matches_patterns(relative, glob_patterns):
                        entries.append(ContextEntry.from_path(entry_path, relative))
            if include_files:
                for name in files:
                    entry_path = Path(base) / name
                    relative = entry_path.relative_to(root).as_posix()
                    if _matches_patterns(relative, glob_patterns):
                        entries.append(ContextEntry.from_path(entry_path, relative))
        return entries


def _matches_patterns(path: str, patterns: list[str] | None) -> bool:
    if not patterns:
        return True
    return any(fnmatch.fnmatch(path, pattern) for pattern in patterns)
