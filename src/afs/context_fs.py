"""Context filesystem helpers for AFS."""

from __future__ import annotations

import fnmatch
import hashlib
import logging
import os
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .agent_scope import assert_mount_allowed
from .grounding_hooks import run_grounding_hooks
from .history import log_event
from .manager import AFSManager
from .models import ContextCategory, MountType
from .policy import PolicyEnforcer

logger = logging.getLogger(__name__)

FileMountType = MountType | ContextCategory


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
    def from_path(cls, path: Path, relative_path: str) -> ContextEntry:
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

    def __init__(
        self,
        manager: AFSManager,
        context_path: Path,
        *,
        scoped_mount_roots: Mapping[FileMountType, Path] | None = None,
    ) -> None:
        self._manager = manager
        self._context_path = context_path.resolve()
        if not self._context_path.exists():
            raise FileNotFoundError(f"No AFS context at {self._context_path}")
        self._scoped_mount_roots = {
            mount_type: root.expanduser().absolute()
            for mount_type, root in (scoped_mount_roots or {}).items()
        }
        for mount_type, root in self._scoped_mount_roots.items():
            canonical_root = self._manager.resolve_mount_root(
                self._context_path, mount_type
            ).resolve(strict=False)
            if root == canonical_root or not root.is_relative_to(canonical_root):
                raise ValueError(
                    f"Scoped {mount_type.value} root must be inside its context category"
                )
            cursor = canonical_root
            for part in root.relative_to(canonical_root).parts:
                cursor /= part
                if cursor.is_symlink():
                    raise ValueError(
                        f"Scoped {mount_type.value} root must not contain symlinks"
                    )
        self._policy = PolicyEnforcer(manager.config.directories)

    @property
    def context_path(self) -> Path:
        return self._context_path

    def resolve_mount_root(self, mount_type: FileMountType) -> Path:
        scoped_root = self._scoped_mount_roots.get(mount_type)
        if scoped_root is not None:
            return scoped_root
        return self._manager.resolve_mount_root(self._context_path, mount_type)

    def canonical_relative_path(
        self, mount_type: FileMountType, relative_path: str | None
    ) -> str:
        """Return the index path relative to the canonical category root."""

        target, _scope_root = self.resolve_path(mount_type, relative_path)
        canonical_root = self._manager.resolve_mount_root(
            self._context_path, mount_type
        ).resolve()
        try:
            return target.relative_to(canonical_root).as_posix()
        except ValueError as exc:
            raise ValueError("Path escapes canonical mount root") from exc

    def resolve_path(
        self, mount_type: FileMountType, relative_path: str | None
    ) -> tuple[Path, Path]:
        mount_root = self.resolve_mount_root(mount_type)
        is_scoped = mount_type in self._scoped_mount_roots
        if not mount_root.exists() and not is_scoped:
            raise FileNotFoundError(f"Mount root not found: {mount_root}")
        target = mount_root if not relative_path else mount_root / relative_path
        root_resolved = mount_root.resolve(strict=False)
        target_resolved = target.resolve(strict=False)
        if target_resolved == root_resolved or target_resolved.is_relative_to(
            root_resolved
        ):
            return target_resolved, root_resolved
        raise ValueError("Path escapes mount root")

    def _ensure_mount_access(
        self, mount_type: FileMountType, *, operation: str
    ) -> None:
        allowed, message = self._policy.validate_operation(mount_type, operation)
        if not allowed:
            raise PermissionError(message)
        assert_mount_allowed(mount_type, operation=operation)

    def _content_history_metadata(self, content: str) -> dict[str, object]:
        return {
            "content_chars": len(content),
            "content_lines": content.count("\n") + (1 if content else 0),
            "content_sha256": hashlib.sha256(content.encode("utf-8")).hexdigest(),
        }

    def read_text(
        self,
        mount_type: FileMountType,
        relative_path: str,
        *,
        encoding: str = "utf-8",
        errors: str = "replace",
    ) -> str:
        self._ensure_mount_access(mount_type, operation="read")
        run_grounding_hooks(
            event="before_context_read",
            payload={
                "mount_type": mount_type.value,
                "relative_path": relative_path,
                "context_path": str(self._context_path),
            },
            config=self._manager.config,
        )
        target, _root = self.resolve_path(mount_type, relative_path)
        if not target.exists():
            raise FileNotFoundError(f"Path not found: {target}")
        if target.is_dir():
            raise IsADirectoryError(f"Path is a directory: {target}")
        content = target.read_text(encoding=encoding, errors=errors)
        log_event(
            "fs",
            "afs.context_fs",
            op="read",
            context_root=self._context_path,
            metadata={
                "mount_type": mount_type.value,
                "relative_path": relative_path,
                "context_path": str(self._context_path),
                "encoding": encoding,
                **self._content_history_metadata(content),
            },
            include_payloads=False,
        )
        return content

    def write_text(
        self,
        mount_type: FileMountType,
        relative_path: str,
        content: str,
        *,
        encoding: str = "utf-8",
        append: bool = False,
        mkdirs: bool = False,
    ) -> Path:
        self._ensure_mount_access(mount_type, operation="write")
        target, root = self.resolve_path(mount_type, relative_path)
        if not root.exists():
            if mount_type not in self._scoped_mount_roots:
                raise FileNotFoundError(f"Mount root not found: {root}")
            root.mkdir(parents=True, exist_ok=True)
        if not target.parent.exists():
            if not mkdirs:
                raise FileNotFoundError(f"Parent directory missing: {target.parent}")
            target.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if append else "w"
        with target.open(mode, encoding=encoding) as handle:
            handle.write(content)
        run_grounding_hooks(
            event="after_context_write",
            payload={
                "mount_type": mount_type.value,
                "relative_path": relative_path,
                "context_path": str(self._context_path),
                "append": append,
                "mkdirs": mkdirs,
            },
            config=self._manager.config,
        )
        log_event(
            "fs",
            "afs.context_fs",
            op="write",
            context_root=self._context_path,
            metadata={
                "mount_type": mount_type.value,
                "relative_path": relative_path,
                "context_path": str(self._context_path),
                "encoding": encoding,
                "append": append,
                "mkdirs": mkdirs,
                **self._content_history_metadata(content),
            },
            include_payloads=False,
        )
        self._sync_index_for_write(
            mount_type,
            self.canonical_relative_path(mount_type, relative_path),
        )
        return target

    def _sync_index_for_write(
        self, mount_type: FileMountType, relative_path: str
    ) -> None:
        try:
            settings = self._manager.config.context_index
            if not settings.enabled:
                return

            from .context_index import ContextSQLiteIndex

            index = ContextSQLiteIndex(self._manager, self._context_path)
            index.upsert_relative_path(
                mount_type,
                relative_path,
                include_content=settings.include_content,
                max_file_size_bytes=settings.max_file_size_bytes,
                max_content_chars=settings.max_content_chars,
            )
        except Exception as exc:  # pragma: no cover - non-critical path
            logger.debug("Context index sync skipped for %s: %s", relative_path, exc)

    def stat_entry(
        self,
        mount_type: FileMountType,
        relative_path: str,
    ) -> ContextEntry:
        self._ensure_mount_access(mount_type, operation="read")
        target, root = self.resolve_path(mount_type, relative_path)
        if not target.exists():
            raise FileNotFoundError(f"Path not found: {target}")
        relative = target.relative_to(root).as_posix()
        return ContextEntry.from_path(target, relative)

    def list_entries(
        self,
        mount_type: FileMountType,
        *,
        relative_path: str | None = None,
        max_depth: int | None = 1,
        glob_patterns: list[str] | None = None,
        include_files: bool = True,
        include_dirs: bool = True,
    ) -> list[ContextEntry]:
        self._ensure_mount_access(mount_type, operation="read")
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
