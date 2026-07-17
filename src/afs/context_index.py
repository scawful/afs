"""SQLite-backed context indexing and query helpers."""

from __future__ import annotations

import hashlib
import os
import re
import sqlite3
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .context_layout import LAYOUT_VERSION, detect_layout_version
from .manager import AFSManager
from .models import ContextCategory, MountType
from .project_registry import assert_no_linklike_components
from .scopes import ResolvedScope, visible_mount_roots, visible_scope_prefixes
from .sensitivity import matches_path_rules

DEFAULT_DB_FILENAME = "context_index.sqlite3"
DEFAULT_MAX_FILE_SIZE_BYTES = 256 * 1024
DEFAULT_MAX_CONTENT_CHARS = 12000
INDEX_HEALTH_MOUNT_TYPES: tuple[MountType, ...] = (
    MountType.MEMORY,
    MountType.KNOWLEDGE,
    MountType.TOOLS,
    MountType.GLOBAL,
)

_TEXT_SUFFIXES = {
    ".asm",
    ".bash",
    ".c",
    ".cc",
    ".cfg",
    ".conf",
    ".cpp",
    ".css",
    ".csv",
    ".h",
    ".html",
    ".inc",
    ".ini",
    ".java",
    ".js",
    ".json",
    ".lua",
    ".md",
    ".ps1",
    ".py",
    ".rs",
    ".s",
    ".sh",
    ".sql",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
    ".xml",
    ".yaml",
    ".yml",
    ".zsh",
    ".65c",
    ".65s",
}


@dataclass
class IndexSummary:
    context_path: str
    db_path: str
    indexed_at: str
    rows_written: int
    rows_deleted: int
    by_mount_type: dict[str, int]
    skipped_large_files: int
    skipped_binary_files: int
    errors: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "context_path": self.context_path,
            "db_path": self.db_path,
            "indexed_at": self.indexed_at,
            "rows_written": self.rows_written,
            "rows_deleted": self.rows_deleted,
            "by_mount_type": dict(self.by_mount_type),
            "skipped_large_files": self.skipped_large_files,
            "skipped_binary_files": self.skipped_binary_files,
            "errors": list(self.errors),
        }


@dataclass(frozen=True)
class MountSnapshot:
    count: int
    max_modified: str | None
    fingerprint: str


class ContextSQLiteIndex:
    """Persistent SQLite index of context mount entries."""

    def __init__(
        self,
        manager: AFSManager,
        context_path: Path,
        *,
        db_path: Path | None = None,
    ) -> None:
        self._manager = manager
        self._context_path = context_path.expanduser().resolve()
        self._managed_v2_db_root: Path | None = None
        self._db_path = db_path.expanduser().resolve() if db_path else self._default_db_path()
        self._assert_managed_db_safe()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._assert_managed_db_safe(require_parent=True)
        self._fts_enabled = False
        self._initialize()

    @property
    def context_path(self) -> Path:
        return self._context_path

    @property
    def db_path(self) -> Path:
        return self._db_path

    def rebuild(
        self,
        *,
        mount_types: list[MountType] | None = None,
        include_content: bool = True,
        max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES,
        max_content_chars: int = DEFAULT_MAX_CONTENT_CHARS,
    ) -> IndexSummary:
        selected_mounts = self._normalize_mount_types(mount_types)
        indexed_at = datetime.now(timezone.utc).isoformat()
        rows: list[tuple[Any, ...]] = []
        by_mount_type: dict[str, int] = {mount_type.value: 0 for mount_type in selected_mounts}
        errors: list[str] = []
        skipped_large_files = 0
        skipped_binary_files = 0

        for mount_type in selected_mounts:
            try:
                mount_root = self._manager.resolve_mount_root(self._context_path, mount_type)
            except Exception as exc:  # pragma: no cover - defensive
                errors.append(f"{mount_type.value}: resolve failed: {exc}")
                continue

            if not mount_root.exists():
                continue

            for entry, relative_path in _iter_mount_entries(mount_root):
                entry = self._safe_v2_index_entry(
                    mount_type,
                    relative_path,
                    entry,
                )
                if entry is None:
                    continue
                if self._should_skip_relative_path(
                    mount_type,
                    relative_path,
                    entry=entry,
                ):
                    continue
                try:
                    row, reason = self._build_row(
                        mount_type,
                        relative_path=relative_path,
                        entry=entry,
                        include_content=include_content,
                        max_file_size_bytes=max_file_size_bytes,
                        max_content_chars=max_content_chars,
                        indexed_at=indexed_at,
                    )
                except OSError as exc:
                    errors.append(f"{mount_type.value}: stat failed for {entry}: {exc}")
                    continue

                if reason == "too_large":
                    skipped_large_files += 1
                elif reason == "binary":
                    skipped_binary_files += 1

                rows.append(row)
                by_mount_type[mount_type.value] = by_mount_type.get(mount_type.value, 0) + 1

        rows_deleted = 0
        with self._connect() as connection:
            rows_deleted = self._delete_rows_for_mounts(selected_mounts, connection=connection)
            if rows:
                connection.executemany(
                    """
                    INSERT INTO file_index (
                        context_path,
                        mount_type,
                        relative_path,
                        absolute_path,
                        is_dir,
                        size_bytes,
                        modified_at,
                        content_text,
                        content_hash,
                        indexed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
            self._commit_mutation(connection, truncate_wal=True)

        return IndexSummary(
            context_path=str(self._context_path),
            db_path=str(self._db_path),
            indexed_at=indexed_at,
            rows_written=len(rows),
            rows_deleted=rows_deleted,
            by_mount_type=by_mount_type,
            skipped_large_files=skipped_large_files,
            skipped_binary_files=skipped_binary_files,
            errors=errors,
        )

    def rebuild_scoped(
        self,
        scoped: ResolvedScope,
        *,
        mount_types: list[MountType] | None = None,
        include_content: bool = True,
        max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES,
        max_content_chars: int = DEFAULT_MAX_CONTENT_CHARS,
    ) -> IndexSummary:
        """Rebuild only rows visible to one v2 project/common requester.

        Rows belonging to other project scopes are preserved.  More
        importantly, their filesystem roots are never traversed while the
        scoped rebuild is collecting entries.
        """

        self._assert_scope_matches(scoped)
        if scoped.layout_version != LAYOUT_VERSION:
            return self.rebuild(
                mount_types=mount_types,
                include_content=include_content,
                max_file_size_bytes=max_file_size_bytes,
                max_content_chars=max_content_chars,
            )

        selected_mounts = self._normalize_scoped_mount_types(scoped, mount_types)
        indexed_at = datetime.now(timezone.utc).isoformat()
        rows: list[tuple[Any, ...]] = []
        by_mount_type: dict[str, int] = {
            mount_type.value: 0 for mount_type in selected_mounts
        }
        errors: list[str] = []
        skipped_large_files = 0
        skipped_binary_files = 0
        prefixes_by_mount: dict[MountType, tuple[str, ...]] = {}

        for mount_type in selected_mounts:
            try:
                mount_root = self._manager.resolve_mount_root(
                    self._context_path, mount_type
                )
                roots = visible_mount_roots(
                    mount_root,
                    mount_type=mount_type,
                    scoped=scoped,
                )
            except Exception as exc:  # pragma: no cover - defensive
                errors.append(f"{mount_type.value}: resolve failed: {exc}")
                continue

            prefixes: list[str] = []
            for root in roots:
                try:
                    prefix = root.relative_to(mount_root).as_posix()
                except ValueError:
                    errors.append(
                        f"{mount_type.value}: scoped root escaped mount: {root}"
                    )
                    continue
                prefixes.append(prefix)
                if not root.exists():
                    continue
                candidates = [(root, prefix), *_iter_mount_entries_with_prefix(root, prefix)]
                for entry, relative_path in candidates:
                    entry = self._safe_v2_index_entry(
                        mount_type,
                        relative_path,
                        entry,
                    )
                    if entry is None:
                        continue
                    if self._should_skip_relative_path(
                        mount_type,
                        relative_path,
                        entry=entry,
                    ):
                        continue
                    try:
                        row, reason = self._build_row(
                            mount_type,
                            relative_path=relative_path,
                            entry=entry,
                            include_content=include_content,
                            max_file_size_bytes=max_file_size_bytes,
                            max_content_chars=max_content_chars,
                            indexed_at=indexed_at,
                        )
                    except OSError as exc:
                        errors.append(
                            f"{mount_type.value}: stat failed for {entry}: {exc}"
                        )
                        continue

                    if reason == "too_large":
                        skipped_large_files += 1
                    elif reason == "binary":
                        skipped_binary_files += 1
                    rows.append(row)
                    by_mount_type[mount_type.value] += 1
            prefixes_by_mount[mount_type] = tuple(prefixes)

        with self._connect() as connection:
            rows_deleted = self._delete_rows_for_scoped_prefixes(
                prefixes_by_mount,
                connection=connection,
            )
            if rows:
                connection.executemany(
                    """
                    INSERT INTO file_index (
                        context_path,
                        mount_type,
                        relative_path,
                        absolute_path,
                        is_dir,
                        size_bytes,
                        modified_at,
                        content_text,
                        content_hash,
                        indexed_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    rows,
                )
            self._commit_mutation(connection, truncate_wal=True)

        return IndexSummary(
            context_path=str(self._context_path),
            db_path=str(self._db_path),
            indexed_at=indexed_at,
            rows_written=len(rows),
            rows_deleted=rows_deleted,
            by_mount_type=by_mount_type,
            skipped_large_files=skipped_large_files,
            skipped_binary_files=skipped_binary_files,
            errors=errors,
        )

    def upsert_relative_path(
        self,
        mount_type: MountType,
        relative_path: str,
        *,
        include_content: bool = True,
        max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES,
        max_content_chars: int = DEFAULT_MAX_CONTENT_CHARS,
    ) -> bool:
        normalized = _normalize_relative_path(relative_path)
        if not normalized:
            return False

        mount_root = self._manager.resolve_mount_root(self._context_path, mount_type)
        entry = mount_root / normalized
        entry = self._safe_v2_index_entry(mount_type, normalized, entry)
        if entry is None:
            self.delete_relative_path(mount_type, normalized)
            return False
        if self._should_skip_relative_path(
            mount_type,
            normalized,
            entry=entry,
        ):
            self.delete_relative_path(mount_type, normalized)
            return False
        if not entry.exists():
            self.delete_relative_path(mount_type, normalized)
            return False

        indexed_at = datetime.now(timezone.utc).isoformat()
        row, _reason = self._build_row(
            mount_type,
            relative_path=normalized,
            entry=entry,
            include_content=include_content,
            max_file_size_bytes=max_file_size_bytes,
            max_content_chars=max_content_chars,
            indexed_at=indexed_at,
        )

        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR REPLACE INTO file_index (
                    context_path,
                    mount_type,
                    relative_path,
                    absolute_path,
                    is_dir,
                    size_bytes,
                    modified_at,
                    content_text,
                    content_hash,
                    indexed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                row,
            )
            self._commit_mutation(connection)
        return True

    def sync_absolute_path(
        self,
        absolute_path: Path,
        *,
        scoped: ResolvedScope | None = None,
        include_content: bool = True,
        max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES,
        max_content_chars: int = DEFAULT_MAX_CONTENT_CHARS,
    ) -> bool:
        inferred = self.infer_relative_for_absolute_path(absolute_path)
        if not inferred:
            return False

        mount_type, relative_path = inferred
        if scoped is not None:
            self._assert_scope_matches(scoped)
        candidate = absolute_path.expanduser().resolve()
        if candidate.exists():
            if candidate.is_dir():
                rebuild_kwargs = {
                    "mount_types": [mount_type],
                    "include_content": include_content,
                    "max_file_size_bytes": max_file_size_bytes,
                    "max_content_chars": max_content_chars,
                }
                if scoped is not None and scoped.layout_version == LAYOUT_VERSION:
                    self.rebuild_scoped(scoped, **rebuild_kwargs)
                else:
                    self.rebuild(**rebuild_kwargs)
                return True
            self.upsert_relative_path(
                mount_type,
                relative_path,
                include_content=include_content,
                max_file_size_bytes=max_file_size_bytes,
                max_content_chars=max_content_chars,
            )
            return True

        self.delete_relative_prefix(mount_type, relative_path)
        return True

    def delete_relative_path(self, mount_type: MountType, relative_path: str) -> int:
        normalized = _normalize_relative_path(relative_path)
        if not normalized:
            return 0

        with self._connect() as connection:
            cursor = connection.execute(
                """
                DELETE FROM file_index
                WHERE context_path = ?
                  AND mount_type = ?
                  AND relative_path = ?
                """,
                (str(self._context_path), mount_type.value, normalized),
            )
            self._commit_mutation(connection)
            if cursor.rowcount is None or cursor.rowcount < 0:
                return 0
            return int(cursor.rowcount)

    def delete_relative_prefix(self, mount_type: MountType, relative_path: str) -> int:
        normalized = _normalize_relative_path(relative_path)
        if not normalized:
            return 0

        with self._connect() as connection:
            cursor = connection.execute(
                """
                DELETE FROM file_index
                WHERE context_path = ?
                  AND mount_type = ?
                  AND (
                    relative_path = ?
                    OR substr(relative_path, 1, length(?)) = ?
                  )
                """,
                (
                    str(self._context_path),
                    mount_type.value,
                    normalized,
                    f"{normalized}/",
                    f"{normalized}/",
                ),
            )
            self._commit_mutation(connection)
            if cursor.rowcount is None or cursor.rowcount < 0:
                return 0
            return int(cursor.rowcount)

    def infer_relative_for_absolute_path(
        self, absolute_path: Path
    ) -> tuple[MountType, str] | None:
        candidate = absolute_path.expanduser().resolve()
        for mount_type in MountType:
            try:
                mount_root = self._manager.resolve_mount_root(self._context_path, mount_type)
            except Exception:
                continue
            if not mount_root.exists():
                continue

            mount_root_resolved = mount_root.resolve()
            try:
                rel = candidate.relative_to(mount_root_resolved).as_posix()
                if rel and rel != ".":
                    return mount_type, rel
            except ValueError:
                pass

            try:
                children = mount_root.iterdir()
            except OSError:
                continue
            for child in children:
                if not child.is_symlink():
                    continue
                try:
                    target = child.resolve()
                except OSError:
                    continue
                if not target.exists():
                    continue
                if candidate == target:
                    return mount_type, child.name
                try:
                    sub = candidate.relative_to(target).as_posix()
                except ValueError:
                    continue
                if not sub or sub == ".":
                    return mount_type, child.name
                return mount_type, f"{child.name}/{sub}"
        return None

    def has_entries(self, *, mount_types: list[MountType] | None = None) -> bool:
        selected_mounts = self._normalize_mount_types(mount_types)
        if not selected_mounts:
            return False
        placeholders = ", ".join("?" for _ in selected_mounts)
        params = [str(self._context_path), *[mount.value for mount in selected_mounts]]
        with self._connect() as connection:
            row = connection.execute(
                f"""
                SELECT COUNT(1) AS count
                FROM file_index
                WHERE context_path = ?
                  AND mount_type IN ({placeholders})
                """,
                params,
            ).fetchone()
        return bool(row and int(row["count"]) > 0)

    def has_entries_scoped(
        self,
        scoped: ResolvedScope,
        *,
        mount_types: list[MountType] | None = None,
    ) -> bool:
        """Return whether the authorized v2 scope has indexed rows."""

        self._assert_scope_matches(scoped)
        if scoped.layout_version != LAYOUT_VERSION:
            return self.has_entries(mount_types=mount_types)
        return self.count_entries_scoped(scoped, mount_types=mount_types) > 0

    def needs_refresh(self, *, mount_types: list[MountType] | None = None) -> bool:
        selected_mounts = self._normalize_mount_types(mount_types)
        if not selected_mounts:
            return False

        fs_snapshot = self._filesystem_snapshot(selected_mounts)
        db_snapshot = self._indexed_snapshot(selected_mounts)

        for mount_type in selected_mounts:
            key = mount_type.value
            if fs_snapshot.get(key, _empty_mount_snapshot()) != db_snapshot.get(
                key, _empty_mount_snapshot()
            ):
                return True
        return False

    def needs_refresh_scoped(
        self,
        scoped: ResolvedScope,
        *,
        mount_types: list[MountType] | None = None,
    ) -> bool:
        """Compare only authorized current/common filesystem and index rows."""

        self._assert_scope_matches(scoped)
        if scoped.layout_version != LAYOUT_VERSION:
            return self.needs_refresh(mount_types=mount_types)
        selected_mounts = self._normalize_scoped_mount_types(scoped, mount_types)
        fs_snapshot = self._filesystem_snapshot_scoped(selected_mounts, scoped)
        db_snapshot = self._indexed_snapshot(
            selected_mounts,
            relative_prefixes=visible_scope_prefixes(scoped),
        )
        return any(
            fs_snapshot.get(mount_type.value, _empty_mount_snapshot())
            != db_snapshot.get(mount_type.value, _empty_mount_snapshot())
            for mount_type in selected_mounts
        )

    @staticmethod
    def health_mount_types() -> list[MountType]:
        """Return the stable mounts used for agent-facing index health checks."""
        return list(INDEX_HEALTH_MOUNT_TYPES)

    def needs_health_refresh(self) -> bool:
        """Return whether stable, query-relevant mounts have drifted."""
        return self.needs_refresh(mount_types=self.health_mount_types())

    def summary(self) -> IndexSummary:
        """Return a lightweight summary of the current index state."""
        by_mount: dict[str, int] = {}
        total = 0
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT mount_type, COUNT(1) AS cnt FROM file_index "
                "WHERE context_path = ? GROUP BY mount_type",
                (str(self._context_path),),
            ).fetchall()
            for row in rows:
                by_mount[row["mount_type"]] = int(row["cnt"])
                total += int(row["cnt"])
        return IndexSummary(
            context_path=str(self._context_path),
            db_path=str(self.db_path),
            indexed_at="",
            rows_written=total,
            rows_deleted=0,
            by_mount_type=by_mount,
            skipped_large_files=0,
            skipped_binary_files=0,
            errors=[],
        )

    @property
    def total_entries(self) -> int:
        """Return total indexed entries for this context path."""
        with self._connect() as connection:
            row = connection.execute(
                "SELECT COUNT(1) AS cnt FROM file_index WHERE context_path = ?",
                (str(self._context_path),),
            ).fetchone()
            return int(row["cnt"]) if row else 0

    def count_entries(
        self,
        *,
        mount_types: list[MountType] | None = None,
        relative_prefixes: Sequence[str] | None = None,
    ) -> int:
        """Count indexed entries after optional mount and scope predicates."""

        selected_mounts = self._normalize_mount_types(mount_types)
        if not selected_mounts:
            return 0
        where = ["context_path = ?"]
        params: list[Any] = [str(self._context_path)]
        placeholders = ", ".join("?" for _ in selected_mounts)
        where.append(f"mount_type IN ({placeholders})")
        params.extend(mount.value for mount in selected_mounts)
        prefixes = tuple(
            str(prefix).replace("\\", "/").lstrip("/")
            for prefix in (relative_prefixes or ())
        )
        if prefixes:
            where.append(
                "("
                + " OR ".join(
                    "substr(relative_path, 1, length(?)) = ?" for _ in prefixes
                )
                + ")"
            )
            for prefix in prefixes:
                params.extend((prefix, prefix))
        with self._connect() as connection:
            row = connection.execute(
                f"SELECT COUNT(1) AS cnt FROM file_index WHERE {' AND '.join(where)}",
                params,
            ).fetchone()
        return int(row["cnt"]) if row else 0

    def count_entries_scoped(
        self,
        scoped: ResolvedScope,
        *,
        mount_types: list[MountType] | None = None,
    ) -> int:
        """Count only category rows visible to a current/common requester."""

        self._assert_scope_matches(scoped)
        if scoped.layout_version != LAYOUT_VERSION:
            return self.count_entries(mount_types=mount_types)
        selected_mounts = self._normalize_scoped_mount_types(scoped, mount_types)
        if not selected_mounts:
            return 0
        placeholders = ", ".join("?" for _ in selected_mounts)
        prefix_clause, prefix_params = _relative_prefix_sql(
            visible_scope_prefixes(scoped),
            leading_and=True,
        )
        params: list[Any] = [
            str(self._context_path),
            *[mount.value for mount in selected_mounts],
            *prefix_params,
        ]
        with self._connect() as connection:
            row = connection.execute(
                f"""
                SELECT COUNT(1) AS cnt
                FROM file_index
                WHERE context_path = ?
                  AND mount_type IN ({placeholders})
                  {prefix_clause}
                """,
                params,
            ).fetchone()
        return int(row["cnt"]) if row else 0

    def diff(
        self,
        *,
        mount_types: list[MountType] | None = None,
        relative_prefixes: Sequence[str] | None = None,
        scoped: ResolvedScope | None = None,
    ) -> dict[str, Any]:
        """Compare filesystem state against the index.

        Returns lists of new, modified, and deleted paths per mount type.
        """
        if scoped is not None:
            self._assert_scope_matches(scoped)
        selected_mounts = (
            self._normalize_scoped_mount_types(scoped, mount_types)
            if scoped is not None and scoped.layout_version == LAYOUT_VERSION
            else self._normalize_mount_types(mount_types)
        )
        normalized_prefixes = _effective_relative_prefixes(
            relative_prefixes,
            scoped=scoped,
        )

        def visible(relative_path: str) -> bool:
            return _relative_path_matches_prefixes(relative_path, normalized_prefixes)
        added: list[dict[str, str]] = []
        modified: list[dict[str, str]] = []
        deleted: list[dict[str, str]] = []

        for mount_type in selected_mounts:
            try:
                mount_root = self._manager.resolve_mount_root(
                    self._context_path, mount_type
                )
            except Exception:
                continue
            if not mount_root.exists():
                continue

            # Build filesystem snapshot: relative_path -> (size, mtime)
            fs_entries: dict[str, tuple[int, str]] = {}
            if scoped is not None and scoped.layout_version == LAYOUT_VERSION:
                scoped_entries = []
                for root in visible_mount_roots(
                    mount_root,
                    mount_type=mount_type,
                    scoped=scoped,
                ):
                    if not root.exists():
                        continue
                    prefix = root.relative_to(mount_root).as_posix()
                    scoped_entries.extend(
                        _iter_mount_entries_with_prefix(root, prefix)
                    )
                mount_entries = scoped_entries
            else:
                mount_entries = _iter_mount_entries(mount_root)
            for entry, relative_path in mount_entries:
                if not visible(relative_path):
                    continue
                if scoped is not None and scoped.layout_version == LAYOUT_VERSION:
                    entry = self._safe_v2_index_entry(
                        mount_type,
                        relative_path,
                        entry,
                    )
                    if entry is None:
                        continue
                if self._should_skip_relative_path(
                    mount_type,
                    relative_path,
                    entry=entry,
                ):
                    continue
                try:
                    stat = entry.stat()
                    mtime = _iso_utc(stat.st_mtime)
                    fs_entries[relative_path] = (stat.st_size, mtime)
                except OSError:
                    continue

            # Build index snapshot: relative_path -> (size, mtime)
            db_entries: dict[str, tuple[int, str | None]] = {}
            with self._connect() as connection:
                prefix_clause, prefix_params = _relative_prefix_sql(
                    normalized_prefixes,
                    leading_and=True,
                )
                rows = connection.execute(
                    "SELECT relative_path, size_bytes, modified_at "
                    "FROM file_index WHERE context_path = ? AND mount_type = ?"
                    f"{prefix_clause}",
                    (str(self._context_path), mount_type.value, *prefix_params),
                ).fetchall()
                for row in rows:
                    if not visible(row["relative_path"]):
                        continue
                    db_entries[row["relative_path"]] = (
                        int(row["size_bytes"]),
                        row["modified_at"],
                    )

            # Compare
            for rel_path, (fs_size, fs_mtime) in fs_entries.items():
                if rel_path not in db_entries:
                    added.append({
                        "mount_type": mount_type.value,
                        "relative_path": rel_path,
                        "size_bytes": fs_size,
                    })
                else:
                    db_size, db_mtime = db_entries[rel_path]
                    if fs_size != db_size or fs_mtime != db_mtime:
                        modified.append({
                            "mount_type": mount_type.value,
                            "relative_path": rel_path,
                            "size_bytes": fs_size,
                        })

            for rel_path in db_entries:
                if rel_path not in fs_entries:
                    deleted.append({
                        "mount_type": mount_type.value,
                        "relative_path": rel_path,
                    })

        return {
            "context_path": str(self._context_path),
            "added": added,
            "modified": modified,
            "deleted": deleted,
            "total_changes": len(added) + len(modified) + len(deleted),
        }

    def freshness_scores(
        self,
        *,
        mount_types: list[MountType] | None = None,
        decay_hours: float = 168.0,
        threshold: float = 0.0,
        relative_prefixes: Sequence[str] | None = None,
        scoped: ResolvedScope | None = None,
    ) -> dict[str, Any]:
        """Compute per-file freshness scores based on index vs filesystem state.

        Score = max(0, 1 - age_seconds / decay_seconds) for indexed files.
        Modified or deleted files get 0.0.
        """
        if scoped is not None:
            self._assert_scope_matches(scoped)
        selected_mounts = (
            self._normalize_scoped_mount_types(scoped, mount_types)
            if scoped is not None and scoped.layout_version == LAYOUT_VERSION
            else self._normalize_mount_types(mount_types)
        )
        normalized_prefixes = _effective_relative_prefixes(
            relative_prefixes,
            scoped=scoped,
        )

        def visible(relative_path: str) -> bool:
            return _relative_path_matches_prefixes(relative_path, normalized_prefixes)
        decay_seconds = decay_hours * 3600.0
        mount_scores: dict[str, float] = {}
        files: dict[str, list[dict[str, Any]]] = {}
        now = datetime.now(timezone.utc).timestamp()

        with self._connect() as connection:
            for mount_type in selected_mounts:
                mount_key = mount_type.value
                try:
                    mount_root = self._manager.resolve_mount_root(
                        self._context_path, mount_type
                    )
                except Exception:
                    continue
                if not mount_root.exists():
                    files[mount_key] = []
                    mount_scores[mount_key] = 1.0
                    continue

                fs_entries: dict[str, Path] = {}
                if scoped is not None and scoped.layout_version == LAYOUT_VERSION:
                    scoped_entries = []
                    for root in visible_mount_roots(
                        mount_root,
                        mount_type=mount_type,
                        scoped=scoped,
                    ):
                        if not root.exists():
                            continue
                        prefix = root.relative_to(mount_root).as_posix()
                        scoped_entries.extend(
                            _iter_mount_entries_with_prefix(root, prefix)
                        )
                    mount_entries = scoped_entries
                else:
                    mount_entries = _iter_mount_entries(mount_root)
                for entry, relative_path in mount_entries:
                    if not visible(relative_path):
                        continue
                    if scoped is not None and scoped.layout_version == LAYOUT_VERSION:
                        entry = self._safe_v2_index_entry(
                            mount_type,
                            relative_path,
                            entry,
                        )
                        if entry is None:
                            continue
                    if self._should_skip_relative_path(
                        mount_type,
                        relative_path,
                        entry=entry,
                    ):
                        continue
                    if not entry.is_file():
                        continue
                    fs_entries[relative_path] = entry

                prefix_clause, prefix_params = _relative_prefix_sql(
                    normalized_prefixes,
                    leading_and=True,
                )
                rows = connection.execute(
                    """
                    SELECT relative_path, modified_at, absolute_path
                    FROM file_index
                    WHERE context_path = ? AND mount_type = ? AND is_dir = 0
                    """
                    + prefix_clause,
                    (str(self._context_path), mount_key, *prefix_params),
                ).fetchall()

                file_entries: list[dict[str, Any]] = []
                score_sum = 0.0
                count = 0
                seen_paths: set[str] = set()

                for rel_path, indexed_modified_at, abs_path in rows:
                    if not visible(rel_path):
                        continue
                    try:
                        row_mount = MountType(mount_key)
                    except ValueError:
                        continue
                    safe_entry = self._safe_v2_index_entry(
                        row_mount,
                        rel_path,
                        Path(abs_path) if abs_path else mount_root / rel_path,
                        require_exists=False,
                    )
                    if safe_entry is None:
                        continue
                    seen_paths.add(rel_path)
                    entry_path = safe_entry
                    if not entry_path.exists():
                        score = 0.0
                        status = "deleted"
                    else:
                        try:
                            current_mtime = entry_path.stat().st_mtime
                        except OSError:
                            score = 0.0
                            status = "deleted"
                            file_entries.append({
                                "relative_path": rel_path,
                                "mount_type": mount_key,
                                "score": score,
                                "status": status,
                            })
                            score_sum += score
                            count += 1
                            continue

                        indexed_ts = self._parse_iso_timestamp(indexed_modified_at)
                        if indexed_ts is not None and current_mtime > indexed_ts + 1.0:
                            score = 0.0
                            status = "modified"
                        else:
                            age = now - (indexed_ts or current_mtime)
                            score = max(0.0, 1.0 - age / decay_seconds) if decay_seconds > 0 else 1.0
                            status = "indexed"

                    if score >= threshold:
                        file_entries.append({
                            "relative_path": rel_path,
                            "mount_type": mount_key,
                            "score": round(score, 4),
                            "status": status,
                        })
                    score_sum += score
                    count += 1

                for rel_path in sorted(fs_entries):
                    if rel_path in seen_paths:
                        continue
                    score = 0.0
                    count += 1
                    if score >= threshold:
                        file_entries.append({
                            "relative_path": rel_path,
                            "mount_type": mount_key,
                            "score": score,
                            "status": "unindexed",
                        })

                file_entries.sort(key=lambda f: f["score"])
                files[mount_key] = file_entries
                mount_scores[mount_key] = round(score_sum / count, 4) if count > 0 else 1.0

        return {
            "mount_scores": mount_scores,
            "files": files,
            "decay_hours": decay_hours,
            "threshold": threshold,
        }

    @staticmethod
    def _parse_iso_timestamp(value: str | None) -> float | None:
        if not value:
            return None
        try:
            normalized = value.strip()
            if normalized.endswith("Z"):
                normalized = normalized[:-1] + "+00:00"
            parsed = datetime.fromisoformat(normalized)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.timestamp()
        except (ValueError, OSError):
            return None

    def query(
        self,
        *,
        query: str | None = None,
        mount_types: list[MountType] | None = None,
        relative_prefix: str | None = None,
        limit: int = 25,
        include_content: bool = False,
    ) -> list[dict[str, Any]]:
        selected_mounts = self._normalize_mount_types(mount_types)
        if not selected_mounts:
            return []

        normalized_prefix = _normalize_relative_prefix(relative_prefix)
        normalized_query = (query or "").strip()
        limit_value = max(1, min(limit, 500))

        query_limit = (
            500
            if detect_layout_version(self._context_path) == LAYOUT_VERSION
            else limit_value
        )
        rows = []
        fts_query = _build_fts_query(normalized_query) if normalized_query else None
        if normalized_query and self._fts_enabled and fts_query:
            rows = self._query_with_fts(
                fts_query=fts_query,
                mount_types=selected_mounts,
                relative_prefix=normalized_prefix,
                limit=query_limit,
            )

        if not rows:
            rows = self._query_with_like(
                query=normalized_query,
                mount_types=selected_mounts,
                relative_prefix=normalized_prefix,
                limit=query_limit,
            )

        payloads: list[dict[str, Any]] = []
        for row in rows:
            try:
                mount_type = MountType(str(row["mount_type"]))
            except ValueError:
                continue
            if self._safe_v2_index_entry(
                mount_type,
                str(row["relative_path"]),
                Path(str(row["absolute_path"])),
                require_exists=True,
            ) is None:
                continue
            payloads.append(
                self._row_to_payload(
                    row=row,
                    query=normalized_query,
                    include_content=include_content,
                )
            )
            if len(payloads) >= limit_value:
                break
        return payloads

    def query_scoped(
        self,
        scoped: ResolvedScope,
        *,
        query: str | None = None,
        mount_types: list[MountType] | None = None,
        limit: int = 25,
        include_content: bool = False,
    ) -> list[dict[str, Any]]:
        """Query only the current/common rows authorized for one requester."""

        self._assert_scope_matches(scoped)
        if scoped.layout_version != LAYOUT_VERSION:
            return self.query(
                query=query,
                mount_types=mount_types,
                limit=limit,
                include_content=include_content,
            )
        selected_mounts = self._normalize_scoped_mount_types(scoped, mount_types)
        if not selected_mounts:
            return []
        limit_value = max(1, min(limit, 500))
        results: list[dict[str, Any]] = []
        seen: set[tuple[str, str]] = set()
        for prefix in visible_scope_prefixes(scoped):
            for item in self.query(
                query=query,
                mount_types=selected_mounts,
                relative_prefix=prefix,
                limit=limit_value,
                include_content=include_content,
            ):
                key = (
                    str(item.get("mount_type", "")),
                    str(item.get("relative_path", "")),
                )
                if key in seen:
                    continue
                seen.add(key)
                results.append(item)
                if len(results) >= limit_value:
                    return results
        return results

    def _query_with_fts(
        self,
        *,
        fts_query: str,
        mount_types: list[MountType],
        relative_prefix: str,
        limit: int,
    ) -> list[sqlite3.Row]:
        where = ["fi.context_path = ?"]
        params: list[Any] = [str(self._context_path)]

        placeholders = ", ".join("?" for _ in mount_types)
        where.append(f"fi.mount_type IN ({placeholders})")
        params.extend([mount.value for mount in mount_types])

        if relative_prefix:
            descendant_prefix = f"{relative_prefix}/"
            where.append(
                "(fi.relative_path = ? OR "
                "substr(fi.relative_path, 1, length(?)) = ?)"
            )
            params.extend((relative_prefix, descendant_prefix, descendant_prefix))

        with self._connect() as connection:
            return connection.execute(
                f"""
                SELECT
                    fi.mount_type,
                    fi.relative_path,
                    fi.absolute_path,
                    fi.is_dir,
                    fi.size_bytes,
                    fi.modified_at,
                    fi.indexed_at,
                    fi.content_text,
                    bm25(file_index_fts) AS relevance_score
                FROM file_index_fts
                JOIN file_index fi ON fi.rowid = file_index_fts.rowid
                WHERE file_index_fts MATCH ?
                  AND {" AND ".join(where)}
                ORDER BY relevance_score ASC, fi.is_dir ASC, fi.modified_at DESC, fi.relative_path ASC
                LIMIT ?
                """,
                [fts_query, *params, limit],
            ).fetchall()

    def _query_with_like(
        self,
        *,
        query: str,
        mount_types: list[MountType],
        relative_prefix: str,
        limit: int,
    ) -> list[sqlite3.Row]:
        where = ["context_path = ?"]
        params: list[Any] = [str(self._context_path)]

        placeholders = ", ".join("?" for _ in mount_types)
        where.append(f"mount_type IN ({placeholders})")
        params.extend([mount.value for mount in mount_types])

        if relative_prefix:
            descendant_prefix = f"{relative_prefix}/"
            where.append(
                "(relative_path = ? OR "
                "substr(relative_path, 1, length(?)) = ?)"
            )
            params.extend((relative_prefix, descendant_prefix, descendant_prefix))

        if query:
            token = f"%{query.lower()}%"
            where.append(
                "(LOWER(relative_path) LIKE ? OR LOWER(COALESCE(content_text, '')) LIKE ?)"
            )
            params.extend([token, token])

        with self._connect() as connection:
            return connection.execute(
                f"""
                SELECT
                    mount_type,
                    relative_path,
                    absolute_path,
                    is_dir,
                    size_bytes,
                    modified_at,
                    indexed_at,
                    content_text,
                    NULL AS relevance_score
                FROM file_index
                WHERE {" AND ".join(where)}
                ORDER BY is_dir ASC, modified_at DESC, relative_path ASC
                LIMIT ?
                """,
                [*params, limit],
            ).fetchall()

    def _row_to_payload(
        self,
        *,
        row: sqlite3.Row,
        query: str,
        include_content: bool,
    ) -> dict[str, Any]:
        content_text = row["content_text"] if isinstance(row["content_text"], str) else ""
        payload: dict[str, Any] = {
            "mount_type": row["mount_type"],
            "relative_path": row["relative_path"],
            "absolute_path": row["absolute_path"],
            "is_dir": bool(row["is_dir"]),
            "size_bytes": int(row["size_bytes"]),
            "modified_at": row["modified_at"],
            "indexed_at": row["indexed_at"],
            "content_excerpt": _build_excerpt(content_text, query),
        }
        score = row["relevance_score"]
        if isinstance(score, (int, float)):
            payload["relevance_score"] = float(score)
        if include_content:
            payload["content"] = content_text
        return payload

    def _build_row(
        self,
        mount_type: MountType,
        *,
        relative_path: str,
        entry: Path,
        include_content: bool,
        max_file_size_bytes: int,
        max_content_chars: int,
        indexed_at: str,
    ) -> tuple[tuple[Any, ...], str | None]:
        normalized = _normalize_relative_path(relative_path)
        if not normalized:
            raise OSError("relative path is empty")

        stat = entry.stat()
        is_dir = entry.is_dir()
        content_text: str | None = None
        content_hash: str | None = None
        reason: str | None = None

        if include_content and entry.is_file():
            content_text, content_hash, reason = _read_text_for_indexing(
                entry,
                size_bytes=stat.st_size,
                max_file_size_bytes=max_file_size_bytes,
                max_content_chars=max_content_chars,
            )

        row = (
            str(self._context_path),
            mount_type.value,
            normalized,
            str(entry),
            1 if is_dir else 0,
            stat.st_size,
            _iso_utc(stat.st_mtime),
            content_text,
            content_hash,
            indexed_at,
        )
        return row, reason

    def _filesystem_snapshot(
        self,
        mount_types: list[MountType],
    ) -> dict[str, MountSnapshot]:
        snapshot: dict[str, MountSnapshot] = {}
        for mount_type in mount_types:
            count = 0
            max_modified: str | None = None
            fingerprint = hashlib.sha256()
            try:
                mount_root = self._manager.resolve_mount_root(self._context_path, mount_type)
            except Exception:
                snapshot[mount_type.value] = _empty_mount_snapshot()
                continue
            if not mount_root.exists():
                snapshot[mount_type.value] = _empty_mount_snapshot()
                continue

            for entry, relative_path in _iter_mount_entries(mount_root):
                entry = self._safe_v2_index_entry(
                    mount_type,
                    relative_path,
                    entry,
                )
                if entry is None:
                    continue
                if self._should_skip_relative_path(
                    mount_type,
                    relative_path,
                    entry=entry,
                ):
                    continue
                try:
                    stat = entry.stat()
                except OSError:
                    continue
                modified = _iso_utc(stat.st_mtime)
                count += 1
                if max_modified is None or modified > max_modified:
                    max_modified = modified
                _update_snapshot_fingerprint(
                    fingerprint,
                    relative_path=relative_path,
                    is_dir=entry.is_dir(),
                    size_bytes=stat.st_size,
                    modified_at=modified,
                )
            snapshot[mount_type.value] = MountSnapshot(
                count=count,
                max_modified=max_modified,
                fingerprint=fingerprint.hexdigest(),
            )
        return snapshot

    def _filesystem_snapshot_scoped(
        self,
        mount_types: list[MountType],
        scoped: ResolvedScope,
    ) -> dict[str, MountSnapshot]:
        snapshot: dict[str, MountSnapshot] = {}
        for mount_type in mount_types:
            count = 0
            max_modified: str | None = None
            fingerprint = hashlib.sha256()
            try:
                mount_root = self._manager.resolve_mount_root(
                    self._context_path, mount_type
                )
                roots = visible_mount_roots(
                    mount_root,
                    mount_type=mount_type,
                    scoped=scoped,
                )
            except Exception:
                snapshot[mount_type.value] = _empty_mount_snapshot()
                continue
            scoped_candidates: list[tuple[Path, str]] = []
            for root in roots:
                try:
                    prefix = root.relative_to(mount_root).as_posix()
                except ValueError:
                    continue
                if not root.exists():
                    continue
                scoped_candidates.append((root, prefix))
                scoped_candidates.extend(
                    _iter_mount_entries_with_prefix(root, prefix)
                )
            for entry, relative_path in sorted(
                scoped_candidates,
                key=lambda candidate: candidate[1],
            ):
                entry = self._safe_v2_index_entry(
                    mount_type,
                    relative_path,
                    entry,
                )
                if entry is None or self._should_skip_relative_path(
                    mount_type,
                    relative_path,
                    entry=entry,
                ):
                    continue
                try:
                    stat = entry.stat()
                except OSError:
                    continue
                modified = _iso_utc(stat.st_mtime)
                count += 1
                if max_modified is None or modified > max_modified:
                    max_modified = modified
                _update_snapshot_fingerprint(
                    fingerprint,
                    relative_path=relative_path,
                    is_dir=entry.is_dir(),
                    size_bytes=stat.st_size,
                    modified_at=modified,
                )
            snapshot[mount_type.value] = MountSnapshot(
                count=count,
                max_modified=max_modified,
                fingerprint=fingerprint.hexdigest(),
            )
        return snapshot

    def _safe_v2_index_entry(
        self,
        mount_type: MountType,
        relative_path: str,
        entry: Path,
        *,
        require_exists: bool = False,
    ) -> Path | None:
        """Return a lexical v2 entry only when it stays in its exact scope."""

        if detect_layout_version(self._context_path) != LAYOUT_VERSION:
            return entry
        if ContextCategory.from_mount_type(mount_type) is None:
            return entry
        normalized = _normalize_relative_path(relative_path)
        if not normalized:
            return None
        parts = Path(normalized).parts
        mount_root = self._manager.resolve_mount_root(self._context_path, mount_type)
        if parts[0] == "common":
            scope_root = mount_root / "common"
        elif parts[0] == "projects":
            if len(parts) == 1:
                scope_root = mount_root
            else:
                scope_root = mount_root / "projects" / parts[1]
        else:
            return None
        expected = mount_root / normalized
        lexical_entry = Path(os.path.abspath(entry.expanduser()))
        try:
            safe_scope_root = assert_no_linklike_components(
                scope_root,
                boundary=self._context_path,
                allow_missing=not require_exists,
            )
            safe = assert_no_linklike_components(
                expected,
                boundary=safe_scope_root,
                allow_missing=not require_exists,
            )
        except (OSError, ValueError):
            return None
        if lexical_entry != safe or (require_exists and not safe.exists()):
            return None
        return safe

    def _should_skip_relative_path(
        self,
        mount_type: MountType,
        relative_path: str,
        *,
        entry: Path | None = None,
    ) -> bool:
        if mount_type != MountType.GLOBAL:
            ignored_global = False
        else:
            db_name = self._db_path.name
            ignored = {
                db_name,
                f"{db_name}-journal",
                f"{db_name}-shm",
                f"{db_name}-wal",
            }
            ignored_global = relative_path in ignored
        if ignored_global:
            return True

        absolute_path = entry
        if absolute_path is None:
            absolute_path = self._manager.resolve_mount_root(self._context_path, mount_type) / relative_path
        virtual_path = f"{mount_type.value}/{relative_path}".strip("/")
        return matches_path_rules(
            absolute_path,
            relative_path=virtual_path,
            patterns=self._manager.config.sensitivity.never_index,
        )

    def _indexed_snapshot(
        self,
        mount_types: list[MountType],
        *,
        relative_prefixes: Sequence[str] | None = None,
    ) -> dict[str, MountSnapshot]:
        placeholders = ", ".join("?" for _ in mount_types)
        params = [str(self._context_path), *[mount.value for mount in mount_types]]
        prefix_clause = ""
        normalized_prefixes = tuple(
            _normalize_relative_prefix(prefix) for prefix in (relative_prefixes or ())
        )
        if normalized_prefixes:
            predicates: list[str] = []
            for prefix in normalized_prefixes:
                predicates.append(
                    "(relative_path = ? OR substr(relative_path, 1, length(?)) = ?)"
                )
                descendant = f"{prefix}/"
                params.extend((prefix, descendant, descendant))
            prefix_clause = f" AND ({' OR '.join(predicates)})"
        snapshot_state: dict[str, dict[str, Any]] = {
            mount.value: {
                "count": 0,
                "max_modified": None,
                "fingerprint": hashlib.sha256(),
            }
            for mount in mount_types
        }
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT mount_type, relative_path, is_dir, size_bytes, modified_at
                FROM file_index
                WHERE context_path = ?
                  AND mount_type IN ({placeholders})
                  {prefix_clause}
                ORDER BY mount_type ASC, relative_path ASC
                """,
                params,
            )
            for row in rows:
                state = snapshot_state[row["mount_type"]]
                modified = row["modified_at"]
                state["count"] += 1
                if isinstance(modified, str) and (
                    state["max_modified"] is None or modified > state["max_modified"]
                ):
                    state["max_modified"] = modified
                _update_snapshot_fingerprint(
                    state["fingerprint"],
                    relative_path=row["relative_path"],
                    is_dir=bool(row["is_dir"]),
                    size_bytes=int(row["size_bytes"]),
                    modified_at=modified,
                )
        snapshot: dict[str, MountSnapshot] = {}
        for mount in mount_types:
            state = snapshot_state[mount.value]
            snapshot[mount.value] = MountSnapshot(
                count=int(state["count"]),
                max_modified=state["max_modified"],
                fingerprint=state["fingerprint"].hexdigest(),
            )
        return snapshot

    def _assert_scope_matches(self, scoped: ResolvedScope) -> None:
        if scoped.context_root.expanduser().resolve() != self._context_path:
            raise ValueError("index scope belongs to a different context root")
        if scoped.layout_version != detect_layout_version(self._context_path):
            raise ValueError("index scope layout does not match context root")

    def _default_db_path(self) -> Path:
        configured_name = DEFAULT_DB_FILENAME
        try:
            raw_name = self._manager.config.context_index.db_filename
        except Exception:
            raw_name = DEFAULT_DB_FILENAME
        if isinstance(raw_name, str) and raw_name.strip():
            configured_name = raw_name.strip()

        configured_path = Path(configured_name).expanduser()
        layout_version = detect_layout_version(self._context_path)
        if configured_path.is_absolute() and layout_version == LAYOUT_VERSION:
            raise ValueError("v2 context index db_filename must be relative to its global root")
        if configured_path.is_absolute():
            return configured_path.resolve()

        global_root = self._manager.resolve_mount_root(self._context_path, MountType.GLOBAL)
        global_root.mkdir(parents=True, exist_ok=True)
        if layout_version == LAYOUT_VERSION:
            self._managed_v2_db_root = global_root
            return assert_no_linklike_components(
                global_root / configured_path,
                boundary=global_root,
            )
        return (global_root / configured_path).resolve()

    def _assert_managed_db_safe(self, *, require_parent: bool = False) -> None:
        """Reject redirected default v2 database and SQLite sidecar paths."""

        root = self._managed_v2_db_root
        if root is None:
            return
        assert_no_linklike_components(
            self._db_path.parent,
            boundary=self._context_path,
            allow_missing=not require_parent,
        )
        for candidate in (
            self._db_path,
            Path(f"{self._db_path}-journal"),
            Path(f"{self._db_path}-shm"),
            Path(f"{self._db_path}-wal"),
        ):
            assert_no_linklike_components(candidate, boundary=self._context_path)

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.execute("PRAGMA synchronous=NORMAL")
            connection.execute("PRAGMA wal_autocheckpoint=1")
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS file_index (
                    context_path TEXT NOT NULL,
                    mount_type TEXT NOT NULL,
                    relative_path TEXT NOT NULL,
                    absolute_path TEXT NOT NULL,
                    is_dir INTEGER NOT NULL,
                    size_bytes INTEGER NOT NULL,
                    modified_at TEXT,
                    content_text TEXT,
                    content_hash TEXT,
                    indexed_at TEXT NOT NULL,
                    PRIMARY KEY (context_path, mount_type, relative_path)
                )
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_file_index_context_mount
                ON file_index (context_path, mount_type)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_file_index_relative_path
                ON file_index (relative_path)
                """
            )
            connection.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_file_index_modified_at
                ON file_index (modified_at)
                """
            )
            self._fts_enabled = self._initialize_fts(connection)
            self._repair_checkpoint_visibility(connection)
            connection.commit()

    def _initialize_fts(self, connection: sqlite3.Connection) -> bool:
        try:
            connection.execute(
                """
                CREATE VIRTUAL TABLE IF NOT EXISTS file_index_fts
                USING fts5(
                    relative_path,
                    content_text,
                    content='file_index',
                    content_rowid='rowid'
                )
                """
            )
            connection.execute(
                """
                CREATE TRIGGER IF NOT EXISTS file_index_ai
                AFTER INSERT ON file_index
                BEGIN
                    INSERT INTO file_index_fts(rowid, relative_path, content_text)
                    VALUES (new.rowid, new.relative_path, COALESCE(new.content_text, ''));
                END
                """
            )
            connection.execute(
                """
                CREATE TRIGGER IF NOT EXISTS file_index_ad
                AFTER DELETE ON file_index
                BEGIN
                    INSERT INTO file_index_fts(file_index_fts, rowid, relative_path, content_text)
                    VALUES ('delete', old.rowid, old.relative_path, COALESCE(old.content_text, ''));
                END
                """
            )
            connection.execute(
                """
                CREATE TRIGGER IF NOT EXISTS file_index_au
                AFTER UPDATE ON file_index
                BEGIN
                    INSERT INTO file_index_fts(file_index_fts, rowid, relative_path, content_text)
                    VALUES ('delete', old.rowid, old.relative_path, COALESCE(old.content_text, ''));
                    INSERT INTO file_index_fts(rowid, relative_path, content_text)
                    VALUES (new.rowid, new.relative_path, COALESCE(new.content_text, ''));
                END
                """
            )

            row_count = connection.execute("SELECT COUNT(1) FROM file_index").fetchone()[0]
            fts_count = connection.execute("SELECT COUNT(1) FROM file_index_fts").fetchone()[0]
            if row_count > 0 and fts_count == 0:
                connection.execute("INSERT INTO file_index_fts(file_index_fts) VALUES ('rebuild')")
            return True
        except sqlite3.OperationalError:
            return False

    def _repair_checkpoint_visibility(self, connection: sqlite3.Connection) -> None:
        if not self._fts_enabled:
            return
        row_count = connection.execute("SELECT COUNT(1) FROM file_index").fetchone()[0]
        fts_count = connection.execute("SELECT COUNT(1) FROM file_index_fts").fetchone()[0]
        if row_count != 0 or fts_count == 0:
            return
        self._checkpoint_wal(connection, truncate=True)

    def _delete_rows_for_mounts(
        self,
        mount_types: list[MountType],
        *,
        connection: sqlite3.Connection | None = None,
    ) -> int:
        if not mount_types:
            return 0
        placeholders = ", ".join("?" for _ in mount_types)
        params = [str(self._context_path), *[mount.value for mount in mount_types]]
        if connection is None:
            with self._connect() as owned_connection:
                cursor = owned_connection.execute(
                    f"""
                    DELETE FROM file_index
                    WHERE context_path = ?
                      AND mount_type IN ({placeholders})
                    """,
                    params,
                )
                self._commit_mutation(owned_connection)
                if cursor.rowcount is None or cursor.rowcount < 0:
                    return 0
                return int(cursor.rowcount)
        cursor = connection.execute(
            f"""
            DELETE FROM file_index
            WHERE context_path = ?
              AND mount_type IN ({placeholders})
            """,
            params,
        )
        if cursor.rowcount is None or cursor.rowcount < 0:
            return 0
        return int(cursor.rowcount)

    def _delete_rows_for_scoped_prefixes(
        self,
        prefixes_by_mount: dict[MountType, tuple[str, ...]],
        *,
        connection: sqlite3.Connection,
    ) -> int:
        deleted = 0
        for mount_type, prefixes in prefixes_by_mount.items():
            normalized = tuple(
                _normalize_relative_prefix(prefix) for prefix in prefixes if prefix
            )
            if not normalized:
                continue
            predicates: list[str] = []
            params: list[Any] = [str(self._context_path), mount_type.value]
            for prefix in normalized:
                predicates.append(
                    "(relative_path = ? OR substr(relative_path, 1, length(?)) = ?)"
                )
                descendant = f"{prefix}/"
                params.extend((prefix, descendant, descendant))
            cursor = connection.execute(
                f"""
                DELETE FROM file_index
                WHERE context_path = ?
                  AND mount_type = ?
                  AND ({' OR '.join(predicates)})
                """,
                params,
            )
            if cursor.rowcount is not None and cursor.rowcount >= 0:
                deleted += int(cursor.rowcount)
        return deleted

    def _commit_mutation(
        self,
        connection: sqlite3.Connection,
        *,
        truncate_wal: bool = False,
    ) -> None:
        connection.commit()
        self._checkpoint_wal(connection, truncate=truncate_wal)

    def _checkpoint_wal(
        self,
        connection: sqlite3.Connection,
        *,
        truncate: bool = False,
    ) -> None:
        mode = "TRUNCATE" if truncate else "PASSIVE"
        try:
            connection.execute(f"PRAGMA wal_checkpoint({mode})").fetchone()
        except sqlite3.OperationalError:
            pass

    def _connect(self) -> sqlite3.Connection:
        self._assert_managed_db_safe(require_parent=True)
        connection = sqlite3.connect(self._db_path, timeout=5.0)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout=5000")
        return connection

    @staticmethod
    def _normalize_mount_types(mount_types: list[MountType] | None) -> list[MountType]:
        source = mount_types or list(MountType)
        normalized: list[MountType] = []
        seen: set[MountType] = set()
        for mount_type in source:
            if mount_type in seen:
                continue
            seen.add(mount_type)
            normalized.append(mount_type)
        return normalized

    @classmethod
    def _normalize_scoped_mount_types(
        cls,
        scoped: ResolvedScope,
        mount_types: list[MountType] | None,
    ) -> list[MountType]:
        selected = cls._normalize_mount_types(mount_types)
        if scoped.layout_version != LAYOUT_VERSION:
            return selected
        return [
            mount_type
            for mount_type in selected
            if ContextCategory.from_mount_type(mount_type) is not None
        ]


INDEX_SCAN_SKIP_NAMES = {
    "__pycache__",
    "node_modules",
    ".venv",
    "venv",
    ".tox",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    ".egg-info",
    "CMakeFiles",
    "cmake-build-debug",
    "cmake-build-release",
    ".cmake",
    ".git",
    ".hg",
    ".svn",
    "dist",
    ".DS_Store",
}


def _is_text_candidate(path: Path) -> bool:
    suffix = path.suffix.lower()
    if suffix in _TEXT_SUFFIXES:
        return True
    return suffix == ""


def _iter_mount_entries(mount_root: Path):
    try:
        children = sorted(mount_root.iterdir(), key=lambda item: item.name)
    except OSError:
        return

    for child in children:
        if child.name in INDEX_SCAN_SKIP_NAMES:
            continue
        yield child, child.relative_to(mount_root).as_posix()

        if not child.is_dir():
            continue

        if child.is_symlink():
            try:
                scan_root = child.resolve()
            except OSError:
                continue
            prefix = child.name
        else:
            scan_root = child
            prefix = ""

        if not scan_root.exists() or not scan_root.is_dir():
            continue

        for nested in _walk_skipping(scan_root):
            nested_relative = nested.relative_to(scan_root).as_posix()
            if not nested_relative:
                continue
            if prefix:
                yield nested, f"{prefix}/{nested_relative}"
            else:
                yield nested, nested.relative_to(mount_root).as_posix()


def _iter_mount_entries_with_prefix(root: Path, prefix: str):
    """Yield lexical descendants without following any link-like directory."""

    for entry in _walk_skipping(root):
        relative = entry.relative_to(root).as_posix()
        yield entry, f"{prefix}/{relative}" if prefix else relative


def _walk_skipping(root: Path):
    """Walk directory tree, skipping known junk directories."""
    try:
        children = sorted(root.iterdir(), key=lambda item: item.name)
    except OSError:
        return
    for child in children:
        if child.name in INDEX_SCAN_SKIP_NAMES:
            continue
        yield child
        if not child.is_symlink() and child.is_dir():
            yield from _walk_skipping(child)


def count_mount_files(mount_root: Path) -> int:
    """Count file entries in a mount, following symlinked directories."""
    total = 0
    for entry, _relative_path in _iter_mount_entries(mount_root):
        try:
            if entry.is_file():
                total += 1
        except OSError:
            continue
    return total


def _read_text_for_indexing(
    path: Path,
    *,
    size_bytes: int,
    max_file_size_bytes: int,
    max_content_chars: int,
) -> tuple[str | None, str | None, str | None]:
    if not _is_text_candidate(path):
        return None, None, None
    if size_bytes > max_file_size_bytes:
        return None, None, "too_large"
    try:
        raw = path.read_bytes()
    except OSError:
        return None, None, None
    if b"\x00" in raw[:4096]:
        return None, None, "binary"
    text = raw.decode("utf-8", errors="replace")
    if max_content_chars > 0:
        text = text[:max_content_chars]
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    return text, digest, None


def _build_excerpt(content: str, query: str) -> str | None:
    if not content:
        return None
    compact = " ".join(content.split())
    if not compact:
        return None
    if not query:
        return compact[:220]
    lowered = compact.lower()
    needle = query.lower()
    index = lowered.find(needle)
    if index < 0:
        return compact[:220]
    start = max(0, index - 90)
    end = min(len(compact), index + len(needle) + 130)
    return compact[start:end]


def _build_fts_query(query: str) -> str | None:
    tokens = [token.lower() for token in re.findall(r"[A-Za-z0-9_./-]+", query)]
    if not tokens:
        return None
    return " AND ".join(f'"{token}"' for token in tokens)


def _normalize_relative_path(path: str) -> str:
    candidate = path.strip().replace("\\", "/")
    while candidate.startswith("./"):
        candidate = candidate[2:]
    candidate = candidate.lstrip("/")
    if not candidate:
        return ""

    raw_parts = [part for part in candidate.split("/") if part and part != "."]
    if any(part == ".." for part in raw_parts):
        return ""
    return "/".join(raw_parts)


def _normalize_relative_prefix(path: str | None) -> str:
    if path is None:
        return ""
    return _normalize_relative_path(path)


_NO_VISIBLE_SCOPE_PREFIX = ".afs-no-visible-scope"


def _effective_relative_prefixes(
    requested: Sequence[str] | None,
    *,
    scoped: ResolvedScope | None,
) -> tuple[str, ...]:
    """Intersect optional caller prefixes with the requester's v2 scope."""

    requested_values = tuple(requested or ())
    normalized_requested = tuple(
        prefix
        for prefix in (
            _normalize_relative_prefix(str(item)) for item in requested_values
        )
        if prefix
    )
    if scoped is None or scoped.layout_version != LAYOUT_VERSION:
        return normalized_requested

    authorized = tuple(
        prefix
        for prefix in (
            _normalize_relative_prefix(item) for item in visible_scope_prefixes(scoped)
        )
        if prefix
    )
    if not requested_values:
        return authorized
    if not normalized_requested:
        return (_NO_VISIBLE_SCOPE_PREFIX,)

    intersections: list[str] = []
    for requested_prefix in normalized_requested:
        for authorized_prefix in authorized:
            if _relative_path_matches_prefixes(requested_prefix, (authorized_prefix,)):
                intersections.append(requested_prefix)
            elif _relative_path_matches_prefixes(authorized_prefix, (requested_prefix,)):
                intersections.append(authorized_prefix)
    return tuple(dict.fromkeys(intersections)) or (_NO_VISIBLE_SCOPE_PREFIX,)


def _relative_path_matches_prefixes(path: str, prefixes: Sequence[str]) -> bool:
    """Return true when a path is exactly at or below one lexical prefix."""

    if not prefixes:
        return True
    normalized_path = _normalize_relative_path(path)
    return any(
        normalized_path == prefix or normalized_path.startswith(f"{prefix}/")
        for prefix in prefixes
    )


def _relative_prefix_sql(
    prefixes: Sequence[str],
    *,
    leading_and: bool = False,
) -> tuple[str, list[str]]:
    """Build an exact-or-descendant SQLite predicate for lexical prefixes."""

    normalized = tuple(
        prefix for prefix in (_normalize_relative_prefix(item) for item in prefixes) if prefix
    )
    if not normalized:
        return "", []
    predicates: list[str] = []
    params: list[str] = []
    for prefix in normalized:
        descendant = f"{prefix}/"
        predicates.append(
            "(relative_path = ? OR substr(relative_path, 1, length(?)) = ?)"
        )
        params.extend((prefix, descendant, descendant))
    clause = f"({' OR '.join(predicates)})"
    return (f" AND {clause}" if leading_and else clause), params


def _empty_mount_snapshot() -> MountSnapshot:
    return MountSnapshot(count=0, max_modified=None, fingerprint=hashlib.sha256(b"").hexdigest())


def _update_snapshot_fingerprint(
    digest: hashlib._Hash,
    *,
    relative_path: str,
    is_dir: bool,
    size_bytes: int,
    modified_at: str | None,
) -> None:
    entry_type = "d" if is_dir else "f"
    digest.update(relative_path.encode("utf-8", errors="replace"))
    digest.update(b"\0")
    digest.update(entry_type.encode("ascii"))
    digest.update(b"\0")
    digest.update(str(size_bytes).encode("ascii"))
    digest.update(b"\0")
    digest.update((modified_at or "").encode("utf-8", errors="replace"))
    digest.update(b"\n")


def _iso_utc(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()
