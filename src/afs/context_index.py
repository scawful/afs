"""SQLite-backed context indexing and query helpers."""

from __future__ import annotations

import hashlib
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .manager import AFSManager
from .models import MountType
from .sensitivity import matches_path_rules

DEFAULT_DB_FILENAME = "context_index.sqlite3"
DEFAULT_MAX_FILE_SIZE_BYTES = 256 * 1024
DEFAULT_MAX_CONTENT_CHARS = 12000

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
        self._db_path = db_path.expanduser().resolve() if db_path else self._default_db_path()
        self._db_path.parent.mkdir(parents=True, exist_ok=True)
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
        include_content: bool = True,
        max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE_BYTES,
        max_content_chars: int = DEFAULT_MAX_CONTENT_CHARS,
    ) -> bool:
        inferred = self.infer_relative_for_absolute_path(absolute_path)
        if not inferred:
            return False

        mount_type, relative_path = inferred
        candidate = absolute_path.expanduser().resolve()
        if candidate.exists():
            if candidate.is_dir():
                self.rebuild(
                    mount_types=[mount_type],
                    include_content=include_content,
                    max_file_size_bytes=max_file_size_bytes,
                    max_content_chars=max_content_chars,
                )
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
                    OR relative_path LIKE ?
                  )
                """,
                (
                    str(self._context_path),
                    mount_type.value,
                    normalized,
                    f"{normalized}/%",
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

    def diff(
        self,
        *,
        mount_types: list[MountType] | None = None,
    ) -> dict[str, Any]:
        """Compare filesystem state against the index.

        Returns lists of new, modified, and deleted paths per mount type.
        """
        selected_mounts = self._normalize_mount_types(mount_types)
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
            for entry, relative_path in _iter_mount_entries(mount_root):
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
                rows = connection.execute(
                    "SELECT relative_path, size_bytes, modified_at "
                    "FROM file_index WHERE context_path = ? AND mount_type = ?",
                    (str(self._context_path), mount_type.value),
                ).fetchall()
                for row in rows:
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
    ) -> dict[str, Any]:
        """Compute per-file freshness scores based on index vs filesystem state.

        Score = max(0, 1 - age_seconds / decay_seconds) for indexed files.
        Modified or deleted files get 0.0.
        """
        selected_mounts = self._normalize_mount_types(mount_types)
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
                for entry, relative_path in _iter_mount_entries(mount_root):
                    if self._should_skip_relative_path(
                        mount_type,
                        relative_path,
                        entry=entry,
                    ):
                        continue
                    if not entry.is_file():
                        continue
                    fs_entries[relative_path] = entry

                rows = connection.execute(
                    """
                    SELECT relative_path, modified_at, absolute_path
                    FROM file_index
                    WHERE context_path = ? AND mount_type = ? AND is_dir = 0
                    """,
                    (str(self._context_path), mount_key),
                ).fetchall()

                file_entries: list[dict[str, Any]] = []
                score_sum = 0.0
                count = 0
                seen_paths: set[str] = set()

                for rel_path, indexed_modified_at, abs_path in rows:
                    seen_paths.add(rel_path)
                    entry_path = Path(abs_path) if abs_path else mount_root / rel_path
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

        rows = []
        fts_query = _build_fts_query(normalized_query) if normalized_query else None
        if normalized_query and self._fts_enabled and fts_query:
            rows = self._query_with_fts(
                fts_query=fts_query,
                mount_types=selected_mounts,
                relative_prefix=normalized_prefix,
                limit=limit_value,
            )

        if not rows:
            rows = self._query_with_like(
                query=normalized_query,
                mount_types=selected_mounts,
                relative_prefix=normalized_prefix,
                limit=limit_value,
            )

        return [
            self._row_to_payload(
                row=row,
                query=normalized_query,
                include_content=include_content,
            )
            for row in rows
        ]

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
            where.append("fi.relative_path LIKE ?")
            params.append(f"{relative_prefix}%")

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
            where.append("relative_path LIKE ?")
            params.append(f"{relative_prefix}%")

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
    ) -> dict[str, MountSnapshot]:
        placeholders = ", ".join("?" for _ in mount_types)
        params = [str(self._context_path), *[mount.value for mount in mount_types]]
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

    def _default_db_path(self) -> Path:
        configured_name = DEFAULT_DB_FILENAME
        try:
            raw_name = self._manager.config.context_index.db_filename
        except Exception:
            raw_name = DEFAULT_DB_FILENAME
        if isinstance(raw_name, str) and raw_name.strip():
            configured_name = raw_name.strip()

        configured_path = Path(configured_name).expanduser()
        if configured_path.is_absolute():
            return configured_path.resolve()

        global_root = self._manager.resolve_mount_root(self._context_path, MountType.GLOBAL)
        global_root.mkdir(parents=True, exist_ok=True)
        return (global_root / configured_path).resolve()

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
        if child.name == ".keep":
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

        for nested in scan_root.rglob("*"):
            if nested.name == ".keep":
                continue
            nested_relative = nested.relative_to(scan_root).as_posix()
            if not nested_relative:
                continue
            if prefix:
                yield nested, f"{prefix}/{nested_relative}"
            else:
                yield nested, nested.relative_to(mount_root).as_posix()


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
