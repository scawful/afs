from __future__ import annotations

import os
import sqlite3
from pathlib import Path

import pytest

from afs.context_index import ContextSQLiteIndex
from afs.context_layout import scaffold_v2
from afs.manager import AFSManager
from afs.models import MountType
from afs.project_registry import ProjectRegistry
from afs.schema import AFSConfig, GeneralConfig
from afs.scopes import resolve_scope


def _make_index(tmp_path: Path) -> tuple[ContextSQLiteIndex, Path]:
    context_root = tmp_path / "context"
    scratchpad = context_root / "scratchpad"
    scratchpad.mkdir(parents=True)
    manager = AFSManager(config=AFSConfig(general=GeneralConfig(context_root=context_root)))
    index = ContextSQLiteIndex(manager, context_root, db_path=tmp_path / "index.sqlite3")
    return index, scratchpad


def _fts_consistent(db_path: Path) -> None:
    with sqlite3.connect(db_path) as connection:
        # rank=1 compares the FTS index against the external content table.
        connection.execute(
            "INSERT INTO file_index_fts(file_index_fts, rank) VALUES ('integrity-check', 1)"
        )


def _query_paths(index: ContextSQLiteIndex, text: str) -> set[str]:
    rows = index.query(query=text, mount_types=[MountType.SCRATCHPAD])
    return {row["relative_path"] for row in rows}


def test_rebuild_without_changes_writes_nothing(tmp_path: Path) -> None:
    index, scratchpad = _make_index(tmp_path)
    for name in ("a.md", "b.md", "c.md"):
        (scratchpad / name).write_text(f"{name} body", encoding="utf-8")

    first = index.rebuild(mount_types=[MountType.SCRATCHPAD])
    second = index.rebuild(mount_types=[MountType.SCRATCHPAD])

    assert first.rows_inserted == first.rows_written >= 3
    assert second.rows_written == first.rows_written
    assert (second.rows_inserted, second.rows_updated, second.rows_deleted) == (0, 0, 0)
    _fts_consistent(index.db_path)


def test_rebuild_applies_only_changed_rows(tmp_path: Path) -> None:
    index, scratchpad = _make_index(tmp_path)
    (scratchpad / "keep.md").write_text("steady", encoding="utf-8")
    (scratchpad / "edit.md").write_text("oldtoken", encoding="utf-8")
    (scratchpad / "gone.md").write_text("doomed", encoding="utf-8")
    index.rebuild(mount_types=[MountType.SCRATCHPAD])

    (scratchpad / "edit.md").write_text("newtoken and more", encoding="utf-8")
    (scratchpad / "gone.md").unlink()
    (scratchpad / "fresh.md").write_text("brandnew", encoding="utf-8")
    summary = index.rebuild(mount_types=[MountType.SCRATCHPAD])

    assert (summary.rows_inserted, summary.rows_updated, summary.rows_deleted) == (1, 1, 1)
    assert _query_paths(index, "newtoken") == {"edit.md"}
    assert _query_paths(index, "oldtoken") == set()
    assert _query_paths(index, "doomed") == set()
    assert _query_paths(index, "brandnew") == {"fresh.md"}
    _fts_consistent(index.db_path)


def test_upsert_relative_path_does_not_orphan_fts_rows(tmp_path: Path) -> None:
    index, scratchpad = _make_index(tmp_path)
    note = scratchpad / "note.md"
    note.write_text("firsttoken", encoding="utf-8")
    assert index.upsert_relative_path(MountType.SCRATCHPAD, "note.md")

    note.write_text("secondtoken", encoding="utf-8")
    assert index.upsert_relative_path(MountType.SCRATCHPAD, "note.md")

    assert _query_paths(index, "firsttoken") == set()
    assert _query_paths(index, "secondtoken") == {"note.md"}
    _fts_consistent(index.db_path)


def test_vacuum_keeps_fts_rowids_aligned(tmp_path: Path) -> None:
    index, scratchpad = _make_index(tmp_path)
    for number in range(20):
        (scratchpad / f"n{number:02d}.md").write_text(f"marker{number:02d}", encoding="utf-8")
    index.rebuild(mount_types=[MountType.SCRATCHPAD])
    for number in range(0, 20, 2):
        (scratchpad / f"n{number:02d}.md").unlink()
    index.rebuild(mount_types=[MountType.SCRATCHPAD])

    with sqlite3.connect(index.db_path) as connection:
        connection.execute("VACUUM")

    assert _query_paths(index, "marker07") == {"n07.md"}
    _fts_consistent(index.db_path)


def test_legacy_implicit_rowid_schema_is_migrated(tmp_path: Path) -> None:
    db_path = tmp_path / "index.sqlite3"
    context_root = (tmp_path / "context").resolve()
    with sqlite3.connect(db_path) as connection:
        connection.executescript(
            """
            CREATE TABLE file_index (
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
            );
            CREATE VIRTUAL TABLE file_index_fts USING fts5(
                relative_path, content_text, content='file_index', content_rowid='rowid'
            );
            CREATE TRIGGER file_index_ai AFTER INSERT ON file_index BEGIN
                INSERT INTO file_index_fts(rowid, relative_path, content_text)
                VALUES (new.rowid, new.relative_path, COALESCE(new.content_text, ''));
            END;
            """
        )
        connection.execute(
            "INSERT INTO file_index VALUES (?, 'scratchpad', 'legacy.md', ?, 0, 9, NULL, "
            "'legacytoken', 'hash', '2026-01-01T00:00:00+00:00')",
            (str(context_root), str(context_root / "scratchpad" / "legacy.md")),
        )

    index, _ = _make_index(tmp_path)

    with sqlite3.connect(db_path) as connection:
        columns = {row[1]: row[5] for row in connection.execute("PRAGMA table_info(file_index)")}
        tables = {
            row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type='table'")
        }
    assert columns["id"] == 1
    assert "file_index_legacy" not in tables
    assert _query_paths(index, "legacytoken") == {"legacy.md"}
    _fts_consistent(db_path)


def _seed_dash_slash_siblings(root: Path) -> None:
    # "-" (0x2D) sorts before "/" (0x2F): a depth-first walk yields
    # dimensions/child.md before dimensions-local.md, SQLite the reverse.
    (root / "dimensions").mkdir(parents=True)
    (root / "dimensions" / "child.md").write_text("child", encoding="utf-8")
    (root / "dimensions-local.md").write_text("sibling", encoding="utf-8")


def test_needs_refresh_is_false_after_rebuild_with_dash_slash_siblings(
    tmp_path: Path,
) -> None:
    index, scratchpad = _make_index(tmp_path)
    _seed_dash_slash_siblings(scratchpad)

    index.rebuild(mount_types=[MountType.SCRATCHPAD])

    assert index.needs_refresh(mount_types=[MountType.SCRATCHPAD]) is False
    (scratchpad / "dimensions" / "child.md").write_text("child edited", encoding="utf-8")
    assert index.needs_refresh(mount_types=[MountType.SCRATCHPAD]) is True


def test_needs_refresh_scoped_is_false_after_rebuild_with_dash_slash_siblings(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "AFS_ALLOWED_MOUNTS",
        "memory,history,scratchpad,hivemind,knowledge,tools,global,items,monorepo",
    )
    context = tmp_path / ".context"
    alpha = tmp_path / "alpha"
    alpha.mkdir()
    scaffold_v2(context)
    record = ProjectRegistry(context).register(alpha)
    _seed_dash_slash_siblings(context / "knowledge" / "projects" / record.project_id)
    _seed_dash_slash_siblings(context / "knowledge" / "common")
    manager = AFSManager(config=AFSConfig(general=GeneralConfig(context_root=context)))
    index = ContextSQLiteIndex(manager, context, db_path=tmp_path / "index.sqlite3")
    scope = resolve_scope(context, requester_path=alpha)

    index.rebuild_scoped(scope, mount_types=[MountType.KNOWLEDGE])

    assert index.needs_refresh_scoped(scope, mount_types=[MountType.KNOWLEDGE]) is False


def _append_with_later_mtime(path: Path, text: str) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(text)
    later = path.stat().st_mtime + 60
    os.utime(path, (later, later))


def test_stale_mounts_lists_only_drifted_mounts(tmp_path: Path) -> None:
    index, scratchpad = _make_index(tmp_path)
    memory = scratchpad.parent / "memory"
    memory.mkdir()
    (scratchpad / "a.md").write_text("a", encoding="utf-8")
    (memory / "b.md").write_text("b", encoding="utf-8")
    mounts = [MountType.SCRATCHPAD, MountType.MEMORY]
    index.rebuild(mount_types=mounts)

    assert index.stale_mounts(mount_types=mounts) == []
    assert index.needs_refresh(mount_types=mounts) is False
    (memory / "b.md").write_text("b edited", encoding="utf-8")
    assert index.stale_mounts(mount_types=mounts) == [MountType.MEMORY]
    assert index.needs_refresh(mount_types=mounts) is True


def test_history_event_log_appends_do_not_mark_history_stale(tmp_path: Path) -> None:
    index, scratchpad = _make_index(tmp_path)
    history = scratchpad.parent / "history"
    history.mkdir()
    events = history / "events_20261028.jsonl"
    events.write_text('{"op": "invoke"}\n', encoding="utf-8")
    index.rebuild(mount_types=[MountType.HISTORY])

    _append_with_later_mtime(events, '{"op": "call"}\n')

    assert index.stale_mounts(mount_types=[MountType.HISTORY]) == []
    (history / "events_20261029.jsonl").write_text("{}\n", encoding="utf-8")
    assert index.stale_mounts(mount_types=[MountType.HISTORY]) == [MountType.HISTORY]


def test_presence_only_rule_covers_only_history_event_logs(tmp_path: Path) -> None:
    index, scratchpad = _make_index(tmp_path)
    history = scratchpad.parent / "history"
    history.mkdir()
    notes = history / "notes.jsonl"
    notes.write_text("{}\n", encoding="utf-8")
    lookalike = scratchpad / "events_20261028.jsonl"
    lookalike.write_text("{}\n", encoding="utf-8")
    mounts = [MountType.SCRATCHPAD, MountType.HISTORY]
    index.rebuild(mount_types=mounts)

    _append_with_later_mtime(notes, "{}\n")
    _append_with_later_mtime(lookalike, "{}\n")

    assert index.stale_mounts(mount_types=mounts) == mounts


def test_scoped_history_event_log_appends_do_not_mark_history_stale(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "AFS_ALLOWED_MOUNTS",
        "memory,history,scratchpad,hivemind,knowledge,tools,global,items,monorepo",
    )
    context = tmp_path / ".context"
    alpha = tmp_path / "alpha"
    alpha.mkdir()
    scaffold_v2(context)
    ProjectRegistry(context).register(alpha)
    events = context / "history" / "common" / "events_20261028.jsonl"
    events.parent.mkdir(parents=True, exist_ok=True)
    events.write_text('{"op": "invoke"}\n', encoding="utf-8")
    guide = context / "knowledge" / "common" / "guide.md"
    guide.parent.mkdir(parents=True, exist_ok=True)
    guide.write_text("guide", encoding="utf-8")
    manager = AFSManager(config=AFSConfig(general=GeneralConfig(context_root=context)))
    index = ContextSQLiteIndex(manager, context, db_path=tmp_path / "index.sqlite3")
    scope = resolve_scope(context, requester_path=alpha)
    mounts = [MountType.HISTORY, MountType.KNOWLEDGE]
    index.rebuild_scoped(scope, mount_types=mounts)

    _append_with_later_mtime(events, '{"op": "call"}\n')

    assert index.stale_mounts_scoped(scope, mount_types=mounts) == []
    assert index.needs_refresh_scoped(scope, mount_types=mounts) is False
    guide.write_text("guide edited", encoding="utf-8")
    assert index.stale_mounts_scoped(scope, mount_types=mounts) == [MountType.KNOWLEDGE]
