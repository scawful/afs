from __future__ import annotations

import gzip
from datetime import date
from pathlib import Path

import pytest

from afs.context_layout import scaffold_v2
from afs.history import archive_history_events, iter_history_events

TODAY = date(2026, 10, 27)


def _write_day(root: Path, stamp: str, body: str | None = None) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"events_{stamp}.jsonl"
    path.write_text(body or f'{{"id": "{stamp}", "type": "cli"}}\n', encoding="utf-8")
    return path


def test_dry_run_reports_without_writing(tmp_path: Path) -> None:
    history = tmp_path / "history"
    old = _write_day(history, "20260901")
    recent = _write_day(history, "20261020")

    report = archive_history_events(history, keep_days=30, dry_run=True, today=TODAY)

    assert [entry.path for entry in report.entries] == [old]
    assert report.bytes_before == old.stat().st_size
    assert report.bytes_after == 0
    assert old.exists() and recent.exists()
    assert not list(history.glob("*.gz"))


def test_apply_gzips_old_days_and_keeps_window(tmp_path: Path) -> None:
    history = tmp_path / "history"
    body = "".join(f'{{"id": "{n}", "type": "cli"}}\n' for n in range(200))
    old = _write_day(history, "20260901", body)
    boundary = _write_day(history, "20260927")
    kept = _write_day(history, "20260928")
    (history / "events_notes.jsonl").write_text("{}\n", encoding="utf-8")

    report = archive_history_events(history, keep_days=30, today=TODAY)

    assert {entry.path.name for entry in report.entries} == {old.name, boundary.name}
    assert all(entry.skipped is None for entry in report.entries)
    assert not old.exists() and not boundary.exists() and kept.exists()
    assert (history / "events_notes.jsonl").exists()
    with gzip.open(history / "events_20260901.jsonl.gz", "rt", encoding="utf-8") as handle:
        assert handle.read() == body
    assert report.bytes_after < report.bytes_before
    ids = {event.get("id") for event in iter_history_events(history)} - {None}
    assert ids == {"20260928"}


def test_existing_archive_is_never_overwritten(tmp_path: Path) -> None:
    history = tmp_path / "history"
    old = _write_day(history, "20260901")
    existing = history / "events_20260901.jsonl.gz"
    existing.write_bytes(b"prior archive")

    report = archive_history_events(history, keep_days=30, today=TODAY)

    assert report.entries[0].skipped is not None
    assert old.exists()
    assert existing.read_bytes() == b"prior archive"


def test_keep_days_must_be_positive(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        archive_history_events(tmp_path, keep_days=0, today=TODAY)


def test_v2_archives_legacy_and_common_ledgers(tmp_path: Path) -> None:
    context_root = tmp_path / "context"
    scaffold_v2(context_root)
    legacy = _write_day(context_root / "history", "20260901")
    common = _write_day(context_root / "history" / "common", "20260901")

    report = archive_history_events(context_root / "history" / "common", today=TODAY)

    assert [entry.path for entry in report.entries] == [legacy, common]
    assert not legacy.exists() and not common.exists()
    assert (context_root / "history" / "events_20260901.jsonl.gz").exists()
    assert (context_root / "history" / "common" / "events_20260901.jsonl.gz").exists()
