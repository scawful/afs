from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

from afs.history import reap_scratchpad, reap_storage
from afs.schema import HistoryConfig, ScratchpadConfig

TODAY = date(2026, 11, 14)


def _touch_aged(path: Path, day: date) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        if path.suffix or path.name.endswith(".md"):
            path.write_text("x\n", encoding="utf-8")
        else:
            path.mkdir(parents=True, exist_ok=True)
            (path / "note.md").write_text("x\n", encoding="utf-8")
    stamp = datetime(day.year, day.month, day.day, tzinfo=timezone.utc).timestamp()
    path.touch()
    import os

    os.utime(path, (stamp, stamp))
    if path.is_dir():
        for child in path.rglob("*"):
            os.utime(child, (stamp, stamp))


def test_history_config_retention_defaults() -> None:
    cfg = HistoryConfig()
    assert cfg.retention_days == 30
    assert cfg.gzip_older is True
    parsed = HistoryConfig.from_dict({"retention_days": 7, "gzip_older": False})
    assert parsed.retention_days == 7
    assert parsed.gzip_older is False


def test_scratchpad_config_protect_defaults() -> None:
    cfg = ScratchpadConfig()
    assert cfg.retention_days == 14
    assert "state.md" in cfg.protect
    assert "handoffs" in cfg.protect


def test_reap_scratchpad_dry_run_and_apply(tmp_path: Path) -> None:
    scratchpad = tmp_path / "scratchpad"
    scratchpad.mkdir()
    (scratchpad / "state.md").write_text("keep\n", encoding="utf-8")
    (scratchpad / "deferred.md").write_text("keep\n", encoding="utf-8")
    (scratchpad / "handoffs").mkdir()
    (scratchpad / "handoffs" / "active.md").write_text("keep\n", encoding="utf-8")

    old_agent = scratchpad / "afs_agents" / "session_bootstrap.json"
    _touch_aged(old_agent, date(2026, 10, 1))
    recent_agent = scratchpad / "afs_agents" / "fresh.json"
    _touch_aged(recent_agent, date(2026, 11, 10))

    old_session = scratchpad / "codex-afs-20261001"
    _touch_aged(old_session, date(2026, 10, 1))
    recent_session = scratchpad / "codex-afs-20261110"
    _touch_aged(recent_session, date(2026, 11, 10))

    dry = reap_scratchpad(scratchpad, keep_days=14, dry_run=True, today=TODAY)
    dry_paths = {entry.path for entry in dry if entry.skipped is None}
    assert old_agent in dry_paths
    assert old_session in dry_paths
    assert recent_agent not in dry_paths
    assert recent_session not in dry_paths
    assert old_agent.exists() and old_session.exists()

    applied = reap_scratchpad(scratchpad, keep_days=14, dry_run=False, today=TODAY)
    assert all(entry.skipped is None for entry in applied)
    assert not old_agent.exists() and not old_session.exists()
    assert recent_agent.exists() and recent_session.exists()
    assert (scratchpad / "state.md").exists()
    assert (scratchpad / "handoffs" / "active.md").exists()


def test_reap_storage_combines_history_and_scratchpad(tmp_path: Path) -> None:
    history = tmp_path / "history"
    history.mkdir()
    old = history / "events_20260901.jsonl"
    old.write_text('{"id":"old"}\n', encoding="utf-8")
    recent = history / "events_20261110.jsonl"
    recent.write_text('{"id":"new"}\n', encoding="utf-8")

    scratchpad = tmp_path / "scratchpad"
    scratchpad.mkdir()
    (scratchpad / "state.md").write_text("keep\n", encoding="utf-8")
    stale = scratchpad / "afs_agents" / "stale.md"
    _touch_aged(stale, date(2026, 9, 1))

    report = reap_storage(
        history_root=history,
        scratchpad_root=scratchpad,
        history_keep_days=30,
        scratchpad_keep_days=14,
        dry_run=False,
        today=TODAY,
    )
    assert report.history is not None
    assert not old.exists()
    assert (history / "events_20260901.jsonl.gz").exists()
    assert recent.exists()
    assert not stale.exists()
    assert (scratchpad / "state.md").exists()
