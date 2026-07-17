"""Behavioral tests for the conservative one-shot default agents."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

from afs.agents import briefing_agent, default_context_warm, index_rebuild, skills_mine
from afs.context_index import IndexSummary
from afs.context_layout import scaffold_v2
from afs.models import MountType
from afs.schema import (
    AFSConfig,
    DirectoryConfig,
    GeneralConfig,
    PolicyType,
)


def _config(context_root: Path, *, scratchpad_name: str = "scratchpad") -> AFSConfig:
    return AFSConfig(
        general=GeneralConfig(context_root=context_root),
        directories=[
            DirectoryConfig(
                name=scratchpad_name,
                policy=PolicyType.WRITABLE,
                role=MountType.SCRATCHPAD,
            )
        ],
    )


def test_default_context_warm_forwards_safe_background_flags(monkeypatch) -> None:
    context_warm_main = MagicMock(return_value=0)
    monkeypatch.setattr(default_context_warm.context_warm, "main", context_warm_main)

    assert default_context_warm.main(["--stdout"]) == 0
    context_warm_main.assert_called_once_with(
        [
            "--skip-workspace-sync",
            "--skip-embeddings",
            "--max-contexts",
            "100",
            "--stdout",
        ]
    )


def test_index_rebuild_limits_mounts_and_reports_summary_errors(
    tmp_path: Path,
    monkeypatch,
) -> None:
    context_root = tmp_path / "context"
    context_root.mkdir()
    config = _config(context_root)
    selected_mounts = [MountType.KNOWLEDGE, MountType.MEMORY]
    calls: list[tuple[str, list[MountType] | None]] = []

    class FakeIndex:
        def has_entries(self, *, mount_types=None):
            calls.append(("has_entries", mount_types))
            return True

        def needs_refresh(self, *, mount_types=None):
            calls.append(("needs_refresh", mount_types))
            return True

        def rebuild(self, *, mount_types=None, **_kwargs):
            calls.append(("rebuild", mount_types))
            return IndexSummary(
                context_path=str(context_root),
                db_path=str(context_root / "global" / "index.sqlite3"),
                indexed_at="2026-07-15T00:00:00+00:00",
                rows_written=1,
                rows_deleted=0,
                by_mount_type={"knowledge": 1, "memory": 0},
                skipped_large_files=0,
                skipped_binary_files=0,
                errors=["knowledge: unreadable mount"],
            )

    emitted = []
    monkeypatch.setattr(index_rebuild, "load_agent_config", lambda _path: config)
    monkeypatch.setattr(index_rebuild, "ContextSQLiteIndex", lambda *_args: FakeIndex())
    monkeypatch.setattr(
        index_rebuild,
        "emit_result",
        lambda result, **_kwargs: emitted.append(result),
    )

    assert index_rebuild.main([]) == 1
    assert calls == [
        ("has_entries", selected_mounts),
        ("needs_refresh", selected_mounts),
        ("rebuild", selected_mounts),
    ]
    assert emitted[0].status == "error"
    assert "knowledge: unreadable mount" in emitted[0].payload["error"]


def test_skills_mine_skips_artifacts_when_no_candidates(
    tmp_path: Path,
    monkeypatch,
) -> None:
    context_root = tmp_path / "context"
    context_root.mkdir()
    config = _config(context_root)
    write_artifacts = MagicMock()
    emitted = []
    monkeypatch.setattr(skills_mine, "load_agent_config", lambda _path: config)
    monkeypatch.setattr(
        skills_mine,
        "mine_skill_candidates",
        lambda *_args, **_kwargs: {
            "candidate_count": 0,
            "successful_sessions": 0,
            "candidates": [],
        },
    )
    monkeypatch.setattr(skills_mine, "write_skill_candidate_artifacts", write_artifacts)
    monkeypatch.setattr(
        skills_mine,
        "emit_result",
        lambda result, **_kwargs: emitted.append(result),
    )

    assert skills_mine.main([]) == 0
    write_artifacts.assert_not_called()
    assert emitted[0].status == "ok"
    assert emitted[0].metrics["candidates"] == 0


def test_skills_mine_reports_operational_exceptions(
    tmp_path: Path,
    monkeypatch,
) -> None:
    context_root = tmp_path / "context"
    context_root.mkdir()
    config = _config(context_root)
    emitted = []
    monkeypatch.setattr(skills_mine, "load_agent_config", lambda _path: config)

    def _raise(*_args, **_kwargs):
        raise RuntimeError("mining database unavailable")

    monkeypatch.setattr(skills_mine, "mine_skill_candidates", _raise)
    monkeypatch.setattr(
        skills_mine,
        "emit_result",
        lambda result, **_kwargs: emitted.append(result),
    )

    assert skills_mine.main([]) == 1
    assert emitted[0].status == "error"
    assert "mining database unavailable" in emitted[0].payload["error"]


def test_briefing_uses_remapped_scratchpad_without_network_or_tasks(
    tmp_path: Path,
    monkeypatch,
) -> None:
    context_root = tmp_path / "context"
    context_root.mkdir()
    config = _config(context_root, scratchpad_name="daily-notes")
    emitted = []
    build_briefing = MagicMock(return_value={"date": "2026-07-15"})
    monkeypatch.setattr(briefing_agent, "load_agent_config", lambda _path: config)
    monkeypatch.setattr("afs.cli.briefing._build_briefing", build_briefing)
    monkeypatch.setattr("afs.cli.briefing._render_text", lambda _briefing: "briefing")
    monkeypatch.setattr(
        briefing_agent,
        "emit_result",
        lambda result, **_kwargs: emitted.append(result),
    )

    assert briefing_agent.main([]) == 0
    build_briefing.assert_called_once_with(
        days=7,
        include_gws=False,
        include_tasks=False,
    )
    target = Path(emitted[0].payload["path"])
    assert target.parent == context_root / "daily-notes" / "briefings"
    assert target.read_text(encoding="utf-8") == "briefing\n"


def test_briefing_uses_v2_common_scope(
    tmp_path: Path,
    monkeypatch,
) -> None:
    context_root = tmp_path / ".context"
    scaffold_v2(context_root)
    config = _config(context_root)
    emitted = []
    monkeypatch.setattr(briefing_agent, "load_agent_config", lambda _path: config)
    monkeypatch.setattr(
        "afs.cli.briefing._build_briefing",
        lambda **_kwargs: {"date": "2026-07-16"},
    )
    monkeypatch.setattr("afs.cli.briefing._render_text", lambda _briefing: "briefing")
    monkeypatch.setattr(
        briefing_agent,
        "emit_result",
        lambda result, **_kwargs: emitted.append(result),
    )

    assert briefing_agent.main([]) == 0
    target = Path(emitted[0].payload["path"])
    assert target.parent == context_root / "scratchpad" / "common" / "briefings"
    assert target.read_text(encoding="utf-8") == "briefing\n"


def test_briefing_rejects_v2_symlinked_output_root(
    tmp_path: Path,
    monkeypatch,
) -> None:
    context_root = tmp_path / ".context"
    scaffold_v2(context_root)
    config = _config(context_root)
    outside = tmp_path / "outside"
    outside.mkdir()
    common = context_root / "scratchpad" / "common"
    common.mkdir()
    try:
        (common / "briefings").symlink_to(outside, target_is_directory=True)
    except OSError as exc:  # pragma: no cover - Windows without symlink privilege
        import pytest

        pytest.skip(f"directory symlinks unavailable: {exc}")
    emitted = []
    monkeypatch.setattr(briefing_agent, "load_agent_config", lambda _path: config)
    monkeypatch.setattr(
        briefing_agent,
        "emit_result",
        lambda result, **_kwargs: emitted.append(result),
    )

    assert briefing_agent.main([]) == 1
    assert emitted[0].status == "error"
    assert list(outside.iterdir()) == []
