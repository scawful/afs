"""Tests for ``.afsignore`` / ``.afsindex.toml`` context index policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pytest

from afs.cli.context import register_parsers
from afs.context_index import ContextSQLiteIndex, _iter_mount_entries
from afs.context_layout import scaffold_v2
from afs.hybrid_search import SourcePolicy, _path_allowed
from afs.index_policy import IndexPathPolicy, ignore_matches, load_ignore_file
from afs.manager import AFSManager
from afs.models import MountType
from afs.project_registry import ProjectRegistry
from afs.schema import AFSConfig, GeneralConfig

HASH_DIR = "0123456789abcdef"

ALLOWLIST_TOML = """
[allowlist.scratchpad]
suffixes = [".md", ".json"]
max_depth = 1
max_bytes = 64
receipt_globs = ["RESULT*"]
receipt_max_bytes = 256
skip_hash_dirs = true
"""


def _write(path: Path, text: str = "x") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _make_index(tmp_path: Path, context_root: Path) -> ContextSQLiteIndex:
    manager = AFSManager(config=AFSConfig(general=GeneralConfig(context_root=context_root)))
    return ContextSQLiteIndex(manager, context_root, db_path=tmp_path / "index.sqlite3")


def _paths(index: ContextSQLiteIndex, mount_type: MountType = MountType.SCRATCHPAD) -> set[str]:
    return {
        row["relative_path"]
        for row in index.query(mount_types=[mount_type], limit=1000)
    }


def _seed_allowlist_tree(scratchpad: Path) -> None:
    _write(scratchpad / "notes.md", "small")
    _write(scratchpad / "big.md", "x" * 100)
    _write(scratchpad / "image.png", "png")
    _write(scratchpad / "a" / "plan.md", "plan")
    _write(scratchpad / "a" / "b" / "deep.md", "deep")
    _write(scratchpad / "a" / "b" / "c" / "RESULT_final.md", "r" * 100)
    _write(scratchpad / "a" / "b" / "c" / "RESULT_huge.md", "r" * 300)
    _write(scratchpad / HASH_DIR / "notes.md", "hashed")
    _write(scratchpad / "a" / HASH_DIR / "RESULT.md", "hashed receipt")


ALLOWLISTED_PATHS = {
    "notes.md",
    "a",
    "a/plan.md",
    "a/b",
    "a/b/c/RESULT_final.md",
}


@pytest.mark.parametrize(
    ("pattern", "relative", "expected"),
    [
        ("scratchpad/**/build/", "scratchpad/build/x.md", True),
        ("scratchpad/**/build/", "scratchpad/a/b/build", True),
        ("scratchpad/**/build/", "knowledge/build/x.md", False),
        ("a/**/b", "a/b", True),
        ("a/**/b", "a/x/y/b", True),
        ("a/**/b", "ab", False),
        ("**/node_modules/", "node_modules/pkg/index.md", True),
        ("**/node_modules/", "scratchpad/x/node_modules/pkg.md", True),
        ("*.log", "scratchpad/deep/run.log", True),
        ("build/", "scratchpad/build/x.md", True),
        ("/scratchpad/tmp", "scratchpad/tmp/x.md", True),
    ],
)
def test_ignore_matches_patterns(pattern: str, relative: str, expected: bool) -> None:
    patterns = [(pattern.lstrip("/"), False)]
    assert ignore_matches(relative, patterns) is expected


def test_ignore_matches_last_match_wins(tmp_path: Path) -> None:
    ignore_file = tmp_path / ".afsignore"
    ignore_file.write_text("# comment\n\n*.log\n!keep.log\n", encoding="utf-8")
    patterns = load_ignore_file(ignore_file)

    assert patterns == [("*.log", False), ("keep.log", True)]
    assert ignore_matches("scratchpad/run.log", patterns)
    assert not ignore_matches("scratchpad/keep.log", patterns)
    assert not ignore_matches("scratchpad/notes.md", [])


def test_hybrid_search_honors_zero_directory_double_star(tmp_path: Path) -> None:
    _write(tmp_path / "docs" / "drafts" / "x.md")
    _write(tmp_path / "docs" / "a" / "drafts" / "y.md")
    _write(tmp_path / "docs" / "final.md")
    gitignore = [("docs/**/drafts/", False)]

    def allowed(relative: str) -> bool:
        return _path_allowed(tmp_path / relative, tmp_path, relative, SourcePolicy(), gitignore)

    assert not allowed("docs/drafts/x.md")
    assert not allowed("docs/a/drafts/y.md")
    assert allowed("docs/final.md")


def test_no_policy_files_leave_index_unchanged(tmp_path: Path) -> None:
    context_root = tmp_path / "context"
    _seed_allowlist_tree(context_root / "scratchpad")
    index = _make_index(tmp_path, context_root)

    summary = index.rebuild(mount_types=[MountType.SCRATCHPAD])

    assert not index._path_policy.active
    assert summary.index_policy == {}
    assert summary.to_dict()["dry_run"] is False
    paths = _paths(index)
    assert ALLOWLISTED_PATHS < paths
    assert {"big.md", "image.png", "a/b/c/RESULT_huge.md", f"{HASH_DIR}/notes.md"} <= paths


def test_afsignore_excludes_virtual_paths(tmp_path: Path) -> None:
    context_root = tmp_path / "context"
    scratchpad = context_root / "scratchpad"
    _write(scratchpad / "keep.md")
    _write(scratchpad / "build" / "x.md")
    _write(scratchpad / "a" / "build" / "y.md")
    _write(context_root / "knowledge" / "build" / "z.md")
    _write(context_root / ".afsignore", "scratchpad/**/build/\n")
    index = _make_index(tmp_path, context_root)

    summary = index.rebuild(mount_types=[MountType.SCRATCHPAD, MountType.KNOWLEDGE])

    assert _paths(index) == {"keep.md", "a"}
    assert _paths(index, MountType.KNOWLEDGE) == {"build", "build/z.md"}
    assert summary.index_policy["ignore_patterns"] == 1
    assert summary.index_policy["config_file"] is None


def test_negation_cannot_reinclude_under_excluded_directory(tmp_path: Path) -> None:
    context_root = tmp_path / "context"
    scratchpad = context_root / "scratchpad"
    _write(scratchpad / "logs" / "keep.md")
    _write(scratchpad / "logs" / "sub" / "deep.md")
    _write(scratchpad / "x.tmp")
    _write(scratchpad / "important.tmp")
    index = _make_index(tmp_path, context_root)
    index.rebuild(mount_types=[MountType.SCRATCHPAD])
    assert {"logs/keep.md", "logs/sub/deep.md"} <= _paths(index)

    _write(
        context_root / ".afsignore",
        "scratchpad/logs/\n!scratchpad/logs/keep.md\n!scratchpad/logs/sub/\n"
        "*.tmp\n!important.tmp\n",
    )

    assert index.upsert_relative_path(MountType.SCRATCHPAD, "logs/keep.md") is False
    assert index.upsert_relative_path(MountType.SCRATCHPAD, "logs/sub/deep.md") is False
    assert not {"logs/keep.md", "logs/sub/deep.md"} & _paths(index)

    index.rebuild(mount_types=[MountType.SCRATCHPAD])
    assert _paths(index) == {"important.tmp"}


def test_scratchpad_allowlist_filters_suffix_depth_size_and_hash_dirs(
    tmp_path: Path,
) -> None:
    context_root = tmp_path / "context"
    scratchpad = context_root / "scratchpad"
    _seed_allowlist_tree(scratchpad)
    _write(context_root / "knowledge" / "a" / "b" / "c" / "deep.png", "kept")
    _write(context_root / ".afsindex.toml", ALLOWLIST_TOML)
    index = _make_index(tmp_path, context_root)

    summary = index.rebuild(mount_types=[MountType.SCRATCHPAD, MountType.KNOWLEDGE])

    assert summary.errors == []
    assert _paths(index) == ALLOWLISTED_PATHS
    assert "a/b/c/deep.png" in _paths(index, MountType.KNOWLEDGE)
    assert summary.index_policy["allowlists"]["scratchpad"]["max_depth"] == 1
    assert index.needs_refresh(mount_types=[MountType.SCRATCHPAD]) is False
    assert index.diff(mount_types=[MountType.SCRATCHPAD])["total_changes"] == 0


def test_allowlist_prunes_hash_directories_during_walk(tmp_path: Path) -> None:
    context_root = tmp_path / "context"
    scratchpad = context_root / "scratchpad"
    _seed_allowlist_tree(scratchpad)
    _write(context_root / ".afsindex.toml", ALLOWLIST_TOML)
    index = _make_index(tmp_path, context_root)

    prune = index._policy_prune(MountType.SCRATCHPAD)
    walked = {relative for _entry, relative in _iter_mount_entries(scratchpad, prune=prune)}

    assert HASH_DIR in walked
    assert f"{HASH_DIR}/notes.md" not in walked
    assert f"a/{HASH_DIR}/RESULT.md" not in walked
    assert "a/b/c/RESULT_final.md" in walked


def test_allowlist_depth_is_measured_below_v2_project_scope(tmp_path: Path) -> None:
    context_root = tmp_path / "context"
    project = tmp_path / "project"
    project.mkdir()
    scaffold_v2(context_root)
    project_id = ProjectRegistry(context_root).register(project).project_id
    scope = context_root / "scratchpad" / "projects" / project_id
    _write(scope / "notes.md", "small")
    _write(scope / "a" / "plan.md", "plan")
    _write(scope / "a" / "b" / "deep.md", "deep")
    _write(context_root / "scratchpad" / "common" / "shared.md", "shared")
    _write(context_root / ".afsindex.toml", ALLOWLIST_TOML)
    index = _make_index(tmp_path, context_root)

    index.rebuild(mount_types=[MountType.SCRATCHPAD])

    paths = _paths(index)
    prefix = f"projects/{project_id}"
    assert {f"{prefix}/notes.md", f"{prefix}/a/plan.md", "common/shared.md"} <= paths
    assert f"{prefix}/a/b/deep.md" not in paths


def test_invalid_policy_config_is_reported_and_ignored(tmp_path: Path) -> None:
    context_root = tmp_path / "context"
    _seed_allowlist_tree(context_root / "scratchpad")
    _write(context_root / ".afsindex.toml", "[allowlist.scratchpad\nmax_depth = 1\n")
    index = _make_index(tmp_path, context_root)

    summary = index.rebuild(mount_types=[MountType.SCRATCHPAD])

    assert any(
        error.startswith("index policy: .afsindex.toml: unreadable")
        for error in summary.errors
    )
    assert "a/b/deep.md" in _paths(index)


def test_invalid_allowlist_sections_are_skipped(tmp_path: Path) -> None:
    context_root = tmp_path / "context"
    _write(
        context_root / ".afsindex.toml",
        '[allowlist.scratchpad]\nmax_depth = "two"\n'
        "[allowlist.bogus]\nmax_depth = 1\n"
        "[allowlist.knowledge]\nmax_depth = 0\nsurprise = true\n"
        '[allowlist.memory]\nsuffixes = [".md"]\n',
    )

    policy = IndexPathPolicy.load(context_root)

    assert set(policy.allowlists) == {"memory"}
    assert len(policy.errors) == 3
    assert any("max_depth must be a non-negative integer" in error for error in policy.errors)
    assert any("[allowlist.bogus] is not a mount type" in error for error in policy.errors)
    assert any("unknown keys: surprise" in error for error in policy.errors)


def test_policy_reloads_when_files_change(tmp_path: Path) -> None:
    context_root = tmp_path / "context"
    _seed_allowlist_tree(context_root / "scratchpad")
    index = _make_index(tmp_path, context_root)
    index.rebuild(mount_types=[MountType.SCRATCHPAD])

    _write(context_root / ".afsindex.toml", ALLOWLIST_TOML)
    assert index.needs_refresh(mount_types=[MountType.SCRATCHPAD]) is True
    index.rebuild(mount_types=[MountType.SCRATCHPAD])
    assert _paths(index) == ALLOWLISTED_PATHS

    (context_root / ".afsindex.toml").unlink()
    index.rebuild(mount_types=[MountType.SCRATCHPAD])
    assert "big.md" in _paths(index)


def test_dry_run_counts_changes_without_writing(tmp_path: Path) -> None:
    context_root = tmp_path / "context"
    scratchpad = context_root / "scratchpad"
    _seed_allowlist_tree(scratchpad)
    index = _make_index(tmp_path, context_root)
    index.rebuild(mount_types=[MountType.SCRATCHPAD])
    before = _paths(index)

    _write(context_root / ".afsindex.toml", ALLOWLIST_TOML)
    _write(scratchpad / "new.md", "new")
    _write(scratchpad / "notes.md", "edited")

    dry = index.rebuild(mount_types=[MountType.SCRATCHPAD], dry_run=True)

    assert dry.dry_run is True
    assert _paths(index) == before
    assert dry.rows_inserted == 1
    assert dry.rows_updated == 1
    assert dry.rows_deleted == len(before - ALLOWLISTED_PATHS)

    real = index.rebuild(mount_types=[MountType.SCRATCHPAD])
    assert (real.rows_inserted, real.rows_updated, real.rows_deleted) == (
        dry.rows_inserted,
        dry.rows_updated,
        dry.rows_deleted,
    )
    assert _paths(index) == ALLOWLISTED_PATHS | {"new.md"}


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    register_parsers(subparsers)
    return parser


def _cli_manager(tmp_path: Path) -> tuple[AFSManager, Path, Path]:
    context_root = tmp_path / "context"
    manager = AFSManager(config=AFSConfig(general=GeneralConfig(context_root=context_root)))
    project = tmp_path / "project"
    project.mkdir()
    manager.ensure(path=project, context_root=context_root)
    return manager, project, context_root


def test_cli_index_rebuild_dry_run_json(capsys, monkeypatch, tmp_path: Path) -> None:
    manager, project, context_root = _cli_manager(tmp_path)
    _seed_allowlist_tree(context_root / "scratchpad")
    _write(context_root / ".afsindex.toml", ALLOWLIST_TOML)
    monkeypatch.setattr("afs.cli.context.load_manager", lambda _config_path: manager)
    parser = _make_parser()
    base = [
        "index",
        "rebuild",
        "--mount",
        "scratchpad",
        "--path",
        str(project),
        "--context-root",
        str(context_root),
    ]

    args = parser.parse_args([*base, "--dry-run", "--json"])
    assert args.func(args) == 0
    payload = json.loads(capsys.readouterr().out)

    assert payload["dry_run"] is True
    assert payload["rows_written"] == len(ALLOWLISTED_PATHS)
    assert payload["rows_inserted"] == len(ALLOWLISTED_PATHS)
    assert payload["index_policy"]["allowlists"]["scratchpad"]["skip_hash_dirs"] is True
    index = ContextSQLiteIndex(manager, context_root)
    assert index.count_entries(mount_types=[MountType.SCRATCHPAD]) == 0

    text_args = parser.parse_args([*base, "--dry-run"])
    assert text_args.func(text_args) == 0
    output = capsys.readouterr().out
    assert "dry_run: true" in output
    assert "- allowlists: scratchpad" in output
    assert index.count_entries(mount_types=[MountType.SCRATCHPAD]) == 0
