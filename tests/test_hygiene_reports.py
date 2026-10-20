from __future__ import annotations

import os
import subprocess
import time
from pathlib import Path

from afs.handoff_folders import folder_report
from afs.handoff_prompt import supersede_folder
from afs.worktree_report import parse_worktree_porcelain, worktree_report


def _age(path: Path, days: float) -> None:
    stamp = time.time() - days * 86400
    for item in [path, *path.rglob("*")]:
        os.utime(item, (stamp, stamp))


def test_folder_report_flags_idle_unreferenced_folders(tmp_path: Path) -> None:
    root = tmp_path / "handoffs"
    old, live, fresh = root / "old_2026-09-01", root / "live_2026-09-01", root / "fresh_2026-10-19"
    for folder in (old, live, fresh):
        (folder / "build").mkdir(parents=True)
        (folder / "build" / "lib.o").write_bytes(b"x" * 100)
        (folder / "rom.sfc").write_bytes(b"r" * 10)
        (folder / "README.md").write_text("notes")
    supersede_folder(old, superseded_by=str(fresh), by="codex")
    _age(old, 30)
    _age(live, 30)

    report = folder_report(root, older_than_days=14, live_texts=("see live_2026-09-01/WORK.md",))

    assert report["folder_count"] == 3
    assert report["bytes_by_kind"]["build"] == 300
    assert report["bytes_by_kind"]["rom"] == 30
    assert [item["name"] for item in report["candidates"]] == ["old_2026-09-01"]
    assert report["candidates"][0]["superseded_by"] == str(fresh)
    assert report["superseded_count"] == 1
    assert old.joinpath("README.md").exists()


def test_parse_worktree_porcelain() -> None:
    text = (
        "worktree /repo\nHEAD aaa\nbranch refs/heads/main\n\n"
        "worktree /tmp/gone\nHEAD bbb\nbranch refs/heads/feat\nprunable gitdir file points to x\n\n"
        "worktree /wt\nHEAD ccc\ndetached\nlocked\n"
    )
    items = parse_worktree_porcelain(text)
    assert [item.path for item in items] == ["/repo", "/tmp/gone", "/wt"]
    assert items[1].branch == "feat" and items[1].prunable
    assert items[2].locked and items[2].branch == ""


def test_worktree_report_marks_merged_clean_idle_worktrees(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    env = {
        **os.environ,
        "GIT_AUTHOR_DATE": "2026-01-01T00:00:00Z",
        "GIT_COMMITTER_DATE": "2026-01-01T00:00:00Z",
    }

    def git(*args: str, cwd: Path = repo) -> None:
        subprocess.run(["git", *args], cwd=cwd, env=env, check=True, capture_output=True)

    repo.mkdir()
    git("init", "-q", "-b", "main")
    git("-c", "user.email=t@t", "-c", "user.name=t", "commit", "-q", "--allow-empty", "-m", "init")
    git("worktree", "add", "-q", "-b", "merged", str(tmp_path / "wt-merged"))
    git("worktree", "add", "-q", "-b", "dirty", str(tmp_path / "wt-dirty"))
    (tmp_path / "wt-dirty" / "f.txt").write_text("x")
    git("add", "f.txt", cwd=tmp_path / "wt-dirty")

    report = worktree_report(repo, older_than_days=7)

    by_branch = {row["branch"]: row for row in report["worktrees"]}
    assert report["default_branch"] == "main"
    assert by_branch["merged"]["candidate"] == "clean, merged into default branch"
    assert by_branch["dirty"]["dirty"] is True and by_branch["dirty"]["candidate"] == ""
    assert report["candidate_count"] == 1
