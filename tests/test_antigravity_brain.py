from __future__ import annotations

import os
import time
from pathlib import Path

from afs.antigravity_brain import import_body, list_conversations, load_conversation

CONVERSATION = "19491a06-7124-4f28-ab01-fb22ed9a7fdf"


def _repo(tmp_path: Path) -> tuple[Path, Path]:
    main = tmp_path / "src" / "app"
    (main / ".git" / "worktrees" / "wt1").mkdir(parents=True)
    worktree = tmp_path / "worktrees" / "wt1"
    worktree.mkdir(parents=True)
    (worktree / ".git").write_text(f"gitdir: {main}/.git/worktrees/wt1\n")
    return main, worktree


def test_load_prefers_handoff_and_maps_worktree_to_main_repo(tmp_path: Path) -> None:
    main, worktree = _repo(tmp_path)
    brain = tmp_path / "brain" / CONVERSATION
    brain.mkdir(parents=True)
    (brain / "task.md").write_text("# Task list\n- [x] step\n")
    (brain / "handoff.md").write_text(
        f"# Handoff - App polish\nSee [x](file://{worktree}/src/a.cc#L1) and {main}/README.md.\n"
    )
    (brain / "handoff.md.resolved").write_text("ignored")

    conversation = load_conversation(brain)

    assert conversation is not None
    assert conversation.files == ["handoff.md", "task.md"]
    assert conversation.title == "Handoff - App polish"
    assert conversation.project == str(main)
    assert conversation.artifact_id() == CONVERSATION.replace("-", "")
    body = import_body(conversation)
    assert body.startswith(f"Imported from Antigravity conversation `{CONVERSATION}`.")
    assert "## handoff.md" in body and "Task list" not in body
    assert str(brain / "task.md") in body


def test_list_filters_by_age_and_project(tmp_path: Path) -> None:
    main, _worktree = _repo(tmp_path)
    root = tmp_path / "brain"
    fresh, stale, other = root / "a", root / "b", root / "c"
    for directory, text in (
        (fresh, f"# Fresh\n{main}/x.py\n"),
        (stale, f"# Stale\n{main}/y.py\n"),
        (other, "# Other\nno paths\n"),
    ):
        directory.mkdir(parents=True)
        (directory / "walkthrough.md").write_text(text)
    old = time.time() - 30 * 86400
    os.utime(stale / "walkthrough.md", (old, old))
    (root / "empty").mkdir()

    assert [c.title for c in list_conversations(root, project=main)] == ["Fresh", "Stale"]
    assert sorted(c.title for c in list_conversations(root, since_days=7)) == ["Fresh", "Other"]
    assert list_conversations(root, limit=0) == []
