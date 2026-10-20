from __future__ import annotations

import json
from pathlib import Path

import pytest

from afs.file_io import FileWriteConflict
from afs.handoff import HandoffStore
from afs.handoff_prompt import (
    SUPERSEDED_FILE,
    render_folder_prompt,
    render_revision_prompt,
    supersede_folder,
)


def test_revision_prompt_warns_when_newer_revision_exists(tmp_path: Path) -> None:
    store = HandoffStore(tmp_path / "context", scope_id="common")
    first = store.create_revision(
        title="Fix CI", agent_name="codex", next_steps=["rerun ci"], blocked=["flaky lane"]
    )
    newer = store.create_revision(
        title="Fix CI v2",
        agent_name="codex",
        stream_id=first.stream_id,
        supersedes=first.revision_id,
    )
    latest = store.read(stream_id=first.stream_id)

    text = render_revision_prompt(first, harness="cursor", latest=latest, project_path=tmp_path)

    assert text.startswith('You are cursor, continuing handoff "Fix CI" from codex.')
    assert "--agent-name cursor" in text
    assert f"superseded by newer revision {newer.revision_id}" in text
    assert "- rerun ci" in text and "- flaky lane" in text
    assert f"afs handoff ack {first.revision_id} --by cursor" in text


def test_folder_prompt_inlines_entry_file_and_supersede_warning(tmp_path: Path) -> None:
    old, new = tmp_path / "topic_2026-10-19", tmp_path / "topic_2026-10-20"
    old.mkdir()
    new.mkdir()
    (old / "README.md").write_text("readme body\n")
    (old / "NEXT_AGENT.md").write_text("do the next thing\n")
    readme_before = (old / "README.md").read_bytes()

    marker = supersede_folder(old, superseded_by=str(new), by="claude", reason="redo")
    text = render_folder_prompt(old, harness="antigravity")

    assert (old / "README.md").read_bytes() == readme_before
    assert json.loads((old / SUPERSEDED_FILE).read_text())["superseded_by"] == str(new)
    assert marker.by == "claude"
    assert f"WARNING: superseded by {new}" in text
    assert "--- NEXT_AGENT.md ---" in text and "do the next thing" in text
    assert "readme body" not in text
    assert SUPERSEDED_FILE not in text.split("Files:")[1].splitlines()[0]


def test_supersede_refuses_second_marker_and_missing_folder(tmp_path: Path) -> None:
    folder = tmp_path / "h"
    folder.mkdir()
    supersede_folder(folder, superseded_by="x", by="codex")
    with pytest.raises(FileWriteConflict):
        supersede_folder(folder, superseded_by="y", by="codex")
    with pytest.raises(FileNotFoundError):
        supersede_folder(tmp_path / "missing", superseded_by="y", by="codex")
