from __future__ import annotations

import json
from argparse import Namespace
from pathlib import Path

from afs.cli.skills import (
    skills_archive_command,
    skills_mine_command,
    skills_promote_command,
    skills_reject_command,
    skills_review_command,
)
from afs.history import append_history_event
from afs.manager import AFSManager
from afs.models import MountType
from afs.schema import AFSConfig, GeneralConfig
from afs.skill_mining import mine_skill_candidates, review_skill_candidates


def _make_manager_and_context(tmp_path: Path) -> tuple[AFSManager, Path]:
    config = AFSConfig(
        general=GeneralConfig(context_root=tmp_path / "contexts"),
    )
    manager = AFSManager(config=config)
    project_path = tmp_path / "project"
    project_path.mkdir()
    context_path = manager.ensure(path=project_path).path
    return manager, project_path


def _write_candidate_artifact(
    manager: AFSManager,
    context_path: Path,
    *,
    stamp: str,
    candidates: list[dict],
) -> Path:
    scratchpad_root = manager.resolve_mount_root(context_path, MountType.SCRATCHPAD)
    artifacts_root = scratchpad_root / "skill_candidates"
    artifacts_root.mkdir(parents=True, exist_ok=True)
    artifact_path = artifacts_root / f"skill_candidates_{stamp}.json"
    artifact_path.write_text(
        json.dumps(
            {
                "generated_at": "2026-04-11T02:02:02+00:00",
                "sessions_analyzed": 4,
                "successful_sessions": 3,
                "candidate_count": len(candidates),
                "candidates": candidates,
            }
        ) + "\n",
        encoding="utf-8",
    )
    return artifact_path


def _append_session_trace(
    history_root: Path,
    *,
    session_id: str,
    prompt_preview: str,
    successful: bool = True,
) -> None:
    append_history_event(
        history_root,
        "session",
        "afs.session",
        op="session_start",
        metadata={"session_id": session_id, "client": "codex"},
    )
    append_history_event(
        history_root,
        "session",
        "afs.session",
        op="user_prompt_submit",
        metadata={
            "session_id": session_id,
            "client": "codex",
            "prompt_preview": prompt_preview,
        },
    )
    append_history_event(
        history_root,
        "mcp_tool",
        "afs.mcp",
        op="call",
        metadata={
            "session_id": session_id,
            "tool_name": "context.query",
            "ok": True,
        },
    )
    append_history_event(
        history_root,
        "mcp_tool",
        "afs.mcp",
        op="call",
        metadata={
            "session_id": session_id,
            "tool_name": "context.read",
            "ok": True,
        },
    )
    append_history_event(
        history_root,
        "fs",
        "afs.fs",
        op="write",
        metadata={
            "session_id": session_id,
            "mount_type": "scratchpad",
            "relative_path": f"notes/{session_id}.md",
        },
    )
    append_history_event(
        history_root,
        "mcp_tool",
        "afs.mcp",
        op="call",
        metadata={
            "session_id": session_id,
            "tool_name": "context.write",
            "ok": successful,
            "error": None if successful else "boom",
        },
    )
    append_history_event(
        history_root,
        "session",
        "afs.session",
        op="task_completed" if successful else "task_failed",
        metadata={"session_id": session_id, "task_id": f"task-{session_id}"},
    )
    append_history_event(
        history_root,
        "session",
        "afs.session",
        op="session_end",
        metadata={
            "session_id": session_id,
            "client": "codex",
            "exit_code": 0 if successful else 1,
        },
    )


def test_mine_skill_candidates_clusters_repeated_successful_sessions(tmp_path: Path) -> None:
    manager, project_path = _make_manager_and_context(tmp_path)
    context_path = project_path / ".context"
    history_root = manager.resolve_mount_root(context_path, MountType.HISTORY)

    _append_session_trace(
        history_root,
        session_id="session-a",
        prompt_preview="Refresh context query notes for scratchpad review",
    )
    _append_session_trace(
        history_root,
        session_id="session-b",
        prompt_preview="Review context query notes and update scratchpad guidance",
    )
    _append_session_trace(
        history_root,
        session_id="session-c",
        prompt_preview="Refresh context query notes for scratchpad review",
        successful=False,
    )

    payload = mine_skill_candidates(
        context_path,
        lookback_hours=24,
        max_sessions=10,
        min_occurrences=2,
        max_candidates=5,
        config=manager.config,
    )

    assert payload["candidate_count"] == 1
    assert payload["sessions_analyzed"] == 3
    assert payload["successful_sessions"] == 2

    candidate = payload["candidates"][0]
    assert candidate["occurrences"] == 2
    assert candidate["tool_sequence"] == [
        "context.query",
        "context.read",
        "context.write",
    ]
    assert set(candidate["source_sessions"]) == {"session-a", "session-b"}
    assert "*.md" in candidate["touched_path_hints"]
    assert candidate["suggested_skill"]["name"].startswith("workflow-")


def test_skills_mine_command_writes_artifacts(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    manager, project_path = _make_manager_and_context(tmp_path)
    context_path = project_path / ".context"
    history_root = manager.resolve_mount_root(context_path, MountType.HISTORY)

    _append_session_trace(
        history_root,
        session_id="session-a",
        prompt_preview="Refresh context query notes for scratchpad review",
    )
    _append_session_trace(
        history_root,
        session_id="session-b",
        prompt_preview="Review context query notes and update scratchpad guidance",
    )

    monkeypatch.setattr(
        "afs.cli.skills.load_runtime_config_from_args",
        lambda args, start_dir=None: (manager.config, tmp_path / "afs.toml"),
    )

    rc = skills_mine_command(
        Namespace(
            config=None,
            path=str(project_path),
            context_root=None,
            context_dir=None,
            lookback_hours=24,
            max_sessions=10,
            min_occurrences=2,
            max_candidates=5,
            replay_limit=200,
            no_write_artifacts=False,
            json=True,
        )
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["candidate_count"] == 1
    assert Path(payload["artifact_paths"]["json"]).exists()
    assert Path(payload["artifact_paths"]["markdown"]).exists()
    markdown = Path(payload["artifact_paths"]["markdown"]).read_text(encoding="utf-8")
    assert "# Skill Candidates" in markdown
    assert "context.query -> context.read -> context.write" in markdown


def test_mine_skill_candidates_falls_back_to_enriched_session_workflow_steps(
    tmp_path: Path,
) -> None:
    manager, project_path = _make_manager_and_context(tmp_path)
    context_path = project_path / ".context"
    history_root = manager.resolve_mount_root(context_path, MountType.HISTORY)

    for session_id, prompt_preview in [
        ("session-a", "Investigate harness progress updates"),
        ("session-b", "Investigate harness progress tracking"),
    ]:
        append_history_event(
            history_root,
            "session",
            "afs.session",
            op="user_prompt_submit",
            metadata={
                "session_id": session_id,
                "client": "codex",
                "prompt_preview": prompt_preview,
            },
        )
        append_history_event(
            history_root,
            "session",
            "afs.session",
            op="task_completed",
            metadata={
                "session_id": session_id,
                "client": "codex",
                "workflow": "edit_fast",
                "tool_profile": "edit_and_verify",
                "workflow_steps": [
                    "user_prompt_submit",
                    "task_created",
                    "task_completed",
                ],
                "matched_skills": ["agentic-background"],
                "outcome": "completed",
            },
        )
        append_history_event(
            history_root,
            "session",
            "afs.session",
            op="session_end",
            metadata={
                "session_id": session_id,
                "client": "codex",
                "workflow": "edit_fast",
                "tool_profile": "edit_and_verify",
                "workflow_steps": [
                    "user_prompt_submit",
                    "task_created",
                    "task_completed",
                ],
                "matched_skills": ["agentic-background"],
                "outcome": "completed",
                "exit_code": 0,
            },
        )

    payload = mine_skill_candidates(
        context_path,
        lookback_hours=24,
        max_sessions=10,
        min_occurrences=2,
        max_candidates=5,
        config=manager.config,
    )

    assert payload["candidate_count"] == 1
    candidate = payload["candidates"][0]
    assert candidate["tool_sequence"] == [
        "workflow:edit_fast",
        "profile:edit_and_verify",
        "skill:agentic-background",
        "user_prompt_submit",
        "task_created",
        "task_completed",
    ]
    assert candidate["occurrences"] == 2


def test_review_skill_candidates_uses_latest_artifact_and_candidate_filter(
    tmp_path: Path,
) -> None:
    manager, project_path = _make_manager_and_context(tmp_path)
    context_path = project_path / ".context"
    old_path = _write_candidate_artifact(
        manager,
        context_path,
        stamp="20260411T010101Z",
        candidates=[
            {
                "id": "workflow-old",
                "name": "workflow-old",
                "confidence": 0.4,
                "occurrences": 2,
                "tool_sequence": ["context.query", "context.read"],
                "trigger_terms": ["old"],
                "clients": ["codex"],
                "source_sessions": ["old-1", "old-2"],
            }
        ],
    )
    new_path = _write_candidate_artifact(
        manager,
        context_path,
        stamp="20260411T020202Z",
        candidates=[
            {
                "id": "workflow-new",
                "name": "workflow-new",
                "confidence": 0.7,
                "occurrences": 3,
                "tool_sequence": ["context.query", "context.read", "context.write"],
                "trigger_terms": ["new"],
                "clients": ["codex"],
                "source_sessions": ["new-1", "new-2", "new-3"],
            },
            {
                "id": "workflow-other",
                "name": "workflow-other",
                "confidence": 0.5,
                "occurrences": 2,
                "tool_sequence": ["workflow:edit_fast", "task_completed"],
                "trigger_terms": ["other"],
                "clients": ["codex"],
                "source_sessions": ["other-1", "other-2"],
            },
        ],
    )

    payload = review_skill_candidates(
        manager,
        context_path,
        candidate_id="workflow-other",
    )

    assert payload["artifact_path"] == str(new_path.resolve())
    assert payload["artifact_count"] == 2
    assert payload["candidate_count"] == 1
    assert payload["total_candidate_count"] == 1
    assert payload["candidates"][0]["id"] == "workflow-other"
    assert payload["candidates"][0]["status"] == "pending"
    assert payload["status_counts"]["pending"] == 2


def test_skills_review_command_outputs_latest_artifact_json(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    manager, project_path = _make_manager_and_context(tmp_path)
    context_path = project_path / ".context"
    history_root = manager.resolve_mount_root(context_path, MountType.HISTORY)

    _append_session_trace(
        history_root,
        session_id="session-a",
        prompt_preview="Refresh context query notes for scratchpad review",
    )
    _append_session_trace(
        history_root,
        session_id="session-b",
        prompt_preview="Review context query notes and update scratchpad guidance",
    )

    monkeypatch.setattr(
        "afs.cli.skills.load_runtime_config_from_args",
        lambda args, start_dir=None: (manager.config, tmp_path / "afs.toml"),
    )

    assert skills_mine_command(
        Namespace(
            config=None,
            path=str(project_path),
            context_root=None,
            context_dir=None,
            lookback_hours=24,
            max_sessions=10,
            min_occurrences=2,
            max_candidates=5,
            replay_limit=200,
            no_write_artifacts=False,
            json=True,
        )
    ) == 0
    capsys.readouterr()

    rc = skills_review_command(
        Namespace(
            config=None,
            path=str(project_path),
            context_root=None,
            context_dir=None,
            artifact=None,
            candidate=None,
            limit=10,
            json=True,
        )
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["candidate_count"] == 1
    assert payload["artifact_path"].endswith(".json")
    assert payload["candidates"][0]["tool_sequence"] == [
        "context.query",
        "context.read",
        "context.write",
    ]
    assert payload["candidates"][0]["status"] == "pending"


def test_skills_promote_command_writes_skill_md(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    manager, project_path = _make_manager_and_context(tmp_path)
    context_path = project_path / ".context"
    _write_candidate_artifact(
        manager,
        context_path,
        stamp="20260411T030303Z",
        candidates=[
            {
                "id": "workflow-harness-progress",
                "name": "workflow-harness-progress",
                "confidence": 0.7,
                "occurrences": 3,
                "tool_sequence": ["context.query", "context.read", "context.write"],
                "trigger_terms": ["harness", "progress"],
                "clients": ["codex"],
                "prompt_examples": [
                    "Investigate harness progress updates",
                    "Investigate harness progress tracking",
                ],
                "source_sessions": ["session-a", "session-b", "session-c"],
                "suggested_skill": {
                    "name": "workflow-harness-progress",
                    "triggers": ["harness", "progress"],
                    "notes": ["Observed in 3 successful session traces."],
                },
            }
        ],
    )

    monkeypatch.setattr(
        "afs.cli.skills.load_runtime_config_from_args",
        lambda args, start_dir=None: (manager.config, tmp_path / "afs.toml"),
    )

    skill_root = tmp_path / "promoted-skills"
    rc = skills_promote_command(
        Namespace(
            config=None,
            path=str(project_path),
            context_root=None,
            context_dir=None,
            artifact=None,
            candidate=None,
            profile=None,
            root=str(skill_root),
            name=None,
            limit=25,
            dry_run=False,
            force=False,
            json=True,
        )
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    skill_path = Path(payload["skill_path"])
    assert skill_path.exists()
    content = skill_path.read_text(encoding="utf-8")
    assert "name: workflow-harness-progress" in content
    assert "triggers:" in content
    assert "  - harness" in content
    assert "profiles:" in content
    assert "  - general" in content
    assert "Generated from mined AFS session traces." in content
    assert payload["review_state"]["status"] == "promoted"
    assert Path(payload["review_state"]["state_path"]).exists()

    reviewed = review_skill_candidates(manager, context_path, candidate_id="workflow-harness-progress")
    assert reviewed["candidates"][0]["status"] == "promoted"
    assert reviewed["candidates"][0]["review_state"]["skill_path"] == str(skill_path)


def test_skills_promote_requires_candidate_when_multiple_candidates(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    manager, project_path = _make_manager_and_context(tmp_path)
    context_path = project_path / ".context"
    _write_candidate_artifact(
        manager,
        context_path,
        stamp="20260411T040404Z",
        candidates=[
            {
                "id": "workflow-one",
                "name": "workflow-one",
                "confidence": 0.6,
                "occurrences": 2,
                "tool_sequence": ["context.query", "context.read"],
                "trigger_terms": ["one"],
                "clients": ["codex"],
                "source_sessions": ["s1", "s2"],
            },
            {
                "id": "workflow-two",
                "name": "workflow-two",
                "confidence": 0.6,
                "occurrences": 2,
                "tool_sequence": ["context.query", "context.write"],
                "trigger_terms": ["two"],
                "clients": ["codex"],
                "source_sessions": ["s3", "s4"],
            },
        ],
    )

    monkeypatch.setattr(
        "afs.cli.skills.load_runtime_config_from_args",
        lambda args, start_dir=None: (manager.config, tmp_path / "afs.toml"),
    )

    rc = skills_promote_command(
        Namespace(
            config=None,
            path=str(project_path),
            context_root=None,
            context_dir=None,
            artifact=None,
            candidate=None,
            profile=None,
            root=str(tmp_path / "promoted-skills"),
            name=None,
            limit=25,
            dry_run=False,
            force=False,
            json=False,
        )
    )

    assert rc == 1
    assert "Multiple candidates matched" in capsys.readouterr().err


def test_skills_promote_refuses_to_overwrite_without_force(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    manager, project_path = _make_manager_and_context(tmp_path)
    context_path = project_path / ".context"
    _write_candidate_artifact(
        manager,
        context_path,
        stamp="20260411T050505Z",
        candidates=[
            {
                "id": "workflow-harness-progress",
                "name": "workflow-harness-progress",
                "confidence": 0.7,
                "occurrences": 3,
                "tool_sequence": ["context.query", "context.read", "context.write"],
                "trigger_terms": ["harness", "progress"],
                "clients": ["codex"],
                "source_sessions": ["session-a", "session-b", "session-c"],
            }
        ],
    )

    monkeypatch.setattr(
        "afs.cli.skills.load_runtime_config_from_args",
        lambda args, start_dir=None: (manager.config, tmp_path / "afs.toml"),
    )

    skill_root = tmp_path / "promoted-skills"
    target = skill_root / "workflow-harness-progress" / "SKILL.md"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("---\nname: workflow-harness-progress\n---\n", encoding="utf-8")

    rc = skills_promote_command(
        Namespace(
            config=None,
            path=str(project_path),
            context_root=None,
            context_dir=None,
            artifact=None,
            candidate=None,
            profile=None,
            root=str(skill_root),
            name=None,
            limit=25,
            dry_run=False,
            force=False,
            json=False,
        )
    )

    assert rc == 1
    assert "Use --force to overwrite" in capsys.readouterr().err


def test_skills_review_command_filters_by_status(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    manager, project_path = _make_manager_and_context(tmp_path)
    context_path = project_path / ".context"
    _write_candidate_artifact(
        manager,
        context_path,
        stamp="20260411T060606Z",
        candidates=[
            {
                "id": "workflow-pending",
                "name": "workflow-pending",
                "confidence": 0.6,
                "occurrences": 2,
                "tool_sequence": ["context.query", "context.read"],
                "trigger_terms": ["pending"],
                "clients": ["codex"],
                "source_sessions": ["s1", "s2"],
            },
            {
                "id": "workflow-promoted",
                "name": "workflow-promoted",
                "confidence": 0.6,
                "occurrences": 2,
                "tool_sequence": ["context.query", "context.write"],
                "trigger_terms": ["promoted"],
                "clients": ["codex"],
                "source_sessions": ["s3", "s4"],
            },
        ],
    )

    monkeypatch.setattr(
        "afs.cli.skills.load_runtime_config_from_args",
        lambda args, start_dir=None: (manager.config, tmp_path / "afs.toml"),
    )
    skills_promote_command(
        Namespace(
            config=None,
            path=str(project_path),
            context_root=None,
            context_dir=None,
            artifact=None,
            candidate="workflow-promoted",
            profile=None,
            root=str(tmp_path / "promoted-skills"),
            name=None,
            limit=25,
            dry_run=False,
            force=False,
            json=True,
        )
    )
    capsys.readouterr()

    rc = skills_review_command(
        Namespace(
            config=None,
            path=str(project_path),
            context_root=None,
            context_dir=None,
            artifact=None,
            candidate=None,
            status="promoted",
            limit=10,
            json=True,
        )
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["candidate_count"] == 1
    assert payload["candidates"][0]["id"] == "workflow-promoted"
    assert payload["candidates"][0]["status"] == "promoted"


def test_skills_reject_command_updates_review_queue(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    manager, project_path = _make_manager_and_context(tmp_path)
    context_path = project_path / ".context"
    _write_candidate_artifact(
        manager,
        context_path,
        stamp="20260411T070707Z",
        candidates=[
            {
                "id": "workflow-pending",
                "name": "workflow-pending",
                "confidence": 0.6,
                "occurrences": 2,
                "tool_sequence": ["context.query", "context.read"],
                "trigger_terms": ["pending"],
                "clients": ["codex"],
                "source_sessions": ["s1", "s2"],
            },
            {
                "id": "workflow-rejected",
                "name": "workflow-rejected",
                "confidence": 0.5,
                "occurrences": 2,
                "tool_sequence": ["context.query", "context.write"],
                "trigger_terms": ["reject"],
                "clients": ["codex"],
                "source_sessions": ["s3", "s4"],
            },
        ],
    )

    monkeypatch.setattr(
        "afs.cli.skills.load_runtime_config_from_args",
        lambda args, start_dir=None: (manager.config, tmp_path / "afs.toml"),
    )

    rc = skills_reject_command(
        Namespace(
            config=None,
            path=str(project_path),
            context_root=None,
            context_dir=None,
            artifact=None,
            candidate="workflow-rejected",
            limit=25,
            json=True,
        )
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["review_state"]["status"] == "rejected"
    assert payload["review_state"]["rejected_at"]

    reviewed = review_skill_candidates(manager, context_path, status_filter="pending")
    assert reviewed["candidate_count"] == 1
    assert reviewed["candidates"][0]["id"] == "workflow-pending"
    assert reviewed["status_counts"] == {"pending": 1, "rejected": 1}


def test_skills_archive_command_records_archived_status(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    manager, project_path = _make_manager_and_context(tmp_path)
    context_path = project_path / ".context"
    _write_candidate_artifact(
        manager,
        context_path,
        stamp="20260411T080808Z",
        candidates=[
            {
                "id": "workflow-archive",
                "name": "workflow-archive",
                "confidence": 0.55,
                "occurrences": 2,
                "tool_sequence": ["context.query", "context.read", "context.write"],
                "trigger_terms": ["archive"],
                "clients": ["codex"],
                "source_sessions": ["s5", "s6"],
            }
        ],
    )

    monkeypatch.setattr(
        "afs.cli.skills.load_runtime_config_from_args",
        lambda args, start_dir=None: (manager.config, tmp_path / "afs.toml"),
    )

    rc = skills_archive_command(
        Namespace(
            config=None,
            path=str(project_path),
            context_root=None,
            context_dir=None,
            artifact=None,
            candidate="workflow-archive",
            limit=25,
            json=True,
        )
    )

    assert rc == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload["review_state"]["status"] == "archived"
    assert payload["review_state"]["archived_at"]

    reviewed = review_skill_candidates(manager, context_path, status_filter="archived")
    assert reviewed["candidate_count"] == 1
    assert reviewed["candidates"][0]["id"] == "workflow-archive"
    assert reviewed["candidates"][0]["status"] == "archived"
