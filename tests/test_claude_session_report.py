import json

from afs.claude.session_report import build_session_report, render_session_report_markdown


def _write_jsonl(path, events) -> None:
    path.write_text("\n".join(json.dumps(e) for e in events) + "\n", encoding="utf-8")


def test_claude_session_report_discovers_artifacts(tmp_path) -> None:
    session_id = "42cc501a-6d5b-4087-af16-91152a81b866"
    claude_root = tmp_path / ".claude"
    projects_dir = claude_root / "projects" / "proj"
    projects_dir.mkdir(parents=True)

    transcript = projects_dir / f"{session_id}.jsonl"
    _write_jsonl(
        transcript,
        [
            {
                "cwd": "/repo",
                "gitBranch": "main",
                "version": "2.1.34",
                "slug": "test-slug",
                "timestamp": "2026-02-07T01:00:00.000Z",
                "type": "user",
                "message": {"role": "user", "content": "Hello."},
            },
            {
                "timestamp": "2026-02-07T01:00:01.000Z",
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "model": "claude-test",
                    "content": [
                        {
                            "type": "tool_use",
                            "id": "toolu_test",
                            "name": "Bash",
                            "input": {"command": "echo hi"},
                        }
                    ],
                },
            },
        ],
    )

    # Debug log
    debug_dir = claude_root / "debug"
    debug_dir.mkdir(parents=True)
    (debug_dir / f"{session_id}.txt").write_text("debug", encoding="utf-8")

    # Artifacts (subagents)
    artifacts = projects_dir / session_id
    subagents_dir = artifacts / "subagents"
    subagents_dir.mkdir(parents=True)
    _write_jsonl(
        subagents_dir / "agent-a1.jsonl",
        [
            {
                "agentId": "a1",
                "cwd": "/repo",
                "gitBranch": "main",
                "slug": "test-slug",
                "timestamp": "2026-02-07T01:10:00.000Z",
                "type": "user",
                "message": {"role": "user", "content": "Please do subtask."},
            },
            {
                "agentId": "a1",
                "timestamp": "2026-02-07T01:10:01.000Z",
                "type": "assistant",
                "message": {
                    "role": "assistant",
                    "model": "claude-sub",
                    "content": [{"type": "text", "text": "Subtask done."}],
                },
            },
        ],
    )

    report = build_session_report(session_id, claude_root=claude_root)
    assert report.paths.session_id == session_id
    assert report.paths.transcript_path == transcript
    assert report.paths.artifacts_dir == artifacts
    assert report.paths.debug_log_path == debug_dir / f"{session_id}.txt"
    assert report.cwd == "/repo"
    assert report.git_branch == "main"
    assert report.version == "2.1.34"
    assert report.slug == "test-slug"
    assert "claude-test" in report.models
    assert "Bash" in report.tool_calls
    assert len(report.subagents) == 1
    assert report.subagents[0].agent_id == "a1"
    assert report.subagents[0].first_user_prompt.startswith("Please do subtask")

    md = render_session_report_markdown(report)
    assert f"# Claude Session {session_id}" in md
    assert str(transcript) in md
    assert "Subagents (1)" in md


def test_claude_session_report_accepts_prefix(tmp_path) -> None:
    session_id = "42cc501a-6d5b-4087-af16-91152a81b866"
    prefix = "42cc501a"
    claude_root = tmp_path / ".claude"
    projects_dir = claude_root / "projects" / "proj"
    projects_dir.mkdir(parents=True)
    transcript = projects_dir / f"{session_id}.jsonl"
    _write_jsonl(
        transcript,
        [
            {
                "timestamp": "2026-02-07T01:00:00.000Z",
                "type": "user",
                "message": {"role": "user", "content": "Hello."},
            }
        ],
    )

    report = build_session_report(prefix, claude_root=claude_root, include_subagents=False)
    assert report.paths.session_id == session_id
    assert report.paths.transcript_path == transcript

