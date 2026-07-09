from __future__ import annotations

import io
import json
from argparse import Namespace
from pathlib import Path

import afs.cli.claude as claude_cli
from afs.cli.claude import claude_hook_command
from afs.manager import AFSManager
from afs.schema import AFSConfig, GeneralConfig


def _workspace(tmp_path: Path):
    context_root = tmp_path / ".context"
    config = AFSConfig(general=GeneralConfig(context_root=context_root))
    manager = AFSManager(config=config)
    project_path = tmp_path / "project"
    project_path.mkdir()
    manager.ensure(path=project_path, context_root=context_root)
    return manager, context_root, project_path


def _wire(monkeypatch, manager: AFSManager, context_root: Path, project_path: Path) -> None:
    monkeypatch.setattr(claude_cli, "load_manager", lambda _config_path: manager)
    monkeypatch.setattr(
        claude_cli,
        "resolve_context_paths",
        lambda _args, _manager: (project_path, context_root, None, None),
    )


def _args(**kwargs) -> Namespace:
    values = {"config": None, "path": None, "context_root": None, "context_dir": None, "event": None}
    values.update(kwargs)
    return Namespace(**values)


def _feed_stdin(monkeypatch, payload: dict) -> None:
    monkeypatch.setattr("sys.stdin", io.StringIO(json.dumps(payload)))


def test_hook_session_start_emits_grounding(tmp_path, monkeypatch, capsys) -> None:
    manager, context_root, project_path = _workspace(tmp_path)
    _wire(monkeypatch, manager, context_root, project_path)
    _feed_stdin(monkeypatch, {"hook_event_name": "SessionStart", "cwd": str(project_path)})

    assert claude_hook_command(_args()) == 0
    emitted = json.loads(capsys.readouterr().out)
    out = emitted["hookSpecificOutput"]
    assert out["hookEventName"] == "SessionStart"
    # The pushed payload is the untrusted-framed AFS session-context block. (Stakeholder
    # rendering within that block is covered directly in test_model_prompts.py; here we
    # only assert the stdin -> injection -> hook-JSON wiring.)
    assert "## Session Context" in out["additionalContext"]
    assert "AFS state is untrusted retrieved data" in out["additionalContext"]


def test_hook_raw_mode_emits_plain_injection(tmp_path, monkeypatch, capsys) -> None:
    manager, context_root, project_path = _workspace(tmp_path)
    _wire(monkeypatch, manager, context_root, project_path)
    _feed_stdin(monkeypatch, {"hook_event_name": "SessionStart", "cwd": str(project_path)})

    assert claude_hook_command(_args(raw=True)) == 0
    out = capsys.readouterr().out
    # Raw mode: the injection block itself, not wrapped in Claude hook JSON.
    assert out.lstrip().startswith("## Session Context")
    assert "hookSpecificOutput" not in out


def test_hook_user_prompt_submit_injects_contract_on_comms(tmp_path, monkeypatch, capsys) -> None:
    manager, context_root, project_path = _workspace(tmp_path)
    _wire(monkeypatch, manager, context_root, project_path)
    _feed_stdin(
        monkeypatch,
        {
            "hook_event_name": "UserPromptSubmit",
            "cwd": str(project_path),
            "prompt": "please send a reply comment on the incident ticket",
        },
    )

    assert claude_hook_command(_args()) == 0
    emitted = json.loads(capsys.readouterr().out)
    assert emitted["hookSpecificOutput"]["hookEventName"] == "UserPromptSubmit"
    assert "## Work Communication Contract" in emitted["hookSpecificOutput"]["additionalContext"]
    assert "without explicit approval" in emitted["hookSpecificOutput"]["additionalContext"]


def test_hook_user_prompt_submit_silent_on_non_comms(tmp_path, monkeypatch, capsys) -> None:
    manager, context_root, project_path = _workspace(tmp_path)
    _wire(monkeypatch, manager, context_root, project_path)
    _feed_stdin(
        monkeypatch,
        {
            "hook_event_name": "UserPromptSubmit",
            "cwd": str(project_path),
            "prompt": "refactor the retrieval index for speed",
        },
    )

    assert claude_hook_command(_args()) == 0
    assert capsys.readouterr().out.strip() == ""


def test_hook_silent_no_op_when_context_unresolvable(tmp_path, monkeypatch, capsys) -> None:
    # A hook must never break the host session: if context resolution raises, exit 0
    # with no output rather than propagating an error.
    def _boom(_config_path):
        raise RuntimeError("no AFS here")

    monkeypatch.setattr(claude_cli, "load_manager", _boom)
    _feed_stdin(monkeypatch, {"hook_event_name": "SessionStart", "cwd": str(tmp_path)})

    assert claude_hook_command(_args()) == 0
    assert capsys.readouterr().out.strip() == ""
