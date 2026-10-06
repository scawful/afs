from __future__ import annotations

import argparse
from pathlib import Path

from afs import session_grounding
from afs.cli import core
from afs.schema import AFSConfig
from afs.session_grounding import (
    DEFAULT_SESSION_GROUNDING_TOKEN_BUDGET,
    resolve_session_grounding_token_budget,
)


def test_session_helper_discovers_config_from_explicit_project_path(tmp_path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    def fake_load_runtime_config_model(**kwargs):
        captured.update(kwargs)
        return AFSConfig(), None

    monkeypatch.setattr(core, "load_runtime_config_model", fake_load_runtime_config_model)
    monkeypatch.setattr(
        core,
        "resolve_context_paths",
        lambda args, manager: (tmp_path, tmp_path / ".context", tmp_path / ".context", None),
    )

    core._load_manager_context_and_config_path(
        argparse.Namespace(
            config=None,
            path=str(tmp_path),
            context_root=None,
            context_dir=None,
        )
    )

    assert captured["start_dir"] == tmp_path.resolve()


def test_session_grounding_builds_read_only_bootstrap(monkeypatch) -> None:
    calls: dict[str, object] = {}

    def fake_bootstrap(manager, context_path, **kwargs):
        calls["bootstrap"] = (manager, context_path, kwargs)
        return {"project": "portable"}

    def fake_injection(**kwargs):
        calls["injection"] = kwargs
        return "grounded"

    monkeypatch.setattr(session_grounding, "build_session_bootstrap", fake_bootstrap)
    monkeypatch.setattr(session_grounding, "build_hook_injection", fake_injection)

    manager = object()
    context_path = Path("/workspace/project/.context")
    project_path = Path("/workspace/project")
    result = session_grounding.build_session_grounding(
        manager,
        context_path,
        project_path=project_path,
        skills_prompt="review portability",
    )

    assert result == "grounded"
    _manager, _context, kwargs = calls["bootstrap"]
    assert kwargs == {
        "project_path": project_path,
        "token_budget": 0,
        "record_event": False,
        "skills_prompt": "review portability",
        "include_skills": True,
    }
    assert calls["injection"] == {
        "event": "SessionStart",
        "context_path": context_path,
        "session_state": {"project": "portable"},
        "prompt": "",
    }


def test_user_prompt_grounding_skips_bootstrap(monkeypatch) -> None:
    def unexpected_bootstrap(*_args, **_kwargs):
        raise AssertionError("UserPromptSubmit must not build a full bootstrap")

    monkeypatch.setattr(session_grounding, "build_session_bootstrap", unexpected_bootstrap)
    monkeypatch.setattr(
        session_grounding,
        "build_hook_injection",
        lambda **kwargs: f"prompt={kwargs['prompt']};state={kwargs['session_state']}",
    )

    result = session_grounding.build_session_grounding(
        object(),
        Path("/workspace/project/.context"),
        event="UserPromptSubmit",
        prompt="draft the release note",
    )

    assert result == "prompt=draft the release note;state=None"


def test_session_grounding_budget_is_bounded_and_overridable(monkeypatch) -> None:
    monkeypatch.delenv("AFS_SESSION_GROUNDING_TOKEN_BUDGET", raising=False)
    assert resolve_session_grounding_token_budget() == DEFAULT_SESSION_GROUNDING_TOKEN_BUDGET

    monkeypatch.setenv("AFS_SESSION_GROUNDING_TOKEN_BUDGET", "600")
    assert resolve_session_grounding_token_budget() == 600
    assert resolve_session_grounding_token_budget(0) == 0


def test_invalid_session_grounding_budget_uses_default(monkeypatch) -> None:
    monkeypatch.setenv("AFS_SESSION_GROUNDING_TOKEN_BUDGET", "not-a-number")
    assert resolve_session_grounding_token_budget() == DEFAULT_SESSION_GROUNDING_TOKEN_BUDGET
