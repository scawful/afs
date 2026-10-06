from __future__ import annotations

from afs.claude_defaults import (
    DEFAULT_CLAUDE_GENERATION_MODEL,
    claude_prompt_cache_enabled,
    claude_system_content,
    default_claude_generation_model,
)


def test_claude_model_is_stable_and_overridable(monkeypatch) -> None:
    monkeypatch.delenv("AFS_CLAUDE_MODEL", raising=False)
    assert default_claude_generation_model() == DEFAULT_CLAUDE_GENERATION_MODEL
    assert DEFAULT_CLAUDE_GENERATION_MODEL == "claude-sonnet-5"

    monkeypatch.setenv("AFS_CLAUDE_MODEL", "company-claude")
    assert default_claude_generation_model() == "company-claude"


def test_claude_prompt_cache_can_be_disabled(monkeypatch) -> None:
    monkeypatch.delenv("AFS_CLAUDE_PROMPT_CACHE", raising=False)
    assert claude_prompt_cache_enabled()
    assert not claude_prompt_cache_enabled({"claude_prompt_cache": False})


def test_claude_system_content_marks_only_stable_prompt() -> None:
    content = claude_system_content("stable", cache=True)
    assert content == [
        {
            "type": "text",
            "text": "stable",
            "cache_control": {"type": "ephemeral"},
        }
    ]
    assert claude_system_content("stable", cache=False) == "stable"
