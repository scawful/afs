from __future__ import annotations

from afs.gemini_defaults import (
    DEFAULT_GEMINI_SUBTASK_MODEL,
    default_gemini_subtask_model,
    supported_gemini_thinking_levels,
)


def test_subtask_model_has_independent_override(monkeypatch) -> None:
    monkeypatch.setenv("AFS_GEMINI_MODEL", "gemini-generation")
    monkeypatch.setenv("AFS_GEMINI_SUBTASK_MODEL", "gemini-subtask")

    assert default_gemini_subtask_model() == "gemini-subtask"


def test_subtask_model_falls_back_to_generation_override(monkeypatch) -> None:
    monkeypatch.delenv("AFS_GEMINI_SUBTASK_MODEL", raising=False)
    monkeypatch.setenv("AFS_GEMINI_MODEL", "gemini-generation")

    assert default_gemini_subtask_model() == "gemini-generation"


def test_subtask_model_uses_stable_default(monkeypatch) -> None:
    monkeypatch.delenv("AFS_GEMINI_SUBTASK_MODEL", raising=False)
    monkeypatch.delenv("AFS_GEMINI_MODEL", raising=False)

    assert default_gemini_subtask_model() == DEFAULT_GEMINI_SUBTASK_MODEL
    assert DEFAULT_GEMINI_SUBTASK_MODEL == "gemini-3.8-flash"
    assert supported_gemini_thinking_levels(DEFAULT_GEMINI_SUBTASK_MODEL) == (
        "low",
        "medium",
        "high",
    )
