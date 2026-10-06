"""Provider-neutral session grounding for interactive agent harnesses."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from .model_prompts import build_hook_injection
from .session_bootstrap import build_session_bootstrap

DEFAULT_SESSION_GROUNDING_TOKEN_BUDGET = 0


def resolve_session_grounding_token_budget(value: int | None = None) -> int:
    """Resolve an optional overall budget; zero keeps built-in section bounds."""
    raw: int | str | None = value
    if raw is None:
        raw = os.getenv(
            "AFS_SESSION_GROUNDING_TOKEN_BUDGET",
            str(DEFAULT_SESSION_GROUNDING_TOKEN_BUDGET),
        )
    try:
        return max(0, int(raw or 0))
    except (TypeError, ValueError):
        return DEFAULT_SESSION_GROUNDING_TOKEN_BUDGET


def build_session_grounding(
    manager: Any,
    context_path: Path,
    *,
    project_path: Path | None = None,
    event: str = "SessionStart",
    prompt: str = "",
    skills_prompt: str | None = None,
    include_skills: bool | None = None,
    token_budget: int | None = None,
) -> str:
    """Return bounded AFS context suitable for any host's system prompt.

    ``SessionStart`` builds the normal read-only bootstrap without writing
    artifacts or recording an event. ``UserPromptSubmit`` only emits the
    just-in-time communication guardrail when the prompt needs it.
    """
    normalized_event = str(event or "").strip() or "SessionStart"
    session_state = None
    if normalized_event != "UserPromptSubmit":
        skills_enabled = (
            os.getenv("AFS_SESSION_SKILLS_MATCH_ENABLED", "1") != "0"
            if include_skills is None
            else include_skills
        )
        effective_skills_prompt = (
            os.getenv("AFS_SESSION_SKILLS_PROMPT", "").strip()[:8192]
            if skills_prompt is None
            else skills_prompt.strip()[:8192]
        )
        if not skills_enabled:
            effective_skills_prompt = ""
        session_state = build_session_bootstrap(
            manager,
            context_path,
            project_path=project_path,
            token_budget=resolve_session_grounding_token_budget(token_budget),
            record_event=False,
            skills_prompt=effective_skills_prompt,
            include_skills=skills_enabled,
        )

    return build_hook_injection(
        event=normalized_event,
        context_path=context_path,
        session_state=session_state,
        prompt=prompt,
    )
