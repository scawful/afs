"""Current, overridable defaults shared by AFS Claude integrations."""

from __future__ import annotations

import os
from typing import Any

DEFAULT_CLAUDE_GENERATION_MODEL = "claude-sonnet-5"


def default_claude_generation_model() -> str:
    """Return the configured Claude model without fixing host transport policy."""
    return os.getenv("AFS_CLAUDE_MODEL", "").strip() or DEFAULT_CLAUDE_GENERATION_MODEL


def claude_prompt_cache_enabled(extra: dict[str, Any] | None = None) -> bool:
    """Resolve whether stable Claude system content should request prompt caching."""
    config = extra or {}
    raw = config.get("claude_prompt_cache")
    if isinstance(raw, dict):
        raw = raw.get("enabled")
    if raw is None:
        raw = config.get("claude_prompt_cache_enabled")
    if raw is None:
        raw = os.getenv("AFS_CLAUDE_PROMPT_CACHE", "on")
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() not in {"0", "false", "no", "off", "disabled"}


def claude_system_content(
    prompt: str,
    *,
    cache: bool,
) -> str | list[dict[str, Any]]:
    """Return Anthropic system content with one cache breakpoint when enabled."""
    if not cache:
        return prompt
    return [
        {
            "type": "text",
            "text": prompt,
            "cache_control": {"type": "ephemeral"},
        }
    ]
