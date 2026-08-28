"""Current, overridable defaults shared by AFS Gemini integrations."""

from __future__ import annotations

import os

DEFAULT_GEMINI_GENERATION_MODEL = "gemini-3.7-flash"
DEFAULT_GEMINI_EMBEDDING_MODEL = "gemini-embedding-2"
GEMINI_THINKING_LEVELS = ("minimal", "low", "medium", "high")


def default_gemini_generation_model() -> str:
    """Return the configured Gemini generation model without fixing host policy."""
    return os.getenv("AFS_GEMINI_MODEL", "").strip() or DEFAULT_GEMINI_GENERATION_MODEL


def supported_gemini_thinking_levels(model: str) -> tuple[str, ...]:
    """Return known thinking levels, retaining broad compatibility for unknown models."""
    normalized = model.strip().lower().removeprefix("models/")
    if normalized.startswith("gemini-3.7-flash"):
        return ("low", "medium", "high")
    if normalized.startswith("gemini-3.1-pro"):
        return ("low", "medium", "high")
    return GEMINI_THINKING_LEVELS


def validate_gemini_thinking_level(model: str, level: str) -> str:
    """Normalize and validate a thinking level against known model constraints."""
    normalized = level.strip().lower()
    if not normalized:
        return ""
    allowed = supported_gemini_thinking_levels(model)
    if normalized not in allowed:
        expected = ", ".join(allowed)
        raise ValueError(
            f"Gemini model {model!r} does not support thinking level {normalized!r}; "
            f"expected one of: {expected}"
        )
    return normalized
