"""LLM-assisted memory summarization for AFS history consolidation.

Replaces mechanical counter-based memory summaries with LLM-synthesized
natural-language entries.  Uses a fallback chain (local → gemini → claude)
so that summarization degrades gracefully when no LLM is available.

The ``LLMSummarizer`` class is the primary entry point.  It exposes a
``summarize_events()`` method that takes a batch of history event dicts
and returns a natural-language summary string.  On any failure the caller
receives ``None`` and should fall back to the existing counter-based
``_build_memory_entry()`` path in ``memory_consolidation.py``.

Design invariants:
    * Never crash — all LLM failures are caught and logged.
    * Lock file prevents concurrent summarization runs.
    * Prompt is kept short to fit local model context windows.
"""

from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .agents.guardrails import ModelRoute, QuotaTracker, resolve_model
from .agents.llm_bridge import query_llm
from .gemini_defaults import DEFAULT_GEMINI_GENERATION_MODEL

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default summarization prompt
# ---------------------------------------------------------------------------

_SUMMARIZATION_SYSTEM_PROMPT = (
    "You are a concise technical summarizer for the AFS (Agent File System) "
    "context management system.  You receive structured history events that "
    "describe file-system operations, context changes, hook executions, and "
    "agent activity.  Your job is to produce a brief, informative summary."
)

_SUMMARIZATION_USER_PROMPT = (
    "Summarize these AFS operations into 2-3 sentences describing what was "
    "accomplished and what changed.  Focus on the high-level intent rather "
    "than listing every event.  Use past tense.  Do not include timestamps "
    "or event IDs in the summary."
)

# Fallback chain for summarization: cheapest first.
_SUMMARIZER_FALLBACK_CHAIN = ["local", "gemini", "claude"]

# Provider → default model mapping for summarization (mirrors guardrails but
# may diverge — e.g. we can use a smaller Gemini model for summaries).
_SUMMARIZER_MODEL_MAP: dict[str, str] = {
    "local": "qwen2.5-coder:14b",
    "gemini": DEFAULT_GEMINI_GENERATION_MODEL,
    "claude": "claude-3-5-sonnet",
}

# Maximum number of events to include in the prompt context.  Larger batches
# are truncated to keep the prompt within local-model context limits.
_MAX_EVENTS_IN_PROMPT = 80

# Lock file name written next to the checkpoint during LLM summarization.
_LOCK_FILENAME = "llm_summarizer.lock"

# Stale lock threshold in seconds — locks older than this are ignored.
_STALE_LOCK_SECONDS = 600


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------


def _format_events_for_prompt(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Distill event dicts into a compact representation for the LLM prompt.

    Only includes the fields that are useful for summarization to keep the
    prompt small enough for local models.
    """
    compact: list[dict[str, Any]] = []
    for event in events[:_MAX_EVENTS_IN_PROMPT]:
        entry: dict[str, Any] = {}
        if isinstance(event.get("timestamp"), str):
            entry["timestamp"] = event["timestamp"]
        if isinstance(event.get("type"), str):
            entry["kind"] = event["type"]
        if isinstance(event.get("op"), str):
            entry["op"] = event["op"]
        if isinstance(event.get("source"), str):
            entry["source"] = event["source"]
        # Extract a one-line summary from metadata when available.
        metadata = event.get("metadata")
        if isinstance(metadata, dict):
            summary_parts: list[str] = []
            mount_type = metadata.get("mount_type")
            relative_path = metadata.get("relative_path")
            alias = metadata.get("alias")
            detail = metadata.get("detail")
            if isinstance(mount_type, str):
                summary_parts.append(mount_type)
            if isinstance(relative_path, str) and relative_path.strip():
                summary_parts.append(relative_path)
            elif isinstance(alias, str) and alias.strip():
                summary_parts.append(alias)
            if isinstance(detail, str) and detail.strip():
                summary_parts.append(detail)
            if summary_parts:
                entry["summary"] = " / ".join(summary_parts)
        if entry:
            compact.append(entry)

    return compact


def _build_prompt_context(
    events: list[dict[str, Any]],
    context_root: Path,
) -> dict[str, Any]:
    """Build the structured context dict sent alongside the prompt."""
    context_label = (
        context_root.parent.name
        if context_root.name == ".context" and context_root.parent.name
        else context_root.name
    )
    formatted = _format_events_for_prompt(events)
    return {
        "context": context_label,
        "event_count": len(events),
        "events": formatted,
    }


# ---------------------------------------------------------------------------
# Lock helpers
# ---------------------------------------------------------------------------


def _acquire_lock(lock_path: Path) -> bool:
    """Write a lock file.  Returns False if a non-stale lock already exists."""
    try:
        if lock_path.exists():
            try:
                age = time.time() - lock_path.stat().st_mtime
                if age < _STALE_LOCK_SECONDS:
                    logger.debug("LLM summarizer lock held (age=%.1fs)", age)
                    return False
                logger.info("Removing stale LLM summarizer lock (age=%.0fs)", age)
            except OSError:
                pass
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.write_text(
            json.dumps(
                {
                    "pid": os.getpid(),
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }
            ),
            encoding="utf-8",
        )
        return True
    except OSError as exc:
        logger.warning("Failed to acquire LLM summarizer lock: %s", exc)
        return False


def _release_lock(lock_path: Path) -> None:
    """Remove the lock file, ignoring errors."""
    try:
        lock_path.unlink(missing_ok=True)
    except OSError:
        pass


# ---------------------------------------------------------------------------
# LLMSummarizer
# ---------------------------------------------------------------------------


class LLMSummarizer:
    """Produce natural-language memory summaries via an LLM fallback chain.

    Constructor parameters:
        preferred_provider: First provider to try (default ``"local"``).
        model_override: Explicit model ID, skipping the default map.
        fallback_chain: Ordered list of providers to try.
        quota_tracker: Shared ``QuotaTracker`` instance (created if omitted).
        task_tier: Guardrail tier for model resolution (default ``"background"``).
        lock_dir: Directory for the lock file (default: next to checkpoint).
    """

    def __init__(
        self,
        *,
        preferred_provider: str = "local",
        model_override: str | None = None,
        fallback_chain: list[str] | None = None,
        quota_tracker: QuotaTracker | None = None,
        task_tier: str = "background",
        lock_dir: Path | None = None,
    ) -> None:
        self._preferred_provider = preferred_provider
        self._model_override = model_override
        self._fallback_chain = fallback_chain or list(_SUMMARIZER_FALLBACK_CHAIN)
        self._quota_tracker = quota_tracker
        self._task_tier = task_tier
        self._lock_dir = lock_dir

    # ------------------------------------------------------------------
    # Model resolution
    # ------------------------------------------------------------------

    def _resolve_route(self) -> ModelRoute:
        """Resolve the model route using the fallback chain."""
        if self._model_override:
            return ModelRoute(
                provider=self._preferred_provider,
                model_id=self._model_override,
                reason="explicit model override for summarization",
            )
        # Try preferred provider first, then the rest of the chain.
        chain = list(self._fallback_chain)
        if self._preferred_provider not in chain:
            chain.insert(0, self._preferred_provider)
        elif chain[0] != self._preferred_provider:
            chain.remove(self._preferred_provider)
            chain.insert(0, self._preferred_provider)

        try:
            return resolve_model(
                preferred=chain[0],
                fallback_chain=chain,
                quota_tracker=self._quota_tracker,
                task_tier=self._task_tier,
            )
        except RuntimeError:
            # All quotas exhausted — return local as last resort.
            return ModelRoute(
                provider="local",
                model_id=_SUMMARIZER_MODEL_MAP.get("local", "qwen2.5-coder:14b"),
                reason="all quotas exhausted, falling back to local",
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def summarize_events(
        self,
        events: list[dict[str, Any]],
        context_root: Path,
        config: Any | None = None,
    ) -> str | None:
        """Summarize *events* into a natural-language string.

        Returns ``None`` on any failure so the caller can fall back to the
        counter-based summary.  Never raises.

        Args:
            events: List of history event dicts (from JSONL).
            context_root: The AFS context root for labeling.
            config: Optional AFSConfig (unused today, reserved for future
                prompt-tuning knobs).
        """
        if not events:
            return None

        lock_path = (self._lock_dir / _LOCK_FILENAME) if self._lock_dir else None

        try:
            return self._summarize_with_lock(events, context_root, lock_path)
        except Exception as exc:
            logger.warning(
                "LLM summarization failed (falling back to counter-based): %s",
                exc,
            )
            return None

    def _summarize_with_lock(
        self,
        events: list[dict[str, Any]],
        context_root: Path,
        lock_path: Path | None,
    ) -> str | None:
        """Inner summarization with optional lock file protection."""
        if lock_path is not None:
            if not _acquire_lock(lock_path):
                logger.info("Skipping LLM summarization — lock held")
                return None

        try:
            return self._call_llm(events, context_root)
        finally:
            if lock_path is not None:
                _release_lock(lock_path)

    def _call_llm(
        self,
        events: list[dict[str, Any]],
        context_root: Path,
    ) -> str | None:
        """Build the prompt, call the LLM, and validate the response."""
        route = self._resolve_route()
        prompt_context = _build_prompt_context(events, context_root)

        logger.info(
            "LLM summarization: provider=%s model=%s events=%d",
            route.provider,
            route.model_id,
            len(events),
        )

        result = query_llm(
            _SUMMARIZATION_USER_PROMPT,
            prompt_context,
            route,
            system_prompt=_SUMMARIZATION_SYSTEM_PROMPT,
            max_retries=2,
            retry_base_seconds=0.5,
        )

        if result.startswith("ERROR:"):
            logger.warning("LLM summarization error: %s", result)
            return None

        # Basic validation: must be non-empty and look like prose.
        cleaned = result.strip()
        if not cleaned or len(cleaned) < 10:
            logger.warning(
                "LLM summarization returned too-short response (%d chars)",
                len(cleaned),
            )
            return None

        return cleaned


# ---------------------------------------------------------------------------
# Convenience factory
# ---------------------------------------------------------------------------


def create_summarizer(
    *,
    summarize_with_llm: bool = False,
    summarizer_provider: str = "local",
    summarizer_model: str = "",
    lock_dir: Path | None = None,
) -> LLMSummarizer | None:
    """Create an ``LLMSummarizer`` if LLM summarization is enabled.

    Returns ``None`` when ``summarize_with_llm`` is ``False``, so callers
    can unconditionally check ``if summarizer is not None``.
    """
    if not summarize_with_llm:
        return None

    return LLMSummarizer(
        preferred_provider=summarizer_provider or "local",
        model_override=summarizer_model or None,
        lock_dir=lock_dir,
    )
