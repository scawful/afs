"""LLM bridge — synchronous inference calls routed by guardrails ModelRoute.

Provides a thin, synchronous ``query_llm()`` function that sends a prompt +
context to the provider identified by a guardrails ``ModelRoute``.  All calls
are wrapped in a timeout and broad exception handling so that callers never
crash if the LLM is unreachable or misconfigured.

Supported providers:
    claude  — Anthropic SDK (graceful fallback if not installed)
    gemini  — google.genai (already used in the project)
    local   — OpenAI-compatible API against Ollama (host from nodes.toml)
    codex   — placeholder, returns structured "not available" response
"""

from __future__ import annotations

import json
import logging
import os
import time
from typing import Any

from .guardrails import ModelRoute

logger = logging.getLogger(__name__)

# Ollama host — override with OLLAMA_HOST or AFS_OLLAMA_HOST env var
OLLAMA_HOST = os.getenv("OLLAMA_HOST") or os.getenv("AFS_OLLAMA_HOST") or "http://localhost:11434"

# Default timeout for all LLM calls (seconds)
LLM_TIMEOUT = 30


# ---------------------------------------------------------------------------
# Retry helper
# ---------------------------------------------------------------------------

# Substrings / status codes that indicate transient (retryable) failures.
_TRANSIENT_MARKERS = ("timeout", "timed out", "connection", "429", "503", "502")

# Substrings that indicate non-retryable failures (auth, bad request, missing SDK).
_PERMANENT_MARKERS = ("401", "403", "400", "not installed", "not set")


def _is_transient_error(error_str: str) -> bool:
    """Return True if *error_str* looks like a transient failure worth retrying."""
    lower = error_str.lower()
    # If any permanent marker matches, don't retry.
    if any(m in lower for m in _PERMANENT_MARKERS):
        return False
    # If any transient marker matches, retry.
    return any(m in lower for m in _TRANSIENT_MARKERS)


def _with_retries(
    fn,
    *args,
    max_retries: int = 3,
    retry_base_seconds: float = 1.0,
    **kwargs,
) -> str:
    """Call *fn* up to *max_retries* times with exponential backoff.

    Only retries when the result is an ``"ERROR: ..."`` string that looks
    transient (timeouts, connection errors, 429/502/503).  Non-transient
    errors (auth, bad request, missing SDK) are returned immediately.

    Returns the successful result, or the last error string if all retries
    are exhausted.
    """
    last_result = ""
    for attempt in range(1, max_retries + 1):
        result = fn(*args, **kwargs)
        # Success — no ERROR prefix.
        if not result.startswith("ERROR:"):
            return result
        last_result = result
        # Non-retryable error — bail immediately.
        if not _is_transient_error(result):
            return result
        # Last attempt — don't sleep, just return the error.
        if attempt == max_retries:
            break
        delay = retry_base_seconds * (2 ** (attempt - 1))
        logger.warning(
            "LLM retry %d/%d after %.1fs — %s",
            attempt, max_retries, delay, result,
        )
        time.sleep(delay)
    return last_result


# ---------------------------------------------------------------------------
# Provider implementations
# ---------------------------------------------------------------------------

def _query_claude(
    prompt: str,
    context: dict[str, Any],
    model_id: str,
    system_prompt: str = "",
) -> str:
    """Call Anthropic Claude via the anthropic SDK."""
    try:
        import anthropic
    except ImportError:
        return "ERROR: anthropic SDK not installed — pip install anthropic"

    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        return "ERROR: ANTHROPIC_API_KEY not set"

    try:
        client = anthropic.Anthropic(api_key=api_key, timeout=LLM_TIMEOUT)
        kwargs: dict[str, Any] = {
            "model": model_id,
            "max_tokens": 4096,
            "messages": [
                {
                    "role": "user",
                    "content": f"{prompt}\n\nContext:\n{json.dumps(context, indent=2, default=str)}",
                },
            ],
        }
        if system_prompt:
            kwargs["system"] = system_prompt
        message = client.messages.create(**kwargs)
        # Extract text from content blocks
        parts = []
        for block in message.content:
            if hasattr(block, "text"):
                parts.append(block.text)
        return "\n".join(parts) if parts else "ERROR: empty response from Claude"
    except Exception as exc:
        return f"ERROR: Claude call failed: {exc}"


def _query_gemini(
    prompt: str,
    context: dict[str, Any],
    model_id: str,
    system_prompt: str = "",
) -> str:
    """Call Google Gemini via google.genai."""
    try:
        from google import genai
        from google.genai import types
    except ImportError:
        return "ERROR: google-genai SDK not installed — pip install google-genai"

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    try:
        if api_key:
            client = genai.Client(api_key=api_key)
        else:
            client = genai.Client()

        full_prompt = f"{prompt}\n\nContext:\n{json.dumps(context, indent=2, default=str)}"
        config = types.GenerateContentConfig(
            system_instruction=system_prompt,
        ) if system_prompt else None
        response = client.models.generate_content(
            model=model_id,
            contents=full_prompt,
            config=config,
        )
        text = response.text if hasattr(response, "text") else ""
        return text or "ERROR: empty response from Gemini"
    except Exception as exc:
        return f"ERROR: Gemini call failed: {exc}"


def _query_local(
    prompt: str,
    context: dict[str, Any],
    model_id: str,
    system_prompt: str = "",
) -> str:
    """Call a local Ollama model via OpenAI-compatible API."""
    try:
        import httpx
    except ImportError:
        return "ERROR: httpx not installed — pip install httpx"

    full_prompt = f"{prompt}\n\nContext:\n{json.dumps(context, indent=2, default=str)}"
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": full_prompt})
    payload = {
        "model": model_id,
        "messages": messages,
        "stream": False,
        "options": {
            "temperature": 0.5,
            "num_predict": 4096,
        },
    }

    try:
        with httpx.Client(timeout=LLM_TIMEOUT) as client:
            response = client.post(f"{OLLAMA_HOST}/api/chat", json=payload)
            response.raise_for_status()
            data = response.json()
            content = data.get("message", {}).get("content", "")
            return content or "ERROR: empty response from local model"
    except Exception as exc:
        return f"ERROR: local model call failed: {exc}"


def _query_codex(
    prompt: str,
    context: dict[str, Any],
    model_id: str,
    system_prompt: str = "",
) -> str:
    """Codex provider — placeholder, not yet available."""
    return json.dumps({
        "status": "not_available",
        "provider": "codex",
        "model_id": model_id,
        "message": "Codex provider is not yet available for LLM inference.",
    })


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_PROVIDER_MAP = {
    "claude": _query_claude,
    "gemini": _query_gemini,
    "local": _query_local,
    "codex": _query_codex,
}


def query_llm(
    prompt: str,
    context: dict[str, Any],
    model_route: ModelRoute,
    *,
    system_prompt: str = "",
    max_retries: int = 3,
    retry_base_seconds: float = 1.0,
) -> str:
    """Send a prompt + context to the LLM identified by *model_route*.

    Returns the response text on success, or a string prefixed with ``"ERROR:"``
    on failure.  Never raises — all exceptions are caught and returned as error
    strings so that callers can treat the LLM as an optional enrichment layer.

    Transient failures (timeouts, connection errors, 429/502/503) are retried
    with exponential backoff up to *max_retries* times.  Non-transient errors
    (auth failures, bad requests, missing SDKs) are returned immediately.

    Args:
        prompt: The instruction / question for the LLM.
        context: Structured data to include as context.
        model_route: A ``ModelRoute`` from guardrails indicating provider + model_id.
        system_prompt: Optional system prompt composed by ``model_prompts.py``.
            When provided, it is passed to the provider as a system message
            instead of being concatenated into the user prompt.
        max_retries: Maximum number of attempts for transient failures (default 3).
        retry_base_seconds: Base delay for exponential backoff (default 1.0s).

    Returns:
        Response text from the LLM, or an ``"ERROR: ..."`` string.
    """
    provider = model_route.provider
    handler = _PROVIDER_MAP.get(provider)
    if handler is None:
        return f"ERROR: unknown provider '{provider}'"

    logger.info(
        "LLM bridge: querying provider=%s model=%s prompt_len=%d system_len=%d",
        provider, model_route.model_id, len(prompt), len(system_prompt),
    )

    # Codex is a placeholder — no retries needed.
    use_retries = provider != "codex"

    try:
        if system_prompt:
            call_args = (prompt, context, model_route.model_id, system_prompt)
        else:
            call_args = (prompt, context, model_route.model_id)

        if use_retries:
            result = _with_retries(
                handler,
                *call_args,
                max_retries=max_retries,
                retry_base_seconds=retry_base_seconds,
            )
        else:
            result = handler(*call_args)
        return result
    except Exception as exc:
        return f"ERROR: unexpected failure in LLM bridge: {exc}"
