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
from typing import Any

from .guardrails import ModelRoute

logger = logging.getLogger(__name__)

# Ollama host from nodes.toml — medical-mechanica workstation
OLLAMA_HOST = "http://100.104.53.21:11434"

# Default timeout for all LLM calls (seconds)
LLM_TIMEOUT = 30


# ---------------------------------------------------------------------------
# Provider implementations
# ---------------------------------------------------------------------------

def _query_claude(prompt: str, context: dict[str, Any], model_id: str) -> str:
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
        message = client.messages.create(
            model=model_id,
            max_tokens=4096,
            messages=[
                {
                    "role": "user",
                    "content": f"{prompt}\n\nContext:\n{json.dumps(context, indent=2, default=str)}",
                },
            ],
        )
        # Extract text from content blocks
        parts = []
        for block in message.content:
            if hasattr(block, "text"):
                parts.append(block.text)
        return "\n".join(parts) if parts else "ERROR: empty response from Claude"
    except Exception as exc:
        return f"ERROR: Claude call failed: {exc}"


def _query_gemini(prompt: str, context: dict[str, Any], model_id: str) -> str:
    """Call Google Gemini via google.genai."""
    try:
        from google import genai
    except ImportError:
        return "ERROR: google-genai SDK not installed — pip install google-genai"

    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    try:
        if api_key:
            client = genai.Client(api_key=api_key)
        else:
            client = genai.Client()

        full_prompt = f"{prompt}\n\nContext:\n{json.dumps(context, indent=2, default=str)}"
        response = client.models.generate_content(
            model=model_id,
            contents=full_prompt,
        )
        text = response.text if hasattr(response, "text") else ""
        return text or "ERROR: empty response from Gemini"
    except Exception as exc:
        return f"ERROR: Gemini call failed: {exc}"


def _query_local(prompt: str, context: dict[str, Any], model_id: str) -> str:
    """Call a local Ollama model via OpenAI-compatible API."""
    try:
        import httpx
    except ImportError:
        return "ERROR: httpx not installed — pip install httpx"

    full_prompt = f"{prompt}\n\nContext:\n{json.dumps(context, indent=2, default=str)}"
    payload = {
        "model": model_id,
        "messages": [{"role": "user", "content": full_prompt}],
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


def _query_codex(prompt: str, context: dict[str, Any], model_id: str) -> str:
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


def query_llm(prompt: str, context: dict[str, Any], model_route: ModelRoute) -> str:
    """Send a prompt + context to the LLM identified by *model_route*.

    Returns the response text on success, or a string prefixed with ``"ERROR:"``
    on failure.  Never raises — all exceptions are caught and returned as error
    strings so that callers can treat the LLM as an optional enrichment layer.

    Args:
        prompt: The instruction / question for the LLM.
        context: Structured data to include as context.
        model_route: A ``ModelRoute`` from guardrails indicating provider + model_id.

    Returns:
        Response text from the LLM, or an ``"ERROR: ..."`` string.
    """
    provider = model_route.provider
    handler = _PROVIDER_MAP.get(provider)
    if handler is None:
        return f"ERROR: unknown provider '{provider}'"

    logger.info(
        "LLM bridge: querying provider=%s model=%s prompt_len=%d",
        provider, model_route.model_id, len(prompt),
    )

    try:
        result = handler(prompt, context, model_route.model_id)
        return result
    except Exception as exc:
        return f"ERROR: unexpected failure in LLM bridge: {exc}"
