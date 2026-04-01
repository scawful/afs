"""Tests for LLM bridge — provider routing, error handling, graceful fallbacks."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from afs.agents.guardrails import ModelRoute
from afs.agents.llm_bridge import (
    LLM_TIMEOUT,
    OLLAMA_HOST,
    _query_claude,
    _query_codex,
    _query_gemini,
    _query_local,
    query_llm,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def claude_route() -> ModelRoute:
    return ModelRoute(provider="claude", model_id="claude-3-5-sonnet")


@pytest.fixture
def gemini_route() -> ModelRoute:
    return ModelRoute(provider="gemini", model_id="gemini-1.5-pro")


@pytest.fixture
def local_route() -> ModelRoute:
    return ModelRoute(provider="local", model_id="qwen2.5-coder:14b")


@pytest.fixture
def codex_route() -> ModelRoute:
    return ModelRoute(provider="codex", model_id="codex")


@pytest.fixture
def sample_context() -> dict[str, Any]:
    return {"findings": [{"type": "test", "detail": "test finding"}]}


# ---------------------------------------------------------------------------
# Claude provider
# ---------------------------------------------------------------------------

class TestClaudeProvider:
    def test_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")

        mock_block = MagicMock()
        mock_block.text = "Claude analysis result"
        mock_message = MagicMock()
        mock_message.content = [mock_block]

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_message

        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            result = _query_claude("test prompt", {"key": "val"}, "claude-3-5-sonnet")

        assert result == "Claude analysis result"
        mock_anthropic.Anthropic.assert_called_once_with(api_key="sk-test-key", timeout=LLM_TIMEOUT)
        mock_client.messages.create.assert_called_once()

    def test_missing_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        mock_anthropic = MagicMock()
        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            result = _query_claude("test", {}, "claude-3-5-sonnet")
        assert result.startswith("ERROR:")
        assert "ANTHROPIC_API_KEY" in result

    def test_sdk_not_installed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Temporarily hide the anthropic module
        with patch.dict(sys.modules, {"anthropic": None}):
            result = _query_claude("test", {}, "claude-3-5-sonnet")
        assert result.startswith("ERROR:")
        assert "anthropic SDK not installed" in result

    def test_api_exception(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = RuntimeError("connection refused")

        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            result = _query_claude("test", {}, "claude-3-5-sonnet")
        assert result.startswith("ERROR:")
        assert "connection refused" in result

    def test_empty_response(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")

        mock_message = MagicMock()
        mock_message.content = []  # empty content blocks

        mock_client = MagicMock()
        mock_client.messages.create.return_value = mock_message

        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            result = _query_claude("test", {}, "claude-3-5-sonnet")
        assert result.startswith("ERROR:")
        assert "empty response" in result


# ---------------------------------------------------------------------------
# Gemini provider
# ---------------------------------------------------------------------------

class TestGeminiProvider:
    def test_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

        mock_response = MagicMock()
        mock_response.text = "Gemini analysis result"

        mock_client = MagicMock()
        mock_client.models.generate_content.return_value = mock_response

        mock_genai = MagicMock()
        mock_genai.Client.return_value = mock_client

        mock_google = MagicMock()
        mock_google.genai = mock_genai

        with patch.dict(sys.modules, {"google": mock_google, "google.genai": mock_genai}):
            result = _query_gemini("test prompt", {"key": "val"}, "gemini-1.5-pro")

        assert result == "Gemini analysis result"

    def test_sdk_not_installed(self) -> None:
        with patch.dict(sys.modules, {"google": None, "google.genai": None}):
            result = _query_gemini("test", {}, "gemini-1.5-pro")
        assert result.startswith("ERROR:")
        assert "google-genai SDK not installed" in result

    def test_api_exception(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("GEMINI_API_KEY", "test-key")

        mock_client = MagicMock()
        mock_client.models.generate_content.side_effect = RuntimeError("quota exceeded")

        mock_genai = MagicMock()
        mock_genai.Client.return_value = mock_client

        mock_google = MagicMock()
        mock_google.genai = mock_genai

        with patch.dict(sys.modules, {"google": mock_google, "google.genai": mock_genai}):
            result = _query_gemini("test", {}, "gemini-1.5-pro")
        assert result.startswith("ERROR:")
        assert "quota exceeded" in result


# ---------------------------------------------------------------------------
# Local (Ollama) provider
# ---------------------------------------------------------------------------

class TestLocalProvider:
    @staticmethod
    def _make_httpx_mock(*, response_json=None, side_effect=None):
        """Build a mock httpx module whose Client works as a context manager."""
        mock_response = MagicMock()
        mock_response.raise_for_status = MagicMock()
        if response_json is not None:
            mock_response.json.return_value = response_json

        mock_client = MagicMock()
        if side_effect:
            mock_client.post.side_effect = side_effect
        else:
            mock_client.post.return_value = mock_response
        mock_client.__enter__ = MagicMock(return_value=mock_client)
        mock_client.__exit__ = MagicMock(return_value=False)

        mock_httpx = MagicMock()
        mock_httpx.Client.return_value = mock_client
        return mock_httpx

    def test_success(self) -> None:
        mock_httpx = self._make_httpx_mock(
            response_json={"message": {"content": "Local model analysis"}},
        )
        with patch.dict(sys.modules, {"httpx": mock_httpx}):
            result = _query_local("test prompt", {"key": "val"}, "qwen2.5-coder:14b")
        assert result == "Local model analysis"

    def test_connection_error(self) -> None:
        mock_httpx = self._make_httpx_mock(side_effect=ConnectionError("unreachable"))
        with patch.dict(sys.modules, {"httpx": mock_httpx}):
            result = _query_local("test", {}, "qwen2.5-coder:14b")
        assert result.startswith("ERROR:")
        assert "unreachable" in result

    def test_empty_response(self) -> None:
        mock_httpx = self._make_httpx_mock(
            response_json={"message": {"content": ""}},
        )
        with patch.dict(sys.modules, {"httpx": mock_httpx}):
            result = _query_local("test", {}, "qwen2.5-coder:14b")
        assert result.startswith("ERROR:")
        assert "empty response" in result

    def test_httpx_not_installed(self) -> None:
        with patch.dict(sys.modules, {"httpx": None}):
            result = _query_local("test", {}, "qwen2.5-coder:14b")
        assert result.startswith("ERROR:")
        assert "httpx not installed" in result


# ---------------------------------------------------------------------------
# Codex provider (placeholder)
# ---------------------------------------------------------------------------

class TestCodexProvider:
    def test_returns_not_available(self) -> None:
        result = _query_codex("test", {}, "codex")
        parsed = json.loads(result)
        assert parsed["status"] == "not_available"
        assert parsed["provider"] == "codex"
        assert "not yet available" in parsed["message"]


# ---------------------------------------------------------------------------
# query_llm() — main public function
# ---------------------------------------------------------------------------

class TestQueryLLM:
    def test_routes_to_claude(
        self, claude_route: ModelRoute, sample_context: dict[str, Any],
    ) -> None:
        mock_fn = MagicMock(return_value="Claude result")
        with patch.dict("afs.agents.llm_bridge._PROVIDER_MAP", {"claude": mock_fn}):
            result = query_llm("test prompt", sample_context, claude_route)
        assert result == "Claude result"
        mock_fn.assert_called_once_with("test prompt", sample_context, "claude-3-5-sonnet")

    def test_routes_to_gemini(
        self, gemini_route: ModelRoute, sample_context: dict[str, Any],
    ) -> None:
        mock_fn = MagicMock(return_value="Gemini result")
        with patch.dict("afs.agents.llm_bridge._PROVIDER_MAP", {"gemini": mock_fn}):
            result = query_llm("test prompt", sample_context, gemini_route)
        assert result == "Gemini result"
        mock_fn.assert_called_once_with("test prompt", sample_context, "gemini-1.5-pro")

    def test_routes_to_local(
        self, local_route: ModelRoute, sample_context: dict[str, Any],
    ) -> None:
        mock_fn = MagicMock(return_value="Local result")
        with patch.dict("afs.agents.llm_bridge._PROVIDER_MAP", {"local": mock_fn}):
            result = query_llm("test prompt", sample_context, local_route)
        assert result == "Local result"
        mock_fn.assert_called_once_with("test prompt", sample_context, "qwen2.5-coder:14b")

    def test_routes_to_codex(
        self, codex_route: ModelRoute, sample_context: dict[str, Any],
    ) -> None:
        result = query_llm("test prompt", sample_context, codex_route)
        parsed = json.loads(result)
        assert parsed["status"] == "not_available"

    def test_unknown_provider(self, sample_context: dict[str, Any]) -> None:
        route = ModelRoute(provider="unknown_provider", model_id="some-model")
        result = query_llm("test", sample_context, route)
        assert result.startswith("ERROR:")
        assert "unknown provider" in result

    def test_handler_exception_caught(
        self, claude_route: ModelRoute, sample_context: dict[str, Any],
    ) -> None:
        mock_fn = MagicMock(side_effect=RuntimeError("unexpected crash"))
        with patch.dict("afs.agents.llm_bridge._PROVIDER_MAP", {"claude": mock_fn}):
            result = query_llm("test", sample_context, claude_route)
        assert result.startswith("ERROR:")
        assert "unexpected failure" in result

    def test_error_prefix_on_provider_failure(
        self, gemini_route: ModelRoute, sample_context: dict[str, Any],
    ) -> None:
        mock_fn = MagicMock(return_value="ERROR: Gemini quota exceeded")
        with patch.dict("afs.agents.llm_bridge._PROVIDER_MAP", {"gemini": mock_fn}):
            result = query_llm("test", sample_context, gemini_route)
        assert result == "ERROR: Gemini quota exceeded"


# ---------------------------------------------------------------------------
# Timeout handling
# ---------------------------------------------------------------------------

class TestTimeoutHandling:
    def test_claude_timeout_returns_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Claude should return ERROR on timeout, not raise."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test-key")

        mock_client = MagicMock()
        mock_client.messages.create.side_effect = TimeoutError("request timed out")

        mock_anthropic = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client

        with patch.dict(sys.modules, {"anthropic": mock_anthropic}):
            result = _query_claude("test", {}, "claude-3-5-sonnet")
        assert result.startswith("ERROR:")

    def test_local_timeout_returns_error(self) -> None:
        """Local provider should return ERROR on timeout, not raise."""
        mock_httpx = TestLocalProvider._make_httpx_mock(
            side_effect=TimeoutError("request timed out"),
        )
        with patch.dict(sys.modules, {"httpx": mock_httpx}):
            result = _query_local("test", {}, "qwen2.5-coder:14b")
        assert result.startswith("ERROR:")


# ---------------------------------------------------------------------------
# Graceful fallback when SDK not installed
# ---------------------------------------------------------------------------

class TestGracefulFallback:
    def test_anthropic_sdk_missing_via_query_llm(
        self, sample_context: dict[str, Any],
    ) -> None:
        """query_llm should return error string when anthropic is missing."""
        route = ModelRoute(provider="claude", model_id="claude-3-5-sonnet")
        with patch.dict(sys.modules, {"anthropic": None}):
            result = query_llm("test", sample_context, route)
        assert result.startswith("ERROR:")
        assert "anthropic SDK not installed" in result

    def test_genai_sdk_missing_via_query_llm(
        self, sample_context: dict[str, Any],
    ) -> None:
        """query_llm should return error string when google.genai is missing."""
        route = ModelRoute(provider="gemini", model_id="gemini-1.5-pro")
        with patch.dict(sys.modules, {"google": None, "google.genai": None}):
            result = query_llm("test", sample_context, route)
        assert result.startswith("ERROR:")
        assert "google-genai SDK not installed" in result
