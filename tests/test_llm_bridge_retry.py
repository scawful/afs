"""Tests for LLM bridge retry logic — _is_transient_error, _with_retries, query_llm retries."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from afs.agents.guardrails import ModelRoute
from afs.agents.llm_bridge import (
    _is_transient_error,
    _with_retries,
    query_llm,
)

# ---------------------------------------------------------------------------
# _is_transient_error
# ---------------------------------------------------------------------------


class TestIsTransientError:
    """Classify error strings as transient (retryable) or permanent."""

    @pytest.mark.parametrize(
        "error_str",
        [
            "ERROR: Claude call failed: timeout",
            "ERROR: request timed out after 30s",
            "ERROR: connection refused",
            "ERROR: Gemini call failed: 429 Too Many Requests",
            "ERROR: local model call failed: 503 Service Unavailable",
            "ERROR: 502 Bad Gateway",
        ],
        ids=["timeout", "timed-out", "connection", "429", "503", "502"],
    )
    def test_transient_errors_return_true(self, error_str: str) -> None:
        assert _is_transient_error(error_str) is True

    @pytest.mark.parametrize(
        "error_str",
        [
            "ERROR: ANTHROPIC_API_KEY not set",
            "ERROR: anthropic SDK not installed",
            "ERROR: Claude call failed: 401 Unauthorized",
            "ERROR: Gemini call failed: 403 Forbidden",
            "ERROR: Claude call failed: 400 Bad Request",
        ],
        ids=["not-set", "not-installed", "401", "403", "400"],
    )
    def test_permanent_errors_return_false(self, error_str: str) -> None:
        assert _is_transient_error(error_str) is False

    def test_unrecognised_error_returns_false(self) -> None:
        """An error that matches no marker at all is not retried."""
        assert _is_transient_error("ERROR: something completely unknown") is False


# ---------------------------------------------------------------------------
# _with_retries
# ---------------------------------------------------------------------------


class TestWithRetries:
    """Exercise the retry wrapper around provider functions."""

    def test_success_on_first_try(self) -> None:
        fn = MagicMock(return_value="all good")
        result = _with_retries(fn, max_retries=3, retry_base_seconds=0.0)
        assert result == "all good"
        assert fn.call_count == 1

    @patch("afs.agents.llm_bridge.time.sleep")
    def test_retries_on_transient_then_succeeds(self, mock_sleep: MagicMock) -> None:
        fn = MagicMock(side_effect=[
            "ERROR: connection refused",
            "success after retry",
        ])
        result = _with_retries(fn, max_retries=3, retry_base_seconds=1.0)
        assert result == "success after retry"
        assert fn.call_count == 2
        mock_sleep.assert_called_once_with(1.0)  # base * 2^0

    def test_no_retry_on_permanent_error(self) -> None:
        fn = MagicMock(return_value="ERROR: ANTHROPIC_API_KEY not set")
        result = _with_retries(fn, max_retries=3, retry_base_seconds=0.0)
        assert result == "ERROR: ANTHROPIC_API_KEY not set"
        assert fn.call_count == 1

    @patch("afs.agents.llm_bridge.time.sleep")
    def test_returns_last_error_after_max_retries(self, mock_sleep: MagicMock) -> None:
        fn = MagicMock(return_value="ERROR: connection refused")
        result = _with_retries(fn, max_retries=3, retry_base_seconds=1.0)
        assert result == "ERROR: connection refused"
        assert fn.call_count == 3
        # Two sleeps for retries 1 and 2; the third attempt returns immediately
        assert mock_sleep.call_count == 2

    @patch("afs.agents.llm_bridge.time.sleep")
    def test_exponential_backoff_delays(self, mock_sleep: MagicMock) -> None:
        fn = MagicMock(return_value="ERROR: 503 Service Unavailable")
        _with_retries(fn, max_retries=4, retry_base_seconds=2.0)
        # Delays: 2*2^0=2, 2*2^1=4, 2*2^2=8 (no sleep on last attempt)
        assert mock_sleep.call_args_list == [
            ((2.0,),),
            ((4.0,),),
            ((8.0,),),
        ]

    def test_forwards_positional_args(self) -> None:
        fn = MagicMock(return_value="ok")
        _with_retries(fn, "arg1", "arg2", max_retries=1, retry_base_seconds=0.0)
        fn.assert_called_once_with("arg1", "arg2")

    def test_forwards_extra_kwargs(self) -> None:
        fn = MagicMock(return_value="ok")
        _with_retries(fn, max_retries=1, retry_base_seconds=0.0, extra="val")
        fn.assert_called_once_with(extra="val")


# ---------------------------------------------------------------------------
# query_llm — retry integration
# ---------------------------------------------------------------------------


class TestQueryLLMRetryKwargs:
    """Verify query_llm passes max_retries and retry_base_seconds through."""

    @patch("afs.agents.llm_bridge.time.sleep")
    def test_query_llm_accepts_max_retries(self, mock_sleep: MagicMock) -> None:
        """max_retries=1 should prevent any retry."""
        route = ModelRoute(provider="claude", model_id="claude-3-5-sonnet")
        mock_fn = MagicMock(return_value="ERROR: connection refused")
        with patch.dict("afs.agents.llm_bridge._PROVIDER_MAP", {"claude": mock_fn}):
            result = query_llm("prompt", {}, route, max_retries=1)
        assert result.startswith("ERROR:")
        assert mock_fn.call_count == 1
        mock_sleep.assert_not_called()

    @patch("afs.agents.llm_bridge.time.sleep")
    def test_query_llm_accepts_retry_base_seconds(self, mock_sleep: MagicMock) -> None:
        route = ModelRoute(provider="gemini", model_id="gemini-1.5-pro")
        mock_fn = MagicMock(side_effect=[
            "ERROR: 429 Too Many Requests",
            "success",
        ])
        with patch.dict("afs.agents.llm_bridge._PROVIDER_MAP", {"gemini": mock_fn}):
            result = query_llm(
                "prompt", {}, route,
                max_retries=3,
                retry_base_seconds=5.0,
            )
        assert result == "success"
        mock_sleep.assert_called_once_with(5.0)

    @patch("afs.agents.llm_bridge.time.sleep")
    def test_codex_provider_skips_retries(self, mock_sleep: MagicMock) -> None:
        """The codex placeholder should never be retried."""
        route = ModelRoute(provider="codex", model_id="codex")
        result = query_llm("prompt", {}, route, max_retries=5)
        # Codex returns a JSON blob, not an ERROR, so no retries
        assert '"not_available"' in result
        mock_sleep.assert_not_called()
