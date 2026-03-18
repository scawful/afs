"""Tests for CLI contextual hint output."""

from __future__ import annotations

import os
from unittest.mock import patch

from afs.cli.core import _hint


def test_hint_plain_no_tty() -> None:
    """Hint returns plain text when not a TTY."""
    with patch("sys.stdout") as mock_stdout:
        mock_stdout.isatty.return_value = False
        result = _hint("try this")
    assert result == "  try this"
    assert "\033" not in result


def test_hint_ansi_when_tty() -> None:
    """Hint includes ANSI dim codes when TTY."""
    with patch("sys.stdout") as mock_stdout, patch.dict(os.environ, {}, clear=False):
        mock_stdout.isatty.return_value = True
        # Ensure NO_COLOR is not set
        os.environ.pop("NO_COLOR", None)
        result = _hint("try this")
    assert "\033[2m" in result
    assert "try this" in result


def test_hint_no_color_env() -> None:
    """Hint skips ANSI when NO_COLOR is set."""
    with patch("sys.stdout") as mock_stdout, patch.dict(os.environ, {"NO_COLOR": "1"}):
        mock_stdout.isatty.return_value = True
        result = _hint("try this")
    assert "\033" not in result
