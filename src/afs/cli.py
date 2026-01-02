"""AFS command-line entry point.

This module provides backward compatibility by delegating to the
modular CLI package structure in afs.cli/.
"""

from __future__ import annotations

from typing import Iterable

from .cli import build_parser, main

__all__ = ["build_parser", "main"]

if __name__ == "__main__":
    raise SystemExit(main())
