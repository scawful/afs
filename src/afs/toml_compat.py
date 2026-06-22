"""Compatibility wrapper for TOML parsing across supported Python versions."""

from __future__ import annotations

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python < 3.11
    import tomli as tomllib  # type: ignore[no-redef]

__all__ = ["tomllib"]
