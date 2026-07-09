"""Release metadata smoke tests."""

from __future__ import annotations

from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib

import afs
from afs.mcp.transport import SERVER_VERSION

ROOT = Path(__file__).resolve().parents[1]


def test_package_version_matches_metadata() -> None:
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert project["version"] == afs.__version__
    assert SERVER_VERSION == afs.__version__


def test_release_docs_reference_current_version() -> None:
    assert f"## [{afs.__version__}]" in (ROOT / "CHANGELOG.md").read_text()
    assert f"version: {afs.__version__}" in (ROOT / "CITATION.cff").read_text()
