#!/usr/bin/env python3
"""Release metadata and package smoke checks for AFS."""

from __future__ import annotations

import argparse
import pathlib
import sys
import zipfile
from typing import Any, cast

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover - Python 3.10 fallback
    import tomli as tomllib


def _project_version() -> str:
    data = cast(dict[str, Any], tomllib.loads(pathlib.Path("pyproject.toml").read_text()))
    return cast(str, data["project"]["version"])


def check_metadata() -> int:
    version = _project_version()
    version_py = pathlib.Path("src/afs/version.py").read_text()
    changelog = pathlib.Path("CHANGELOG.md").read_text()
    citation = pathlib.Path("CITATION.cff").read_text()
    errors: list[str] = []
    if f'__version__ = "{version}"' not in version_py:
        errors.append("src/afs/version.py does not match pyproject.toml")
    if f"## [{version}]" not in changelog:
        errors.append("CHANGELOG.md missing release entry")
    if f"version: {version}" not in citation:
        errors.append("CITATION.cff version does not match pyproject.toml")
    if errors:
        for error in errors:
            print(f"release metadata error: {error}", file=sys.stderr)
        return 1
    print(f"release metadata ok: {version}")
    return 0


def check_dist() -> int:
    version = _project_version()
    wheels = sorted(pathlib.Path("dist").glob("afs-*.whl"))
    sdists = sorted(pathlib.Path("dist").glob("afs-*.tar.gz"))
    if not wheels or not sdists:
        print("release package error: expected wheel and sdist under dist/", file=sys.stderr)
        return 1
    wheel = wheels[-1]
    if version not in wheel.name:
        print(f"release package error: wheel name does not include {version}: {wheel}", file=sys.stderr)
        return 1
    with zipfile.ZipFile(wheel) as zf:
        names = set(zf.namelist())
    required = {
        "afs/version.py",
        "afs/mcp_server.py",
        "afs/bundled_skills/context-setup/SKILL.md",
    }
    missing = [name for name in required if not any(item.endswith(name) for item in names)]
    if missing:
        print(f"release package error: missing from wheel: {missing}", file=sys.stderr)
        return 1
    print(f"package ok: {wheel.name} ({version})")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dist", action="store_true", help="also validate built dist artifacts")
    args = parser.parse_args()
    status = check_metadata()
    if status:
        return status
    if args.dist:
        return check_dist()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
