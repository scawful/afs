"""Agent harness manifest loading, validation, and export helpers."""

from __future__ import annotations

import os
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def default_manifest_path() -> Path:
    """Return the repo-owned default agent manifest path."""
    override = os.getenv("AFS_AGENT_MANIFEST", "").strip()
    if override:
        return Path(override).expanduser()
    return Path(__file__).resolve().parents[2] / "configs" / "agent_manifest.toml"


@dataclass(frozen=True)
class ManifestIssue:
    level: str
    message: str

    def to_dict(self) -> dict[str, str]:
        return {"level": self.level, "message": self.message}


def load_manifest(path: Path | None = None) -> dict[str, Any]:
    manifest_path = (path or default_manifest_path()).expanduser()
    return tomllib.loads(manifest_path.read_text(encoding="utf-8"))


def _as_list(value: Any) -> list[Any]:
    return value if isinstance(value, list) else []


def _name_set(items: list[Any]) -> set[str]:
    names: set[str] = set()
    for item in items:
        if isinstance(item, dict):
            name = str(item.get("name", "")).strip()
            if name:
                names.add(name)
    return names


def validate_manifest(data: dict[str, Any], *, check_paths: bool = False) -> list[ManifestIssue]:
    issues: list[ManifestIssue] = []

    if int(data.get("version", 0) or 0) <= 0:
        issues.append(ManifestIssue("error", "version must be a positive integer"))

    paths = data.get("paths")
    if not isinstance(paths, dict):
        issues.append(ManifestIssue("error", "[paths] table is required"))
        paths = {}

    harnesses = _as_list(data.get("harnesses"))
    skills = _as_list(data.get("skills"))
    mcp_servers = _as_list(data.get("mcp_servers"))
    harness_names = _name_set(harnesses)
    skill_names = _name_set(skills)
    mcp_names = _name_set(mcp_servers)

    if not harness_names:
        issues.append(ManifestIssue("error", "at least one [[harnesses]] entry is required"))
    if not skill_names:
        issues.append(ManifestIssue("warning", "no [[skills]] entries declared"))

    for index, harness in enumerate(harnesses, start=1):
        if not isinstance(harness, dict):
            issues.append(ManifestIssue("error", f"harness entry {index} must be a table"))
            continue
        name = str(harness.get("name", "")).strip()
        if not name:
            issues.append(ManifestIssue("error", f"harness entry {index} is missing name"))
        for server in _as_list(harness.get("mcp_servers")):
            if str(server) not in mcp_names:
                issues.append(
                    ManifestIssue("warning", f"harness {name or index} references unknown MCP server {server}")
                )

    for index, skill in enumerate(skills, start=1):
        if not isinstance(skill, dict):
            issues.append(ManifestIssue("error", f"skill entry {index} must be a table"))
            continue
        name = str(skill.get("name", "")).strip()
        if not name:
            issues.append(ManifestIssue("error", f"skill entry {index} is missing name"))
        for target in _as_list(skill.get("targets")):
            if str(target) not in harness_names:
                issues.append(
                    ManifestIssue("warning", f"skill {name or index} targets unknown harness {target}")
                )

    if check_paths:
        path_values: list[str] = []
        for value in paths.values():
            if isinstance(value, str):
                path_values.append(value)
        for harness in harnesses:
            if isinstance(harness, dict):
                path_values.extend(str(p) for p in _as_list(harness.get("instructions")))
                path_values.extend(str(p) for p in _as_list(harness.get("skill_roots")))
        for skill in skills:
            if isinstance(skill, dict) and isinstance(skill.get("canonical_path"), str):
                path_values.append(str(skill["canonical_path"]))
        for raw in path_values:
            if raw.startswith("afs ") or raw.startswith("source "):
                continue
            expanded = Path(raw).expanduser()
            if not expanded.exists():
                issues.append(ManifestIssue("warning", f"path does not exist: {raw}"))

    return issues


def summarize_manifest(data: dict[str, Any]) -> dict[str, Any]:
    harnesses = _as_list(data.get("harnesses"))
    skills = _as_list(data.get("skills"))
    mcp_servers = _as_list(data.get("mcp_servers"))
    return {
        "version": data.get("version"),
        "last_reviewed": data.get("last_reviewed"),
        "description": data.get("description", ""),
        "paths": data.get("paths") if isinstance(data.get("paths"), dict) else {},
        "harnesses": [item.get("name") for item in harnesses if isinstance(item, dict)],
        "skills": [item.get("name") for item in skills if isinstance(item, dict)],
        "mcp_servers": [item.get("name") for item in mcp_servers if isinstance(item, dict)],
    }


def export_for_harness(data: dict[str, Any], harness_name: str) -> dict[str, Any]:
    target = harness_name.strip()
    harnesses = [h for h in _as_list(data.get("harnesses")) if isinstance(h, dict)]
    skills = [s for s in _as_list(data.get("skills")) if isinstance(s, dict)]
    mcp_servers = [m for m in _as_list(data.get("mcp_servers")) if isinstance(m, dict)]
    harness = next((h for h in harnesses if str(h.get("name", "")) == target), None)
    if harness is None:
        raise KeyError(f"harness not found: {target}")
    server_names = {str(name) for name in _as_list(harness.get("mcp_servers"))}
    return {
        "paths": data.get("paths") if isinstance(data.get("paths"), dict) else {},
        "harness": harness,
        "skills": [
            skill
            for skill in skills
            if target in {str(name) for name in _as_list(skill.get("targets"))}
        ],
        "mcp_servers": [
            server for server in mcp_servers if str(server.get("name", "")) in server_names
        ],
    }
