"""Workspace sync helpers for AFS."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

from .schema import AFSConfig, WorkspaceDirectory


def load_workspace_entries(
    root: Path,
    *,
    include_sections: bool = True,
    include_items: bool = True,
    include_local: bool = True,
) -> list[WorkspaceDirectory]:
    workspace_toml = root / "WORKSPACE.toml"
    if not workspace_toml.exists():
        raise FileNotFoundError(f"missing WORKSPACE.toml at {workspace_toml}")

    data = _load_toml(workspace_toml)
    if include_local:
        local_data = _load_toml(root / "WORKSPACE.local.toml")
        data = _merge_workspace_data(data, local_data)

    entries: list[WorkspaceDirectory] = []
    if include_sections:
        entries.extend(_parse_workspace_section(root, data.get("section", [])))
    if include_items:
        entries.extend(_parse_workspace_items(root, data.get("item", [])))

    return _dedupe_entries(entries)


def sync_workspace_config(
    config: AFSConfig,
    entries: Iterable[WorkspaceDirectory],
    *,
    merge: bool,
) -> list[WorkspaceDirectory]:
    new_entries = list(entries)
    if merge:
        combined = _merge_workspace_entries(config.general.workspace_directories, new_entries)
        config.general.workspace_directories = combined
    else:
        config.general.workspace_directories = new_entries
    return config.general.workspace_directories


def resolve_config_output(config_path: Path | None) -> Path:
    if config_path:
        return config_path.expanduser().resolve()
    return Path.home() / ".config" / "afs" / "afs.toml"


def _merge_workspace_entries(
    existing: Iterable[WorkspaceDirectory],
    new_entries: Iterable[WorkspaceDirectory],
) -> list[WorkspaceDirectory]:
    combined: list[WorkspaceDirectory] = []
    seen: set[Path] = set()
    for entry in [*existing, *new_entries]:
        resolved = entry.path.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        combined.append(entry)
    return combined


def _parse_workspace_section(
    root: Path,
    sections: Iterable[dict],
) -> list[WorkspaceDirectory]:
    entries: list[WorkspaceDirectory] = []
    for section in sections:
        if not isinstance(section, dict):
            continue
        rel = section.get("path")
        if not isinstance(rel, str):
            continue
        path = (root / rel).expanduser().resolve()
        if not path.exists() or not path.is_dir():
            continue
        desc = section.get("description") if isinstance(section.get("description"), str) else None
        entries.append(WorkspaceDirectory(path=path, description=desc))
    return entries


def _parse_workspace_items(
    root: Path,
    items: Iterable[dict],
) -> list[WorkspaceDirectory]:
    entries: list[WorkspaceDirectory] = []
    for item in items:
        if not isinstance(item, dict):
            continue
        rel = item.get("path")
        if not isinstance(rel, str):
            continue
        path = (root / rel).expanduser().resolve()
        if not path.exists() or not path.is_dir():
            continue
        desc = item.get("description") if isinstance(item.get("description"), str) else None
        entries.append(WorkspaceDirectory(path=path, description=desc))
    return entries


def _dedupe_entries(entries: Iterable[WorkspaceDirectory]) -> list[WorkspaceDirectory]:
    deduped: list[WorkspaceDirectory] = []
    seen: set[Path] = set()
    for entry in entries:
        resolved = entry.path.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        deduped.append(entry)
    return deduped


def _load_toml(path: Path) -> dict:
    if not path.exists():
        return {}
    try:
        import tomllib
    except ImportError:
        return {}
    try:
        with path.open("rb") as handle:
            return tomllib.load(handle)
    except Exception:
        return {}


def _merge_workspace_data(base: dict, override: dict) -> dict:
    if not override:
        return base
    merged = dict(base)
    for key in ("section", "item"):
        base_list = merged.get(key, [])
        override_list = override.get(key, [])
        if isinstance(base_list, list) and isinstance(override_list, list):
            merged[key] = [*base_list, *override_list]
    return merged
