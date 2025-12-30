"""Minimal config loader for AFS."""

from __future__ import annotations

import os
import tomllib
from pathlib import Path
from typing import Any

from .schema import AFSConfig


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def _expand_path(path: str | Path) -> Path:
    return Path(path).expanduser().resolve()


def _parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _expand_config_paths(config_data: dict[str, Any]) -> None:
    if "general" in config_data:
        general = config_data["general"]
        if "context_root" in general:
            general["context_root"] = _expand_path(general["context_root"])
        if "agent_workspaces_dir" in general:
            general["agent_workspaces_dir"] = _expand_path(general["agent_workspaces_dir"])
        if "python_executable" in general:
            python_exec = general["python_executable"]
            if isinstance(python_exec, str) and python_exec.startswith("~"):
                general["python_executable"] = _expand_path(python_exec)
        if "workspace_directories" in general:
            for ws_dir in general["workspace_directories"]:
                if "path" in ws_dir:
                    ws_dir["path"] = _expand_path(ws_dir["path"])

    if "plugins" in config_data and "plugin_dirs" in config_data["plugins"]:
        config_data["plugins"]["plugin_dirs"] = [
            _expand_path(p) for p in config_data["plugins"]["plugin_dirs"]
        ]

    if "projects" in config_data:
        for project in config_data["projects"]:
            if "path" in project:
                project["path"] = _expand_path(project["path"])
            if "knowledge_roots" in project:
                project["knowledge_roots"] = [
                    _expand_path(p) for p in project["knowledge_roots"]
                ]


def load_config(config_path: Path | None = None, merge_user: bool = True) -> dict[str, Any]:
    """Load configuration with basic precedence and path expansion."""
    env_config = os.environ.get("AFS_CONFIG_PATH")
    if config_path is None and env_config:
        config_path = Path(env_config).expanduser()

    prefer_user = _parse_bool(os.environ.get("AFS_PREFER_USER_CONFIG"), default=True)
    prefer_repo = _parse_bool(os.environ.get("AFS_PREFER_REPO_CONFIG"))
    if prefer_repo:
        prefer_user = False

    config_data: dict[str, Any] = {}
    user_raw: dict[str, Any] = {}
    local_raw: dict[str, Any] = {}
    explicit_raw: dict[str, Any] = {}

    if merge_user:
        user_path = Path.home() / ".config" / "afs" / "config.toml"
        if user_path.exists():
            with open(user_path, "rb") as f:
                user_raw = tomllib.load(f)

    local_path = Path("afs.toml")
    if local_path.exists():
        with open(local_path, "rb") as f:
            local_raw = tomllib.load(f)

    if prefer_user:
        config_data = _deep_merge(config_data, local_raw)
        config_data = _deep_merge(config_data, user_raw)
    else:
        config_data = _deep_merge(config_data, user_raw)
        config_data = _deep_merge(config_data, local_raw)

    if config_path and config_path.exists():
        with open(config_path, "rb") as f:
            explicit_raw = tomllib.load(f)
        config_data = _deep_merge(config_data, explicit_raw)

    _merge_workspace_registry(config_data)
    _expand_config_paths(config_data)
    return config_data


def load_config_model(
    config_path: Path | None = None,
    merge_user: bool = True,
) -> AFSConfig:
    """Load configuration and return a typed model."""
    data = load_config(config_path=config_path, merge_user=merge_user)
    return AFSConfig.from_dict(data)


def _merge_workspace_registry(config_data: dict[str, Any]) -> None:
    general = config_data.setdefault("general", {})
    raw_context_root = general.get("context_root", Path.home() / ".context")
    context_root = _expand_path(raw_context_root)
    registry_path = context_root / "workspaces.toml"
    if not registry_path.exists():
        return

    try:
        with open(registry_path, "rb") as f:
            registry = tomllib.load(f)
    except Exception:
        return

    entries = registry.get("workspaces", [])
    if not isinstance(entries, list):
        return

    existing = general.get("workspace_directories")
    if not isinstance(existing, list):
        existing = []

    merged = list(existing)
    seen = {
        item.get("path")
        for item in merged
        if isinstance(item, dict) and item.get("path")
    }

    for entry in entries:
        if not isinstance(entry, dict):
            continue
        path = entry.get("path")
        if not path or path in seen:
            continue
        merged.append(entry)
        seen.add(path)

    general["workspace_directories"] = merged
