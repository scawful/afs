"""Minimal config loader for AFS."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import tomllib

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


def find_repo_config(start_dir: Path | None = None) -> Path | None:
    """Find the nearest ``afs.toml`` by walking upward from ``start_dir``."""
    current = (start_dir or Path.cwd()).expanduser().resolve()
    for parent in [current, *current.parents]:
        candidate = parent / "afs.toml"
        if candidate.exists():
            return candidate.resolve()
    return None


def resolve_runtime_config_path(
    config_path: str | Path | None = None,
    *,
    start_dir: Path | None = None,
) -> Path | None:
    """Resolve the runtime config path from explicit args, env, or nearest repo config."""
    if config_path is not None:
        return Path(config_path).expanduser().resolve()

    env_config = os.environ.get("AFS_CONFIG_PATH")
    if env_config:
        return Path(env_config).expanduser().resolve()

    return find_repo_config(start_dir)


def _expand_config_paths(config_data: dict[str, Any]) -> None:
    if "general" in config_data:
        general = config_data["general"]
        if "context_root" in general:
            general["context_root"] = _expand_path(general["context_root"])
        if "agent_workspaces_dir" in general:
            general["agent_workspaces_dir"] = _expand_path(general["agent_workspaces_dir"])
        if "mcp_allowed_roots" in general and isinstance(general["mcp_allowed_roots"], list):
            general["mcp_allowed_roots"] = [
                _expand_path(path)
                for path in general["mcp_allowed_roots"]
                if isinstance(path, (str, Path))
            ]
        if "python_executable" in general:
            python_exec = general["python_executable"]
            if isinstance(python_exec, str) and python_exec.startswith("~"):
                general["python_executable"] = _expand_path(python_exec)
        if "workspace_directories" in general:
            expanded_ws = []
            for ws_dir in general["workspace_directories"]:
                if isinstance(ws_dir, str):
                    expanded_ws.append({"path": _expand_path(ws_dir)})
                elif isinstance(ws_dir, dict) and "path" in ws_dir:
                    ws_dir["path"] = _expand_path(ws_dir["path"])
                    expanded_ws.append(ws_dir)
                else:
                    expanded_ws.append(ws_dir)
            general["workspace_directories"] = expanded_ws

    if "plugins" in config_data and "plugin_dirs" in config_data["plugins"]:
        config_data["plugins"]["plugin_dirs"] = [
            _expand_path(p) for p in config_data["plugins"]["plugin_dirs"]
        ]

    if "extensions" in config_data and "extension_dirs" in config_data["extensions"]:
        config_data["extensions"]["extension_dirs"] = [
            _expand_path(p) for p in config_data["extensions"]["extension_dirs"]
        ]

    for profile_root_key in ("profiles", "profile"):
        profile_root = config_data.get(profile_root_key)
        if not isinstance(profile_root, dict):
            continue
        for profile_name, profile_data in profile_root.items():
            if profile_name in {"active_profile", "auto_apply", "profiles"}:
                continue
            if not isinstance(profile_data, dict):
                continue
            for key in ("knowledge_mounts", "skill_roots", "model_registries"):
                if key in profile_data and isinstance(profile_data[key], list):
                    profile_data[key] = [
                        _expand_path(p) for p in profile_data[key] if isinstance(p, (str, Path))
                    ]

        nested_profiles = profile_root.get("profiles")
        if isinstance(nested_profiles, dict):
            for _name, profile_data in nested_profiles.items():
                if not isinstance(profile_data, dict):
                    continue
                for key in ("knowledge_mounts", "skill_roots", "model_registries"):
                    if key in profile_data and isinstance(profile_data[key], list):
                        profile_data[key] = [
                            _expand_path(p) for p in profile_data[key] if isinstance(p, (str, Path))
                        ]

    if "projects" in config_data:
        for project in config_data["projects"]:
            if "path" in project:
                project["path"] = _expand_path(project["path"])
            if "knowledge_roots" in project:
                project["knowledge_roots"] = [
                    _expand_path(p) for p in project["knowledge_roots"]
                ]

    if "memory_export" in config_data:
        memory_export = config_data["memory_export"]
        if "dataset_output" in memory_export:
            memory_export["dataset_output"] = _expand_path(memory_export["dataset_output"])
        if "report_output" in memory_export and memory_export["report_output"]:
            memory_export["report_output"] = _expand_path(memory_export["report_output"])
        routes = memory_export.get("routes")
        if isinstance(routes, list):
            for route in routes:
                if isinstance(route, dict) and "output" in route:
                    route["output"] = _expand_path(route["output"])

    if "memory_consolidation" in config_data:
        memory_consolidation = config_data["memory_consolidation"]
        if (
            "report_output" in memory_consolidation
            and memory_consolidation["report_output"]
        ):
            memory_consolidation["report_output"] = _expand_path(
                memory_consolidation["report_output"]
            )


def load_config(
    config_path: Path | None = None,
    merge_user: bool = True,
    *,
    start_dir: Path | None = None,
    prefer_local: bool | None = None,
) -> dict[str, Any]:
    """Load configuration with basic precedence and path expansion."""
    explicit_path = Path(config_path).expanduser().resolve() if config_path else None
    env_config = os.environ.get("AFS_CONFIG_PATH")
    env_path = (
        Path(env_config).expanduser().resolve()
        if explicit_path is None and env_config
        else None
    )
    local_path = find_repo_config(start_dir)
    effective_explicit_path = explicit_path or env_path
    explicit_override = explicit_path is not None or env_path is not None

    prefer_user = _parse_bool(
        os.environ.get("AFS_PREFER_USER_CONFIG"),
        default=not bool(prefer_local),
    )
    prefer_repo = _parse_bool(
        os.environ.get("AFS_PREFER_REPO_CONFIG"),
        default=bool(prefer_local),
    )
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

    if local_path and local_path.exists() and not explicit_override:
        with open(local_path, "rb") as f:
            local_raw = tomllib.load(f)

    if prefer_user:
        config_data = _deep_merge(config_data, local_raw)
        config_data = _deep_merge(config_data, user_raw)
    else:
        config_data = _deep_merge(config_data, user_raw)
        config_data = _deep_merge(config_data, local_raw)

    if effective_explicit_path and effective_explicit_path.exists() and explicit_override:
        with open(effective_explicit_path, "rb") as f:
            explicit_raw = tomllib.load(f)
        config_data = _deep_merge(config_data, explicit_raw)

    _merge_env_allowed_roots(config_data)
    _merge_workspace_registry(config_data)
    _expand_config_paths(config_data)
    return config_data


def load_config_model(
    config_path: Path | None = None,
    merge_user: bool = True,
    *,
    start_dir: Path | None = None,
    prefer_local: bool | None = None,
) -> AFSConfig:
    """Load configuration and return a typed model."""
    data = load_config(
        config_path=config_path,
        merge_user=merge_user,
        start_dir=start_dir,
        prefer_local=prefer_local,
    )
    return AFSConfig.from_dict(data)


def load_runtime_config_model(
    config_path: Path | None = None,
    merge_user: bool = True,
    *,
    start_dir: Path | None = None,
    prefer_local: bool = True,
) -> tuple[AFSConfig, Path | None]:
    """Load runtime config and return both the model and the resolved config path."""
    resolved_path = resolve_runtime_config_path(config_path=config_path, start_dir=start_dir)
    model = load_config_model(
        config_path=config_path,
        merge_user=merge_user,
        start_dir=start_dir,
        prefer_local=prefer_local,
    )
    return model, resolved_path


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


def _merge_env_allowed_roots(config_data: dict[str, Any]) -> None:
    raw = os.environ.get("AFS_MCP_ALLOWED_ROOTS", "").strip()
    if not raw:
        return

    general = config_data.setdefault("general", {})
    existing = general.get("mcp_allowed_roots")
    if not isinstance(existing, list):
        existing = []

    merged = list(existing)
    seen = {
        str(item).strip()
        for item in merged
        if isinstance(item, (str, Path)) and str(item).strip()
    }
    for item in raw.split(os.pathsep):
        value = item.strip()
        if not value or value in seen:
            continue
        merged.append(value)
        seen.add(value)

    general["mcp_allowed_roots"] = merged
