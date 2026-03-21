"""Detect AFS MCP registrations across supported client config surfaces."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import tomllib

SUPPORTED_MCP_CLIENTS = ("gemini", "claude", "codex")

_JSON_CONFIG_CANDIDATES: dict[str, tuple[str, ...]] = {
    "gemini": (
        ".config/gemini/mcp.json",
        ".config/gemini/settings.json",
        ".gemini/mcp.json",
        ".gemini/settings.json",
    ),
    "claude": (
        ".claude/settings.json",
        "Library/Application Support/Claude/claude_desktop_config.json",
    ),
}

_TOML_CONFIG_CANDIDATES: dict[str, tuple[str, ...]] = {
    "codex": (
        ".codex/config.toml",
        ".config/codex/config.toml",
    ),
}


def discover_mcp_config_paths(
    *,
    home: Path | None = None,
    cwd: Path | None = None,
) -> dict[str, list[Path]]:
    """Return known MCP config files for supported clients."""
    home_dir = (home or Path.home()).expanduser()
    cwd_dir = (cwd or Path.cwd()).expanduser()

    configs: dict[str, list[Path]] = {client: [] for client in SUPPORTED_MCP_CLIENTS}
    for client, candidates in _JSON_CONFIG_CANDIDATES.items():
        paths = [home_dir / candidate for candidate in candidates]
        # Also check project-level .claude/settings.json
        if client == "claude":
            project_candidate = cwd_dir / ".claude" / "settings.json"
            if project_candidate not in paths:
                paths.append(project_candidate)
        configs[client] = _existing_paths(paths)

    for client, candidates in _TOML_CONFIG_CANDIDATES.items():
        paths = [home_dir / candidate for candidate in candidates]
        project_candidate = cwd_dir / ".codex" / "config.toml"
        if project_candidate not in paths:
            paths.append(project_candidate)
        configs[client] = _existing_paths(paths)

    return configs


def find_afs_mcp_registrations(
    *,
    home: Path | None = None,
    cwd: Path | None = None,
) -> dict[str, list[str]]:
    """Return config files that register the AFS MCP server."""
    hits: dict[str, list[str]] = {}
    for client, paths in discover_mcp_config_paths(home=home, cwd=cwd).items():
        matched: list[str] = []
        for path in paths:
            if path.suffix.lower() == ".toml":
                if _toml_config_has_afs_entry(path):
                    matched.append(str(path))
            elif _json_config_has_afs_entry(path):
                matched.append(str(path))
        hits[client] = matched
    return hits


def _existing_paths(paths: list[Path] | tuple[Path, ...] | Any) -> list[Path]:
    seen: set[str] = set()
    existing: list[Path] = []
    for path in paths:
        marker = str(path)
        if marker in seen:
            continue
        seen.add(marker)
        if path.exists():
            existing.append(path)
    return existing


def _json_config_has_afs_entry(path: Path) -> bool:
    try:
        data = json.loads(path.read_text(encoding="utf-8", errors="ignore"))
    except (OSError, json.JSONDecodeError):
        return False
    return _has_afs_mcp_entry(data)


def _toml_config_has_afs_entry(path: Path) -> bool:
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError):
        return False
    return _has_afs_mcp_entry(data)


def _has_afs_mcp_entry(data: Any) -> bool:
    if not isinstance(data, dict):
        return False
    for key in ("mcpServers", "mcp_servers"):
        servers = data.get(key)
        if isinstance(servers, dict) and _servers_include_afs(servers):
            return True
    return False


def _servers_include_afs(servers: dict[str, Any]) -> bool:
    for name, config in servers.items():
        if _server_looks_like_afs(name, config):
            return True
    return False


def _server_looks_like_afs(name: Any, config: Any) -> bool:
    if isinstance(name, str) and name.strip().lower() == "afs":
        return True
    if not isinstance(config, dict):
        return False

    command = str(config.get("command", "") or "").strip()
    args = config.get("args")
    normalized_args = [str(arg) for arg in args] if isinstance(args, list) else []
    joined = " ".join([command, *normalized_args])
    if "afs.mcp_server" in joined:
        return True

    command_name = Path(command).name.lower()
    if command_name == "afs" and normalized_args[:2] == ["mcp", "serve"]:
        return True
    if command_name in {"python", "python3", "python.exe", "python3.exe"}:
        return normalized_args[:2] == ["-m", "afs.mcp_server"] or (
            normalized_args[:3] == ["-m", "afs", "mcp"]
            and len(normalized_args) >= 4
            and normalized_args[3] == "serve"
        )
    return False

