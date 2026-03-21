"""Claude Code integration helpers for AFS."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any


def generate_claude_settings(
    project_path: Path,
    config: Any = None,
    *,
    config_path: Path | None = None,
) -> dict[str, Any]:
    """Build the mcpServers.afs entry for Claude Code settings."""
    resolved_project = project_path.expanduser().resolve()
    env: dict[str, str] = {}
    resolved_config_path = config_path or _find_project_config(resolved_project)
    if resolved_config_path is not None:
        env["AFS_CONFIG_PATH"] = str(resolved_config_path)
        env["AFS_PREFER_REPO_CONFIG"] = "1"
    if config is not None:
        context_root = getattr(getattr(config, "general", None), "context_root", None)
        if context_root:
            env["AFS_CONTEXT_ROOT"] = str(context_root)

    entry: dict[str, Any] = {
        "command": sys.executable,
        "args": ["-m", "afs.mcp_server"],
    }
    if env:
        entry["env"] = env
    return {"mcpServers": {"afs": entry}}


def merge_claude_settings(existing: dict[str, Any], afs_entry: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge AFS MCP entry into existing Claude settings, preserving other servers."""
    merged = dict(existing)
    mcp_key = "mcpServers"
    if mcp_key not in merged:
        merged[mcp_key] = {}
    if not isinstance(merged[mcp_key], dict):
        merged[mcp_key] = {}
    merged[mcp_key]["afs"] = afs_entry.get(mcp_key, {}).get("afs", {})
    return merged


def generate_claude_md(project_name: str, context_path: str) -> str:
    """Generate project CLAUDE.md content with AFS bootstrap instructions."""
    return f"""# Claude Workspace Bootstrap

Use AFS (Agent File System) for context management in this project.

## Session Startup

Before major work:
1. Run `afs session bootstrap --json` or use the MCP prompt `afs.session.bootstrap`.
2. Read scratchpad state/deferred notes.
3. Check queued tasks and recent hivemind messages.
4. Use `context.query` before asking for already-known context.

## Context

- Project: {project_name}
- Context path: {context_path}

## Handoff Protocol

Before ending a session:
1. Use `handoff.create` to record accomplished work, blockers, and next steps.
2. Update scratchpad state if needed.
3. The next session's bootstrap will include the handoff automatically.
"""


def generate_hooks_config() -> dict[str, Any]:
    """Generate Claude Code hooks entry for logging tool calls to AFS history."""
    return {
        "hooks": {
            "PostToolUse": [
                {
                    "matcher": "*",
                    "command": f"{sys.executable} -m afs events tail --limit 1 --json",
                }
            ]
        }
    }


def _find_project_config(project_path: Path) -> Path | None:
    search_root = project_path if project_path.is_dir() else project_path.parent
    for candidate in [search_root, *search_root.parents]:
        config_path = candidate / "afs.toml"
        if config_path.exists():
            return config_path
    return None
