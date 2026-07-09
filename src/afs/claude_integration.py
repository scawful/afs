"""Claude Code integration helpers for AFS."""

from __future__ import annotations

import shlex
import sys
from pathlib import Path
from typing import Any

from .mcp_runtime import build_afs_mcp_entry, build_afs_runtime_env

# Marker used to find/replace AFS-owned lifecycle hooks idempotently on re-setup.
_AFS_HOOK_MARKER = "afs claude hook"
# Lifecycle events AFS pushes grounding into. SessionStart grounds the session once;
# UserPromptSubmit injects the work-communication contract just-in-time on comms turns.
_AFS_HOOK_EVENTS = ("SessionStart", "UserPromptSubmit")


def generate_claude_settings(
    project_path: Path,
    config: Any = None,
    *,
    config_path: Path | None = None,
    include_project_context: bool = True,
) -> dict[str, Any]:
    """Build the mcpServers.afs entry plus AFS push hooks for Claude Code settings."""
    resolved_project = project_path.expanduser().resolve()
    config_path_for_env: Path | None = None
    context_root_for_env: Path | None = None
    if include_project_context:
        resolved_config_path = config_path or _find_project_config(resolved_project)
        if resolved_config_path is not None:
            config_path_for_env = resolved_config_path
        if config is not None:
            context_root = getattr(getattr(config, "general", None), "context_root", None)
            if context_root:
                context_root_for_env = context_root

    entry = build_afs_mcp_entry(
        "python-module",
        prefer_repo_config=include_project_context,
        config_path=config_path_for_env,
        context_root=context_root_for_env,
    )
    hooks = generate_afs_hook_settings(
        resolved_project,
        prefer_repo_config=include_project_context,
        config_path=config_path_for_env,
        context_root=context_root_for_env,
    )
    return {"mcpServers": {"afs": entry}, **hooks}


def generate_afs_hook_settings(
    project_path: Path,
    *,
    prefer_repo_config: bool = True,
    config_path: Path | None = None,
    context_root: Path | None = None,
) -> dict[str, Any]:
    """Build Claude Code ``hooks`` entries that push AFS grounding into a session.

    Claude Code runs hooks as plain shell command strings without a per-hook ``env``
    block, so the AFS runtime environment (PYTHONPATH, context root, config) is baked
    in as a command prefix that mirrors the MCP server entry. The command degrades to
    a silent no-op when context can't be resolved, so it is safe to register widely.
    """
    resolved_project = project_path.expanduser().resolve()
    env = build_afs_runtime_env(
        prefer_repo_config=prefer_repo_config,
        config_path=config_path,
        context_root=context_root,
    )
    prefix = " ".join(f"{key}={shlex.quote(value)}" for key, value in sorted(env.items()))
    python = shlex.quote(sys.executable)
    base = f"{python} -m afs claude hook --path {shlex.quote(str(resolved_project))}"
    if context_root is not None:
        base += f" --context-root {shlex.quote(str(context_root.expanduser().resolve()))}"

    def _command(event: str) -> str:
        command = f"{base} --event {event}"
        return f"{prefix} {command}" if prefix else command

    return {
        "hooks": {
            event: [{"hooks": [{"type": "command", "command": _command(event)}]}]
            for event in _AFS_HOOK_EVENTS
        }
    }


def default_claude_user_settings_path(home: Path | None = None) -> Path:
    """Return the default user-level Claude settings path."""
    home_dir = (home or Path.home()).expanduser()
    return home_dir / ".claude" / "settings.json"


def merge_claude_settings(existing: dict[str, Any], afs_entry: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge AFS MCP entry and push hooks into existing Claude settings.

    Preserves other MCP servers and other users' hooks. AFS-owned hooks (identified
    by the ``afs claude hook`` marker) are replaced rather than duplicated, so repeated
    ``afs claude setup`` runs stay idempotent.
    """
    merged = dict(existing)
    mcp_key = "mcpServers"
    if not isinstance(merged.get(mcp_key), dict):
        merged[mcp_key] = {}
    else:
        merged[mcp_key] = dict(merged[mcp_key])
    merged[mcp_key]["afs"] = afs_entry.get(mcp_key, {}).get("afs", {})

    new_hooks = afs_entry.get("hooks")
    if isinstance(new_hooks, dict):
        existing_hooks = merged.get("hooks")
        merged_hooks = dict(existing_hooks) if isinstance(existing_hooks, dict) else {}
        for event, entries in new_hooks.items():
            prior = merged_hooks.get(event)
            kept = [
                entry
                for entry in (prior if isinstance(prior, list) else [])
                if not _is_afs_hook_entry(entry)
            ]
            merged_hooks[event] = kept + list(entries)
        merged["hooks"] = merged_hooks
    return merged


def _is_afs_hook_entry(entry: Any) -> bool:
    """True when a Claude hook entry is an AFS-owned push hook."""
    if not isinstance(entry, dict):
        return False
    for hook in entry.get("hooks", []) or []:
        if isinstance(hook, dict) and _AFS_HOOK_MARKER in str(hook.get("command", "")):
            return True
    return False


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

## Session Recovery

If Claude notices MCP sluggishness, session tool timeouts, repeated missing-tool errors, or obvious stale-session buildup:
1. Run `afs claude doctor --json` first to inspect session counts, bridge protection, and recent debug signals.
2. If cleanup is needed, run `afs claude reap --limit 20` as a dry-run before making changes.
3. Claude may run `afs claude reap --limit 20 --apply` to archive stale or zombie sessions in bounded batches.
4. Never reap `protected` sessions or any project with an active `bridge-pointer.json`.
5. Re-run `afs claude doctor --json` after each batch and stop once the blocking condition clears.

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
