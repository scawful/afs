"""Tests for Claude Code integration (Feature 5)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from afs.claude_integration import (
    default_claude_user_settings_path,
    generate_afs_hook_settings,
    generate_claude_md,
    generate_claude_settings,
    generate_hooks_config,
    merge_claude_settings,
)


def _afs_hook_commands(settings: dict, event: str) -> list[str]:
    return [
        hook["command"]
        for entry in settings.get("hooks", {}).get(event, [])
        for hook in entry.get("hooks", [])
        if "afs claude hook" in hook.get("command", "")
    ]


def test_generate_claude_settings_basic() -> None:
    settings = generate_claude_settings(Path("/tmp/test"))
    assert "mcpServers" in settings
    assert "afs" in settings["mcpServers"]
    entry = settings["mcpServers"]["afs"]
    assert entry["command"] == sys.executable
    assert entry["args"] == ["-m", "afs.mcp_server"]


def test_generate_claude_settings_includes_runtime_env() -> None:
    settings = generate_claude_settings(Path("/tmp/test"))
    entry = settings["mcpServers"]["afs"]
    env = entry["env"]
    repo_root = Path(__file__).resolve().parents[1]

    assert env["AFS_ROOT"] == str(repo_root)
    assert env["PYTHONPATH"] == str(repo_root / "src")
    if (repo_root / ".venv").exists():
        assert env["AFS_VENV"] == str(repo_root / ".venv")


def test_generate_claude_settings_with_context_root() -> None:
    class FakeConfig:
        class general:
            context_root = Path("/home/user/.context")
    settings = generate_claude_settings(Path("/tmp/test"), config=FakeConfig())
    entry = settings["mcpServers"]["afs"]
    assert "env" in entry
    assert entry["env"]["AFS_CONTEXT_ROOT"] == str(Path("/home/user/.context").expanduser().resolve())


def test_generate_claude_settings_prefers_project_config(tmp_path: Path) -> None:
    project_path = tmp_path / "repo"
    project_path.mkdir()
    config_path = project_path / "afs.toml"
    config_path.write_text("[general]\ncontext_root = \"/tmp/context\"\n", encoding="utf-8")

    settings = generate_claude_settings(project_path)
    entry = settings["mcpServers"]["afs"]

    assert entry["env"]["AFS_CONFIG_PATH"] == str(config_path)
    assert entry["env"]["AFS_PREFER_REPO_CONFIG"] == "1"


def test_generate_claude_settings_user_scope_omits_project_context(tmp_path: Path) -> None:
    project_path = tmp_path / "repo"
    project_path.mkdir()
    config_path = project_path / "afs.toml"
    config_path.write_text("[general]\ncontext_root = \"/tmp/context\"\n", encoding="utf-8")

    class FakeConfig:
        class general:
            context_root = Path("/home/user/.context")

    settings = generate_claude_settings(
        project_path,
        config=FakeConfig(),
        config_path=config_path,
        include_project_context=False,
    )
    entry = settings["mcpServers"]["afs"]
    env = entry["env"]

    assert "AFS_CONFIG_PATH" not in env
    assert "AFS_PREFER_REPO_CONFIG" not in env
    assert "AFS_CONTEXT_ROOT" not in env


def test_merge_preserves_other_servers() -> None:
    existing = {
        "mcpServers": {
            "other-server": {"command": "other", "args": []},
        }
    }
    afs_entry = generate_claude_settings(Path("/tmp/test"))
    merged = merge_claude_settings(existing, afs_entry)
    assert "other-server" in merged["mcpServers"]
    assert "afs" in merged["mcpServers"]


def test_merge_does_not_mutate_existing() -> None:
    existing = {
        "mcpServers": {
            "other-server": {"command": "other", "args": []},
        }
    }
    afs_entry = generate_claude_settings(Path("/tmp/test"))
    merge_claude_settings(existing, afs_entry)
    assert "afs" not in existing["mcpServers"]


def test_merge_creates_mcp_servers_if_missing() -> None:
    existing = {"someKey": "someValue"}
    afs_entry = generate_claude_settings(Path("/tmp/test"))
    merged = merge_claude_settings(existing, afs_entry)
    assert "mcpServers" in merged
    assert "afs" in merged["mcpServers"]
    assert merged["someKey"] == "someValue"


def test_generate_claude_md() -> None:
    md = generate_claude_md("my-project", "/home/user/.context")
    assert "my-project" in md
    assert "/home/user/.context" in md
    assert "afs session bootstrap" in md
    assert "afs claude doctor --json" in md
    assert "afs claude reap --limit 20 --apply" in md
    assert "Never reap `protected` sessions" in md
    assert "handoff.create" in md


def test_generate_hooks_config() -> None:
    hooks = generate_hooks_config()
    assert "hooks" in hooks
    assert "PostToolUse" in hooks["hooks"]
    assert len(hooks["hooks"]["PostToolUse"]) == 1
    assert "-m afs events tail" in hooks["hooks"]["PostToolUse"][0]["command"]


def test_generate_claude_settings_includes_push_hooks() -> None:
    settings = generate_claude_settings(Path("/tmp/test"))
    assert "SessionStart" in _hook_events(settings)
    assert "UserPromptSubmit" in _hook_events(settings)
    session_cmds = _afs_hook_commands(settings, "SessionStart")
    assert len(session_cmds) == 1
    assert "-m afs claude hook" in session_cmds[0]
    assert "--event SessionStart" in session_cmds[0]


def test_generate_afs_hook_settings_bakes_runtime_env(tmp_path: Path) -> None:
    hooks = generate_afs_hook_settings(
        tmp_path,
        context_root=tmp_path / ".context",
    )
    command = hooks["hooks"]["UserPromptSubmit"][0]["hooks"][0]["command"]
    # Env is baked in as a prefix since Claude hooks take no per-hook env block.
    assert "PYTHONPATH=" in command
    assert f"AFS_CONTEXT_ROOT={tmp_path / '.context'}" in command
    assert command.strip().endswith("--event UserPromptSubmit")


def test_merge_hooks_is_idempotent_and_preserves_user_hooks() -> None:
    existing = {
        "hooks": {
            "SessionStart": [
                {"hooks": [{"type": "command", "command": "echo user-owned"}]}
            ]
        }
    }
    afs_entry = generate_claude_settings(Path("/tmp/test"))
    merged_once = merge_claude_settings(existing, afs_entry)
    merged_twice = merge_claude_settings(merged_once, afs_entry)

    session = merged_twice["hooks"]["SessionStart"]
    afs_hooks = [
        hook
        for entry in session
        for hook in entry["hooks"]
        if "afs claude hook" in hook["command"]
    ]
    user_hooks = [
        hook
        for entry in session
        for hook in entry["hooks"]
        if hook["command"] == "echo user-owned"
    ]
    # AFS hook appears exactly once after two merges; the user's hook is preserved.
    assert len(afs_hooks) == 1
    assert len(user_hooks) == 1


def _hook_events(settings: dict) -> set:
    return set(settings.get("hooks", {}).keys())


def test_default_claude_user_settings_path(tmp_path: Path) -> None:
    assert default_claude_user_settings_path(home=tmp_path) == tmp_path / ".claude" / "settings.json"


def test_mcp_registration_detects_project_claude_settings(tmp_path: Path) -> None:
    from afs.health.mcp_registration import discover_mcp_config_paths

    # Create project-level .claude/settings.json
    claude_dir = tmp_path / ".claude"
    claude_dir.mkdir()
    settings_path = claude_dir / "settings.json"
    settings_path.write_text(json.dumps({"mcpServers": {"afs": {}}}), encoding="utf-8")

    paths = discover_mcp_config_paths(home=tmp_path / "fakehome", cwd=tmp_path)
    assert any(str(settings_path) == str(p) for p in paths.get("claude", []))
