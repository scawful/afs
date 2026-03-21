"""Tests for Claude Code integration (Feature 5)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from afs.claude_integration import (
    generate_claude_md,
    generate_claude_settings,
    generate_hooks_config,
    merge_claude_settings,
)


def test_generate_claude_settings_basic() -> None:
    settings = generate_claude_settings(Path("/tmp/test"))
    assert "mcpServers" in settings
    assert "afs" in settings["mcpServers"]
    entry = settings["mcpServers"]["afs"]
    assert entry["command"] == sys.executable
    assert entry["args"] == ["-m", "afs.mcp_server"]


def test_generate_claude_settings_with_context_root() -> None:
    class FakeConfig:
        class general:
            context_root = Path("/home/user/.context")
    settings = generate_claude_settings(Path("/tmp/test"), config=FakeConfig())
    entry = settings["mcpServers"]["afs"]
    assert "env" in entry
    assert entry["env"]["AFS_CONTEXT_ROOT"] == "/home/user/.context"


def test_generate_claude_settings_prefers_project_config(tmp_path: Path) -> None:
    project_path = tmp_path / "repo"
    project_path.mkdir()
    config_path = project_path / "afs.toml"
    config_path.write_text("[general]\ncontext_root = \"/tmp/context\"\n", encoding="utf-8")

    settings = generate_claude_settings(project_path)
    entry = settings["mcpServers"]["afs"]

    assert entry["env"]["AFS_CONFIG_PATH"] == str(config_path)
    assert entry["env"]["AFS_PREFER_REPO_CONFIG"] == "1"


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
    assert "handoff.create" in md


def test_generate_hooks_config() -> None:
    hooks = generate_hooks_config()
    assert "hooks" in hooks
    assert "PostToolUse" in hooks["hooks"]
    assert len(hooks["hooks"]["PostToolUse"]) == 1
    assert "-m afs events tail" in hooks["hooks"]["PostToolUse"][0]["command"]


def test_mcp_registration_detects_project_claude_settings(tmp_path: Path) -> None:
    from afs.health.mcp_registration import discover_mcp_config_paths

    # Create project-level .claude/settings.json
    claude_dir = tmp_path / ".claude"
    claude_dir.mkdir()
    settings_path = claude_dir / "settings.json"
    settings_path.write_text(json.dumps({"mcpServers": {"afs": {}}}), encoding="utf-8")

    paths = discover_mcp_config_paths(home=tmp_path / "fakehome", cwd=tmp_path)
    assert any(str(settings_path) == str(p) for p in paths.get("claude", []))
