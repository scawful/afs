from __future__ import annotations

from pathlib import Path

from afs.health.mcp_registration import (
    discover_mcp_config_paths,
    find_afs_mcp_registrations,
)


def test_find_afs_mcp_registrations_detects_supported_clients(tmp_path: Path) -> None:
    home = tmp_path / "home"
    cwd = tmp_path / "workspace"
    home.mkdir()
    cwd.mkdir()

    gemini_config = home / ".gemini" / "mcp.json"
    gemini_config.parent.mkdir(parents=True)
    gemini_config.write_text(
        '{"mcpServers":{"afs":{"command":"afs","args":["mcp","serve"]}}}',
        encoding="utf-8",
    )

    claude_settings = home / ".claude" / "settings.json"
    claude_settings.parent.mkdir(parents=True)
    claude_settings.write_text('{"statusLine":{"type":"command"}}', encoding="utf-8")

    claude_desktop = (
        home
        / "Library"
        / "Application Support"
        / "Claude"
        / "claude_desktop_config.json"
    )
    claude_desktop.parent.mkdir(parents=True)
    claude_desktop.write_text(
        '{"mcpServers":{"afs":{"command":"python3","args":["-m","afs.mcp_server"]}}}',
        encoding="utf-8",
    )

    codex_config = home / ".codex" / "config.toml"
    codex_config.parent.mkdir(parents=True)
    codex_config.write_text(
        '[mcp_servers.afs]\ncommand = "afs"\nargs = ["mcp", "serve"]\n',
        encoding="utf-8",
    )

    hits = find_afs_mcp_registrations(home=home, cwd=cwd)

    assert hits["gemini"] == [str(gemini_config)]
    assert hits["claude"] == [str(claude_desktop)]
    assert hits["codex"] == [str(codex_config)]


def test_discover_mcp_config_paths_includes_project_codex_config(tmp_path: Path) -> None:
    home = tmp_path / "home"
    cwd = tmp_path / "repo"
    home.mkdir()
    cwd.mkdir()

    project_codex = cwd / ".codex" / "config.toml"
    project_codex.parent.mkdir(parents=True)
    project_codex.write_text(
        '[mcp_servers.afs]\ncommand = "python3"\nargs = ["-m", "afs", "mcp", "serve"]\n',
        encoding="utf-8",
    )

    configs = discover_mcp_config_paths(home=home, cwd=cwd)

    assert configs["codex"] == [project_codex]

