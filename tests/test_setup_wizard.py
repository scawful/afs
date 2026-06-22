from __future__ import annotations

from pathlib import Path

import afs.cli as cli
from afs.cli import build_parser
from afs.cli.setup_wizard import build_setup_plan


def test_setup_plan_defaults_to_helpers_without_agent_routing(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()

    plan = build_setup_plan(
        workspace=workspace,
        context_root=workspace / ".context",
        config_path=workspace / "afs.toml",
        config_scope="project",
        context_mode="project",
        link_context=False,
        shell_mode="helpers",
        mcp_mode="none",
        gws_mode="skip",
        worker=False,
        afs_command=["afs"],
    )

    commands = [step.command for step in plan.steps]
    assert commands[0][:2] == ["afs", "init"]
    assert commands[1][:3] == ["afs", "context", "repair"]
    shell_command = next(command for command in commands if "install-shell" in command)
    assert "--helpers-only" in shell_command
    assert "install-worker" not in " ".join(" ".join(command) for command in commands)


def test_setup_plan_can_include_mcp_gws_and_worker(tmp_path: Path) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()

    plan = build_setup_plan(
        workspace=workspace,
        context_root=Path.home() / ".context",
        config_path=Path.home() / ".config" / "afs" / "config.toml",
        config_scope="user",
        context_mode="shared",
        link_context=False,
        shell_mode="agent-hooks",
        mcp_mode="both",
        gws_mode="check",
        worker=True,
        afs_command=["afs"],
    )

    shell = [step.to_dict()["shell"] for step in plan.steps]
    assert any("agent-hooks install-worker" in item for item in shell)
    assert any("claude setup --scope user" in item for item in shell)
    assert any("gemini setup --scope user" in item for item in shell)
    assert any("gws status" in item for item in shell)


def test_setup_command_json_is_noninteractive(monkeypatch, tmp_path: Path, capsys) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    config_path = tmp_path / "base-afs.toml"
    config_path.write_text(
        "[extensions]\nauto_discover = false\n\n[plugins]\nauto_discover = false\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("AFS_CONFIG_PATH", str(config_path))
    monkeypatch.delenv("AFS_EXTENSION_DIRS", raising=False)
    monkeypatch.delenv("AFS_ENABLED_EXTENSIONS", raising=False)
    monkeypatch.setattr(cli, "log_cli_invocation", lambda *_args, **_kwargs: None)

    exit_code = cli.main(
        [
            "setup",
            "--workspace",
            str(workspace),
            "--config-scope",
            "project",
            "--context-mode",
            "project",
            "--shell",
            "helpers",
            "--mcp",
            "none",
            "--google-workspace",
            "skip",
            "--no-worker",
            "--json",
        ]
    )

    out = capsys.readouterr().out
    assert exit_code == 0
    assert '"workspace"' in out
    assert '"steps"' in out
    assert "install-shell" in out
    assert "manager open" in out


def test_parser_registers_setup_and_guide(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "base-afs.toml"
    config_path.write_text(
        "[extensions]\nauto_discover = false\n\n[plugins]\nauto_discover = false\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("AFS_CONFIG_PATH", str(config_path))
    monkeypatch.delenv("AFS_EXTENSION_DIRS", raising=False)
    monkeypatch.delenv("AFS_ENABLED_EXTENSIONS", raising=False)
    parser = build_parser()
    setup = parser.parse_args(["setup", "--yes", "--shell", "none"])
    assert setup.command == "setup"
    assert setup.yes is True
    assert hasattr(setup, "func")

    guide = parser.parse_args(["guide", "shell"])
    assert guide.command == "guide"
    assert guide.topic == "shell"
    assert hasattr(guide, "func")

    manager = parser.parse_args(["manager", "snapshot", "--json"])
    assert manager.command == "manager"
    assert manager.manager_command == "snapshot"
    assert hasattr(manager, "func")

    manager_open = parser.parse_args(["manager", "--json"])
    assert manager_open.command == "manager"
    assert manager_open.manager_command is None
    assert manager_open._allow_missing_subcommand is True
