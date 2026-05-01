from __future__ import annotations

import json
from pathlib import Path

import afs.cli as cli
from afs.cli import build_parser
from afs.manager import AFSManager
from afs.manager_gui import collect_manager_snapshot
from afs.schema import AFSConfig, GeneralConfig
from afs.tasks import TaskQueue


def test_manager_snapshot_reads_context_clients_tasks_and_extension_hooks(
    monkeypatch,
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    context = workspace / ".context"
    config_path = tmp_path / "afs.toml"
    config_path.write_text(
        f"[general]\ncontext_root = \"{context}\"\n\n[profiles]\nauto_apply = false\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("AFS_CONFIG_PATH", str(config_path))
    manager = AFSManager(config=AFSConfig(general=GeneralConfig(context_root=context)))
    manager.ensure(path=workspace, context_root=context)
    TaskQueue(context).create("write setup docs", created_by="test")

    gemini = workspace / ".gemini"
    gemini.mkdir()
    (gemini / "settings.json").write_text(
        json.dumps({"mcpServers": {"afs": {"command": "afs"}}}),
        encoding="utf-8",
    )
    opencode = workspace / ".opencode"
    opencode.mkdir()
    (opencode / "opencode.jsonc").write_text(
        '{\n  // project-local MCP bridge\n  "mcp": {"afs": {"command": "afs"}},\n}\n',
        encoding="utf-8",
    )

    ext_root = tmp_path / "extensions" / "demo"
    ext_root.mkdir(parents=True)
    (ext_root / "extension.toml").write_text(
        "name = \"demo\"\n"
        "description = \"Demo manager hook\"\n"
        "\n"
        "[manager]\n"
        "actions = [\"afs status\"]\n"
        "\n"
        "[hooks]\n"
        "session_start = [\"echo start\"]\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("AFS_EXTENSION_DIRS", str(tmp_path / "extensions"))
    monkeypatch.setenv("AFS_ENABLED_EXTENSIONS", "demo")

    snapshot = collect_manager_snapshot(workspace, home=tmp_path / "home")

    assert snapshot.context_exists is True
    assert snapshot.context_healthy is not None
    assert snapshot.tasks[0].title == "write setup docs"
    assert any(client.name == "Gemini project" and client.registered for client in snapshot.clients)
    assert any(client.name == "OpenCode project" and client.registered for client in snapshot.clients)
    assert snapshot.extensions[0].name == "demo"
    assert snapshot.extensions[0].manager_actions == ["afs status"]
    assert snapshot.extensions[0].hooks == {"session_start": ["echo start"]}
    assert snapshot.discovery_path["steps"][0]["tool"] == "context.status"
    assert "manager open" in snapshot.commands["open_manager"]


def test_manager_parser_and_json_command(monkeypatch, tmp_path: Path, capsys) -> None:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    context = workspace / ".context"
    AFSManager(config=AFSConfig(general=GeneralConfig(context_root=context))).ensure(
        path=workspace,
        context_root=context,
    )
    monkeypatch.setattr(cli, "log_cli_invocation", lambda *_args, **_kwargs: None)

    parser = build_parser()
    parsed = parser.parse_args(["manager", "snapshot", "--path", str(workspace), "--json"])
    assert parsed.command == "manager"
    assert parsed.manager_command == "snapshot"
    assert hasattr(parsed, "func")

    exit_code = cli.main(["manager", "snapshot", "--path", str(workspace), "--json"])
    out = capsys.readouterr().out

    assert exit_code == 0
    assert '"workspace"' in out
    assert '"clients"' in out
    assert '"discovery_path"' in out
