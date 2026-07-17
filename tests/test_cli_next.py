from __future__ import annotations

import json
from pathlib import Path

import afs.cli as cli
from afs.cli import build_parser
from afs.history import append_history_event
from afs.manager import AFSManager
from afs.next_action import build_next_action, summarize_next_usage
from afs.schema import AFSConfig, GeneralConfig


def _workspace(tmp_path: Path) -> tuple[Path, Path]:
    workspace = tmp_path / "repo"
    workspace.mkdir()
    context = workspace / ".context"
    AFSManager(config=AFSConfig(general=GeneralConfig(context_root=context))).ensure(
        path=workspace,
        context_root=context,
    )
    return workspace, context


def test_build_next_action_routes_intent_to_commands(tmp_path: Path) -> None:
    workspace, context = _workspace(tmp_path)

    action = build_next_action("work reply", workspace=workspace)

    assert action.canonical_intent == "work-writing"
    assert action.context_path == context
    assert action.first_step == "work communication preflight"
    assert action.slash_command == "/afs-work-preflight"
    assert action.commands[0].command.endswith(f"work communication preflight --json --path {workspace}")
    assert "context.status" in action.discovery_path["default_mcp_tools"]

    catchup = build_next_action("continue", workspace=workspace)
    assert catchup.commands[0].command == f"afs status --start-dir {workspace}"
    assert "--mount memory --mount scratchpad" in catchup.commands[1].command

    handoff = build_next_action("handoff", workspace=workspace)
    assert handoff.first_step == "list canonical handoff threads or create one immutable revision"
    assert handoff.mcp_sequence[0] == "handoff.list"
    assert len(handoff.commands) == 1
    assert "scratchpad/handoffs" not in json.dumps(handoff.to_dict())


def test_next_command_json_and_parser(monkeypatch, tmp_path: Path, capsys) -> None:
    workspace, _context = _workspace(tmp_path)
    monkeypatch.setattr(cli, "log_cli_invocation", lambda *_args, **_kwargs: None)

    parser = build_parser()
    parsed = parser.parse_args(["next", "--intent", "verify", "--path", str(workspace), "--json"])
    assert parsed.command == "next"
    assert parsed.next_command is None
    assert hasattr(parsed, "func")

    exit_code = cli.main(["next", "--intent", "verify", "--path", str(workspace), "--json"])
    payload = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert payload["canonical_intent"] == "verify"
    assert payload["commands"][0]["command"].startswith("afs verify plan --json --cwd")
    assert "context.status" in payload["mcp_sequence"]


def test_next_report_summarizes_funnel_usage(tmp_path: Path) -> None:
    _workspace_path, context = _workspace(tmp_path)
    history = context / "history"
    append_history_event(history, "agent_route", "afs.next", op="route", metadata={"intent": "verify"})
    append_history_event(history, "mcp_tool", "afs.mcp", op="call", metadata={"tool_name": "context.status"})
    append_history_event(history, "mcp_tool", "afs.mcp", op="call", metadata={"tool_name": "session.pack"})
    append_history_event(history, "cli", "afs.cli", op="invoke", metadata={"argv": ["afs", "verify", "plan"]})

    report = summarize_next_usage(context, limit=20)

    assert report["next_routes"] == {"verify": 1}
    assert report["mcp_tools"]["context.status"] == 1
    assert report["heavy_mcp_calls"] == {"session.pack": 1}
    assert report["cli_commands"]["verify plan"] == 1
