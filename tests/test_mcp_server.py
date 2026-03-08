from __future__ import annotations

from pathlib import Path

from afs.manager import AFSManager
from afs.mcp_server import _handle_request, build_mcp_registry
from afs.schema import AFSConfig, ExtensionsConfig, GeneralConfig


def _make_manager(tmp_path: Path) -> AFSManager:
    context_root = tmp_path / "context"
    context_root.mkdir(parents=True)
    (context_root / "scratchpad").mkdir()
    general = GeneralConfig(
        context_root=context_root,
        agent_workspaces_dir=context_root / "workspaces",
    )
    return AFSManager(config=AFSConfig(general=general))


def test_tools_list_returns_afs_tools(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    response = _handle_request({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, manager)
    assert response is not None
    tools = response["result"]["tools"]
    names = {tool["name"] for tool in tools}
    assert {"fs.read", "fs.write", "fs.list", "context.discover", "context.mount"}.issubset(names)


def test_fs_write_and_read_tool_calls(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    target = manager.config.general.context_root / "scratchpad" / "notes.txt"

    write_response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "fs.write",
                "arguments": {"path": str(target), "content": "hello", "mkdirs": True},
            },
        },
        manager,
    )
    assert write_response is not None
    assert write_response["result"]["structuredContent"]["bytes"] == 5

    read_response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "fs.read",
                "arguments": {"path": str(target)},
            },
        },
        manager,
    )
    assert read_response is not None
    assert read_response["result"]["structuredContent"]["content"] == "hello"


def test_extension_mcp_tools_are_registered_and_callable(tmp_path: Path) -> None:
    ext_root = tmp_path / "extensions"
    ext_dir = ext_root / "ext_workspace"
    ext_dir.mkdir(parents=True)
    (ext_dir / "extension.toml").write_text(
        "name = \"ext_workspace\"\n"
        "\n"
        "[mcp_tools]\n"
        "module = \"ext_mcp\"\n"
        "factory = \"register_mcp_tools\"\n",
        encoding="utf-8",
    )
    (ext_dir / "ext_mcp.py").write_text(
        "def register_mcp_tools(_manager):\n"
        "    def echo(arguments):\n"
        "        value = arguments.get('value', '')\n"
        "        return {'echo': value}\n"
        "    return [\n"
        "        {\n"
        "            'name': 'workspace.echo',\n"
        "            'description': 'Echo test payload',\n"
        "            'inputSchema': {\n"
        "                'type': 'object',\n"
        "                'properties': {'value': {'type': 'string'}},\n"
        "                'additionalProperties': False,\n"
        "            },\n"
        "            'handler': echo,\n"
        "        }\n"
        "    ]\n",
        encoding="utf-8",
    )

    context_root = tmp_path / "context"
    context_root.mkdir(parents=True)
    (context_root / "scratchpad").mkdir()
    manager = AFSManager(
        config=AFSConfig(
            general=GeneralConfig(
                context_root=context_root,
                agent_workspaces_dir=context_root / "workspaces",
            ),
            extensions=ExtensionsConfig(
                enabled_extensions=["ext_workspace"],
                extension_dirs=[ext_root],
            ),
        )
    )

    registry = build_mcp_registry(manager)
    assert "workspace.echo" in registry.tools

    call_response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 10,
            "method": "tools/call",
            "params": {
                "name": "workspace.echo",
                "arguments": {"value": "ok"},
            },
        },
        manager,
        registry=registry,
    )
    assert call_response is not None
    assert call_response["result"]["structuredContent"]["echo"] == "ok"
