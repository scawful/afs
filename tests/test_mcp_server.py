from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

from afs.agents.supervisor import AgentSupervisor
from afs.history import append_history_event
from afs.manager import AFSManager
from afs.mcp_server import PROTOCOL_VERSION, _handle_request, _read_message, build_mcp_registry
from afs.models import MountType
from afs.schema import (
    AFSConfig,
    ContextIndexConfig,
    DirectoryConfig,
    ExtensionsConfig,
    GeneralConfig,
    ProfileConfig,
    ProfilesConfig,
    WorkspaceDirectory,
    default_directory_configs,
)


def _make_manager(tmp_path: Path) -> AFSManager:
    context_root = tmp_path / "context"
    general = GeneralConfig(
        context_root=context_root,
    )
    manager = AFSManager(config=AFSConfig(general=general))
    project_path = tmp_path / "project"
    project_path.mkdir()
    manager.ensure(path=project_path, context_root=context_root)
    return manager


def _remap_directories(**overrides: str) -> list[DirectoryConfig]:
    directories: list[DirectoryConfig] = []
    for directory in default_directory_configs():
        name = (
            overrides.get(directory.role.value, directory.name)
            if directory.role
            else directory.name
        )
        directories.append(
            DirectoryConfig(
                name=name,
                policy=directory.policy,
                description=directory.description,
                role=directory.role,
            )
        )
    return directories


def _make_remapped_manager(tmp_path: Path, **overrides: str) -> AFSManager:
    context_root = tmp_path / "context"
    general = GeneralConfig(
        context_root=context_root,
    )
    manager = AFSManager(
        config=AFSConfig(
            general=general,
            directories=_remap_directories(**overrides),
        )
    )
    project_path = tmp_path / "project"
    project_path.mkdir()
    manager.ensure(path=project_path, context_root=context_root)
    return manager


PREFERRED_FILE_TOOLS = {
    "context.read",
    "context.write",
    "context.delete",
    "context.move",
    "context.list",
}

COMPATIBILITY_FILE_TOOL_ALIASES = {
    "fs.read",
    "fs.write",
    "fs.delete",
    "fs.move",
    "fs.list",
}


def test_tools_list_returns_preferred_and_compatibility_file_tools(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    response = _handle_request({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, manager)
    assert response is not None
    tools = response["result"]["tools"]
    names = {tool["name"] for tool in tools}
    assert {
        "context.discover",
        "context.init",
        "context.mount",
        "context.unmount",
        "context.index.rebuild",
        "context.query",
        "context.diff",
        "context.status",
        "operator.digest",
        "context.repair",
        "session.pack",
        "events.analytics",
        "events.replay",
        "hivemind.reap",
    }.issubset(names)
    assert PREFERRED_FILE_TOOLS.issubset(names)
    assert COMPATIBILITY_FILE_TOOL_ALIASES.issubset(names)


# Preferred file tool surface: daily agent-facing behavior should exercise
# context.* directly, with fs.* covered separately as compatibility aliases.
def test_context_write_and_read_tool_calls(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    target = manager.config.general.context_root / "scratchpad" / "notes.txt"

    write_response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "context.write",
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
                "name": "context.read",
                "arguments": {"path": str(target)},
            },
        },
        manager,
    )
    assert read_response is not None
    assert read_response["result"]["structuredContent"]["content"] == "hello"


# Compatibility aliases: keep explicit coverage that legacy fs.* names still
# behave identically while the rest of the suite prefers context.*.
def test_fs_file_aliases_match_context_tool_behavior(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    context_root = manager.config.general.context_root
    source = context_root / "scratchpad" / "notes.txt"
    moved = context_root / "scratchpad" / "renamed.txt"

    write_response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 230,
            "method": "tools/call",
            "params": {
                "name": "fs.write",
                "arguments": {"path": str(source), "content": "alias hello", "mkdirs": True},
            },
        },
        manager,
    )
    assert write_response is not None
    assert write_response["result"]["structuredContent"]["bytes"] == 11

    read_response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 231,
            "method": "tools/call",
            "params": {
                "name": "fs.read",
                "arguments": {"path": str(source)},
            },
        },
        manager,
    )
    assert read_response is not None
    assert read_response["result"]["structuredContent"]["content"] == "alias hello"

    list_response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 232,
            "method": "tools/call",
            "params": {
                "name": "fs.list",
                "arguments": {"path": str(context_root / "scratchpad"), "max_depth": 1},
            },
        },
        manager,
    )
    assert list_response is not None
    entries = list_response["result"]["structuredContent"]["entries"]
    assert any(entry["path"].endswith("notes.txt") for entry in entries)

    move_response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 233,
            "method": "tools/call",
            "params": {
                "name": "fs.move",
                "arguments": {"source": str(source), "destination": str(moved)},
            },
        },
        manager,
    )
    assert move_response is not None
    assert moved.exists()
    assert not source.exists()

    delete_response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 234,
            "method": "tools/call",
            "params": {
                "name": "fs.delete",
                "arguments": {"path": str(moved)},
            },
        },
        manager,
    )
    assert delete_response is not None
    assert not moved.exists()


def test_events_analytics_tool_reports_mcp_usage(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    history_root = manager.config.general.context_root / "history"
    history_root.mkdir(parents=True, exist_ok=True)
    append_history_event(
        history_root,
        "mcp_tool",
        "afs.mcp",
        op="call",
        metadata={"tool_name": "context.status", "duration_ms": 15, "ok": True},
    )

    response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 500,
            "method": "tools/call",
            "params": {
                "name": "events.analytics",
                "arguments": {
                    "context_path": str(manager.config.general.context_root),
                    "hours": 24,
                },
            },
        },
        manager,
    )

    assert response is not None
    metrics = response["result"]["structuredContent"]["mcp_tools"]["context.status"]
    assert metrics["count"] == 1


def test_events_replay_tool_filters_by_session_id(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    history_root = manager.config.general.context_root / "history"
    history_root.mkdir(parents=True, exist_ok=True)
    append_history_event(
        history_root,
        "session",
        "afs.session",
        op="bootstrap",
        metadata={"session_id": "session-a"},
    )
    append_history_event(
        history_root,
        "session",
        "afs.session",
        op="bootstrap",
        metadata={"session_id": "session-b"},
    )

    response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 501,
            "method": "tools/call",
            "params": {
                "name": "events.replay",
                "arguments": {
                    "context_path": str(manager.config.general.context_root),
                    "session_id": "session-a",
                },
            },
        },
        manager,
    )

    assert response is not None
    content = response["result"]["structuredContent"]
    assert content["session_id"] == "session-a"
    assert content["count"] == 1


def test_context_list_allows_configured_workspace_root(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace-root"
    workspace_root.mkdir()
    project_dir = workspace_root / "repo"
    project_dir.mkdir()
    (project_dir / "README.md").write_text("workspace root access", encoding="utf-8")

    context_root = tmp_path / "context"
    context_root.mkdir(parents=True)
    (context_root / "scratchpad").mkdir()
    manager = AFSManager(
        config=AFSConfig(
            general=GeneralConfig(
                context_root=context_root,
                workspace_directories=[WorkspaceDirectory(path=workspace_root)],
            )
        )
    )

    response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 60,
            "method": "tools/call",
            "params": {
                "name": "context.list",
                "arguments": {"path": str(workspace_root), "max_depth": 2},
            },
        },
        manager,
    )
    assert response is not None
    entries = response["result"]["structuredContent"]["entries"]
    assert any(entry["path"].endswith("README.md") for entry in entries)


def test_context_list_allows_configured_mcp_allowed_root(tmp_path: Path) -> None:
    allowed_root = tmp_path / "workspace-root"
    allowed_root.mkdir()
    (allowed_root / "WORKSPACE").write_text("configured root", encoding="utf-8")

    context_root = tmp_path / "context"
    context_root.mkdir(parents=True)
    (context_root / "scratchpad").mkdir()
    manager = AFSManager(
        config=AFSConfig(
            general=GeneralConfig(
                context_root=context_root,
                mcp_allowed_roots=[allowed_root],
            )
        )
    )

    response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 66,
            "method": "tools/call",
            "params": {
                "name": "context.list",
                "arguments": {"path": str(allowed_root), "max_depth": 1},
            },
        },
        manager,
    )
    assert response is not None
    entries = response["result"]["structuredContent"]["entries"]
    assert any(entry["path"].endswith("WORKSPACE") for entry in entries)


def test_context_list_allows_env_configured_mcp_root(tmp_path: Path, monkeypatch) -> None:
    allowed_root = tmp_path / "workspace-root"
    allowed_root.mkdir()
    (allowed_root / "WORKSPACE").write_text("env root", encoding="utf-8")
    manager = _make_manager(tmp_path)
    monkeypatch.setenv("AFS_MCP_ALLOWED_ROOTS", str(allowed_root))

    response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 61,
            "method": "tools/call",
            "params": {
                "name": "context.list",
                "arguments": {"path": str(allowed_root), "max_depth": 1},
            },
        },
        manager,
    )
    assert response is not None
    entries = response["result"]["structuredContent"]["entries"]
    assert any(entry["path"].endswith("WORKSPACE") for entry in entries)


def test_context_init_tool_creates_project_context(tmp_path: Path, monkeypatch) -> None:
    manager = _make_manager(tmp_path)
    monkeypatch.chdir(tmp_path)
    project_root = tmp_path / "workspace_project"
    project_root.mkdir(parents=True)

    init_response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 31,
            "method": "tools/call",
            "params": {
                "name": "context.init",
                "arguments": {"project_path": str(project_root)},
            },
        },
        manager,
    )
    assert init_response is not None
    structured = init_response["result"]["structuredContent"]
    assert structured["context_path"] == str(project_root / ".context")
    assert (project_root / ".context").exists()


def test_context_init_rejects_project_outside_cwd_without_allowed_context_root(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    project_root = tmp_path / "workspace_project"
    project_root.mkdir(parents=True)

    init_response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 50,
            "method": "tools/call",
            "params": {
                "name": "context.init",
                "arguments": {"project_path": str(project_root)},
            },
        },
        manager,
    )
    assert init_response is not None
    assert "error" in init_response
    assert "current working directory" in init_response["error"]["message"]


def test_context_init_allows_explicit_allowed_context_root_outside_cwd(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    project_root = tmp_path / "workspace_project"
    project_root.mkdir(parents=True)
    context_root = manager.config.general.context_root / "projects" / "workspace_project"

    init_response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 51,
            "method": "tools/call",
            "params": {
                "name": "context.init",
                "arguments": {
                    "project_path": str(project_root),
                    "context_root": str(context_root),
                },
            },
        },
        manager,
    )
    assert init_response is not None
    structured = init_response["result"]["structuredContent"]
    assert structured["context_path"] == str(context_root)
    assert context_root.exists()


def test_context_init_allows_project_under_workspace_root_outside_cwd(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace-root"
    workspace_root.mkdir()
    context_root = tmp_path / "context"
    context_root.mkdir(parents=True)
    (context_root / "scratchpad").mkdir()
    manager = AFSManager(
        config=AFSConfig(
            general=GeneralConfig(
                context_root=context_root,
                workspace_directories=[WorkspaceDirectory(path=workspace_root)],
            )
        )
    )
    project_root = workspace_root / "workspace_project"
    project_root.mkdir(parents=True)

    init_response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 62,
            "method": "tools/call",
            "params": {
                "name": "context.init",
                "arguments": {"project_path": str(project_root)},
            },
        },
        manager,
    )
    assert init_response is not None
    structured = init_response["result"]["structuredContent"]
    assert structured["context_path"] == str(project_root / ".context")
    assert (project_root / ".context").exists()


def test_context_unmount_tool_removes_alias(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    context_root = manager.config.general.context_root
    (context_root / "knowledge").mkdir(exist_ok=True)

    source_docs = tmp_path / "docs_source"
    source_docs.mkdir(parents=True)

    mount_response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 32,
            "method": "tools/call",
            "params": {
                "name": "context.mount",
                "arguments": {
                    "context_path": str(context_root),
                    "source": str(source_docs),
                    "mount_type": "knowledge",
                    "alias": "docs",
                },
            },
        },
        manager,
    )
    assert mount_response is not None
    assert (context_root / "knowledge" / "docs").exists()

    unmount_response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 33,
            "method": "tools/call",
            "params": {
                "name": "context.unmount",
                "arguments": {
                    "context_path": str(context_root),
                    "mount_type": "knowledge",
                    "alias": "docs",
                },
            },
        },
        manager,
    )
    assert unmount_response is not None
    structured = unmount_response["result"]["structuredContent"]
    assert structured["removed"] is True
    assert not (context_root / "knowledge" / "docs").exists()


def test_context_index_rebuild_tool(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    context_root = manager.config.general.context_root
    notes = context_root / "scratchpad" / "notes.txt"
    notes.write_text("portable sqlite index", encoding="utf-8")

    rebuild_response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "context.index.rebuild",
                "arguments": {
                    "context_path": str(context_root),
                    "mount_types": ["scratchpad"],
                    "include_content": True,
                },
            },
        },
        manager,
    )
    assert rebuild_response is not None
    structured = rebuild_response["result"]["structuredContent"]
    assert structured["rows_written"] >= 1
    assert structured["by_mount_type"]["scratchpad"] >= 1
    assert structured["db_path"].endswith("context_index.sqlite3")


def test_context_index_uses_configured_db_filename(tmp_path: Path) -> None:
    context_root = tmp_path / "context"
    context_root.mkdir(parents=True)
    (context_root / "scratchpad").mkdir()
    manager = AFSManager(
        config=AFSConfig(
            general=GeneralConfig(
                context_root=context_root,
            ),
            context_index=ContextIndexConfig(db_filename="sqlite/context.db"),
        )
    )
    notes = context_root / "scratchpad" / "notes.txt"
    notes.write_text("custom db path", encoding="utf-8")

    rebuild_response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 40,
            "method": "tools/call",
            "params": {
                "name": "context.index.rebuild",
                "arguments": {
                    "context_path": str(context_root),
                    "mount_types": ["scratchpad"],
                },
            },
        },
        manager,
    )
    assert rebuild_response is not None
    db_path = rebuild_response["result"]["structuredContent"]["db_path"]
    assert db_path.endswith("global/sqlite/context.db")


def test_context_query_tool_auto_indexes(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    context_root = manager.config.general.context_root
    note_path = context_root / "scratchpad" / "gemini_notes.md"
    note_path.write_text("Gemini-compatible context query support", encoding="utf-8")

    query_response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {
                "name": "context.query",
                "arguments": {
                    "context_path": str(context_root),
                    "mount_types": ["scratchpad"],
                    "query": "Gemini-compatible",
                    "limit": 10,
                },
            },
        },
        manager,
    )
    assert query_response is not None
    structured = query_response["result"]["structuredContent"]
    assert structured["count"] >= 1
    assert any(entry["relative_path"] == "gemini_notes.md" for entry in structured["entries"])
    assert "index_rebuild" in structured


def test_context_query_tool_resolves_parent_context_from_nested_path(
    tmp_path: Path, monkeypatch
) -> None:
    general = GeneralConfig(
        context_root=tmp_path / "shared-context",
        mcp_allowed_roots=[tmp_path],
    )
    manager = AFSManager(config=AFSConfig(general=general))
    project_path = tmp_path / "project"
    project_path.mkdir()
    manager.ensure(path=project_path)
    context_root = project_path / ".context"
    nested_path = project_path / "docs" / "dev"
    nested_path.mkdir(parents=True)
    (nested_path / "afs.toml").write_text("[project]\nname = 'docs-dev'\n", encoding="utf-8")
    note_path = context_root / "scratchpad" / "nested_route.md"
    note_path.write_text("nested path route marker", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    query_response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {
                "name": "context.query",
                "arguments": {
                    "path": str(nested_path),
                    "mount_types": ["scratchpad"],
                    "query": "route marker",
                    "limit": 10,
                },
            },
        },
        manager,
    )
    assert query_response is not None
    structured = query_response["result"]["structuredContent"]
    assert structured["context_path"] == str(context_root)
    assert structured["count"] >= 1
    assert any(entry["relative_path"] == "nested_route.md" for entry in structured["entries"])


def test_context_query_tool_errors_when_no_existing_context(tmp_path: Path) -> None:
    general = GeneralConfig(
        context_root=tmp_path / "shared-context",
        mcp_allowed_roots=[tmp_path],
    )
    manager = AFSManager(config=AFSConfig(general=general))
    project_path = tmp_path / "orphan-project"
    project_path.mkdir()

    query_response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/call",
            "params": {
                "name": "context.query",
                "arguments": {
                    "path": str(project_path),
                    "mount_types": ["scratchpad"],
                    "query": "missing",
                    "limit": 10,
                },
            },
        },
        manager,
    )
    assert query_response is not None
    assert "error" in query_response
    assert str(project_path) in query_response["error"]["message"]
    assert "context.ensure" in query_response["error"]["message"]


def test_context_status_and_diff_tools(tmp_path: Path, monkeypatch) -> None:
    for name in (
        "AFS_PROFILE",
        "AFS_ENABLED_EXTENSIONS",
        "AFS_KNOWLEDGE_MOUNTS",
        "AFS_SKILL_ROOTS",
        "AFS_MODEL_REGISTRIES",
        "AFS_POLICIES",
    ):
        monkeypatch.delenv(name, raising=False)
    manager = _make_manager(tmp_path)
    context_root = manager.config.general.context_root
    notes_dir = context_root / "scratchpad"
    (context_root / "knowledge").mkdir(exist_ok=True)
    note_path = notes_dir / "daily.md"
    note_path.write_text("initial note", encoding="utf-8")
    docs_root = tmp_path / "docs"
    docs_root.mkdir()
    (docs_root / "README.md").write_text("knowledge mount", encoding="utf-8")
    manager.mount(docs_root, MountType.KNOWLEDGE, alias="docs", context_path=context_root)

    rebuild_response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 63,
            "method": "tools/call",
            "params": {
                "name": "context.index.rebuild",
                "arguments": {
                    "context_path": str(context_root),
                    "mount_types": ["scratchpad"],
                },
            },
        },
        manager,
    )
    assert rebuild_response is not None

    status_response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 64,
            "method": "tools/call",
            "params": {
                "name": "context.status",
                "arguments": {"context_path": str(context_root)},
            },
        },
        manager,
    )
    assert status_response is not None
    status_structured = status_response["result"]["structuredContent"]
    assert status_structured["mount_counts"]["scratchpad"] >= 1
    assert status_structured["mount_counts"]["knowledge"] == 1
    assert status_structured["mount_health"]["healthy"] is True
    assert status_structured["actions"] == []
    assert status_structured["index"]["enabled"] is True
    assert status_structured["index"]["has_entries"] is True
    assert status_structured["index"]["total_entries"] >= 1

    note_path.write_text("updated note", encoding="utf-8")
    added_path = notes_dir / "extra.md"
    added_path.write_text("extra", encoding="utf-8")
    note_path.unlink()

    diff_response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 65,
            "method": "tools/call",
            "params": {
                "name": "context.diff",
                "arguments": {
                    "context_path": str(context_root),
                    "mount_types": ["scratchpad"],
                },
            },
        },
        manager,
    )
    assert diff_response is not None
    diff_structured = diff_response["result"]["structuredContent"]
    assert any(entry["relative_path"] == "extra.md" for entry in diff_structured["added"])
    assert any(entry["relative_path"] == "daily.md" for entry in diff_structured["deleted"])
    assert diff_structured["total_changes"] >= 2


def test_context_repair_tool_remaps_missing_mount(tmp_path: Path) -> None:
    workspace_root = tmp_path / "workspace-root"
    workspace_root.mkdir()
    context_root = tmp_path / "context"
    manager = AFSManager(
        config=AFSConfig(
            general=GeneralConfig(
                context_root=context_root,
                workspace_directories=[WorkspaceDirectory(path=workspace_root)],
            )
        )
    )
    context_root.mkdir(parents=True, exist_ok=True)
    for mount_type in MountType:
        (context_root / mount_type.value).mkdir(exist_ok=True)

    legacy_docs = tmp_path / "legacy-docs"
    legacy_docs.mkdir()
    target = context_root / "knowledge" / "docs"
    target.symlink_to(legacy_docs, target_is_directory=True)
    metadata = {
        "created_at": "2026-01-01T00:00:00",
        "description": "repair target",
        "directories": {mount.value: mount.value for mount in MountType},
        "mount_provenance": {
            "knowledge": {
                "docs": {
                    "alias": "docs",
                    "mount_type": "knowledge",
                    "source": str(legacy_docs),
                    "managed_by": "manual",
                }
            }
        },
    }
    (context_root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    new_docs = workspace_root / legacy_docs.name
    new_docs.mkdir()
    (new_docs / "README.md").write_text("remapped", encoding="utf-8")
    legacy_docs.rmdir()

    repair_response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 66,
            "method": "tools/call",
            "params": {
                "name": "context.repair",
                "arguments": {
                    "context_path": str(context_root),
                    "reapply_profile": False,
                },
            },
        },
        manager,
    )
    assert repair_response is not None
    structured = repair_response["result"]["structuredContent"]
    assert len(structured["remapped_mounts"]) == 1
    assert structured["health_after"]["healthy"] is True


def test_context_query_respects_auto_index_config_default(tmp_path: Path) -> None:
    context_root = tmp_path / "context"
    context_root.mkdir(parents=True)
    (context_root / "scratchpad").mkdir()
    manager = AFSManager(
        config=AFSConfig(
            general=GeneralConfig(
                context_root=context_root,
            ),
            context_index=ContextIndexConfig(auto_index=False),
        )
    )
    note_path = context_root / "scratchpad" / "manual_index.md"
    note_path.write_text("manual indexing required", encoding="utf-8")

    query_response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 41,
            "method": "tools/call",
            "params": {
                "name": "context.query",
                "arguments": {
                    "context_path": str(context_root),
                    "mount_types": ["scratchpad"],
                    "query": "manual indexing required",
                },
            },
        },
        manager,
    )
    assert query_response is not None
    structured = query_response["result"]["structuredContent"]
    assert structured["count"] == 0
    assert "index_rebuild" not in structured


def test_context_query_indexes_symlink_mount_content(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    context_root = manager.config.general.context_root
    knowledge_root = context_root / "knowledge"
    knowledge_root.mkdir(exist_ok=True)

    source_docs = tmp_path / "source_docs"
    source_docs.mkdir()
    (source_docs / "design.md").write_text("SQLite indexing for mounted docs", encoding="utf-8")
    (knowledge_root / "docs").symlink_to(source_docs, target_is_directory=True)

    query_response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/call",
            "params": {
                "name": "context.query",
                "arguments": {
                    "context_path": str(context_root),
                    "mount_types": ["knowledge"],
                    "query": "mounted docs",
                    "limit": 10,
                },
            },
        },
        manager,
    )
    assert query_response is not None
    structured = query_response["result"]["structuredContent"]
    assert any(entry["relative_path"] == "docs/design.md" for entry in structured["entries"])


def test_context_write_keeps_context_query_fresh_without_rebuild(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    context_root = manager.config.general.context_root
    target = context_root / "scratchpad" / "state.md"
    target.write_text("initial context", encoding="utf-8")

    rebuild_response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 7,
            "method": "tools/call",
            "params": {
                "name": "context.index.rebuild",
                "arguments": {
                    "context_path": str(context_root),
                    "mount_types": ["scratchpad"],
                },
            },
        },
        manager,
    )
    assert rebuild_response is not None

    write_response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 8,
            "method": "tools/call",
            "params": {
                "name": "context.write",
                "arguments": {
                    "path": str(target),
                    "content": "incremental freshness check",
                },
            },
        },
        manager,
    )
    assert write_response is not None
    assert write_response["result"]["structuredContent"]["index_updated"] is True

    query_response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 9,
            "method": "tools/call",
            "params": {
                "name": "context.query",
                "arguments": {
                    "context_path": str(context_root),
                    "mount_types": ["scratchpad"],
                    "query": "freshness",
                    "auto_index": False,
                },
            },
        },
        manager,
    )
    assert query_response is not None
    structured = query_response["result"]["structuredContent"]
    assert any(entry["relative_path"] == "state.md" for entry in structured["entries"])
    assert "index_rebuild" not in structured


def test_context_delete_updates_context_index(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    context_root = manager.config.general.context_root
    target = context_root / "scratchpad" / "delete_me.md"
    target.write_text("remove from index", encoding="utf-8")

    rebuild_response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 13,
            "method": "tools/call",
            "params": {
                "name": "context.index.rebuild",
                "arguments": {
                    "context_path": str(context_root),
                    "mount_types": ["scratchpad"],
                },
            },
        },
        manager,
    )
    assert rebuild_response is not None

    delete_response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 14,
            "method": "tools/call",
            "params": {
                "name": "context.delete",
                "arguments": {"path": str(target)},
            },
        },
        manager,
    )
    assert delete_response is not None
    assert delete_response["result"]["structuredContent"]["index_updated"] is True

    query_response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 15,
            "method": "tools/call",
            "params": {
                "name": "context.query",
                "arguments": {
                    "context_path": str(context_root),
                    "mount_types": ["scratchpad"],
                    "query": "remove from index",
                    "auto_index": False,
                },
            },
        },
        manager,
    )
    assert query_response is not None
    entries = query_response["result"]["structuredContent"]["entries"]
    assert not any(entry["relative_path"] == "delete_me.md" for entry in entries)


def test_context_delete_directory_removes_nested_index_entries(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    context_root = manager.config.general.context_root
    docs_dir = context_root / "scratchpad" / "docs"
    docs_dir.mkdir()
    nested = docs_dir / "guide.md"
    nested.write_text("nested delete marker", encoding="utf-8")

    rebuild_response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 42,
            "method": "tools/call",
            "params": {
                "name": "context.index.rebuild",
                "arguments": {
                    "context_path": str(context_root),
                    "mount_types": ["scratchpad"],
                },
            },
        },
        manager,
    )
    assert rebuild_response is not None

    delete_response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 43,
            "method": "tools/call",
            "params": {
                "name": "context.delete",
                "arguments": {"path": str(docs_dir), "recursive": True},
            },
        },
        manager,
    )
    assert delete_response is not None
    assert delete_response["result"]["structuredContent"]["index_updated"] is True

    query_response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 44,
            "method": "tools/call",
            "params": {
                "name": "context.query",
                "arguments": {
                    "context_path": str(context_root),
                    "mount_types": ["scratchpad"],
                    "query": "nested delete marker",
                    "auto_index": False,
                },
            },
        },
        manager,
    )
    assert query_response is not None
    entries = query_response["result"]["structuredContent"]["entries"]
    assert not any(entry["relative_path"].startswith("docs/") for entry in entries)


def test_context_move_updates_context_index(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    context_root = manager.config.general.context_root
    source = context_root / "scratchpad" / "before.md"
    destination = context_root / "scratchpad" / "after.md"
    source.write_text("moved content marker", encoding="utf-8")

    rebuild_response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 16,
            "method": "tools/call",
            "params": {
                "name": "context.index.rebuild",
                "arguments": {
                    "context_path": str(context_root),
                    "mount_types": ["scratchpad"],
                },
            },
        },
        manager,
    )
    assert rebuild_response is not None

    move_response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 17,
            "method": "tools/call",
            "params": {
                "name": "context.move",
                "arguments": {"source": str(source), "destination": str(destination)},
            },
        },
        manager,
    )
    assert move_response is not None
    payload = move_response["result"]["structuredContent"]
    assert payload["index_updated"] is True
    assert destination.exists()
    assert not source.exists()

    query_response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 18,
            "method": "tools/call",
            "params": {
                "name": "context.query",
                "arguments": {
                    "context_path": str(context_root),
                    "mount_types": ["scratchpad"],
                    "query": "moved content marker",
                    "auto_index": False,
                },
            },
        },
        manager,
    )
    assert query_response is not None
    entries = query_response["result"]["structuredContent"]["entries"]
    assert any(entry["relative_path"] == "after.md" for entry in entries)
    assert not any(entry["relative_path"] == "before.md" for entry in entries)


def test_context_move_directory_updates_nested_context_index_entries(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    context_root = manager.config.general.context_root
    source_dir = context_root / "scratchpad" / "before"
    destination_dir = context_root / "scratchpad" / "after"
    source_dir.mkdir()
    (source_dir / "guide.md").write_text("moved directory marker", encoding="utf-8")

    rebuild_response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 45,
            "method": "tools/call",
            "params": {
                "name": "context.index.rebuild",
                "arguments": {
                    "context_path": str(context_root),
                    "mount_types": ["scratchpad"],
                },
            },
        },
        manager,
    )
    assert rebuild_response is not None

    move_response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 46,
            "method": "tools/call",
            "params": {
                "name": "context.move",
                "arguments": {
                    "source": str(source_dir),
                    "destination": str(destination_dir),
                },
            },
        },
        manager,
    )
    assert move_response is not None
    assert move_response["result"]["structuredContent"]["index_updated"] is True

    query_response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 47,
            "method": "tools/call",
            "params": {
                "name": "context.query",
                "arguments": {
                    "context_path": str(context_root),
                    "mount_types": ["scratchpad"],
                    "query": "moved directory marker",
                    "auto_index": False,
                },
            },
        },
        manager,
    )
    assert query_response is not None
    entries = query_response["result"]["structuredContent"]["entries"]
    assert any(entry["relative_path"] == "after/guide.md" for entry in entries)
    assert not any(entry["relative_path"] == "before/guide.md" for entry in entries)


def test_context_query_auto_refreshes_after_external_write(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    context_root = manager.config.general.context_root
    target = context_root / "scratchpad" / "external.md"
    target.write_text("before external edit", encoding="utf-8")

    rebuild_response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 10,
            "method": "tools/call",
            "params": {
                "name": "context.index.rebuild",
                "arguments": {
                    "context_path": str(context_root),
                    "mount_types": ["scratchpad"],
                },
            },
        },
        manager,
    )
    assert rebuild_response is not None

    target.write_text("after external edit", encoding="utf-8")

    query_response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 11,
            "method": "tools/call",
            "params": {
                "name": "context.query",
                "arguments": {
                    "context_path": str(context_root),
                    "mount_types": ["scratchpad"],
                    "query": "external edit",
                    "auto_index": True,
                },
            },
        },
        manager,
    )
    assert query_response is not None
    structured = query_response["result"]["structuredContent"]
    assert any(entry["relative_path"] == "external.md" for entry in structured["entries"])
    assert "index_rebuild" in structured


def test_context_query_auto_refreshes_after_external_rename(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    context_root = manager.config.general.context_root
    source = context_root / "scratchpad" / "before.md"
    destination = context_root / "scratchpad" / "after.md"
    source.write_text("rename freshness marker", encoding="utf-8")

    rebuild_response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 52,
            "method": "tools/call",
            "params": {
                "name": "context.index.rebuild",
                "arguments": {
                    "context_path": str(context_root),
                    "mount_types": ["scratchpad"],
                },
            },
        },
        manager,
    )
    assert rebuild_response is not None

    source.rename(destination)

    query_response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 53,
            "method": "tools/call",
            "params": {
                "name": "context.query",
                "arguments": {
                    "context_path": str(context_root),
                    "mount_types": ["scratchpad"],
                    "query": "rename freshness marker",
                    "auto_index": True,
                },
            },
        },
        manager,
    )
    assert query_response is not None
    structured = query_response["result"]["structuredContent"]
    assert any(entry["relative_path"] == "after.md" for entry in structured["entries"])
    assert not any(entry["relative_path"] == "before.md" for entry in structured["entries"])
    assert "index_rebuild" in structured


def test_initialize_advertises_resources_and_prompts(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    response = _handle_request(
        {"jsonrpc": "2.0", "id": 20, "method": "initialize", "params": {}}, manager
    )
    assert response is not None
    caps = response["result"]["capabilities"]
    assert "tools" in caps
    assert "resources" in caps
    assert "prompts" in caps
    assert response["result"]["protocolVersion"] == PROTOCOL_VERSION
    assert caps["resources"]["subscribe"] is False


def test_initialize_negotiates_supported_protocol_version(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 21,
            "method": "initialize",
            "params": {"protocolVersion": PROTOCOL_VERSION},
        },
        manager,
    )
    assert response is not None
    assert response["result"]["protocolVersion"] == PROTOCOL_VERSION


def test_read_message_accepts_lf_header_terminator() -> None:
    body = b'{"jsonrpc":"2.0","id":1,"method":"ping"}'
    stream = BytesIO(
        f"Content-Length: {len(body)}\n\n".encode("ascii") + body
    )

    payload, mode = _read_message(stream)
    assert mode == "content-length"
    assert payload == {"jsonrpc": "2.0", "id": 1, "method": "ping"}


def test_read_message_accepts_cr_header_terminator() -> None:
    body = b'{"jsonrpc":"2.0","id":2,"method":"ping"}'
    stream = BytesIO(
        f"Content-Length: {len(body)}\r\r".encode("ascii") + body
    )

    payload, mode = _read_message(stream)
    assert mode == "content-length"
    assert payload == {"jsonrpc": "2.0", "id": 2, "method": "ping"}


def test_read_message_accepts_jsonl_transport() -> None:
    stream = BytesIO(b'{"jsonrpc":"2.0","id":3,"method":"ping"}\n')

    payload, mode = _read_message(stream)
    assert mode == "jsonl"
    assert payload == {"jsonrpc": "2.0", "id": 3, "method": "ping"}


def test_resources_list_returns_contexts_resource(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    response = _handle_request(
        {"jsonrpc": "2.0", "id": 21, "method": "resources/list"}, manager
    )
    assert response is not None
    resources = response["result"]["resources"]
    uris = [r["uri"] for r in resources]
    assert "afs://contexts" in uris
    assert "afs://schemas/plan" in uris
    assert "afs://schemas/verification-summary" in uris
    assert f"afs://context/{manager.config.general.context_root}/bootstrap" in uris


def test_resources_read_contexts(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 22,
            "method": "resources/read",
            "params": {"uri": "afs://contexts"},
        },
        manager,
    )
    assert response is not None
    contents = response["result"]["contents"]
    assert len(contents) == 1
    assert contents[0]["uri"] == "afs://contexts"
    assert contents[0]["mimeType"] == "application/json"
    data = json.loads(contents[0]["text"])
    assert isinstance(data, list)


def test_resources_read_schema_contract(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 221,
            "method": "resources/read",
            "params": {"uri": "afs://schemas/plan"},
        },
        manager,
    )
    assert response is not None
    contents = response["result"]["contents"]
    assert len(contents) == 1
    assert contents[0]["uri"] == "afs://schemas/plan"
    assert contents[0]["mimeType"] == "application/schema+json"
    schema = json.loads(contents[0]["text"])
    assert schema["title"] == "AFS Plan"
    assert schema["required"] == ["goal", "steps", "completion_signal", "confidence"]
    assert schema["additionalProperties"] is False


def test_resources_read_unknown_schema_uri_returns_error(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 222,
            "method": "resources/read",
            "params": {"uri": "afs://schemas/not-real"},
        },
        manager,
    )
    assert response is not None
    assert "error" in response
    assert "Unknown resource URI" in response["error"]["message"]


def test_resources_read_contexts_filters_out_disallowed_contexts(
    tmp_path: Path, monkeypatch
) -> None:
    manager = _make_manager(tmp_path)
    allowed_context = manager.config.general.context_root
    outside_context = tmp_path / "outside" / ".context"
    outside_context.mkdir(parents=True)
    (outside_context / "scratchpad").mkdir()

    allowed_root = manager.list_context(context_path=allowed_context)
    outside_root = manager.list_context(context_path=outside_context)

    monkeypatch.setattr(
        "afs.mcp_server.discover_contexts",
        lambda config=None: [allowed_root, outside_root],
    )

    response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 54,
            "method": "resources/read",
            "params": {"uri": "afs://contexts"},
        },
        manager,
    )
    assert response is not None
    contents = response["result"]["contents"]
    data = json.loads(contents[0]["text"])
    paths = {entry["path"] for entry in data}
    assert str(allowed_context) in paths
    assert str(outside_context) not in paths


def test_resources_read_metadata(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    context_root = manager.config.general.context_root
    (context_root / "metadata.json").write_text(
        json.dumps({"created_at": "2025-01-01", "description": "test", "agents": []}),
        encoding="utf-8",
    )
    uri = f"afs://context/{context_root}/metadata"
    response = _handle_request(
        {"jsonrpc": "2.0", "id": 23, "method": "resources/read", "params": {"uri": uri}},
        manager,
    )
    assert response is not None
    contents = response["result"]["contents"]
    assert len(contents) == 1
    data = json.loads(contents[0]["text"])
    assert data["description"] == "test"


def test_resources_read_index(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    context_root = manager.config.general.context_root
    uri = f"afs://context/{context_root}/index"
    response = _handle_request(
        {"jsonrpc": "2.0", "id": 24, "method": "resources/read", "params": {"uri": uri}},
        manager,
    )
    assert response is not None
    contents = response["result"]["contents"]
    assert len(contents) == 1
    data = json.loads(contents[0]["text"])
    assert "has_entries" in data
    assert "needs_refresh" in data


def test_resources_read_bootstrap(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    context_root = manager.config.general.context_root
    (context_root / "scratchpad" / "state.md").write_text(
        "bootstrap state",
        encoding="utf-8",
    )
    uri = f"afs://context/{context_root}/bootstrap"
    response = _handle_request(
        {"jsonrpc": "2.0", "id": 124, "method": "resources/read", "params": {"uri": uri}},
        manager,
    )
    assert response is not None
    contents = response["result"]["contents"]
    payload = json.loads(contents[0]["text"])
    assert payload["context_path"] == str(context_root)
    assert payload["scratchpad"]["state_text"] == "bootstrap state"


def test_resources_read_unknown_uri(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 25,
            "method": "resources/read",
            "params": {"uri": "afs://unknown"},
        },
        manager,
    )
    assert response is not None
    assert "error" in response


def test_resources_read_rejects_context_outside_allowed_roots(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    outside_context = tmp_path / "outside" / ".context"
    outside_context.mkdir(parents=True)
    (outside_context / "metadata.json").write_text(
        json.dumps({"created_at": "2025-01-01", "description": "outside", "agents": []}),
        encoding="utf-8",
    )

    response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 48,
            "method": "resources/read",
            "params": {"uri": f"afs://context/{outside_context}/metadata"},
        },
        manager,
    )
    assert response is not None
    assert "error" in response
    assert "Path outside allowed roots" in response["error"]["message"]


def test_prompts_list_returns_expected_prompts(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    response = _handle_request(
        {"jsonrpc": "2.0", "id": 26, "method": "prompts/list"}, manager
    )
    assert response is not None
    prompts = response["result"]["prompts"]
    names = {p["name"] for p in prompts}
    assert {
        "afs.session.bootstrap",
        "afs.session.pack",
        "afs.workflow.structured",
        "afs.context.overview",
        "afs.query.search",
        "afs.scratchpad.review",
    }.issubset(names)
    # Verify argument schemas
    for prompt in prompts:
        assert "arguments" in prompt
        assert isinstance(prompt["arguments"], list)


def test_prompts_get_session_bootstrap(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    context_root = manager.config.general.context_root
    (context_root / "scratchpad" / "state.md").write_text(
        "bootstrap state",
        encoding="utf-8",
    )
    response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 125,
            "method": "prompts/get",
            "params": {
                "name": "afs.session.bootstrap",
                "arguments": {"context_path": str(context_root)},
            },
        },
        manager,
    )
    assert response is not None
    messages = response["result"]["messages"]
    text = messages[0]["content"]["text"]
    assert "AFS Session Bootstrap" in text
    assert "bootstrap state" in text


def test_tool_session_pack(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    context_root = manager.config.general.context_root
    (context_root / "knowledge").mkdir(exist_ok=True)
    (context_root / "knowledge" / "guide.md").write_text(
        "codex service guide",
        encoding="utf-8",
    )
    from afs.context_index import ContextSQLiteIndex

    ContextSQLiteIndex(manager, context_root).rebuild(
        mount_types=[MountType.KNOWLEDGE],
        include_content=True,
    )
    response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 126,
            "method": "tools/call",
            "params": {
                "name": "session.pack",
                "arguments": {
                    "context_path": str(context_root),
                    "query": "service guide",
                    "task": "Implement the service guide fix.",
                    "model": "codex",
                    "workflow": "edit_fast",
                    "tool_profile": "edit_and_verify",
                    "pack_mode": "retrieval",
                },
            },
        },
        manager,
    )
    assert response is not None
    payload = response["result"]["structuredContent"]
    assert payload["model"] == "codex"
    assert payload["task"] == "Implement the service guide fix."
    assert payload["execution_profile"]["workflow"] == "edit_fast"
    assert payload["execution_profile"]["loop_policy"].startswith("Prompt-only rail.")
    assert payload["execution_profile"]["retry_hint"]
    assert payload["pack_mode"] == "retrieval"
    assert any("guide.md" in source for source in payload["sources"])


def test_tool_operator_digest(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 128,
            "method": "tools/call",
            "params": {
                "name": "operator.digest",
                "arguments": {
                    "label": "pytest -q",
                    "text": (
                        "FAILED tests/test_pack.py::test_cache - AssertionError: miss\n"
                        "========================= 88 passed, 1 failed in 2.10s ========================"
                    ),
                },
            },
        },
        manager,
    )
    assert response is not None
    payload = response["result"]["structuredContent"]
    assert payload["label"] == "pytest -q"
    assert payload["kind"] == "pytest"
    assert payload["details"]["counts"]["passed"] == 88
    assert payload["details"]["counts"]["failed"] == 1
    assert "Summary: pytest failed" in payload["digest_text"]


def test_prompts_get_session_pack(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    context_root = manager.config.general.context_root
    (context_root / "knowledge").mkdir(exist_ok=True)
    (context_root / "knowledge" / "guide.md").write_text(
        "gemini service guide",
        encoding="utf-8",
    )
    from afs.context_index import ContextSQLiteIndex

    ContextSQLiteIndex(manager, context_root).rebuild(
        mount_types=[MountType.KNOWLEDGE],
        include_content=True,
    )
    response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 127,
            "method": "prompts/get",
            "params": {
                "name": "afs.session.pack",
                "arguments": {
                    "context_path": str(context_root),
                    "query": "service guide",
                    "task": "Review the service guide findings.",
                    "model": "gemini",
                    "workflow": "review_deep",
                    "pack_mode": "full_slice",
                },
            },
        },
        manager,
    )
    assert response is not None
    text = response["result"]["messages"][0]["content"]["text"]
    assert "AFS Context Pack" in text
    assert "gemini" in text.lower()
    assert "Pack mode: full_slice" in text
    assert "## Task" in text


def test_prompts_get_workflow_structured(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    context_root = manager.config.general.context_root
    (context_root / "knowledge").mkdir(exist_ok=True)
    (context_root / "knowledge" / "guide.md").write_text(
        "gemini planning guide",
        encoding="utf-8",
    )
    from afs.context_index import ContextSQLiteIndex

    ContextSQLiteIndex(manager, context_root).rebuild(
        mount_types=[MountType.KNOWLEDGE],
        include_content=True,
    )
    response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 1281,
            "method": "prompts/get",
            "params": {
                "name": "afs.workflow.structured",
                "arguments": {
                    "context_path": str(context_root),
                    "schema_name": "plan",
                    "task": "Plan the guide update.",
                    "query": "planning guide",
                    "model": "gemini",
                    "workflow": "scan_fast",
                    "pack_mode": "retrieval",
                },
            },
        },
        manager,
    )
    assert response is not None
    text = response["result"]["messages"][0]["content"]["text"]
    assert "# AFS Structured Workflow Prompt" in text
    assert "Schema resource: afs://schemas/plan" in text
    assert '"completion_signal"' in text
    assert "Pack mode: retrieval" in text
    assert "Plan the guide update." in text


def test_prompts_get_workflow_structured_rejects_unknown_schema(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 1282,
            "method": "prompts/get",
            "params": {
                "name": "afs.workflow.structured",
                "arguments": {"schema_name": "nope", "task": "Plan it."},
            },
        },
        manager,
    )
    assert response is not None
    assert "error" in response
    assert "schema_name must be one of" in response["error"]["message"]


def test_prompts_get_context_overview(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    context_root = manager.config.general.context_root
    (context_root / "metadata.json").write_text(
        json.dumps({
            "created_at": "2025-01-01",
            "description": "test project",
            "agents": ["claude"],
            "directories": {},
            "manual_only": [],
        }),
        encoding="utf-8",
    )
    response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 27,
            "method": "prompts/get",
            "params": {
                "name": "afs.context.overview",
                "arguments": {"context_path": str(context_root)},
            },
        },
        manager,
    )
    assert response is not None
    messages = response["result"]["messages"]
    assert len(messages) >= 1
    assert messages[0]["role"] == "user"
    assert "AFS Context" in messages[0]["content"]["text"]


def test_prompts_get_query_search(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    context_root = manager.config.general.context_root
    (context_root / "scratchpad" / "notes.txt").write_text(
        "prompt search test content", encoding="utf-8"
    )
    response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 28,
            "method": "prompts/get",
            "params": {
                "name": "afs.query.search",
                "arguments": {"query": "prompt search"},
            },
        },
        manager,
    )
    assert response is not None
    messages = response["result"]["messages"]
    assert len(messages) >= 1
    assert messages[0]["role"] == "user"
    text = messages[0]["content"]["text"]
    assert "prompt search" in text.lower() or "Search results" in text


def test_prompts_get_query_search_supports_context_path_and_auto_refresh(
    tmp_path: Path,
) -> None:
    manager = _make_manager(tmp_path)
    context_root = manager.config.general.context_root
    source = context_root / "scratchpad" / "before.md"
    destination = context_root / "scratchpad" / "after.md"
    source.write_text("prompt rename marker", encoding="utf-8")

    rebuild_response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 55,
            "method": "tools/call",
            "params": {
                "name": "context.index.rebuild",
                "arguments": {
                    "context_path": str(context_root),
                    "mount_types": ["scratchpad"],
                },
            },
        },
        manager,
    )
    assert rebuild_response is not None

    source.rename(destination)

    response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 56,
            "method": "prompts/get",
            "params": {
                "name": "afs.query.search",
                "arguments": {
                    "context_path": str(context_root),
                    "query": "prompt rename marker",
                    "mount_types": "scratchpad",
                },
            },
        },
        manager,
    )
    assert response is not None
    messages = response["result"]["messages"]
    text = messages[0]["content"]["text"]
    assert "after.md" in text
    assert "before.md" not in text
    assert "Index refreshed before search." in text


def test_prompts_get_scratchpad_review(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    context_root = manager.config.general.context_root
    (context_root / "scratchpad" / "state.md").write_text(
        "current state info", encoding="utf-8"
    )
    (context_root / "scratchpad" / "deferred.md").write_text(
        "deferred task list", encoding="utf-8"
    )
    response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 29,
            "method": "prompts/get",
            "params": {
                "name": "afs.scratchpad.review",
                "arguments": {"context_path": str(context_root)},
            },
        },
        manager,
    )
    assert response is not None
    messages = response["result"]["messages"]
    assert len(messages) >= 1
    text = messages[0]["content"]["text"]
    assert "current state info" in text
    assert "deferred task list" in text


def test_prompts_get_scratchpad_review_uses_remapped_directory(tmp_path: Path) -> None:
    manager = _make_remapped_manager(tmp_path, scratchpad="notes")
    context_root = manager.config.general.context_root
    (context_root / "notes" / "state.md").write_text(
        "remapped state", encoding="utf-8"
    )
    (context_root / "notes" / "deferred.md").write_text(
        "remapped deferred", encoding="utf-8"
    )
    response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 129,
            "method": "prompts/get",
            "params": {
                "name": "afs.scratchpad.review",
                "arguments": {"context_path": str(context_root)},
            },
        },
        manager,
    )
    assert response is not None
    text = response["result"]["messages"][0]["content"]["text"]
    assert "remapped state" in text
    assert "remapped deferred" in text


def test_prompts_get_unknown_prompt(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 30,
            "method": "prompts/get",
            "params": {"name": "nonexistent", "arguments": {}},
        },
        manager,
    )
    assert response is not None
    assert "error" in response


def test_prompts_get_rejects_context_outside_allowed_roots(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    outside_context = tmp_path / "outside" / ".context"
    scratchpad = outside_context / "scratchpad"
    scratchpad.mkdir(parents=True)
    (scratchpad / "state.md").write_text("outside state", encoding="utf-8")

    response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 49,
            "method": "prompts/get",
            "params": {
                "name": "afs.scratchpad.review",
                "arguments": {"context_path": str(outside_context)},
            },
        },
        manager,
    )
    assert response is not None
    assert "error" in response
    assert "Path outside allowed roots" in response["error"]["message"]


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
            "id": 19,
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


def test_extension_mcp_server_registers_tools_resources_and_prompts(
    tmp_path: Path,
) -> None:
    ext_root = tmp_path / "extensions"
    ext_dir = ext_root / "ext_workspace"
    ext_dir.mkdir(parents=True)
    (ext_dir / "extension.toml").write_text(
        "name = \"ext_workspace\"\n"
        "\n"
        "[mcp_server]\n"
        "module = \"ext_surface\"\n"
        "factory = \"register_mcp_server\"\n",
        encoding="utf-8",
    )
    (ext_dir / "ext_surface.py").write_text(
        "def register_mcp_server(_manager):\n"
        "    def echo(arguments):\n"
        "        return {'echo': arguments.get('value', '')}\n"
        "\n"
        "    def status_resource(manager):\n"
        "        return {'text': '{\"status\": \"ok\"}'}\n"
        "\n"
        "    def review_prompt(arguments):\n"
        "        value = arguments.get('value', '')\n"
        "        return f'Extension review: {value}'\n"
        "\n"
        "    return {\n"
        "        'tools': [\n"
        "            {\n"
        "                'name': 'workspace.echo',\n"
        "                'description': 'Echo test payload',\n"
        "                'inputSchema': {\n"
        "                    'type': 'object',\n"
        "                    'properties': {'value': {'type': 'string'}},\n"
        "                    'additionalProperties': False,\n"
        "                },\n"
        "                'handler': echo,\n"
        "            }\n"
        "        ],\n"
        "        'resources': [\n"
        "            {\n"
        "                'uri': 'afs://ext/status',\n"
        "                'name': 'Extension status',\n"
        "                'description': 'Status resource from extension',\n"
        "                'mimeType': 'application/json',\n"
        "                'handler': status_resource,\n"
        "            }\n"
        "        ],\n"
        "        'prompts': [\n"
        "            {\n"
        "                'name': 'ext.review',\n"
        "                'description': 'Extension review prompt',\n"
        "                'arguments': [\n"
        "                    {\n"
        "                        'name': 'value',\n"
        "                        'description': 'Value to echo',\n"
        "                        'required': False,\n"
        "                    }\n"
        "                ],\n"
        "                'handler': review_prompt,\n"
        "            }\n"
        "        ],\n"
        "    }\n",
        encoding="utf-8",
    )

    context_root = tmp_path / "context"
    context_root.mkdir(parents=True)
    (context_root / "scratchpad").mkdir()
    manager = AFSManager(
        config=AFSConfig(
            general=GeneralConfig(
                context_root=context_root,
            ),
            extensions=ExtensionsConfig(
                enabled_extensions=["ext_workspace"],
                extension_dirs=[ext_root],
            ),
        )
    )

    registry = build_mcp_registry(manager)
    assert "workspace.echo" in registry.tools
    assert "afs://ext/status" in registry.resources
    assert "ext.review" in registry.prompts

    tool_response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 31,
            "method": "tools/call",
            "params": {
                "name": "workspace.echo",
                "arguments": {"value": "ok"},
            },
        },
        manager,
        registry=registry,
    )
    assert tool_response is not None
    assert tool_response["result"]["structuredContent"]["echo"] == "ok"

    resources_response = _handle_request(
        {"jsonrpc": "2.0", "id": 32, "method": "resources/list"},
        manager,
        registry=registry,
    )
    assert resources_response is not None
    resource_uris = {item["uri"] for item in resources_response["result"]["resources"]}
    assert "afs://ext/status" in resource_uris

    resource_read_response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 33,
            "method": "resources/read",
            "params": {"uri": "afs://ext/status"},
        },
        manager,
        registry=registry,
    )
    assert resource_read_response is not None
    contents = resource_read_response["result"]["contents"]
    assert contents[0]["uri"] == "afs://ext/status"
    assert json.loads(contents[0]["text"]) == {"status": "ok"}

    prompts_response = _handle_request(
        {"jsonrpc": "2.0", "id": 34, "method": "prompts/list"},
        manager,
        registry=registry,
    )
    assert prompts_response is not None
    prompt_names = {item["name"] for item in prompts_response["result"]["prompts"]}
    assert "ext.review" in prompt_names

    prompt_response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 35,
            "method": "prompts/get",
            "params": {
                "name": "ext.review",
                "arguments": {"value": "ready"},
            },
        },
        manager,
        registry=registry,
    )
    assert prompt_response is not None
    messages = prompt_response["result"]["messages"]
    assert messages[0]["content"]["text"] == "Extension review: ready"


def test_extension_mcp_server_conflicts_do_not_override_core_surface(
    tmp_path: Path,
) -> None:
    ext_root = tmp_path / "extensions"
    ext_dir = ext_root / "ext_workspace"
    ext_dir.mkdir(parents=True)
    (ext_dir / "extension.toml").write_text(
        "name = \"ext_workspace\"\n"
        "\n"
        "[mcp_server]\n"
        "module = \"ext_surface\"\n"
        "factory = \"register_mcp_server\"\n",
        encoding="utf-8",
    )
    (ext_dir / "ext_surface.py").write_text(
        "def register_mcp_server(_manager):\n"
        "    def duplicate_tool(arguments):\n"
        "        return {'duplicate': True}\n"
        "\n"
        "    def duplicate_resource(manager):\n"
        "        return {'text': 'duplicate'}\n"
        "\n"
        "    def duplicate_prompt(arguments):\n"
        "        return 'duplicate'\n"
        "\n"
        "    return {\n"
        "        'tools': [\n"
        "            {\n"
        "                'name': 'context.read',\n"
        "                'description': 'duplicate tool',\n"
        "                'handler': duplicate_tool,\n"
        "            }\n"
        "        ],\n"
        "        'resources': [\n"
        "            {\n"
        "                'uri': 'afs://contexts',\n"
        "                'name': 'duplicate resource',\n"
        "                'description': 'duplicate resource',\n"
        "                'handler': duplicate_resource,\n"
        "            },\n"
        "            {\n"
        "                'uri': 'afs://schemas/plan',\n"
        "                'name': 'duplicate schema resource',\n"
        "                'description': 'duplicate schema resource',\n"
        "                'handler': duplicate_resource,\n"
        "            }\n"
        "        ],\n"
        "        'prompts': [\n"
        "            {\n"
        "                'name': 'afs.context.overview',\n"
        "                'description': 'duplicate prompt',\n"
        "                'handler': duplicate_prompt,\n"
        "            }\n"
        "        ],\n"
        "    }\n",
        encoding="utf-8",
    )

    context_root = tmp_path / "context"
    context_root.mkdir(parents=True)
    (context_root / "scratchpad").mkdir()
    manager = AFSManager(
        config=AFSConfig(
            general=GeneralConfig(
                context_root=context_root,
            ),
            extensions=ExtensionsConfig(
                enabled_extensions=["ext_workspace"],
                extension_dirs=[ext_root],
            ),
        )
    )

    registry = build_mcp_registry(manager)
    errors = registry.load_errors
    assert any("Tool 'context.read' already registered by core" in message for message in errors.values())
    assert any(
        "Resource 'afs://contexts' already registered by core" in message
        for message in errors.values()
    )
    assert any(
        "Resource 'afs://schemas/plan' already registered by core" in message
        for message in errors.values()
    )
    assert any(
        "Prompt 'afs.context.overview' already registered by core" in message
        for message in errors.values()
    )


def test_profile_mcp_tools_modules_are_loaded_into_registry(
    monkeypatch,
    tmp_path: Path,
) -> None:
    module_path = tmp_path / "profile_mcp.py"
    module_path.write_text(
        "def register_mcp_server(_manager):\n"
        "    def echo(arguments):\n"
        "        return {'echo': arguments.get('value', '')}\n"
        "\n"
        "    def status(_manager):\n"
        "        return {'text': '{\"status\": \"profile\"}'}\n"
        "\n"
        "    def review(arguments):\n"
        "        return f\"Profile review: {arguments.get('value', '')}\"\n"
        "\n"
        "    return {\n"
        "        'tools': [\n"
        "            {\n"
        "                'name': 'profile.echo',\n"
        "                'description': 'Echo from profile module',\n"
        "                'inputSchema': {\n"
        "                    'type': 'object',\n"
        "                    'properties': {'value': {'type': 'string'}},\n"
        "                    'additionalProperties': False,\n"
        "                },\n"
        "                'handler': echo,\n"
        "            }\n"
        "        ],\n"
        "        'resources': [\n"
        "            {\n"
        "                'uri': 'afs://profile/status',\n"
        "                'name': 'Profile status',\n"
        "                'description': 'Status from profile MCP module',\n"
        "                'mimeType': 'application/json',\n"
        "                'handler': status,\n"
        "            }\n"
        "        ],\n"
        "        'prompts': [\n"
        "            {\n"
        "                'name': 'profile.review',\n"
        "                'description': 'Review from profile MCP module',\n"
        "                'arguments': [{'name': 'value', 'required': False}],\n"
        "                'handler': review,\n"
        "            }\n"
        "        ],\n"
        "    }\n",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(tmp_path))

    context_root = tmp_path / "context"
    context_root.mkdir(parents=True)
    (context_root / "scratchpad").mkdir()
    manager = AFSManager(
        config=AFSConfig(
            general=GeneralConfig(
                context_root=context_root,
            ),
            profiles=ProfilesConfig(
                active_profile="work",
                profiles={
                    "work": ProfileConfig(mcp_tools=["profile_mcp"]),
                },
            ),
        )
    )

    registry = build_mcp_registry(manager)
    assert "profile.echo" in registry.tools
    assert "afs://profile/status" in registry.resources
    assert "profile.review" in registry.prompts

    tool_response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 60,
            "method": "tools/call",
            "params": {
                "name": "profile.echo",
                "arguments": {"value": "ok"},
            },
        },
        manager,
        registry=registry,
    )
    assert tool_response is not None
    assert tool_response["result"]["structuredContent"]["echo"] == "ok"

    resource_response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 61,
            "method": "resources/read",
            "params": {"uri": "afs://profile/status"},
        },
        manager,
        registry=registry,
    )
    assert resource_response is not None
    assert json.loads(resource_response["result"]["contents"][0]["text"]) == {
        "status": "profile"
    }

    prompt_response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 62,
            "method": "prompts/get",
            "params": {
                "name": "profile.review",
                "arguments": {"value": "ready"},
            },
        },
        manager,
        registry=registry,
    )
    assert prompt_response is not None
    assert prompt_response["result"]["messages"][0]["content"]["text"] == "Profile review: ready"


def test_tools_list_includes_agent_tools(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    response = _handle_request({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, manager)
    assert response is not None
    tools = response["result"]["tools"]
    names = {tool["name"] for tool in tools}
    assert "agent.spawn" in names
    assert "agent.ps" in names
    assert "agent.stop" in names


def test_agent_ps_returns_empty_list(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {"name": "agent.ps", "arguments": {}},
        },
        manager,
    )
    assert response is not None
    content = response["result"]["structuredContent"]
    assert content["agents"] == []


def test_agent_stop_missing_agent(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "agent.stop", "arguments": {"name": "nonexistent"}},
        },
        manager,
    )
    assert response is not None
    content = response["result"]["structuredContent"]
    assert content["stopped"] is False


def test_hivemind_task_and_agent_logs_tools_use_context_path(tmp_path: Path) -> None:
    manager = _make_remapped_manager(
        tmp_path,
        history="ledger",
        hivemind="bus",
        items="queue",
    )
    context_path = manager.config.general.context_root
    history_file = context_path / "ledger" / "events_20260317.jsonl"
    history_file.write_text(
        json.dumps(
            {
                "timestamp": "2026-03-17T12:00:00+00:00",
                "source": "agent.worker-1",
                "op": "progress",
                "metadata": {"detail": "queued task"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    send_response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "hivemind.send",
                "arguments": {
                    "context_path": str(context_path),
                    "from": "agent-a",
                    "type": "status",
                    "payload": {"state": "ok"},
                },
            },
        },
        manager,
    )
    assert send_response is not None
    send_payload = send_response["result"]["structuredContent"]
    assert send_payload["from"] == "agent-a"
    assert list((context_path / "bus").glob("agent-a/*.json"))

    read_response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {
                "name": "hivemind.read",
                "arguments": {"context_path": str(context_path)},
            },
        },
        manager,
    )
    assert read_response is not None
    assert len(read_response["result"]["structuredContent"]["messages"]) == 1

    create_response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/call",
            "params": {
                "name": "task.create",
                "arguments": {
                    "context_path": str(context_path),
                    "title": "Review runtime wiring",
                    "created_by": "agent-a",
                },
            },
        },
        manager,
    )
    assert create_response is not None
    task_payload = create_response["result"]["structuredContent"]
    assert task_payload["title"] == "Review runtime wiring"
    assert list((context_path / "queue").glob("task-*.json"))

    list_response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 7,
            "method": "tools/call",
            "params": {
                "name": "task.list",
                "arguments": {"context_path": str(context_path)},
            },
        },
        manager,
    )
    assert list_response is not None
    assert len(list_response["result"]["structuredContent"]["tasks"]) == 1

    logs_response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 8,
            "method": "tools/call",
            "params": {
                "name": "agent.logs",
                "arguments": {
                    "context_path": str(context_path),
                    "name": "worker-1",
                },
            },
        },
        manager,
    )
    assert logs_response is not None
    events = logs_response["result"]["structuredContent"]["events"]
    assert len(events) == 1
    assert events[0]["metadata"]["detail"] == "queued task"


def test_review_tools_use_manager_config(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    supervisor = AgentSupervisor(config=manager.config)
    mock_proc = type("MockProc", (), {"pid": 42424})()

    with patch("afs.agents.supervisor.subprocess.Popen", return_value=mock_proc):
        supervisor.spawn("review-agent", "some.module")
    with (
        patch("afs.agents.supervisor.os.kill"),
        patch.object(supervisor, "_pid_alive", side_effect=[True, False]),
    ):
        assert supervisor.set_awaiting_review("review-agent") is True

    response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 9,
            "method": "tools/call",
            "params": {"name": "review.list", "arguments": {}},
        },
        manager,
    )
    assert response is not None
    agents = response["result"]["structuredContent"]["agents"]
    assert [agent["name"] for agent in agents] == ["review-agent"]
