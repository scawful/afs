from __future__ import annotations

import json
import os
import re
import time
from io import BytesIO
from pathlib import Path
from unittest.mock import patch

import pytest

from afs.agents.supervisor import AgentSupervisor
from afs.context_index import ContextSQLiteIndex
from afs.context_layout import scaffold_v2
from afs.extensions import load_extension_manifest
from afs.history import append_history_event
from afs.manager import AFSManager
from afs.mcp.registry import MCPToolDefinition
from afs.mcp_server import (
    DEFAULT_MCP_TOOL_CATALOG,
    MAX_SKILL_MATCH_PROMPT_CHARS,
    MCP_TOOL_CATALOG_ENV,
    MCP_TOOL_NAME_STYLE_ENV,
    PROTOCOL_VERSION,
    _handle_request,
    _normalize_extension_tools,
    _read_message,
    build_mcp_registry,
)
from afs.models import MountType
from afs.project_registry import ProjectRegistry
from afs.schema import (
    AFSConfig,
    ContextIndexConfig,
    DirectoryConfig,
    ExtensionsConfig,
    GeneralConfig,
    ProfileConfig,
    ProfilesConfig,
    SensitivityConfig,
    WorkspaceDirectory,
    default_directory_configs,
)
from afs.skills import (
    MAX_SKILL_BODIES_CHARS,
    MAX_SKILL_BODY_CHARS,
    MAX_SKILL_BODY_MATCHES,
    MAX_SKILL_METADATA_ITEM_CHARS,
    MAX_SKILL_NAME_CHARS,
    SkillMetadata,
)
from afs.work_assistant import WorkAssistantStore


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


def _make_v2_manager(tmp_path: Path) -> tuple[AFSManager, Path, Path, Path, str, str]:
    context_root = tmp_path / "central" / ".context"
    alpha = tmp_path / "projects" / "alpha"
    beta = tmp_path / "projects" / "beta"
    alpha.mkdir(parents=True)
    beta.mkdir(parents=True)
    manager = AFSManager(
        config=AFSConfig(
            general=GeneralConfig(
                context_root=context_root,
                mcp_allowed_roots=[tmp_path],
            )
        )
    )
    manager.ensure(path=alpha, context_root=context_root, layout_version=2)
    registry = ProjectRegistry(context_root)
    alpha_record = registry.resolve(alpha)
    assert alpha_record is not None
    beta_record = registry.register(beta)
    return (
        manager,
        context_root,
        alpha,
        beta,
        alpha_record.project_id,
        beta_record.project_id,
    )


def _call_tool(
    manager: AFSManager,
    name: str,
    arguments: dict[str, object],
    *,
    request_id: int = 900,
) -> dict[str, object]:
    response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": request_id,
            "method": "tools/call",
            "params": {
                "name": name,
                "arguments": arguments,
            },
        },
        manager,
    )
    assert response is not None
    return response


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


def test_legacy_retrieval_tools_fail_before_reading_v2_indexes(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manager, context_root, _alpha, _beta, _alpha_id, _beta_id = _make_v2_manager(
        tmp_path
    )

    def forbidden(*_args, **_kwargs):
        raise AssertionError("legacy retrieval touched an unscoped v2 index")

    monkeypatch.setattr("afs.mcp_server._query_context_index", forbidden)
    monkeypatch.setattr("afs.mcp_server._resolve_embedding_index_dir", forbidden)
    monkeypatch.setattr("afs.codebase_index.build_codebase_index", forbidden)
    monkeypatch.setattr("afs.codebase_index.search_codebase_index", forbidden)

    calls = (
        ("afs.search", {"query": "private marker"}),
        ("afs.codebase.symbols", {"query": "PrivateSymbol"}),
        ("afs.codebase.index", {}),
    )
    for request_id, (name, arguments) in enumerate(calls, start=910):
        response = _call_tool(
            manager,
            name,
            {"context_path": str(context_root), **arguments},
            request_id=request_id,
        )
        message = str(response["error"]["message"])
        assert f"{name} is unavailable for context v2" in message
        assert "context.search" in message


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

PLAIN_SCOPED_TOOLS = {
    "context.search",
    "messages.send",
    "messages.read",
    "note.create",
    "note.read",
    "note.list",
    "handoff.create",
    "handoff.read",
    "handoff.list",
}


def test_tools_list_defaults_to_slim_catalog(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("AFS_ALLOWED_TOOLS", raising=False)
    monkeypatch.delenv("AFS_TOOL_PROFILE", raising=False)
    monkeypatch.delenv(MCP_TOOL_CATALOG_ENV, raising=False)
    monkeypatch.delenv(MCP_TOOL_NAME_STYLE_ENV, raising=False)
    manager = _make_manager(tmp_path)
    response = _handle_request({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, manager)
    assert response is not None
    tools = response["result"]["tools"]
    names = {tool["name"] for tool in tools}
    assert names == DEFAULT_MCP_TOOL_CATALOG | {
        "skill.match",
        "skill.read",
        *PLAIN_SCOPED_TOOLS,
    }
    assert "context.repair" not in names
    assert "agent.spawn" not in names
    assert not COMPATIBILITY_FILE_TOOL_ALIASES.intersection(names)


def test_claude_tool_name_style_lists_safe_aliases_and_accepts_calls(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("AFS_ALLOWED_TOOLS", raising=False)
    monkeypatch.delenv("AFS_TOOL_PROFILE", raising=False)
    monkeypatch.delenv(MCP_TOOL_CATALOG_ENV, raising=False)
    monkeypatch.setenv(MCP_TOOL_NAME_STYLE_ENV, "claude")
    manager = _make_manager(tmp_path)
    registry = build_mcp_registry(manager)

    response = _handle_request(
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        manager,
        registry=registry,
    )
    assert response is not None
    tools = response["result"]["tools"]
    names = {tool["name"] for tool in tools}

    assert names == {
        "context_status",
        "context_query",
        "context_read",
        "context_list",
        "context_write",
        "context_search",
        "skill_match",
        "skill_read",
        "messages_send",
        "messages_read",
        "note_create",
        "note_read",
        "note_list",
        "handoff_create",
        "handoff_read",
        "handoff_list",
    }
    assert all(re.fullmatch(r"^[a-zA-Z0-9_-]{1,64}$", name) for name in names)

    call_response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "skill_match",
                "arguments": {"prompt": "write an implementation plan"},
            },
        },
        manager,
        registry=registry,
    )
    assert call_response is not None
    assert "result" in call_response
    content = call_response["result"]["content"][0]
    assert "implementation-planning" in content["text"]


def test_hidden_extension_cannot_steal_core_safe_alias(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("AFS_ALLOWED_TOOLS", raising=False)
    monkeypatch.delenv("AFS_TOOL_PROFILE", raising=False)
    monkeypatch.delenv(MCP_TOOL_CATALOG_ENV, raising=False)
    monkeypatch.setenv(MCP_TOOL_NAME_STYLE_ENV, "claude")
    manager = _make_manager(tmp_path)
    registry = build_mcp_registry(manager)
    registry.add_tool(
        MCPToolDefinition(
            name="skill_match",
            description="Collision probe.",
            input_schema={"type": "object"},
            handler=lambda _arguments, _manager: {"extension_called": True},
            source="extension:collision",
            catalog="full",
        )
    )

    listed = _handle_request(
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        manager,
        registry=registry,
    )
    names = {tool["name"] for tool in listed["result"]["tools"]}
    assert "skill_match" in names
    assert "skill_match_2" not in names

    called = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "skill_match",
                "arguments": {"prompt": "write an implementation plan"},
            },
        },
        manager,
        registry=registry,
    )
    assert "implementation-planning" in called["result"]["content"][0]["text"]
    assert "extension_called" not in called["result"]["content"][0]["text"]

    monkeypatch.setenv(MCP_TOOL_CATALOG_ENV, "full")
    full = _handle_request(
        {"jsonrpc": "2.0", "id": 3, "method": "tools/list"},
        manager,
        registry=registry,
    )
    full_names = {tool["name"] for tool in full["result"]["tools"]}
    assert {"skill_match", "skill_match_2"}.issubset(full_names)


def test_tools_list_can_expose_full_catalog(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("AFS_ALLOWED_TOOLS", raising=False)
    monkeypatch.delenv("AFS_TOOL_PROFILE", raising=False)
    monkeypatch.setenv(MCP_TOOL_CATALOG_ENV, "full")
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
        "context.repair",
        "session.pack",
        "work.communication.add",
        "work.approvals.request",
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


def test_v2_context_file_tools_enforce_registered_project_scope(tmp_path: Path) -> None:
    manager, context_root, alpha, beta, alpha_id, beta_id = _make_v2_manager(tmp_path)
    alpha_root = context_root / "scratchpad" / "projects" / alpha_id
    beta_root = context_root / "scratchpad" / "projects" / beta_id
    beta_root.mkdir(parents=True)
    beta_note = beta_root / "private.md"
    beta_note.write_text("beta only", encoding="utf-8")

    write_response = _call_tool(
        manager,
        "context.write",
        {
            "context_path": str(context_root),
            "project_path": str(alpha),
            "path": "scratchpad/notes/alpha.md",
            "content": "alpha only",
            "mkdirs": True,
        },
    )
    written = Path(write_response["result"]["structuredContent"]["path"])
    assert written == alpha_root / "notes" / "alpha.md"
    assert written.read_text(encoding="utf-8") == "alpha only"

    listed = _call_tool(
        manager,
        "context.list",
        {
            "context_path": str(context_root),
            "project_path": str(alpha),
            "path": "scratchpad",
            "max_depth": 3,
        },
    )
    listed_paths = {entry["path"] for entry in listed["result"]["structuredContent"]["entries"]}
    assert str(written) in listed_paths
    assert str(beta_note) not in listed_paths

    denied = _call_tool(
        manager,
        "context.read",
        {
            "context_path": str(context_root),
            "project_path": str(alpha),
            "path": str(beta_note),
        },
    )
    assert "error" in denied
    assert "outside the authorized" in denied["error"]["message"]

    common_only = _call_tool(
        manager,
        "context.read",
        {"context_path": str(context_root), "path": str(written)},
    )
    assert "error" in common_only
    assert "outside the authorized common scope" in common_only["error"]["message"]


def test_v2_context_directory_move_sync_preserves_unrelated_index_rows(
    monkeypatch, tmp_path: Path
) -> None:
    manager, context_root, alpha, _beta, alpha_id, beta_id = _make_v2_manager(
        tmp_path
    )
    alpha_root = context_root / "scratchpad" / "projects" / alpha_id
    beta_root = context_root / "scratchpad" / "projects" / beta_id
    (alpha_root / "before").mkdir(parents=True)
    (alpha_root / "before" / "note.md").write_text("alpha move", encoding="utf-8")
    beta_root.mkdir(parents=True)
    (beta_root / "private.md").write_text("beta private", encoding="utf-8")
    index = ContextSQLiteIndex(manager, context_root)
    index.rebuild(mount_types=[MountType.SCRATCHPAD], include_content=True)

    real_iterdir = Path.iterdir

    def guarded_iterdir(path: Path):
        if path == beta_root:
            raise AssertionError("beta scope was traversed")
        return real_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", guarded_iterdir)
    response = _call_tool(
        manager,
        "context.move",
        {
            "context_path": str(context_root),
            "project_path": str(alpha),
            "source": "scratchpad/before",
            "destination": "knowledge/archive",
            "mkdirs": True,
        },
    )

    assert "result" in response
    assert index.count_entries(
        mount_types=[MountType.SCRATCHPAD],
        relative_prefixes=[f"projects/{beta_id}/"],
    ) >= 1


def test_v2_context_list_empty_scope_is_non_mutating_success(tmp_path: Path) -> None:
    manager, context_root, alpha, _beta, alpha_id, _beta_id = _make_v2_manager(tmp_path)
    expected_root = context_root / "knowledge" / "projects" / alpha_id
    assert not expected_root.exists()

    response = _call_tool(
        manager,
        "context.list",
        {
            "context_path": str(context_root),
            "project_path": str(alpha),
            "path": "knowledge",
        },
    )

    assert response["result"]["structuredContent"] == {
        "path": str(expected_root),
        "entries": [],
    }
    assert not expected_root.exists()


def test_v2_context_list_skips_cross_scope_and_outside_links(tmp_path: Path) -> None:
    manager, context_root, alpha, _beta, alpha_id, beta_id = _make_v2_manager(
        tmp_path
    )
    alpha_root = context_root / "scratchpad" / "projects" / alpha_id
    beta_root = context_root / "scratchpad" / "projects" / beta_id
    alpha_root.mkdir(parents=True)
    beta_root.mkdir(parents=True)
    alpha_note = alpha_root / "alpha.md"
    alpha_note.write_text("alpha", encoding="utf-8")
    (beta_root / "beta-private.md").write_text("beta", encoding="utf-8")
    outside = tmp_path / "outside-poison.md"
    outside.write_text("outside poison metadata", encoding="utf-8")
    try:
        (alpha_root / "beta-link").symlink_to(beta_root, target_is_directory=True)
        (alpha_root / "outside-link.md").symlink_to(outside)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    response = _call_tool(
        manager,
        "context.list",
        {
            "context_path": str(context_root),
            "project_path": str(alpha),
            "path": "scratchpad",
            "max_depth": 3,
        },
    )

    assert response["result"]["structuredContent"]["entries"] == [
        {"path": str(alpha_note), "is_dir": False}
    ]


def test_v2_context_list_accepts_registered_project_path_without_context_path(
    tmp_path: Path,
) -> None:
    manager, context_root, alpha, _beta, alpha_id, _beta_id = _make_v2_manager(
        tmp_path
    )
    alpha_root = context_root / "scratchpad" / "projects" / alpha_id
    alpha_root.mkdir(parents=True)
    alpha_note = alpha_root / "alpha.md"
    alpha_note.write_text("alpha", encoding="utf-8")

    response = _call_tool(
        manager,
        "context.list",
        {
            "project_path": str(alpha),
            "path": "scratchpad",
            "max_depth": 1,
        },
    )

    assert response["result"]["structuredContent"]["entries"] == [
        {"path": str(alpha_note), "is_dir": False}
    ]


@pytest.mark.parametrize("tool_prefix", ["context", "fs"])
@pytest.mark.parametrize("operation", ["delete", "move"])
@pytest.mark.parametrize("target_kind", ["file", "directory"])
def test_v2_absolute_mcp_file_tools_reject_linklike_source_and_preserve_target(
    tool_prefix: str,
    operation: str,
    target_kind: str,
    tmp_path: Path,
) -> None:
    manager, context_root, alpha, _beta, alpha_id, _beta_id = _make_v2_manager(
        tmp_path
    )
    alpha_root = context_root / "scratchpad" / "projects" / alpha_id
    alpha_root.mkdir(parents=True)
    target = alpha_root / f"real-{target_kind}"
    if target_kind == "directory":
        target.mkdir()
        (target / "keep.md").write_text("keep", encoding="utf-8")
    else:
        target.write_text("keep", encoding="utf-8")
    link = alpha_root / f"{target_kind}-link"
    destination = alpha_root / f"moved-{target_kind}"
    try:
        link.symlink_to(target, target_is_directory=target_kind == "directory")
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    arguments: dict[str, object] = {
        "context_path": str(context_root),
        "project_path": str(alpha),
    }
    if operation == "delete":
        arguments.update({"path": str(link), "recursive": True})
    else:
        arguments.update(
            {"source": str(link), "destination": str(destination)}
        )
    response = _call_tool(
        manager,
        f"{tool_prefix}.{operation}",
        arguments,
    )

    assert "error" in response
    assert "symbolic link or reparse point" in response["error"]["message"]
    assert link.is_symlink()
    assert target.exists()
    assert not destination.exists()
    if target_kind == "directory":
        assert (target / "keep.md").read_text(encoding="utf-8") == "keep"


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


def test_work_mcp_tools_capture_style_and_request_approval(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    context_path = manager.config.general.context_root

    add_response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 240,
            "method": "tools/call",
            "params": {
                "name": "work.communication.add",
                "arguments": {
                    "context_path": str(context_path),
                    "text": "Findings first, exact file evidence, short follow-up.",
                    "source_system": "github",
                    "source_id": "comment-1",
                    "channel": "pr_review",
                    "purpose": "responding_to_comments",
                    "style_notes": ["findings-first", "direct"],
                },
            },
        },
        manager,
    )
    assert add_response is not None
    assert add_response["result"]["structuredContent"]["sample_id"]

    guide_response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 241,
            "method": "tools/call",
            "params": {
                "name": "work.communication.guide",
                "arguments": {
                    "context_path": str(context_path),
                    "purpose": "responding_to_comments",
                },
            },
        },
        manager,
    )
    assert guide_response is not None
    guide = guide_response["result"]["structuredContent"]
    assert guide["sample_count"] == 1
    assert guide["style_notes"] == ["findings-first", "direct"]
    assert any("explicit approval" in line for line in guide["guidance"])

    preflight_response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 245,
            "method": "tools/call",
            "params": {
                "name": "work.communication.preflight",
                "arguments": {
                    "context_path": str(context_path),
                    "purpose": "responding_to_comments",
                    "approval_limit": 5,
                },
            },
        },
        manager,
    )
    assert preflight_response is not None
    preflight = preflight_response["result"]["structuredContent"]
    assert preflight["style"]["sample_count"] == 1
    assert preflight["approval_guardrail"]["requires_explicit_approval"] is True
    assert preflight["approval_guardrail"]["ready_to_post"] is False
    assert any(
        item["status"] == "required" and "external post" in item["step"]
        for item in preflight["checklist"]
    )

    request_response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 242,
            "method": "tools/call",
            "params": {
                "name": "work.approvals.request",
                "arguments": {
                    "context_path": str(context_path),
                    "target_system": "github",
                    "target_id": "PR-1",
                    "action": "post_pr_comment",
                    "summary": "Post drafted PR response",
                    "preview": {"body": "Thanks, fixed in src/afs/work_assistant.py."},
                },
            },
        },
        manager,
    )
    assert request_response is not None
    approval_id = request_response["result"]["structuredContent"]["approval_id"]
    approval = WorkAssistantStore(context_path).get_approval(approval_id)
    assert approval is not None
    assert approval["status"] == "pending"

    list_response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 243,
            "method": "tools/call",
            "params": {
                "name": "work.approvals.list",
                "arguments": {"context_path": str(context_path)},
            },
        },
        manager,
    )
    assert list_response is not None
    approvals = list_response["result"]["structuredContent"]["approvals"]
    assert approvals[0]["approval_id"] == approval_id
    assert approvals[0]["status"] == "pending"

    show_response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 244,
            "method": "tools/call",
            "params": {
                "name": "work.approvals.show",
                "arguments": {"context_path": str(context_path), "approval_id": approval_id},
            },
        },
        manager,
    )
    assert show_response is not None
    assert show_response["result"]["structuredContent"]["approval"]["preview"]["body"].startswith(
        "Thanks"
    )


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


def test_context_init_rejects_project_outside_cwd_without_allowed_context_root(
    tmp_path: Path,
) -> None:
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


def test_context_mount_tool_fails_closed_for_v2_and_removes_legacy_alias(
    tmp_path: Path,
) -> None:
    manager, context_root, alpha, _beta, _alpha_id, _beta_id = _make_v2_manager(
        tmp_path
    )
    source_docs = tmp_path / "docs_source"
    source_docs.mkdir()

    response = _call_tool(
        manager,
        "context.mount",
        {
            "context_path": str(context_root),
            "project_path": str(alpha),
            "source": str(source_docs),
            "mount_type": "knowledge",
            "alias": "docs",
        },
    )

    assert "error" in response
    assert "manual filesystem mounts are not supported for layout v2" in response[
        "error"
    ]["message"]
    assert not (context_root / "knowledge" / "docs").exists()

    legacy_alias = context_root / "knowledge" / "legacy-docs"
    legacy_alias.symlink_to(source_docs)
    cleanup = _call_tool(
        manager,
        "context.unmount",
        {
            "context_path": str(context_root),
            "project_path": str(alpha),
            "mount_type": "knowledge",
            "alias": "legacy-docs",
        },
    )
    assert cleanup["result"]["structuredContent"]["removed"] is True
    assert not legacy_alias.exists()


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


def test_v2_context_query_filters_scope_before_ranking(tmp_path: Path) -> None:
    manager, context_root, alpha, _beta, alpha_id, beta_id = _make_v2_manager(tmp_path)
    scratchpad = context_root / "scratchpad"
    paths = {
        "common": scratchpad / "common" / "shared.md",
        "alpha": scratchpad / "projects" / alpha_id / "alpha.md",
        "beta": scratchpad / "projects" / beta_id / "beta.md",
    }
    for label, path in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"scope-token {label}", encoding="utf-8")

    scoped_response = _call_tool(
        manager,
        "context.query",
        {
            "context_path": str(context_root),
            "project_path": str(alpha),
            "mount_types": ["scratchpad"],
            "query": "scope-token",
            "refresh": True,
            "limit": 10,
        },
    )
    scoped = scoped_response["result"]["structuredContent"]
    assert scoped["scope_id"] == f"project:{alpha_id}"
    scoped_paths = {entry["relative_path"] for entry in scoped["entries"]}
    assert scoped_paths == {
        f"projects/{alpha_id}/alpha.md",
        "common/shared.md",
    }
    assert f"projects/{beta_id}/beta.md" not in scoped_paths

    common_response = _call_tool(
        manager,
        "context.query",
        {
            "context_path": str(context_root),
            "mount_types": ["scratchpad"],
            "query": "scope-token",
            "limit": 10,
        },
    )
    common_paths = {
        entry["relative_path"]
        for entry in common_response["result"]["structuredContent"]["entries"]
    }
    assert common_paths == {"common/shared.md"}

    all_response = _call_tool(
        manager,
        "context.query",
        {
            "context_path": str(context_root),
            "mount_types": ["scratchpad"],
            "query": "scope-token",
            "all_projects": True,
            "limit": 10,
        },
    )
    all_paths = {
        entry["relative_path"] for entry in all_response["result"]["structuredContent"]["entries"]
    }
    assert all_paths == {path.relative_to(scratchpad).as_posix() for path in paths.values()}


def test_v2_context_query_auto_index_does_not_traverse_or_replace_beta(
    monkeypatch, tmp_path: Path
) -> None:
    manager, context_root, alpha, _beta, alpha_id, beta_id = _make_v2_manager(
        tmp_path
    )
    knowledge = context_root / "knowledge"
    alpha_root = knowledge / "projects" / alpha_id
    beta_root = knowledge / "projects" / beta_id
    common_root = knowledge / "common"
    for root, marker in (
        (alpha_root, "scoped-auto-index alpha"),
        (beta_root, "scoped-auto-index beta"),
        (common_root, "scoped-auto-index common"),
    ):
        root.mkdir(parents=True)
        (root / "note.md").write_text(marker, encoding="utf-8")
    index = ContextSQLiteIndex(manager, context_root)
    index.rebuild(mount_types=[MountType.KNOWLEDGE], include_content=True)
    index.delete_relative_prefix(MountType.KNOWLEDGE, f"projects/{alpha_id}")
    index.delete_relative_prefix(MountType.KNOWLEDGE, "common")
    statements: list[str] = []
    original_connect = ContextSQLiteIndex._connect

    def traced_connect(current: ContextSQLiteIndex):
        connection = original_connect(current)
        connection.set_trace_callback(statements.append)
        return connection

    monkeypatch.setattr(ContextSQLiteIndex, "_connect", traced_connect)

    real_iterdir = Path.iterdir

    def guarded_iterdir(path: Path):
        if path == beta_root:
            raise AssertionError("beta scope was traversed")
        return real_iterdir(path)

    monkeypatch.setattr(Path, "iterdir", guarded_iterdir)
    response = _call_tool(
        manager,
        "context.query",
        {
            "context_path": str(context_root),
            "project_path": str(alpha),
            "mount_types": ["knowledge"],
            "query": "scoped-auto-index",
            "limit": 10,
        },
    )
    payload = response["result"]["structuredContent"]
    assert {entry["relative_path"] for entry in payload["entries"]} == {
        f"projects/{alpha_id}/note.md",
        "common/note.md",
    }
    assert index.count_entries(
        mount_types=[MountType.KNOWLEDGE],
        relative_prefixes=[f"projects/{beta_id}/"],
    ) >= 1
    diff_response = _call_tool(
        manager,
        "context.diff",
        {
            "context_path": str(context_root),
            "project_path": str(alpha),
            "mount_types": ["knowledge"],
        },
    )
    assert "result" in diff_response
    freshness_response = _call_tool(
        manager,
        "context.freshness",
        {
            "context_path": str(context_root),
            "project_path": str(alpha),
            "mount_type": "knowledge",
        },
    )
    assert "result" in freshness_response
    scoped_metadata_selects = [
        statement
        for statement in statements
        if "SELECT relative_path, size_bytes, modified_at" in statement
        or "SELECT relative_path, modified_at, absolute_path" in statement
    ]
    assert len(scoped_metadata_selects) >= 2
    assert all(alpha_id in statement for statement in scoped_metadata_selects)
    assert all(beta_id not in statement for statement in scoped_metadata_selects)


def test_v2_index_rebuild_tool_is_scoped_unless_all_projects(tmp_path: Path) -> None:
    manager, context_root, alpha, _beta, alpha_id, beta_id = _make_v2_manager(
        tmp_path
    )
    for project_id, marker in ((alpha_id, "alpha"), (beta_id, "beta")):
        path = context_root / "knowledge" / "projects" / project_id / "note.md"
        path.parent.mkdir(parents=True)
        path.write_text(marker, encoding="utf-8")

    scoped_response = _call_tool(
        manager,
        "context.index.rebuild",
        {
            "context_path": str(context_root),
            "project_path": str(alpha),
            "mount_types": ["knowledge"],
        },
    )
    assert scoped_response["result"]["structuredContent"]["scope_id"] == (
        f"project:{alpha_id}"
    )
    index = ContextSQLiteIndex(manager, context_root)
    assert index.count_entries(
        mount_types=[MountType.KNOWLEDGE],
        relative_prefixes=[f"projects/{beta_id}/"],
    ) == 0

    all_response = _call_tool(
        manager,
        "context.index.rebuild",
        {
            "context_path": str(context_root),
            "mount_types": ["knowledge"],
            "all_projects": True,
        },
    )
    assert all_response["result"]["structuredContent"]["scope_id"] == "all-projects"
    assert index.count_entries(
        mount_types=[MountType.KNOWLEDGE],
        relative_prefixes=[f"projects/{beta_id}/"],
    ) >= 1


def test_v2_empty_scope_root_does_not_auto_rebuild_repeatedly(tmp_path: Path) -> None:
    manager, context_root, alpha, _beta, alpha_id, _beta_id = _make_v2_manager(
        tmp_path
    )
    (context_root / "knowledge" / "projects" / alpha_id).mkdir(parents=True)
    (context_root / "knowledge" / "common").mkdir(parents=True)
    arguments = {
        "context_path": str(context_root),
        "project_path": str(alpha),
        "mount_types": ["knowledge"],
        "query": "nothing",
    }

    first = _call_tool(manager, "context.query", arguments)
    second = _call_tool(manager, "context.query", arguments)

    assert "index_rebuild" in first["result"]["structuredContent"]
    assert "index_rebuild" not in second["result"]["structuredContent"]


def test_v2_no_argument_mcp_scope_uses_registered_cwd_only(
    monkeypatch, tmp_path: Path
) -> None:
    manager, context_root, alpha, _beta, alpha_id, _beta_id = _make_v2_manager(
        tmp_path
    )
    monkeypatch.chdir(alpha)

    implicit = _call_tool(manager, "context.status", {})
    implicit_payload = implicit["result"]["structuredContent"]
    assert implicit_payload["scope_id"] == f"project:{alpha_id}"

    explicit = _call_tool(
        manager,
        "context.status",
        {"context_path": str(context_root)},
    )
    explicit_payload = explicit["result"]["structuredContent"]
    assert explicit_payload["scope_id"] == "common"


def test_v2_context_status_slim_tool_counts_only_current_and_common_scope(
    tmp_path: Path,
) -> None:
    manager, context_root, alpha, _beta, alpha_id, beta_id = _make_v2_manager(tmp_path)
    for relative, marker in (
        (Path("common") / "shared.md", "common"),
        (Path("projects") / alpha_id / "alpha.md", "alpha"),
        (Path("projects") / beta_id / "beta.md", "beta-private"),
    ):
        path = context_root / "knowledge" / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(marker, encoding="utf-8")

    response = _call_tool(
        manager,
        "context.status",
        {
            "context_path": str(context_root),
            "project_path": str(alpha),
        },
    )
    payload = response["result"]["structuredContent"]

    assert payload["scope_id"] == f"project:{alpha_id}"
    assert payload["mount_counts"]["knowledge"] == 2
    assert payload["total_files"] == sum(payload["mount_counts"].values())
    assert "beta-private" not in json.dumps(payload)
    assert beta_id not in json.dumps(payload)


def test_context_file_tool_schemas_accept_scope_routing_fields(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    registry = build_mcp_registry(manager)

    for name in PREFERRED_FILE_TOOLS:
        properties = registry.tools[name].input_schema["properties"]
        assert {"context_path", "project_path", "scope_id"}.issubset(properties)
    query_properties = registry.tools["context.query"].input_schema["properties"]
    assert "all_projects" in query_properties


def test_v2_context_search_uses_hybrid_scope_filter(tmp_path: Path) -> None:
    from afs.hybrid_search import HybridSearchEngine, HybridSource

    manager, context_root, alpha, beta, alpha_id, beta_id = _make_v2_manager(tmp_path)
    common = context_root / "knowledge" / "common"
    common.mkdir(parents=True)
    (alpha / "alpha.md").write_text("hybrid-token alpha", encoding="utf-8")
    (beta / "beta.md").write_text("hybrid-token beta", encoding="utf-8")
    (common / "common.md").write_text("hybrid-token common", encoding="utf-8")
    engine = HybridSearchEngine(context_root / ".afs" / "search")
    engine.build(
        [
            HybridSource(alpha, scope_id=f"project:{alpha_id}", project_id=alpha_id),
            HybridSource(beta, scope_id=f"project:{beta_id}", project_id=beta_id),
            HybridSource(common, scope_id="common"),
        ]
    )

    response = _call_tool(
        manager,
        "context.search",
        {
            "context_path": str(context_root),
            "project_path": str(alpha),
            "query": "hybrid-token",
            "limit": 10,
        },
    )
    results = response["result"]["structuredContent"]["results"]
    assert {Path(item["source_path"]).name for item in results} == {
        "alpha.md",
        "common.md",
    }
    assert all(Path(item["source_path"]).name != "beta.md" for item in results)


def test_v2_context_search_filters_never_export_and_requires_literal_all_projects(
    tmp_path: Path,
) -> None:
    from afs.hybrid_search import HybridSearchEngine, HybridSource

    base, context_root, alpha, beta, alpha_id, beta_id = _make_v2_manager(tmp_path)
    manager = AFSManager(
        config=AFSConfig(
            general=base.config.general,
            sensitivity=SensitivityConfig(never_export=["knowledge/private/*"]),
        )
    )
    private = context_root / "knowledge" / "common" / "private"
    private.mkdir(parents=True)
    (private / "secret.md").write_text("export-token secret", encoding="utf-8")
    (alpha / "alpha.md").write_text("export-token alpha", encoding="utf-8")
    (beta / "beta.md").write_text("export-token beta", encoding="utf-8")
    HybridSearchEngine(context_root / ".afs" / "search").build(
        [
            HybridSource(alpha, scope_id=f"project:{alpha_id}", project_id=alpha_id),
            HybridSource(beta, scope_id=f"project:{beta_id}", project_id=beta_id),
            HybridSource(private, scope_id="common"),
        ]
    )

    denied = _call_tool(
        manager,
        "context.search",
        {
            "context_path": str(context_root),
            "project_path": str(alpha),
            "query": "export-token",
            "all_projects": "false",
        },
    )["result"]["structuredContent"]
    denied_names = {Path(item["source_path"]).name for item in denied["results"]}
    assert denied["scope_id"] == f"project:{alpha_id}"
    assert denied_names == {"alpha.md"}

    allowed = _call_tool(
        manager,
        "context.search",
        {
            "context_path": str(context_root),
            "project_path": str(alpha),
            "query": "export-token",
            "all_projects": True,
        },
    )["result"]["structuredContent"]
    allowed_names = {Path(item["source_path"]).name for item in allowed["results"]}
    assert allowed["scope_id"] == "all-projects"
    assert allowed_names == {"alpha.md", "beta.md"}


def test_v2_message_tools_default_to_project_plus_common(tmp_path: Path) -> None:
    manager, context_root, alpha, beta, _alpha_id, _beta_id = _make_v2_manager(tmp_path)

    for project, sender, payload, scope_id in (
        (alpha, "alpha-agent", {"text": "alpha"}, None),
        (beta, "beta-agent", {"text": "beta"}, None),
        (alpha, "common-agent", {"text": "common"}, "common"),
    ):
        arguments: dict[str, object] = {
            "context_path": str(context_root),
            "project_path": str(project),
            "from": sender,
            "type": "status",
            "payload": payload,
        }
        if scope_id is not None:
            arguments["scope_id"] = scope_id
        response = _call_tool(manager, "messages.send", arguments)
        assert "result" in response

    scoped_response = _call_tool(
        manager,
        "messages.read",
        {
            "context_path": str(context_root),
            "project_path": str(alpha),
            "limit": 10,
        },
    )
    scoped = scoped_response["result"]["structuredContent"]
    assert {message["payload"]["text"] for message in scoped["messages"]} == {
        "alpha",
        "common",
    }

    raw_false = _call_tool(
        manager,
        "messages.read",
        {
            "context_path": str(context_root),
            "project_path": str(alpha),
            "all_projects": "false",
            "include_legacy": "false",
            "limit": 10,
        },
    )["result"]["structuredContent"]
    assert raw_false["scope_id"] != "all-projects"
    assert {message["payload"]["text"] for message in raw_false["messages"]} == {
        "alpha",
        "common",
    }

    refused_cleanup = _call_tool(
        manager,
        "messages.clean",
        {
            "context_path": str(context_root),
            "all_projects": "false",
            "apply": "false",
        },
    )
    assert "error" in refused_cleanup

    preview_cleanup = _call_tool(
        manager,
        "messages.clean",
        {
            "context_path": str(context_root),
            "all_projects": True,
            "apply": "false",
        },
    )["result"]["structuredContent"]
    assert preview_cleanup["applied"] is False

    common_response = _call_tool(
        manager,
        "messages.read",
        {"context_path": str(context_root), "limit": 10},
    )
    common = common_response["result"]["structuredContent"]
    assert [message["payload"]["text"] for message in common["messages"]] == ["common"]

    all_response = _call_tool(
        manager,
        "messages.read",
        {"context_path": str(context_root), "all_projects": True, "limit": 10},
    )
    all_messages = all_response["result"]["structuredContent"]["messages"]
    assert {message["payload"]["text"] for message in all_messages} == {
        "alpha",
        "beta",
        "common",
    }


def test_v2_legacy_hivemind_reap_requires_literal_all_projects(
    tmp_path: Path,
) -> None:
    manager, context_root, alpha, beta, _alpha_id, _beta_id = _make_v2_manager(
        tmp_path
    )
    for project, sender in ((alpha, "alpha-agent"), (beta, "beta-agent")):
        response = _call_tool(
            manager,
            "messages.send",
            {
                "context_path": str(context_root),
                "project_path": str(project),
                "from": sender,
                "type": "status",
                "payload": {"sender": sender},
            },
        )
        assert "result" in response

    queue_root = manager.resolve_mount_root(context_root, MountType.HIVEMIND)
    message_files = sorted(queue_root.glob("*/*.json"))
    assert len(message_files) == 2
    old_timestamp = time.time() - (2 * 3600)
    for path in message_files:
        os.utime(path, (old_timestamp, old_timestamp))

    base_arguments: dict[str, object] = {
        "context_path": str(context_root),
        "project_path": str(alpha),
        "max_age_hours": 1,
        "dry_run": False,
    }
    for supplied_value in (None, False, "false"):
        arguments = dict(base_arguments)
        if supplied_value is not None:
            arguments["all_projects"] = supplied_value
        refused = _call_tool(manager, "hivemind.reap", arguments)
        assert "error" in refused
        assert "all_projects=true" in refused["error"]["message"]
        assert all(path.exists() for path in message_files)

    applied = _call_tool(
        manager,
        "hivemind.reap",
        {**base_arguments, "all_projects": True},
    )["result"]["structuredContent"]
    assert applied["removed_count"] == 2
    assert not any(path.exists() for path in message_files)

    schema = build_mcp_registry(manager).tools["hivemind.reap"].input_schema
    assert schema["properties"]["all_projects"]["type"] == "boolean"
    assert "all_projects" in schema["required"]


def test_v1_legacy_hivemind_reap_keeps_omitted_gate_compatibility(
    tmp_path: Path,
) -> None:
    manager = _make_manager(tmp_path)
    context_root = manager.config.general.context_root
    sent = _call_tool(
        manager,
        "messages.send",
        {
            "context_path": str(context_root),
            "from": "legacy-agent",
            "type": "status",
            "payload": {"legacy": True},
        },
    )
    assert "result" in sent
    queue_root = manager.resolve_mount_root(context_root, MountType.HIVEMIND)
    message_file = next(queue_root.glob("*/*.json"))
    old_timestamp = time.time() - (2 * 3600)
    os.utime(message_file, (old_timestamp, old_timestamp))

    applied = _call_tool(
        manager,
        "hivemind.reap",
        {
            "context_path": str(context_root),
            "max_age_hours": 1,
            "dry_run": False,
        },
    )["result"]["structuredContent"]

    assert applied["removed_count"] == 1
    assert not message_file.exists()


def test_v2_note_and_handoff_tools_create_readable_scoped_artifacts(tmp_path: Path) -> None:
    manager, context_root, alpha, beta, alpha_id, _beta_id = _make_v2_manager(tmp_path)
    scope_args = {"context_path": str(context_root), "project_path": str(alpha)}

    note_response = _call_tool(
        manager,
        "note.create",
        {
            **scope_args,
            "title": "Review parser edge cases",
            "body": "Keep the empty-input regression pinned.",
            "agent_name": "codex",
        },
    )
    note = note_response["result"]["structuredContent"]
    note_name = Path(note["path"]).name
    assert re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{6}Z--review-parser-edge-cases--[0-9a-f]{10}\.md",
        note_name,
    )
    assert note["metadata"]["scope_id"] == f"project:{alpha_id}"

    note_list = _call_tool(manager, "note.list", scope_args)
    assert [
        item["metadata"]["artifact_id"]
        for item in note_list["result"]["structuredContent"]["notes"]
    ] == [note["metadata"]["artifact_id"]]
    beta_notes = _call_tool(
        manager,
        "note.list",
        {"context_path": str(context_root), "project_path": str(beta)},
    )
    assert beta_notes["result"]["structuredContent"]["notes"] == []

    handoff_response = _call_tool(
        manager,
        "handoff.create",
        {
            **scope_args,
            "title": "Continue scoped MCP review",
            "agent_name": "codex",
            "accomplished": ["Scoped file and message tools"],
            "next_steps": ["Run the full suite"],
        },
    )
    handoff = handoff_response["result"]["structuredContent"]
    assert handoff["scope_id"] == f"project:{alpha_id}"
    assert re.fullmatch(
        r"\d{4}-\d{2}-\d{2}T\d{6}Z--continue-scoped-mcp-review--[0-9a-f]{10}\.md",
        Path(handoff["artifact_path"]).name,
    )


def test_context_tools_block_never_export_paths(tmp_path: Path) -> None:
    context_root = tmp_path / "context"
    manager = AFSManager(
        config=AFSConfig(
            general=GeneralConfig(context_root=context_root),
            sensitivity=SensitivityConfig(never_export=["knowledge/private/*"]),
        )
    )
    manager.ensure(context_root=context_root)
    knowledge_root = manager.resolve_mount_root(context_root, MountType.KNOWLEDGE)
    (knowledge_root / "private").mkdir(parents=True, exist_ok=True)
    (knowledge_root / "public").mkdir(parents=True, exist_ok=True)
    (knowledge_root / "public" / "guide.md").write_text(
        "shared-token public guide",
        encoding="utf-8",
    )
    private_note = knowledge_root / "private" / "secret.md"
    private_note.write_text("shared-token private secret", encoding="utf-8")

    query_response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 501,
            "method": "tools/call",
            "params": {
                "name": "context.query",
                "arguments": {
                    "context_path": str(context_root),
                    "query": "shared-token",
                    "mount_types": ["knowledge"],
                    "include_content": True,
                    "refresh": True,
                    "limit": 1,
                },
            },
        },
        manager,
    )
    entries = query_response["result"]["structuredContent"]["entries"]
    assert [entry["relative_path"] for entry in entries] == ["public/guide.md"]
    assert "private secret" not in json.dumps(query_response)

    read_response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 502,
            "method": "tools/call",
            "params": {
                "name": "context.read",
                "arguments": {"path": str(private_note)},
            },
        },
        manager,
    )
    assert "error" in read_response
    assert "Blocked by sensitivity rule" in read_response["error"]["message"]


def test_context_file_mutations_respect_sensitivity_rules(tmp_path: Path) -> None:
    context_root = tmp_path / "context"
    manager = AFSManager(
        config=AFSConfig(
            general=GeneralConfig(context_root=context_root),
            sensitivity=SensitivityConfig(never_export=["knowledge/private/*"]),
        )
    )
    manager.ensure(context_root=context_root)
    knowledge_root = manager.resolve_mount_root(context_root, MountType.KNOWLEDGE)
    private_root = knowledge_root / "private"
    public_root = knowledge_root / "public"
    private_root.mkdir(parents=True, exist_ok=True)
    public_root.mkdir(parents=True, exist_ok=True)
    private_note = private_root / "secret.md"
    private_note.write_text("secret", encoding="utf-8")
    public_note = public_root / "guide.md"
    public_note.write_text("guide", encoding="utf-8")

    list_response = _call_tool(
        manager,
        "context.list",
        {"path": str(knowledge_root), "max_depth": 2},
        request_id=510,
    )
    listed_paths = [
        entry["path"] for entry in list_response["result"]["structuredContent"]["entries"]
    ]
    assert str(public_note) in listed_paths
    assert str(private_note) not in listed_paths

    write_response = _call_tool(
        manager,
        "context.write",
        {"path": str(private_root / "new.md"), "content": "new secret"},
        request_id=511,
    )
    assert "error" in write_response
    assert "Blocked by sensitivity rule" in write_response["error"]["message"]

    delete_response = _call_tool(
        manager,
        "context.delete",
        {"path": str(private_note)},
        request_id=512,
    )
    assert "error" in delete_response
    assert private_note.exists()

    move_into_private_response = _call_tool(
        manager,
        "context.move",
        {"source": str(public_note), "destination": str(private_root / "moved.md")},
        request_id=513,
    )
    assert "error" in move_into_private_response
    assert public_note.exists()


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
    assert status_structured["discovery_path"]["default_mcp_tools"] == [
        "context.status",
        "context.query",
        "context.read",
        "context.list",
        "context.write",
    ]
    assert "session.pack" in status_structured["discovery_path"]["do_not_default"]

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
    stream = BytesIO(f"Content-Length: {len(body)}\n\n".encode("ascii") + body)

    payload, mode = _read_message(stream)
    assert mode == "content-length"
    assert payload == {"jsonrpc": "2.0", "id": 1, "method": "ping"}


def test_read_message_accepts_cr_header_terminator() -> None:
    body = b'{"jsonrpc":"2.0","id":2,"method":"ping"}'
    stream = BytesIO(f"Content-Length: {len(body)}\r\r".encode("ascii") + body)

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
    response = _handle_request({"jsonrpc": "2.0", "id": 21, "method": "resources/list"}, manager)
    assert response is not None
    resources = response["result"]["resources"]
    uris = [r["uri"] for r in resources]
    assert "afs://contexts" in uris
    assert "afs://schemas/plan" in uris
    assert "afs://schemas/design-brief" in uris
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


def test_resources_read_v2_compat_metadata(tmp_path: Path) -> None:
    manager, context_root, _alpha, _beta, _alpha_id, _beta_id = _make_v2_manager(
        tmp_path
    )
    metadata_path = context_root / ".afs" / "compat" / "metadata.json"
    metadata_path.parent.mkdir(parents=True, exist_ok=True)
    metadata_path.write_text(
        json.dumps(
            {
                "created_at": "2026-07-16T00:00:00+00:00",
                "description": "central v2 metadata",
                "agents": [],
            }
        ),
        encoding="utf-8",
    )

    uri = f"afs://context/{context_root}/metadata"
    response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 223,
            "method": "resources/read",
            "params": {"uri": uri},
        },
        manager,
    )

    assert response is not None and "result" in response
    data = json.loads(response["result"]["contents"][0]["text"])
    assert data["description"] == "central v2 metadata"


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
    response = _handle_request({"jsonrpc": "2.0", "id": 26, "method": "prompts/list"}, manager)
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
    bootstrap = next(prompt for prompt in prompts if prompt["name"] == "afs.session.bootstrap")
    argument_names = {argument["name"] for argument in bootstrap["arguments"]}
    assert {"project_path", "skills_prompt", "skills_top_k"}.issubset(argument_names)
    for prompt_name in ("afs.session.pack", "afs.workflow.structured"):
        prompt = next(prompt for prompt in prompts if prompt["name"] == prompt_name)
        assert "semantic" in {argument["name"] for argument in prompt["arguments"]}
    query_prompt = next(prompt for prompt in prompts if prompt["name"] == "afs.query.search")
    assert {"project_path", "scope_id"}.issubset(
        {argument["name"] for argument in query_prompt["arguments"]}
    )


def test_prompt_bootstrap_forwards_registered_project_path(tmp_path: Path) -> None:
    manager, context_root, alpha, _beta, _alpha_id, _beta_id = _make_v2_manager(tmp_path)
    with (
        patch("afs.mcp_server.build_session_bootstrap", return_value={}) as build,
        patch("afs.mcp_server.render_session_bootstrap", return_value="scoped bootstrap"),
    ):
        response = _handle_request(
            {
                "jsonrpc": "2.0",
                "id": 124,
                "method": "prompts/get",
                "params": {
                    "name": "afs.session.bootstrap",
                    "arguments": {
                        "context_path": str(context_root),
                        "project_path": str(alpha),
                    },
                },
            },
            manager,
        )

    assert response is not None and "result" in response
    assert build.call_args.kwargs["project_path"] == alpha.resolve()


def test_prompts_get_session_bootstrap(tmp_path: Path, monkeypatch) -> None:
    manager = _make_manager(tmp_path)
    afs_root = tmp_path / "afs-root"
    skill = afs_root / "skills" / "quantum" / "SKILL.md"
    skill.parent.mkdir(parents=True)
    skill.write_text(
        "---\n"
        "name: quantum-frobnicate\n"
        "triggers: [quantumfrobnicate]\n"
        "profiles: [general]\n"
        "---\n\n"
        "# Quantum Frobnication\n\n"
        "Validate the flux boundary before frobnication.\n",
        encoding="utf-8",
    )
    for index in range(6):
        extra = afs_root / "skills" / f"z-quantum-extra-{index}" / "SKILL.md"
        extra.parent.mkdir(parents=True)
        extra.write_text(
            "---\n"
            f"name: z-quantum-extra-{index}\n"
            "triggers: [quantumfrobnicate]\n"
            "profiles: [general]\n"
            "---\n\n"
            f"Extra quantum guidance {index}.\n",
            encoding="utf-8",
        )
    monkeypatch.setenv("AFS_ROOT", str(afs_root))
    context_root = manager.config.general.context_root
    (context_root / "scratchpad" / "state.md").write_text(
        "bootstrap state",
        encoding="utf-8",
    )
    project_root = context_root.parent
    (project_root / "README.md").write_text("# Demo\n", encoding="utf-8")
    (project_root / "src").mkdir(exist_ok=True)
    (project_root / "src" / "demo.py").write_text(
        "def demo() -> int:\n    return 1\n", encoding="utf-8"
    )
    response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 125,
            "method": "prompts/get",
            "params": {
                "name": "afs.session.bootstrap",
                "arguments": {
                    "context_path": str(context_root),
                    "skills_prompt": "quantumfrobnicate",
                    "skills_top_k": "7",
                },
            },
        },
        manager,
    )
    assert response is not None
    messages = response["result"]["messages"]
    text = messages[0]["content"]["text"]
    assert "AFS Session Bootstrap" in text
    assert "bootstrap state" in text
    assert "## Codebase" in text
    assert "## Relevant Skills" in text
    assert "Validate the flux boundary before frobnication." in text
    assert "### z-quantum-extra-5" in text


def test_prompts_get_context_overview_without_existing_context(tmp_path: Path) -> None:
    project_path = tmp_path / "scawfulbot"
    project_path.mkdir()
    (project_path / "config").mkdir()
    (project_path / "config" / "registry.json").write_text('{"models": []}\n', encoding="utf-8")
    (project_path / "config" / "system_prompt.md").write_text("# Prompt\n", encoding="utf-8")
    (project_path / "scripts").mkdir()
    (project_path / "scripts" / "train.py").write_text(
        "def train() -> None:\n    pass\n", encoding="utf-8"
    )
    (project_path / "training").mkdir()
    (project_path / "training" / "dataset.py").write_text(
        "def load_dataset() -> list[str]:\n    return []\n", encoding="utf-8"
    )

    manager = AFSManager(
        config=AFSConfig(
            general=GeneralConfig(
                context_root=tmp_path / "context",
                workspace_directories=[WorkspaceDirectory(path=tmp_path, description="tmp")],
            )
        )
    )

    response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 126,
            "method": "prompts/get",
            "params": {
                "name": "afs.context.overview",
                "arguments": {"path": str(project_path)},
            },
        },
        manager,
    )

    assert response is not None
    text = response["result"]["messages"][0]["content"]["text"]
    assert "Context available: no" in text
    assert "## Codebase" in text
    assert "workflow_roots: config, training" in text or "workflow_roots: training, config" in text
    assert "script_roots: scripts" in text


def test_prompts_get_context_overview_prefers_requested_project_over_ancestor_context(
    tmp_path: Path,
) -> None:
    lab_path = tmp_path / "lab"
    lab_path.mkdir()
    project_path = lab_path / "scawfulbot"
    project_path.mkdir()
    (project_path / "config").mkdir()
    (project_path / "config" / "registry.json").write_text('{"models": []}\n', encoding="utf-8")
    (project_path / "scripts").mkdir()
    (project_path / "scripts" / "train.py").write_text(
        "def train() -> None:\n    pass\n", encoding="utf-8"
    )
    (project_path / "training").mkdir()
    (project_path / "training" / "dataset.py").write_text(
        "def load_dataset() -> list[str]:\n    return []\n", encoding="utf-8"
    )

    manager = AFSManager(
        config=AFSConfig(
            general=GeneralConfig(
                context_root=tmp_path / "context",
                workspace_directories=[WorkspaceDirectory(path=tmp_path, description="tmp")],
            )
        )
    )
    manager.ensure(path=lab_path)

    response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 127,
            "method": "prompts/get",
            "params": {
                "name": "afs.context.overview",
                "arguments": {"path": str(project_path)},
            },
        },
        manager,
    )

    assert response is not None
    text = response["result"]["messages"][0]["content"]["text"]
    assert "Context available: yes" in text
    assert f"Project path: {project_path}" in text
    assert "Nearest context project: lab" in text
    assert "workflow_roots: config, training" in text or "workflow_roots: training, config" in text


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


def test_v2_session_pack_tool_and_prompt_propagate_project_scope(tmp_path: Path) -> None:
    manager, context_root, alpha, _beta, alpha_id, beta_id = _make_v2_manager(tmp_path)
    for project_id, marker in (
        (alpha_id, "alpha-pack-marker"),
        (beta_id, "beta-pack-marker"),
    ):
        path = context_root / "knowledge" / "projects" / project_id / "scope.md"
        path.parent.mkdir(parents=True)
        path.write_text(f"pack-scope-query {marker}", encoding="utf-8")
    ContextSQLiteIndex(manager, context_root).rebuild(
        mount_types=[MountType.KNOWLEDGE],
        include_content=True,
    )

    response = _call_tool(
        manager,
        "session.pack",
        {
            "context_path": str(context_root),
            "project_path": str(alpha),
            "query": "pack-scope-query",
            "include_content": True,
            "token_budget": 2000,
        },
    )
    structured = response["result"]["structuredContent"]
    assert structured["scope_id"] == f"project:{alpha_id}"
    assert "alpha-pack-marker" in json.dumps(structured)
    assert "beta-pack-marker" not in json.dumps(structured)

    prompt_response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 991,
            "method": "prompts/get",
            "params": {
                "name": "afs.session.pack",
                "arguments": {
                    "context_path": str(context_root),
                    "project_path": str(alpha),
                    "query": "pack-scope-query",
                    "token_budget": 2000,
                },
            },
        },
        manager,
    )
    assert prompt_response is not None
    text = prompt_response["result"]["messages"][0]["content"]["text"]
    assert f"Scope: project:{alpha_id}" in text
    assert "beta-pack-marker" not in text

    schema = build_mcp_registry(manager).tools["session.pack"].input_schema
    assert "project_path" in schema["properties"]
    assert schema["properties"]["semantic"]["default"] is False


def test_session_pack_prompt_string_false_does_not_enable_semantic_retrieval(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manager = _make_manager(tmp_path)
    context_root = manager.config.general.context_root

    def forbidden(*_args, **_kwargs):
        raise AssertionError("semantic retrieval requires explicit true consent")

    monkeypatch.setattr("afs.context_pack._embedding_section", forbidden)
    response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 992,
            "method": "prompts/get",
            "params": {
                "name": "afs.session.pack",
                "arguments": {
                    "context_path": str(context_root),
                    "query": "local-only",
                    "semantic": "false",
                },
            },
        },
        manager,
    )

    assert response is not None
    assert "error" not in response


def test_session_pack_tool_string_true_does_not_enable_semantic_retrieval(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manager = _make_manager(tmp_path)

    def forbidden(*_args, **_kwargs):
        raise AssertionError("semantic retrieval requires a literal JSON boolean true")

    monkeypatch.setattr("afs.context_pack._embedding_section", forbidden)
    response = _call_tool(
        manager,
        "session.pack",
        {
            "context_path": str(manager.config.general.context_root),
            "query": "local-only",
            "semantic": "true",
        },
        request_id=993,
    )

    assert "error" not in response


def test_structured_prompt_semantic_requires_literal_true(
    tmp_path: Path,
    monkeypatch,
) -> None:
    manager = _make_manager(tmp_path)
    calls: list[str] = []

    def record_embedding(*_args, **_kwargs):
        calls.append("called")
        return None

    monkeypatch.setattr("afs.context_pack._embedding_section", record_embedding)
    base_arguments = {
        "context_path": str(manager.config.general.context_root),
        "schema_name": "plan",
        "task": "Plan local retrieval.",
        "query": "local-only",
    }
    for request_id, semantic in ((994, "false"), (995, "true"), (996, True)):
        response = _handle_request(
            {
                "jsonrpc": "2.0",
                "id": request_id,
                "method": "prompts/get",
                "params": {
                    "name": "afs.workflow.structured",
                    "arguments": {**base_arguments, "semantic": semantic},
                },
            },
            manager,
        )
        assert response is not None
        assert "error" not in response

    assert calls == ["called"]


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
    (tmp_path / ".afs").mkdir()
    (tmp_path / ".afs" / "policy.toml").write_text(
        "[review]\n"
        'focus = ["order findings by severity"]\n\n'
        "[design]\n"
        'constraints = ["preserve compatibility"]\n',
        encoding="utf-8",
    )
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
    assert "## Repo Policy" in text
    assert "- order findings by severity" in text


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
        json.dumps(
            {
                "created_at": "2025-01-01",
                "description": "test project",
                "agents": ["claude"],
                "directories": {},
                "manual_only": [],
            }
        ),
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


def test_v2_query_search_prompt_is_scope_isolated(tmp_path: Path) -> None:
    manager, context_root, alpha, _beta, alpha_id, beta_id = _make_v2_manager(tmp_path)
    alpha_note = context_root / "knowledge" / "projects" / alpha_id / "alpha-result.md"
    beta_note = context_root / "knowledge" / "projects" / beta_id / "beta-secret.md"
    alpha_note.parent.mkdir(parents=True)
    beta_note.parent.mkdir(parents=True)
    alpha_note.write_text("shared-prompt-query alpha", encoding="utf-8")
    beta_note.write_text("shared-prompt-query beta", encoding="utf-8")
    ContextSQLiteIndex(manager, context_root).rebuild(
        mount_types=[MountType.KNOWLEDGE],
        include_content=True,
    )

    response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 995,
            "method": "prompts/get",
            "params": {
                "name": "afs.query.search",
                "arguments": {
                    "context_path": str(context_root),
                    "project_path": str(alpha),
                    "query": "shared-prompt-query",
                    "mount_types": "knowledge",
                },
            },
        },
        manager,
    )

    assert response is not None
    text = response["result"]["messages"][0]["content"]["text"]
    assert "alpha-result.md" in text
    assert "beta-secret.md" not in text
    assert beta_id not in text


def test_v2_context_overview_never_infers_central_parent_as_project(
    tmp_path: Path,
) -> None:
    manager, context_root, alpha, beta, _alpha_id, _beta_id = _make_v2_manager(tmp_path)
    (context_root.parent / "central-parent-secret").mkdir()
    (alpha / "alpha-visible-dir").mkdir()
    (beta / "beta-private-dir").mkdir()

    common_response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 996,
            "method": "prompts/get",
            "params": {
                "name": "afs.context.overview",
                "arguments": {"context_path": str(context_root)},
            },
        },
        manager,
    )
    assert common_response is not None
    common_text = common_response["result"]["messages"][0]["content"]["text"]
    assert "common scope; no project selected" in common_text
    assert "central-parent-secret" not in common_text

    project_response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 997,
            "method": "prompts/get",
            "params": {
                "name": "afs.context.overview",
                "arguments": {
                    "context_path": str(context_root),
                    "project_path": str(alpha),
                },
            },
        },
        manager,
    )
    assert project_response is not None
    project_text = project_response["result"]["messages"][0]["content"]["text"]
    assert "alpha-visible-dir" in project_text
    assert "beta-private-dir" not in project_text


def test_v2_mcp_context_overview_prunes_nested_project_and_visible_context(
    tmp_path: Path,
) -> None:
    alpha = tmp_path / "workspace"
    beta = alpha / "nested-beta"
    context_root = alpha / "central-context"
    beta.mkdir(parents=True)
    scaffold_v2(context_root)
    registry = ProjectRegistry(context_root)
    registry.register(alpha)
    registry.register(beta)
    (alpha / "alpha_safe.py").write_text("ALPHA_SAFE = True\n", encoding="utf-8")
    (beta / "beta_confidential_canary.py").write_text(
        "BETA_PRIVATE = True\n",
        encoding="utf-8",
    )
    manager = AFSManager(
        config=AFSConfig(general=GeneralConfig(context_root=context_root))
    )

    response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 998,
            "method": "prompts/get",
            "params": {
                "name": "afs.context.overview",
                "arguments": {
                    "context_path": str(context_root),
                    "project_path": str(alpha),
                },
            },
        },
        manager,
    )
    assert response is not None
    text = response["result"]["messages"][0]["content"]["text"]
    codebase_text = text.split("## Codebase\n", maxsplit=1)[1]
    assert "alpha_safe.py" in codebase_text
    assert "nested-beta" not in codebase_text
    assert "beta_confidential_canary" not in codebase_text
    assert "central-context" not in codebase_text


def test_prompts_get_scratchpad_review(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    context_root = manager.config.general.context_root
    (context_root / "scratchpad" / "state.md").write_text("current state info", encoding="utf-8")
    (context_root / "scratchpad" / "deferred.md").write_text("deferred task list", encoding="utf-8")
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
    (context_root / "notes" / "state.md").write_text("remapped state", encoding="utf-8")
    (context_root / "notes" / "deferred.md").write_text("remapped deferred", encoding="utf-8")
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
        'name = "ext_workspace"\n'
        "\n"
        "[mcp_tools]\n"
        'module = "ext_mcp"\n'
        'factory = "register_mcp_tools"\n',
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
        'name = "ext_workspace"\n'
        "\n"
        "[mcp_server]\n"
        'module = "ext_surface"\n'
        'factory = "register_mcp_server"\n',
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
        'name = "ext_workspace"\n'
        "\n"
        "[mcp_server]\n"
        'module = "ext_surface"\n'
        'factory = "register_mcp_server"\n',
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
    assert any(
        "Tool 'context.read' already registered by core" in message for message in errors.values()
    )
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
    assert json.loads(resource_response["result"]["contents"][0]["text"]) == {"status": "profile"}

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


def test_full_tools_list_includes_agent_tools(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv(MCP_TOOL_CATALOG_ENV, "full")
    manager = _make_manager(tmp_path)
    response = _handle_request({"jsonrpc": "2.0", "id": 1, "method": "tools/list"}, manager)
    assert response is not None
    tools = response["result"]["tools"]
    names = {tool["name"] for tool in tools}
    assert "agent.spawn" in names
    assert "agent.ps" in names
    assert "agent.stop" in names
    assert "agent.job.status" in names
    assert "agent.job.seed" in names
    assert "agent.job.inbox" in names
    assert "agent.job.review" in names
    assert "agent.job.archive" in names
    assert "agent.job.promote" in names


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


@pytest.mark.parametrize("tool_name", ["agent.spawn", "agent.stop"])
def test_agent_tools_reject_name_path_traversal(
    tmp_path: Path,
    tool_name: str,
) -> None:
    manager = _make_manager(tmp_path)
    arguments = {"name": "../../outside"}
    if tool_name == "agent.spawn":
        arguments["module"] = "afs.agents.context_warm"

    response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {"name": tool_name, "arguments": arguments},
        },
        manager,
    )

    assert response is not None
    assert "error" in response
    assert "one safe filesystem segment" in response["error"]["message"]


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

    manifest_response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 71,
            "method": "tools/call",
            "params": {
                "name": "agent.manifest.show",
                "arguments": {"harness": "codex", "validate": True},
            },
        },
        manager,
    )
    assert manifest_response is not None
    manifest_payload = manifest_response["result"]["structuredContent"]
    assert manifest_payload["harness"]["name"] == "codex"

    run_response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 72,
            "method": "tools/call",
            "params": {
                "name": "agent.run.start",
                "arguments": {
                    "context_path": str(context_path),
                    "task": "Record MCP run",
                    "harness": "codex",
                },
            },
        },
        manager,
    )
    assert run_response is not None
    run_payload = run_response["result"]["structuredContent"]
    assert run_payload["status"] == "running"

    show_run_response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 721,
            "method": "tools/call",
            "params": {
                "name": "agent.run.show",
                "arguments": {
                    "context_path": str(context_path),
                    "run_id": run_payload["id"],
                },
            },
        },
        manager,
    )
    assert show_run_response is not None
    assert show_run_response["result"]["structuredContent"]["task"] == "Record MCP run"

    finish_response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 73,
            "method": "tools/call",
            "params": {
                "name": "agent.run.finish",
                "arguments": {
                    "context_path": str(context_path),
                    "run_id": run_payload["id"],
                    "summary": "done",
                    "verification": [{"command": "smoke", "status": "passed"}],
                },
            },
        },
        manager,
    )
    assert finish_response is not None
    assert finish_response["result"]["structuredContent"]["status"] == "done"

    job_response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 74,
            "method": "tools/call",
            "params": {
                "name": "agent.job.create",
                "arguments": {
                    "context_path": str(context_path),
                    "title": "MCP queued job",
                    "prompt": "Do it.",
                    "allow_destructive": True,
                },
            },
        },
        manager,
    )
    assert job_response is not None
    job_payload = job_response["result"]["structuredContent"]
    assert job_payload["status"] == "queue"
    assert job_payload["allow_destructive"] is True

    job_status_response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 742,
            "method": "tools/call",
            "params": {
                "name": "agent.job.status",
                "arguments": {
                    "context_path": str(context_path),
                    "stale_after_seconds": 60,
                    "recent_runs": 3,
                },
            },
        },
        manager,
    )
    assert job_status_response is not None
    job_status_payload = job_status_response["result"]["structuredContent"]
    assert job_status_payload["counts"]["queue"] == 1
    assert "watchdog" in job_status_payload

    seed_response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 743,
            "method": "tools/call",
            "params": {
                "name": "agent.job.seed",
                "arguments": {
                    "context_path": str(context_path),
                    "profile": "repo-maintenance",
                    "dry_run": True,
                },
            },
        },
        manager,
    )
    assert seed_response is not None
    seed_payload = seed_response["result"]["structuredContent"]
    assert seed_payload["profile"] == "repo-maintenance"
    assert seed_payload["would_create"] == 6

    show_job_response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 741,
            "method": "tools/call",
            "params": {
                "name": "agent.job.show",
                "arguments": {
                    "context_path": str(context_path),
                    "job_id": job_payload["id"],
                },
            },
        },
        manager,
    )
    assert show_job_response is not None
    assert show_job_response["result"]["structuredContent"]["title"] == "MCP queued job"

    claim_response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 75,
            "method": "tools/call",
            "params": {
                "name": "agent.job.claim",
                "arguments": {
                    "context_path": str(context_path),
                    "job_id": job_payload["id"],
                    "agent_name": "worker",
                },
            },
        },
        manager,
    )
    assert claim_response is not None
    assert claim_response["result"]["structuredContent"]["status"] == "running"

    move_job_response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 751,
            "method": "tools/call",
            "params": {
                "name": "agent.job.move",
                "arguments": {
                    "context_path": str(context_path),
                    "job_id": job_payload["id"],
                    "status": "done",
                    "result": "done via MCP",
                },
            },
        },
        manager,
    )
    assert move_job_response is not None
    assert move_job_response["result"]["structuredContent"]["status"] == "done"

    inbox_response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 752,
            "method": "tools/call",
            "params": {
                "name": "agent.job.inbox",
                "arguments": {"context_path": str(context_path)},
            },
        },
        manager,
    )
    assert inbox_response is not None
    inbox_payload = inbox_response["result"]["structuredContent"]
    assert inbox_payload["attention_count"] == 1
    assert "afs agent-jobs inbox" in inbox_payload["command"]

    review_job_response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 753,
            "method": "tools/call",
            "params": {
                "name": "agent.job.review",
                "arguments": {
                    "context_path": str(context_path),
                    "job_id": job_payload["id"],
                },
            },
        },
        manager,
    )
    assert review_job_response is not None
    assert review_job_response["result"]["structuredContent"]["job"]["result"] == "done via MCP"

    promote_job_response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 754,
            "method": "tools/call",
            "params": {
                "name": "agent.job.promote",
                "arguments": {
                    "context_path": str(context_path),
                    "job_id": job_payload["id"],
                    "handoff_name": "mcp-job.md",
                },
            },
        },
        manager,
    )
    assert promote_job_response is not None
    handoff_path = Path(promote_job_response["result"]["structuredContent"]["path"])
    assert handoff_path.exists()

    archive_job_response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 755,
            "method": "tools/call",
            "params": {
                "name": "agent.job.archive",
                "arguments": {
                    "context_path": str(context_path),
                    "job_id": job_payload["id"],
                },
            },
        },
        manager,
    )
    assert archive_job_response is not None
    assert archive_job_response["result"]["structuredContent"]["status"] == "archived"

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


def test_companion_repo_mcp_server_uses_src_layout(tmp_path: Path) -> None:
    workspace_root = tmp_path / "lab"
    repo = workspace_root / "afs_example"
    package = repo / "src" / "afs_example"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("", encoding="utf-8")
    (package / "mcp_surface.py").write_text(
        "def register_mcp_server(_manager):\n"
        "    def echo(arguments):\n"
        "        return {'echo': arguments.get('value', '')}\n"
        "    return {'tools': [{'name': 'google.echo', 'description': 'echo', 'handler': echo}]}\n",
        encoding="utf-8",
    )
    (repo / "extension.toml").write_text(
        'name = "afs_example"\n'
        "\n"
        "[mcp_server]\n"
        'module = "afs_example.mcp_surface"\n'
        'factory = "register_mcp_server"\n',
        encoding="utf-8",
    )

    context_root = tmp_path / "context"
    context_root.mkdir(parents=True)
    (context_root / "scratchpad").mkdir()
    manager = AFSManager(
        config=AFSConfig(
            general=GeneralConfig(context_root=context_root),
            extensions=ExtensionsConfig(
                enabled_extensions=["afs_example"],
                extension_dirs=[],
                extension_repo_roots=[workspace_root],
                auto_discover=False,
            ),
        )
    )

    registry = build_mcp_registry(manager)
    assert "google.echo" in registry.tools

    response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 98,
            "method": "tools/call",
            "params": {"name": "google.echo", "arguments": {"value": "ok"}},
        },
        manager,
        registry=registry,
    )
    assert response is not None
    assert response["result"]["structuredContent"]["echo"] == "ok"


def _write_matching_skill(root: Path, name: str, *, body_size: int = 2_600) -> None:
    skill_dir = root / "skills" / name
    skill_dir.mkdir(parents=True)
    (skill_dir / "SKILL.md").write_text(
        "---\n"
        f"name: {name}\n"
        "triggers: [quantumfrobnicate]\n"
        "enforcement:\n"
        "  - Keep the operation bounded.\n"
        "---\n"
        f"# {name}\n\n" + ("x" * body_size) + "\n",
        encoding="utf-8",
    )


def test_skill_match_and_read_are_bounded(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("AFS_ALLOWED_TOOLS", raising=False)
    monkeypatch.delenv("AFS_TOOL_PROFILE", raising=False)
    afs_root = tmp_path / "afs-root"
    for name in ("alpha", "beta", "delta", "gamma"):
        _write_matching_skill(afs_root, name)
    monkeypatch.setenv("AFS_ROOT", str(afs_root))
    manager = _make_manager(tmp_path)

    metadata_only = _call_tool(
        manager,
        "skill.match",
        {"prompt": "please quantumfrobnicate this", "top_k": 4},
    )
    matches = metadata_only["result"]["structuredContent"]["matches"]
    assert len(matches) == 4
    assert all(match["body"] == "" for match in matches)
    assert all(match["body_omitted"] == "not_requested" for match in matches)

    with_bodies = _call_tool(
        manager,
        "skill.match",
        {
            "prompt": "please quantumfrobnicate this",
            "top_k": 4,
            "include_bodies": True,
        },
    )
    body_matches = with_bodies["result"]["structuredContent"]["matches"]
    assert len([match for match in body_matches if match["body"]]) == MAX_SKILL_BODY_MATCHES
    assert all(len(match["body"]) <= MAX_SKILL_BODY_CHARS for match in body_matches)
    assert sum(len(match["body"]) for match in body_matches) <= MAX_SKILL_BODIES_CHARS
    assert body_matches[3]["body_omitted"] == "match_limit"

    read_response = _call_tool(manager, "skill.read", {"name": "alpha"})
    skill = read_response["result"]["structuredContent"]
    assert skill["body_truncated"] is True
    assert skill["body_chars"] == len(skill["body"]) <= MAX_SKILL_BODY_CHARS
    assert skill["enforcement"] == ["Keep the operation bounded."]


@pytest.mark.parametrize("top_k", [0, 11, True, "2", 1.5])
def test_skill_match_rejects_invalid_top_k(
    tmp_path: Path,
    monkeypatch,
    top_k: object,
) -> None:
    afs_root = tmp_path / "afs-root"
    _write_matching_skill(afs_root, "alpha", body_size=20)
    monkeypatch.setenv("AFS_ROOT", str(afs_root))
    response = _call_tool(
        _make_manager(tmp_path),
        "skill.match",
        {"prompt": "quantumfrobnicate", "top_k": top_k},
    )
    assert "error" in response
    assert "top_k must be an integer from 1 to 10" in response["error"]["message"]


@pytest.mark.parametrize("include_bodies", ["false", 0, 1, [], {}])
def test_skill_match_rejects_non_boolean_include_bodies(
    tmp_path: Path,
    monkeypatch,
    include_bodies: object,
) -> None:
    afs_root = tmp_path / "afs-root"
    _write_matching_skill(afs_root, "alpha", body_size=20)
    monkeypatch.setenv("AFS_ROOT", str(afs_root))
    response = _call_tool(
        _make_manager(tmp_path),
        "skill.match",
        {"prompt": "quantumfrobnicate", "include_bodies": include_bodies},
    )
    assert "error" in response
    assert "include_bodies must be a boolean" in response["error"]["message"]


@pytest.mark.parametrize(
    "prompt",
    [
        "x" * (MAX_SKILL_MATCH_PROMPT_CHARS + 1),
        (" " * MAX_SKILL_MATCH_PROMPT_CHARS) + "focus",
    ],
)
def test_skill_match_rejects_oversized_prompt(tmp_path: Path, prompt: str) -> None:
    response = _call_tool(
        _make_manager(tmp_path),
        "skill.match",
        {"prompt": prompt},
    )
    assert "error" in response
    assert "prompt must be at most" in response["error"]["message"]


def test_skill_read_rejects_whitespace_padded_oversized_name(tmp_path: Path) -> None:
    response = _call_tool(
        _make_manager(tmp_path),
        "skill.read",
        {"name": (" " * MAX_SKILL_NAME_CHARS) + "skill"},
    )
    assert "error" in response
    assert "name must be at most" in response["error"]["message"]


@pytest.mark.parametrize(
    "name",
    [
        "line\nbreak",
        "osc\x1b]8;;https://example.invalid\x07click\x1b]8;;\x07",
        "delete\x7fcontrol",
        "c1\x85control",
    ],
)
def test_skill_read_rejects_control_characters_without_echoing_them(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    name: str,
) -> None:
    manager = _make_manager(tmp_path)
    capsys.readouterr()

    response = _call_tool(manager, "skill.read", {"name": name})

    expected = "name must not contain ASCII or C1 control characters"
    assert response["error"]["message"] == expected
    assert capsys.readouterr().err == f"[afs-mcp] tool error: skill.read: {expected}\n"

    history_files = sorted((manager.config.general.context_root / "history").glob("*.jsonl"))
    event = json.loads(history_files[-1].read_text(encoding="utf-8").splitlines()[-1])
    assert "arguments" not in event["metadata"]


def test_skill_read_sanitizes_control_characters_in_known_skill_preview(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    hostile_name = "known\x1b]8;;https://example.invalid\x07click\x1b]8;;\x07\nnext\x85line"
    metadata = SkillMetadata(
        name=hostile_name,
        path=tmp_path / "known" / "SKILL.md",
    )
    manager = _make_manager(tmp_path)
    capsys.readouterr()

    with patch("afs.mcp_server.discover_skills", return_value=[metadata]):
        response = _call_tool(manager, "skill.read", {"name": "missing"})

    message = response["error"]["message"]
    stderr = capsys.readouterr().err
    assert "\\u001b" in message
    assert "\\u0007" in message
    assert "\\u000a" in message
    assert "\\u0085" in message
    assert not any(ord(char) < 0x20 or 0x7F <= ord(char) <= 0x9F for char in message)
    assert stderr == f"[afs-mcp] tool error: skill.read: {message}\n"
    assert len(message) <= 1_400


@pytest.mark.parametrize(
    ("tool_name", "arguments"),
    [
        (
            "skill.match",
            {"prompt": "focus", "unexpected\x1b": "x" * 4_096},
        ),
        (
            "skill.read",
            {"name": "missing", "unexpected\x1b": "x" * 4_096},
        ),
    ],
)
def test_skill_tools_reject_unknown_arguments_without_logging_values(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    tool_name: str,
    arguments: dict[str, object],
) -> None:
    manager = _make_manager(tmp_path)
    capsys.readouterr()

    response = _call_tool(manager, tool_name, arguments)

    expected = f"{tool_name} received unsupported arguments"
    assert response["error"]["message"] == expected
    assert capsys.readouterr().err == f"[afs-mcp] tool error: {tool_name}: {expected}\n"
    history_files = sorted((manager.config.general.context_root / "history").glob("*.jsonl"))
    event_text = history_files[-1].read_text(encoding="utf-8").splitlines()[-1]
    event = json.loads(event_text)
    assert "arguments" not in event["metadata"]
    assert "x" * 128 not in event_text


def test_skill_read_rechecks_configured_root_containment(
    tmp_path: Path,
    monkeypatch,
) -> None:
    afs_root = tmp_path / "afs-root"
    (afs_root / "skills").mkdir(parents=True)
    monkeypatch.setenv("AFS_ROOT", str(afs_root))
    outside = tmp_path / "outside" / "SKILL.md"
    outside.parent.mkdir()
    outside.write_text("# Outside\n", encoding="utf-8")
    escaped_path = afs_root / "skills" / "escaped" / "SKILL.md"
    escaped_path.parent.mkdir()
    try:
        escaped_path.symlink_to(outside)
    except OSError:
        pytest.skip("symlinks are unavailable on this platform")
    escaped = SkillMetadata(name="escaped", path=escaped_path)

    with patch("afs.mcp_server.discover_skills", return_value=[escaped]):
        response = _call_tool(_make_manager(tmp_path), "skill.read", {"name": "escaped"})

    assert "error" in response
    assert "outside configured roots" in response["error"]["message"]


def test_skill_read_rejects_symlink_swap_after_containment(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    afs_root = tmp_path / "afs-root"
    skill_path = afs_root / "skills" / "safe" / "SKILL.md"
    skill_path.parent.mkdir(parents=True)
    skill_path.write_text(
        "---\nname: safe\ntriggers: [safe]\n---\n\nTrusted body.\n",
        encoding="utf-8",
    )
    outside = tmp_path / "outside-secret.md"
    outside.write_text("SECRET_OUTSIDE_ROOT\n", encoding="utf-8")
    monkeypatch.setenv("AFS_ROOT", str(afs_root))

    import afs.mcp_server as mcp_server_module

    original = mcp_server_module._skill_path_within_roots

    def swap_after_check(path: Path, roots: list[Path]) -> tuple[Path, Path]:
        checked = original(path, roots)
        skill_path.unlink()
        try:
            skill_path.symlink_to(outside)
        except OSError as exc:
            pytest.skip(f"symlinks are unavailable on this platform: {exc}")
        return checked

    monkeypatch.setattr(
        mcp_server_module,
        "_skill_path_within_roots",
        swap_after_check,
    )
    response = _call_tool(_make_manager(tmp_path), "skill.read", {"name": "safe"})

    assert "error" in response
    assert "SECRET_OUTSIDE_ROOT" not in json.dumps(response)


def test_skill_read_unknown_name_preview_is_bounded(tmp_path: Path, monkeypatch) -> None:
    afs_root = tmp_path / "afs-root"
    for index in range(14):
        _write_matching_skill(afs_root, f"skill-{index:02d}", body_size=10)
    monkeypatch.setenv("AFS_ROOT", str(afs_root))

    response = _call_tool(_make_manager(tmp_path), "skill.read", {"name": "missing"})

    message = response["error"]["message"]
    preview = message.split("Known skills: ", 1)[1].split(" (and", 1)[0]
    assert len(preview.split(", ")) == 10
    assert "(and " in message
    assert len(message) < 300


def test_skill_tools_bound_adversarial_metadata(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    huge = "x" * 60_000
    metadata = SkillMetadata(
        name="large",
        path=tmp_path / "large" / "SKILL.md",
        triggers=["focus"],
        enforcement=[huge],
        verification=[huge],
    )

    with (
        patch("afs.mcp_server.discover_skills", return_value=[metadata]),
        patch("afs.skills.discover_skills", return_value=[metadata]),
    ):
        match_response = _call_tool(
            manager,
            "skill.match",
            {"prompt": "focus", "include_bodies": False},
        )
        read_response = _call_tool(manager, "skill.read", {"name": "missing"})

    match = match_response["result"]["structuredContent"]["matches"][0]
    assert len(match["enforcement"][0]) <= MAX_SKILL_METADATA_ITEM_CHARS
    assert len(match["verification"][0]) <= MAX_SKILL_METADATA_ITEM_CHARS
    assert {"enforcement", "verification"}.issubset(match["metadata_truncated"])
    assert len(json.dumps(match_response)) < 10_000
    assert len(read_response["error"]["message"]) < 1_400


def _make_catalog_extension_manager(
    tmp_path: Path,
    *,
    manifest_catalog: str | None,
    tool_catalog: object | None,
) -> AFSManager:
    extension_root = tmp_path / "extensions" / "catalog_extension"
    extension_root.mkdir(parents=True)
    catalog_line = f'catalog = "{manifest_catalog}"\n' if manifest_catalog is not None else ""
    (extension_root / "extension.toml").write_text(
        'name = "catalog_extension"\n\n'
        "[mcp_tools]\n"
        'module = "catalog_extension_mcp"\n' + catalog_line,
        encoding="utf-8",
    )
    tool_catalog_entry = f", 'catalog': {tool_catalog!r}" if tool_catalog is not None else ""
    (extension_root / "catalog_extension_mcp.py").write_text(
        "def register_mcp_tools(_manager):\n"
        "    def echo(arguments):\n"
        "        return {'echo': arguments.get('value', '')}\n"
        "    return [{'name': 'catalog.echo', 'description': 'echo', "
        f"'handler': echo{tool_catalog_entry}}}]\n",
        encoding="utf-8",
    )
    context_root = tmp_path / "context"
    context_root.mkdir()
    (context_root / "scratchpad").mkdir()
    return AFSManager(
        config=AFSConfig(
            general=GeneralConfig(context_root=context_root),
            extensions=ExtensionsConfig(
                enabled_extensions=["catalog_extension"],
                extension_dirs=[tmp_path / "extensions"],
            ),
        )
    )


def test_extension_tool_catalog_defaults_to_full(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("AFS_ALLOWED_TOOLS", raising=False)
    monkeypatch.delenv("AFS_TOOL_PROFILE", raising=False)
    monkeypatch.delenv(MCP_TOOL_CATALOG_ENV, raising=False)
    manager = _make_catalog_extension_manager(
        tmp_path,
        manifest_catalog=None,
        tool_catalog=None,
    )
    registry = build_mcp_registry(manager)

    assert registry.tools["catalog.echo"].catalog == "full"
    response = _handle_request(
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        manager,
        registry=registry,
    )
    assert response is not None
    names = {tool["name"] for tool in response["result"]["tools"]}
    assert "catalog.echo" not in names


def test_extension_manifest_slim_default_and_tool_override(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("AFS_ALLOWED_TOOLS", raising=False)
    monkeypatch.delenv("AFS_TOOL_PROFILE", raising=False)
    monkeypatch.delenv(MCP_TOOL_CATALOG_ENV, raising=False)
    manager = _make_catalog_extension_manager(
        tmp_path,
        manifest_catalog="slim",
        tool_catalog=None,
    )
    registry = build_mcp_registry(manager)
    assert registry.tools["catalog.echo"].catalog == "slim"

    response = _handle_request(
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        manager,
        registry=registry,
    )
    assert response is not None
    names = {tool["name"] for tool in response["result"]["tools"]}
    assert "catalog.echo" in names

    opt_out_manager = _make_catalog_extension_manager(
        tmp_path / "opt-out",
        manifest_catalog="slim",
        tool_catalog="full",
    )
    opt_out_registry = build_mcp_registry(opt_out_manager)
    assert opt_out_registry.tools["catalog.echo"].catalog == "full"


@pytest.mark.parametrize("tool_catalog", ["wide", "", 1, True])
def test_invalid_extension_tool_catalog_fails_surface_closed(
    tmp_path: Path,
    tool_catalog: object,
) -> None:
    manager = _make_catalog_extension_manager(
        tmp_path,
        manifest_catalog=None,
        tool_catalog=tool_catalog,
    )
    registry = build_mcp_registry(manager)

    assert "catalog.echo" not in registry.tools
    errors = [status.error for status in registry.extension_status if status.error]
    assert any("catalog must be 'full' or 'slim'" in error for error in errors)


@pytest.mark.parametrize("catalog", ["wide", "", 1, True])
def test_invalid_extension_manifest_catalog_is_rejected(
    tmp_path: Path,
    catalog: object,
) -> None:
    extension_root = tmp_path / "invalid-catalog"
    extension_root.mkdir()
    rendered = json.dumps(catalog) if not isinstance(catalog, str) else f'"{catalog}"'
    manifest = extension_root / "extension.toml"
    manifest.write_text(
        'name = "invalid-catalog"\n\n'
        "[mcp_tools]\n"
        'module = "invalid_catalog_mcp"\n'
        f"catalog = {rendered}\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="catalog must be 'full' or 'slim'"):
        load_extension_manifest(manifest)


def test_tool_definition_rejects_invalid_catalog() -> None:
    with pytest.raises(ValueError, match="catalog must be one of"):
        MCPToolDefinition(
            name="bad.catalog",
            description="invalid",
            input_schema={"type": "object"},
            handler=lambda _arguments, _manager: {},
            catalog="wide",
        )


def test_extension_tool_normalization_preserves_definition_fields() -> None:
    def pre_hook(arguments, _manager):
        return arguments

    def post_hook(_arguments, result, _manager):
        return result

    definition = MCPToolDefinition(
        name="clone.test",
        description="clone",
        input_schema={"type": "object"},
        handler=lambda _arguments, _manager: {},
        deferred=True,
        concurrent_safe=True,
        pre_hook=pre_hook,
        post_hook=post_hook,
        catalog="slim",
    )

    normalized = _normalize_extension_tools(
        "clone-extension",
        definition,
        source="extension:clone-extension",
    )[0]

    assert normalized.deferred is True
    assert normalized.concurrent_safe is True
    assert normalized.pre_hook is pre_hook
    assert normalized.post_hook is post_hook
    assert normalized.catalog == "slim"
