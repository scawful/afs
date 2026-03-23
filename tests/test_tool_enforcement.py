"""Tests for tool-bundle enforcement via AFS_TOOL_PROFILE and AFS_ALLOWED_TOOLS."""

from __future__ import annotations

from pathlib import Path

import pytest

from afs.agent_scope import allowed_tools, assert_tool_allowed, is_tool_allowed
from afs.manager import AFSManager
from afs.mcp_server import _handle_request, build_mcp_registry
from afs.schema import AFSConfig, GeneralConfig
from afs.session_workflows import (
    TOOL_PROFILE_DEFINITIONS,
    get_tool_profile_surfaces,
    is_tool_in_profile,
)

# ---------------------------------------------------------------------------
# session_workflows helpers
# ---------------------------------------------------------------------------


def test_get_tool_profile_surfaces_returns_known_profile() -> None:
    surfaces = get_tool_profile_surfaces("context_readonly")
    assert isinstance(surfaces, frozenset)
    assert "operator.digest" in surfaces
    assert "context.query" in surfaces
    assert "context.read" in surfaces
    # context_readonly should NOT include repair tools
    assert "context.repair" not in surfaces


def test_get_tool_profile_surfaces_returns_empty_for_unknown() -> None:
    assert get_tool_profile_surfaces("nonexistent_profile") == frozenset()
    assert get_tool_profile_surfaces(None) == frozenset()
    assert get_tool_profile_surfaces("") == frozenset()


def test_is_tool_in_profile_allows_listed_tool() -> None:
    assert is_tool_in_profile("session.bootstrap", "context_readonly") is True
    assert is_tool_in_profile("context.query", "context_readonly") is True


def test_is_tool_in_profile_rejects_unlisted_tool() -> None:
    assert is_tool_in_profile("context.repair", "context_readonly") is False


def test_is_tool_in_profile_allows_everything_for_unknown_profile() -> None:
    assert is_tool_in_profile("anything.at.all", "bogus_profile") is True


def test_all_defined_profiles_have_nonempty_surfaces() -> None:
    for name in TOOL_PROFILE_DEFINITIONS:
        surfaces = get_tool_profile_surfaces(name)
        assert len(surfaces) > 0, f"profile {name!r} has no surfaces"


# ---------------------------------------------------------------------------
# agent_scope: AFS_TOOL_PROFILE env var
# ---------------------------------------------------------------------------


def test_allowed_tools_reads_tool_profile_env(monkeypatch) -> None:
    monkeypatch.delenv("AFS_ALLOWED_TOOLS", raising=False)
    monkeypatch.setenv("AFS_TOOL_PROFILE", "context_readonly")
    result = allowed_tools()
    assert result is not None
    assert "context.query" in result
    assert "context.repair" not in result


def test_allowed_tools_explicit_takes_precedence(monkeypatch) -> None:
    monkeypatch.setenv("AFS_ALLOWED_TOOLS", "session.*,context.status")
    monkeypatch.setenv("AFS_TOOL_PROFILE", "context_readonly")
    result = allowed_tools()
    assert result is not None
    # Should be the explicit list, not the profile
    assert "session.*" in result
    assert "context.status" in result


def test_allowed_tools_returns_none_when_no_env(monkeypatch) -> None:
    monkeypatch.delenv("AFS_ALLOWED_TOOLS", raising=False)
    monkeypatch.delenv("AFS_TOOL_PROFILE", raising=False)
    assert allowed_tools() is None


def test_is_tool_allowed_with_profile(monkeypatch) -> None:
    monkeypatch.delenv("AFS_ALLOWED_TOOLS", raising=False)
    monkeypatch.setenv("AFS_TOOL_PROFILE", "handoff_only")
    assert is_tool_allowed("session.bootstrap") is True
    assert is_tool_allowed("handoff.create") is True
    assert is_tool_allowed("context.query") is False


def test_assert_tool_allowed_raises_with_profile(monkeypatch) -> None:
    monkeypatch.delenv("AFS_ALLOWED_TOOLS", raising=False)
    monkeypatch.setenv("AFS_TOOL_PROFILE", "handoff_only")
    monkeypatch.setenv("AFS_AGENT_NAME", "test-agent")
    with pytest.raises(PermissionError, match="test-agent"):
        assert_tool_allowed("context.query")


def test_is_tool_allowed_unrestricted_without_env(monkeypatch) -> None:
    monkeypatch.delenv("AFS_ALLOWED_TOOLS", raising=False)
    monkeypatch.delenv("AFS_TOOL_PROFILE", raising=False)
    assert is_tool_allowed("literally.anything") is True


# ---------------------------------------------------------------------------
# MCP server: tools/list filtering
# ---------------------------------------------------------------------------


def _make_manager(tmp_path: Path) -> AFSManager:
    context_root = tmp_path / "context"
    manager = AFSManager(config=AFSConfig(general=GeneralConfig(context_root=context_root)))
    project_path = tmp_path / "project"
    project_path.mkdir()
    manager.ensure(path=project_path, context_root=context_root)
    return manager


def test_tools_list_filtered_by_tool_profile(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("AFS_ALLOWED_TOOLS", raising=False)
    monkeypatch.setenv("AFS_TOOL_PROFILE", "context_readonly")
    manager = _make_manager(tmp_path)
    registry = build_mcp_registry(manager)

    response = _handle_request(
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        manager,
        registry=registry,
    )
    tools = response["result"]["tools"]
    tool_names = {t["name"] for t in tools}

    # Should include context_readonly surfaces
    readonly_surfaces = get_tool_profile_surfaces("context_readonly")
    for surface in readonly_surfaces:
        if surface in {t.name for t in registry.tools.values()}:
            assert surface in tool_names, f"{surface} should be in filtered tools/list"

    assert "operator.digest" in tool_names
    # Should NOT include tools outside the profile
    assert "context.repair" not in tool_names
    assert "context.write" not in tool_names


def test_tools_list_unfiltered_without_profile(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("AFS_ALLOWED_TOOLS", raising=False)
    monkeypatch.delenv("AFS_TOOL_PROFILE", raising=False)
    manager = _make_manager(tmp_path)
    registry = build_mcp_registry(manager)

    response = _handle_request(
        {"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
        manager,
        registry=registry,
    )
    tools = response["result"]["tools"]
    tool_names = {t["name"] for t in tools}

    # Without a profile, all registered tools should appear
    assert "context.repair" in tool_names
    assert "session.pack" in tool_names


def test_tool_call_rejected_by_profile(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("AFS_ALLOWED_TOOLS", raising=False)
    monkeypatch.setenv("AFS_TOOL_PROFILE", "context_readonly")
    manager = _make_manager(tmp_path)
    registry = build_mcp_registry(manager)

    response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "context.repair",
                "arguments": {"context_path": str(tmp_path / "context")},
            },
        },
        manager,
        registry=registry,
    )
    # Should get an error response (PermissionError from assert_tool_allowed)
    assert "error" in response


def test_tool_call_allowed_by_profile(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("AFS_ALLOWED_TOOLS", raising=False)
    monkeypatch.setenv("AFS_TOOL_PROFILE", "context_readonly")
    manager = _make_manager(tmp_path)
    registry = build_mcp_registry(manager)

    response = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "context.status",
                "arguments": {"context_path": str(tmp_path / "context")},
            },
        },
        manager,
        registry=registry,
    )
    assert "result" in response
