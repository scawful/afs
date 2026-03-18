"""Tests for P5-P7 improvements: config preservation, monorepo optional, workspace shorthand."""

from __future__ import annotations

from pathlib import Path

import pytest

from afs.context_fs import ContextFileSystem
from afs.manager import AFSManager
from afs.mcp_server import _handle_request
from afs.models import OPTIONAL_MOUNT_TYPES, ContextRoot, MountType, ProjectMetadata
from afs.schema import AFSConfig, AgentConfig, GeneralConfig
from afs.validator import AFSValidator

# --- P6: Monorepo optional ---

def test_context_valid_without_monorepo(tmp_path: Path) -> None:
    ctx = tmp_path / ".context"
    ctx.mkdir()
    for mt in MountType:
        if mt not in OPTIONAL_MOUNT_TYPES:
            (ctx / mt.value).mkdir()
    root = ContextRoot(path=ctx, project_name="test", metadata=ProjectMetadata())
    assert root.is_valid


def test_context_invalid_without_required_dir(tmp_path: Path) -> None:
    ctx = tmp_path / ".context"
    ctx.mkdir()
    # Only create some dirs — missing memory
    for mt in MountType:
        if mt not in OPTIONAL_MOUNT_TYPES and mt != MountType.MEMORY:
            (ctx / mt.value).mkdir()
    root = ContextRoot(path=ctx, project_name="test", metadata=ProjectMetadata())
    assert not root.is_valid


def test_validator_passes_without_monorepo(tmp_path: Path) -> None:
    ctx = tmp_path / ".context"
    ctx.mkdir()
    for mt in MountType:
        if mt not in OPTIONAL_MOUNT_TYPES:
            (ctx / mt.value).mkdir()
    validator = AFSValidator(ctx)
    result = validator.check_integrity()
    assert result["valid"] is True
    assert result["missing"] == []


# --- P7: Workspace directory shorthand ---

def test_workspace_directories_string_shorthand() -> None:
    gc = GeneralConfig.from_dict({"workspace_directories": ["/tmp/lab"]})
    assert len(gc.workspace_directories) == 1
    assert gc.workspace_directories[0].path == Path("/tmp/lab").resolve()


def test_workspace_directories_mixed_forms() -> None:
    gc = GeneralConfig.from_dict({
        "workspace_directories": [
            "/tmp/a",
            {"path": "/tmp/b", "description": "desc-b"},
        ]
    })
    assert len(gc.workspace_directories) == 2
    assert gc.workspace_directories[0].path == Path("/tmp/a").resolve()
    assert gc.workspace_directories[1].description == "desc-b"


# --- P5: Config round-trip ---

def test_write_config_preserves_unknown_sections(tmp_path: Path) -> None:
    from afs.cli._utils import write_config
    from afs.schema import AFSConfig

    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        "[llamacpp]\nenabled = true\nmodel = \"qwen3-14b\"\n\n"
        "[general]\ncontext_root = \"/old\"\n",
        encoding="utf-8",
    )
    config = AFSConfig()
    config.general.context_root = Path("/new")
    write_config(cfg_path, config)

    text = cfg_path.read_text()
    assert "llamacpp" in text
    assert "qwen3-14b" in text
    assert "/new" in text


def test_write_config_updates_known_sections(tmp_path: Path) -> None:
    from afs.cli._utils import write_config
    from afs.schema import AFSConfig

    cfg_path = tmp_path / "config.toml"
    cfg_path.write_text(
        '[profiles]\nactive_profile = "old"\n',
        encoding="utf-8",
    )
    config = AFSConfig()
    config.profiles.active_profile = "new-profile"
    write_config(cfg_path, config)

    text = cfg_path.read_text()
    assert "new-profile" in text


# --- P4: Agent sandbox fields ---

def test_agent_config_sandbox_fields() -> None:
    ac = AgentConfig.from_dict({
        "name": "test",
        "allowed_mounts": ["scratchpad", "history"],
        "allowed_tools": ["fs.read"],
        "workspace_isolated": True,
    })
    assert ac.allowed_mounts == ["scratchpad", "history"]
    assert ac.allowed_tools == ["fs.read"]
    assert ac.workspace_isolated is True


def test_agent_config_sandbox_defaults() -> None:
    ac = AgentConfig.from_dict({"name": "test"})
    assert ac.allowed_mounts == []
    assert ac.allowed_tools == []
    assert ac.workspace_isolated is False


def test_agent_config_sandbox_round_trip() -> None:
    ac = AgentConfig.from_dict({
        "name": "test",
        "allowed_mounts": ["items"],
        "workspace_isolated": True,
    })
    d = ac.to_dict()
    assert d["allowed_mounts"] == ["items"]
    assert d["workspace_isolated"] is True
    restored = AgentConfig.from_dict(d)
    assert restored.allowed_mounts == ["items"]


def test_context_fs_enforces_allowed_mounts_for_reads(tmp_path: Path, monkeypatch) -> None:
    context_root = tmp_path / ".context"
    general = GeneralConfig(
        context_root=tmp_path / "context-root",
        agent_workspaces_dir=tmp_path / "context-root" / "workspaces",
    )
    manager = AFSManager(config=AFSConfig(general=general))
    project_path = tmp_path / "project"
    project_path.mkdir()
    manager.ensure(path=project_path, context_root=context_root)
    (context_root / "memory" / "note.md").write_text("secret", encoding="utf-8")

    monkeypatch.setenv("AFS_AGENT_NAME", "scoped-agent")
    monkeypatch.setenv("AFS_ALLOWED_MOUNTS", "scratchpad")

    fs = ContextFileSystem(manager, context_root)
    with pytest.raises(PermissionError, match="not allowed to read memory"):
        fs.read_text(MountType.MEMORY, "note.md")
    with pytest.raises(PermissionError, match="not allowed to read memory"):
        fs.list_entries(MountType.MEMORY)


def test_mcp_tool_scope_enforces_allowed_tools(tmp_path: Path, monkeypatch) -> None:
    context_root = tmp_path / "context"
    general = GeneralConfig(
        context_root=context_root,
        agent_workspaces_dir=context_root / "workspaces",
    )
    manager = AFSManager(config=AFSConfig(general=general))
    project_path = tmp_path / "project"
    project_path.mkdir()
    manager.ensure(path=project_path, context_root=context_root)
    monkeypatch.setenv("AFS_AGENT_NAME", "scoped-agent")
    monkeypatch.setenv("AFS_ALLOWED_TOOLS", "task.list")

    blocked = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "hivemind.read",
                "arguments": {"context_path": str(context_root)},
            },
        },
        manager,
    )
    assert blocked is not None
    assert blocked["error"]["message"] == "agent scoped-agent not allowed to use tool hivemind.read"

    allowed = _handle_request(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "task.list",
                "arguments": {"context_path": str(context_root)},
            },
        },
        manager,
    )
    assert allowed is not None
    structured = allowed["result"]["structuredContent"]
    assert structured["tasks"] == []
