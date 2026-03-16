"""Tests for schema enhancements: AgentConfig new fields, ProfileConfig new fields, BundleManifest."""

from __future__ import annotations

from afs.schema import AgentConfig, BundleManifest, ProfileConfig


def test_agent_config_new_fields_defaults() -> None:
    agent = AgentConfig(name="test")
    assert agent.triggers == []
    assert agent.schedule == ""
    assert agent.module == ""
    assert agent.watch_paths == []


def test_agent_config_from_dict_with_new_fields() -> None:
    data = {
        "name": "watcher",
        "role": "observer",
        "triggers": ["on_mount", "file_watch"],
        "schedule": "5m",
        "module": "afs.agents.context_warm",
        "watch_paths": ["/tmp/test"],
    }
    agent = AgentConfig.from_dict(data)
    assert agent.name == "watcher"
    assert agent.triggers == ["on_mount", "file_watch"]
    assert agent.schedule == "5m"
    assert agent.module == "afs.agents.context_warm"
    assert len(agent.watch_paths) == 1


def test_agent_config_backward_compat() -> None:
    data = {"name": "legacy", "role": "general", "auto_start": True}
    agent = AgentConfig.from_dict(data)
    assert agent.name == "legacy"
    assert agent.auto_start is True
    assert agent.triggers == []
    assert agent.module == ""


def test_profile_config_new_fields_defaults() -> None:
    profile = ProfileConfig()
    assert profile.mcp_tools == []
    assert profile.cli_modules == []
    assert profile.agent_configs == []


def test_profile_config_from_dict_with_new_fields() -> None:
    data = {
        "mcp_tools": ["my_mcp.tools"],
        "cli_modules": ["my_cli.module"],
        "agent_configs": [
            {"name": "agent1", "module": "afs.agents.context_warm"},
            {"name": "agent2", "auto_start": True},
        ],
    }
    profile = ProfileConfig.from_dict(data)
    assert profile.mcp_tools == ["my_mcp.tools"]
    assert profile.cli_modules == ["my_cli.module"]
    assert len(profile.agent_configs) == 2
    assert profile.agent_configs[0].name == "agent1"
    assert profile.agent_configs[0].module == "afs.agents.context_warm"
    assert profile.agent_configs[1].auto_start is True


def test_profile_config_backward_compat() -> None:
    data = {
        "inherits": ["base"],
        "knowledge_mounts": ["/tmp/knowledge"],
        "policies": ["no_zelda"],
    }
    profile = ProfileConfig.from_dict(data)
    assert profile.inherits == ["base"]
    assert profile.policies == ["no_zelda"]
    assert profile.mcp_tools == []
    assert profile.agent_configs == []


def test_bundle_manifest_from_dict() -> None:
    data = {
        "name": "test-bundle",
        "version": "1.0.0",
        "description": "A test bundle",
        "author": "test",
    }
    manifest = BundleManifest.from_dict(data)
    assert manifest.name == "test-bundle"
    assert manifest.version == "1.0.0"
    assert manifest.description == "A test bundle"
    assert manifest.author == "test"
    assert manifest.skills_dir == "skills"
    assert manifest.knowledge_dir == "knowledge"


def test_bundle_manifest_round_trip() -> None:
    manifest = BundleManifest(
        name="round-trip",
        version="2.0.0",
        description="Test round trip",
    )
    data = manifest.to_dict()
    assert data["name"] == "round-trip"
    assert data["version"] == "2.0.0"
    restored = BundleManifest.from_dict(data)
    assert restored.name == manifest.name
    assert restored.version == manifest.version


def test_bundle_manifest_defaults() -> None:
    manifest = BundleManifest(name="minimal")
    assert manifest.version == "0.1.0"
    assert manifest.tools_dir == "tools"
    assert manifest.agents_dir == "agents"
    assert manifest.mcp_tools_dir == "mcp_tools"
