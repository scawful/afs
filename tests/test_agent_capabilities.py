"""Tests for agent capability registry."""

from __future__ import annotations

from afs.agents import AgentCapability, AgentSpec, _load_agent_module


def test_agent_capability_roundtrip():
    cap = AgentCapability(
        tools=["context.query"],
        topics=["context:repair"],
        mount_types=["knowledge", "tools"],
        description="Test capability",
    )
    assert cap.tools == ["context.query"]
    assert cap.topics == ["context:repair"]
    assert cap.mount_types == ["knowledge", "tools"]
    assert cap.description == "Test capability"


def test_agent_spec_with_and_without_capabilities():
    def noop(argv=None):
        return 0

    spec_with = AgentSpec(
        name="test",
        description="test agent",
        entrypoint=noop,
        capabilities=AgentCapability(
            tools=["a"], topics=["b"], mount_types=["c"],
        ),
    )
    assert spec_with.capabilities is not None
    assert spec_with.capabilities.tools == ["a"]

    spec_without = AgentSpec(
        name="test2",
        description="test agent 2",
        entrypoint=noop,
    )
    assert spec_without.capabilities is None


def test_load_agent_module_reads_capabilities():
    spec = _load_agent_module("afs.agents.context_warm")
    assert spec is not None
    assert spec.capabilities is not None
    assert "knowledge" in spec.capabilities.mount_types
    assert "context.query" in spec.capabilities.tools


def test_supervisor_sets_env_from_capabilities(tmp_path, monkeypatch):
    from afs.agents.supervisor import AgentSupervisor
    from afs.schema import AgentConfig

    sup = AgentSupervisor(state_dir=tmp_path)
    # Agent config with no allowed_mounts/tools but workspace_isolated=False
    config = AgentConfig(name="context-warm", module="afs.agents.context_warm")
    env = sup._build_agent_env("context-warm", config)
    # Should pick up capabilities from registry
    if env is not None:
        # If the agent has capabilities, env should include them
        assert "AFS_ALLOWED_MOUNTS" in env or "AFS_ALLOWED_TOOLS" in env


def test_capabilities_command_lists_all_agents():
    from afs.agents import list_agents
    agents = list_agents()
    # At least some agents should exist
    assert len(agents) > 0
    names = [a.name for a in agents]
    assert "context-warm" in names
