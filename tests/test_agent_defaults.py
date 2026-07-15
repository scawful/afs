"""Tests for the shipped default supervised agent set."""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from afs.agent_defaults import (
    DEFAULT_AGENT_TAG,
    DEFAULT_AGENTS_ENV,
    default_agent_configs,
    default_agents_enabled,
    merge_default_agent_configs,
)
from afs.agents import get_agent_registry
from afs.agents.supervisor import AgentSupervisor
from afs.models import MountType
from afs.profiles import resolve_active_profile
from afs.schema import (
    AFSConfig,
    AgentConfig,
    AgentsConfig,
    DirectoryConfig,
    GeneralConfig,
    PolicyType,
)

DEFAULT_NAMES = {"context-warm", "index-rebuild", "skills-mine", "morning-briefing"}


def _config(tmp_path: Path, **agents_kwargs) -> AFSConfig:
    return AFSConfig(
        general=GeneralConfig(context_root=tmp_path / "context"),
        agents=AgentsConfig(**agents_kwargs),
    )


def test_default_agent_configs_shape(tmp_path: Path) -> None:
    config = _config(tmp_path)
    defaults = {agent.name: agent for agent in default_agent_configs(config)}
    assert set(defaults) == DEFAULT_NAMES

    warm = defaults["context-warm"]
    assert warm.triggers == []
    assert warm.schedule == "daily"
    assert warm.module == "afs.agents.default_context_warm"

    rebuild = defaults["index-rebuild"]
    assert rebuild.module == "afs.agents.index_rebuild"
    context_root = config.general.context_root
    assert rebuild.watch_paths == [context_root / "knowledge", context_root / "memory"]

    assert defaults["skills-mine"].schedule == "weekly"
    assert defaults["morning-briefing"].schedule == "daily"
    for agent in defaults.values():
        assert DEFAULT_AGENT_TAG in agent.tags
        assert agent.module
        assert not agent.auto_start


def test_default_agents_enabled_gates(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv(DEFAULT_AGENTS_ENV, raising=False)
    assert default_agents_enabled(_config(tmp_path)) is True
    assert default_agents_enabled(_config(tmp_path, default_set=False)) is False

    monkeypatch.setenv(DEFAULT_AGENTS_ENV, "off")
    assert default_agents_enabled(_config(tmp_path)) is False
    monkeypatch.setenv(DEFAULT_AGENTS_ENV, "on")
    assert default_agents_enabled(_config(tmp_path, default_set=False)) is True


def test_default_agents_invalid_environment_value_fails_closed(
    tmp_path: Path,
    monkeypatch,
    caplog,
) -> None:
    monkeypatch.setenv(DEFAULT_AGENTS_ENV, "enable-ish")

    assert default_agents_enabled(_config(tmp_path)) is False
    assert DEFAULT_AGENTS_ENV in caplog.text


def test_defaults_only_populate_an_empty_profile(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv(DEFAULT_AGENTS_ENV, raising=False)
    config = _config(tmp_path)
    custom = AgentConfig(name="context-warm", module="my.custom.module")

    populated = merge_default_agent_configs([custom], config=config)
    assert populated == [custom]
    assert {agent.name for agent in merge_default_agent_configs([], config=config)} == DEFAULT_NAMES


def test_default_watch_paths_honor_directory_role_mappings(tmp_path: Path) -> None:
    context_root = tmp_path / "context"
    config = AFSConfig(
        general=GeneralConfig(context_root=context_root),
        directories=[
            DirectoryConfig(
                name="durable-notes",
                policy=PolicyType.READ_ONLY,
                role=MountType.KNOWLEDGE,
            ),
            DirectoryConfig(
                name="long-term-memory",
                policy=PolicyType.READ_ONLY,
                role=MountType.MEMORY,
            ),
        ],
    )

    defaults = {agent.name: agent for agent in default_agent_configs(config)}
    assert defaults["index-rebuild"].watch_paths == [
        context_root / "durable-notes",
        context_root / "long-term-memory",
    ]


def test_resolve_active_profile_includes_defaults(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv(DEFAULT_AGENTS_ENV, raising=False)
    monkeypatch.delenv("AFS_PROFILE", raising=False)
    profile = resolve_active_profile(_config(tmp_path))
    names = {agent.name for agent in profile.agent_configs}
    assert DEFAULT_NAMES <= names

    disabled = resolve_active_profile(_config(tmp_path, default_set=False))
    assert not {agent.name for agent in disabled.agent_configs} & DEFAULT_NAMES


def test_supervisor_routes_default_triggers(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv(DEFAULT_AGENTS_ENV, raising=False)
    config = _config(tmp_path)
    defaults = default_agent_configs(config)
    supervisor = AgentSupervisor(
        config=config,
        state_dir=tmp_path / "supervisor-state",
    )

    booted = supervisor.evaluate_triggers_from("on_boot", defaults)
    assert booted == []

    changed = config.general.context_root / "knowledge" / "notes.md"
    watched = supervisor.evaluate_watch_paths([changed], defaults)
    assert [agent.name for agent in watched] == ["index-rebuild"]

    due = supervisor.due_schedules(
        defaults,
        now=datetime(2026, 7, 15, 7, 0, tzinfo=timezone.utc),
    )
    assert {agent.name for agent in due} == {
        "context-warm",
        "skills-mine",
        "morning-briefing",
    }


def test_default_agent_modules_are_registered() -> None:
    registry = get_agent_registry()
    assert "index-rebuild" in registry
    assert "skills-mine" in registry
    assert "morning-briefing" in registry


def test_agents_config_from_dict_roundtrip() -> None:
    assert AgentsConfig.from_dict({}).default_set is True
    assert AgentsConfig.from_dict({"default_set": False}).default_set is False
    assert AgentsConfig(default_set=False).to_dict() == {"default_set": False}
    parsed = AFSConfig.from_dict({"agents": {"default_set": False}})
    assert parsed.agents.default_set is False
    with pytest.raises(ValueError, match="agents.default_set must be a boolean"):
        AgentsConfig.from_dict({"default_set": "sometimes"})
