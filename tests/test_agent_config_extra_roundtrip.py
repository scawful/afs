"""write_config must preserve AgentConfig.extra across load/write/load."""

from __future__ import annotations

from pathlib import Path

import pytest

from afs.cli._utils import write_config
from afs.config import load_config_model
from afs.profiles import resolve_active_profile
from afs.schema import AFSConfig, AgentConfig, OrchestratorConfig, ProfileConfig, ProfilesConfig


def test_write_config_round_trips_agent_extra_fields(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("AFS_CONFIG_PATH", raising=False)
    context_root = tmp_path / "context"
    workspace_dir = tmp_path / "workspace"
    config_path = tmp_path / "afs.toml"
    config_path.write_text(
        "[general]\n"
        f'context_root = "{context_root.as_posix()}"\n\n'
        "[profiles]\n"
        'active_profile = "work"\n\n'
        "[profiles.work]\n"
        "agent_configs = [\n"
        "  { name = \"scope-agent\", module = \"afs.agents.tooling_watch\", "
        f'context_root = "{context_root.as_posix()}", '
        f'project_name = "oracle-of-secrets", '
        f'workspace_directory = "{workspace_dir.as_posix()}", '
        'custom_scalar = 7, custom_list = [1, "two"], '
        "custom_nested = { flag = true, note = \"nested\" } },\n"
        "]\n\n"
        "[orchestrator]\n"
        "enabled = true\n"
        "default_agents = [\n"
        '  { name = "orch-agent", role = "general", orch_marker = "keep-me" },\n'
        "]\n",
        encoding="utf-8",
    )

    loaded = load_config_model(config_path=config_path, merge_user=False)
    profile_agent = resolve_active_profile(loaded).agent_configs[0]
    assert profile_agent.extra["project_name"] == "oracle-of-secrets"
    assert profile_agent.extra["custom_nested"] == {"flag": True, "note": "nested"}

    before_orchestrator = loaded.orchestrator.default_agents[0].extra.copy()

    write_config(config_path, loaded)
    reloaded = load_config_model(config_path=config_path, merge_user=False)

    roundtrip_profile = resolve_active_profile(reloaded).agent_configs[0]
    assert roundtrip_profile.name == "scope-agent"
    assert roundtrip_profile.module == "afs.agents.tooling_watch"
    assert roundtrip_profile.extra["context_root"] == str(context_root.resolve())
    assert roundtrip_profile.extra["project_name"] == "oracle-of-secrets"
    assert roundtrip_profile.extra["workspace_directory"] == str(workspace_dir.resolve())
    assert roundtrip_profile.extra["custom_scalar"] == 7
    assert roundtrip_profile.extra["custom_list"] == [1, "two"]
    assert roundtrip_profile.extra["custom_nested"] == {"flag": True, "note": "nested"}

    roundtrip_orchestrator = reloaded.orchestrator.default_agents[0]
    assert roundtrip_orchestrator.name == "orch-agent"
    assert roundtrip_orchestrator.extra == before_orchestrator
    assert roundtrip_orchestrator.extra["orch_marker"] == "keep-me"


def test_agent_config_to_dict_canonical_fields_override_extra_collision() -> None:
    agent = AgentConfig(
        name="canonical-name",
        role="worker",
        extra={
            "name": "shadow-name",
            "role": "intruder",
            "tags": ["from-extra"],
        },
    )
    payload = agent.to_dict()
    assert payload["name"] == "canonical-name"
    assert payload["role"] == "worker"
    assert payload["tags"] == []

    restored = AgentConfig.from_dict(payload)
    assert restored.name == "canonical-name"
    assert restored.role == "worker"
    assert restored.tags == []
    assert "name" not in restored.extra
    assert "role" not in restored.extra


def test_write_config_leaves_disposable_file_unchanged_on_serialization_failure(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.delenv("AFS_CONFIG_PATH", raising=False)
    config_path = tmp_path / "afs.toml"
    original_text = (
        "[general]\n"
        f'context_root = "{(tmp_path / "context").as_posix()}"\n'
        "# preserved comment\n"
    )
    config_path.write_text(original_text, encoding="utf-8")

    profile = ProfileConfig(
        agent_configs=[
            AgentConfig(
                name="broken",
                extra={"unserializable": object()},
            )
        ]
    )
    config = AFSConfig(
        profiles=ProfilesConfig(
            active_profile="default",
            profiles={"default": profile},
        ),
        orchestrator=OrchestratorConfig(),
    )

    with pytest.raises((TypeError, ValueError)):
        write_config(config_path, config)

    assert config_path.read_text(encoding="utf-8") == original_text
