from __future__ import annotations

from afs.schema import AFSConfig, ServiceConfig, ServicesConfig
from afs.services.manager import ServiceManager


def test_service_manager_lists_builtins() -> None:
    manager = ServiceManager(config=AFSConfig(), platform_name="linux")
    names = [definition.name for definition in manager.list_definitions()]
    assert "orchestrator" in names
    assert "agent-supervisor" in names
    assert "context-watch" in names
    assert "gemini-workspace-brief" in names


def test_gemini_workspace_brief_service_uses_agent_entrypoint() -> None:
    manager = ServiceManager(config=AFSConfig(), platform_name="linux")
    definition = manager.get_definition("gemini-workspace-brief")
    assert definition is not None
    assert "afs.agents.gemini_workspace_brief" in definition.command


def test_context_warm_service_rebuilds_stale_indexes() -> None:
    manager = ServiceManager(config=AFSConfig(), platform_name="linux")
    definition = manager.get_definition("context-warm")
    assert definition is not None
    assert "--repair-mounts" in definition.command
    assert "--rebuild-stale-indexes" in definition.command


def test_context_watch_service_uses_watch_mode() -> None:
    manager = ServiceManager(config=AFSConfig(), platform_name="linux")
    definition = manager.get_definition("context-watch")
    assert definition is not None
    assert "--watch" in definition.command
    assert "--skip-embeddings" in definition.command


def test_agent_supervisor_service_uses_agent_entrypoint() -> None:
    manager = ServiceManager(config=AFSConfig(), platform_name="linux")
    definition = manager.get_definition("agent-supervisor")
    assert definition is not None
    assert "afs.agents.supervisor" in definition.command


def test_service_config_can_disable_service() -> None:
    services = ServicesConfig(
        enabled=True,
        services={
            "orchestrator": ServiceConfig(name="orchestrator", enabled=False),
        },
    )
    manager = ServiceManager(config=AFSConfig(services=services), platform_name="linux")
    names = [definition.name for definition in manager.list_definitions()]
    assert "orchestrator" not in names


def test_service_render_contains_execstart() -> None:
    manager = ServiceManager(config=AFSConfig(), platform_name="linux")
    unit = manager.render_unit("orchestrator")
    assert "ExecStart=" in unit


def test_service_render_launchd_contains_label() -> None:
    manager = ServiceManager(config=AFSConfig(), platform_name="darwin")
    payload = manager.render_unit("orchestrator")
    assert "\"Label\"" in payload


def test_service_manager_propagates_explicit_config_path(tmp_path) -> None:
    config_path = tmp_path / "afs.toml"
    config_path.write_text("[general]\ncontext_root = \"/tmp/context\"\n", encoding="utf-8")

    manager = ServiceManager(
        config=AFSConfig(),
        config_path=config_path,
        platform_name="linux",
    )
    definition = manager.get_definition("context-warm")

    assert definition is not None
    assert definition.environment["AFS_CONFIG_PATH"] == str(config_path.resolve())


def test_service_manager_preserves_repo_preference_flag(monkeypatch) -> None:
    monkeypatch.setenv("AFS_PREFER_REPO_CONFIG", "1")

    manager = ServiceManager(config=AFSConfig(), platform_name="linux")
    definition = manager.get_definition("context-warm")

    assert definition is not None
    assert definition.environment["AFS_PREFER_REPO_CONFIG"] == "1"
    assert "AFS_PREFER_USER_CONFIG" not in definition.environment
