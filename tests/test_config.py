from __future__ import annotations

from pathlib import Path

from afs.config import load_config, load_config_model


def test_load_config_merges_workspace_registry(tmp_path, monkeypatch) -> None:
    context_root = tmp_path / "context"
    context_root.mkdir()
    workspace_dir = tmp_path / "workspace"
    workspace_dir.mkdir()

    registry_path = context_root / "workspaces.toml"
    registry_path.write_text(
        "[[workspaces]]\n"
        f"path = \"{workspace_dir}\"\n"
        "description = \"Example\"\n",
        encoding="utf-8",
    )

    config_path = tmp_path / "afs.toml"
    config_path.write_text(
        f"[general]\ncontext_root = \"{context_root}\"\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(tmp_path)
    data = load_config(merge_user=False)
    workspaces = data["general"]["workspace_directories"]
    assert workspaces
    assert Path(workspaces[0]["path"]).resolve() == workspace_dir.resolve()


def test_load_config_model_uses_explicit_path(tmp_path) -> None:
    context_root = tmp_path / "context"
    config_path = tmp_path / "custom.toml"
    config_path.write_text(
        f"[general]\ncontext_root = \"{context_root}\"\n",
        encoding="utf-8",
    )

    model = load_config_model(config_path=config_path, merge_user=False)
    assert model.general.context_root == context_root.resolve()


def test_load_config_model_parses_profiles_extensions_hooks(tmp_path) -> None:
    config_path = tmp_path / "profiles.toml"
    config_path.write_text(
        "[extensions]\n"
        "enabled_extensions = [\"afs_google\"]\n"
        f"extension_dirs = [\"{tmp_path / 'extensions'}\"]\n\n"
        "[profiles]\n"
        "active_profile = \"work\"\n"
        "auto_apply = true\n\n"
        "[profiles.work]\n"
        "knowledge_mounts = [\"~/Journal/logs\"]\n"
        "skill_roots = [\"~/skills\"]\n"
        "model_registries = [\"~/registry/chat_registry.toml\"]\n"
        "enabled_extensions = [\"afs_google\"]\n"
        "policies = [\"no_zelda\"]\n\n"
        "[hooks]\n"
        "before_context_read = [\"scripts/hooks/read.sh\"]\n",
        encoding="utf-8",
    )

    model = load_config_model(config_path=config_path, merge_user=False)
    assert model.extensions.enabled_extensions == ["afs_google"]
    assert model.profiles.active_profile == "work"
    assert "work" in model.profiles.profiles
    work = model.profiles.profiles["work"]
    assert work.policies == ["no_zelda"]
    assert model.hooks.before_context_read == ["scripts/hooks/read.sh"]


def test_load_config_model_parses_context_index_settings(tmp_path) -> None:
    config_path = tmp_path / "index.toml"
    config_path.write_text(
        "[context_index]\n"
        "enabled = true\n"
        "db_filename = \"sqlite/context.db\"\n"
        "auto_index = false\n"
        "auto_refresh = false\n"
        "include_content = false\n"
        "max_file_size_bytes = 8192\n"
        "max_content_chars = 1024\n",
        encoding="utf-8",
    )

    model = load_config_model(config_path=config_path, merge_user=False)
    assert model.context_index.enabled is True
    assert model.context_index.db_filename == "sqlite/context.db"
    assert model.context_index.auto_index is False
    assert model.context_index.auto_refresh is False
    assert model.context_index.include_content is False
    assert model.context_index.max_file_size_bytes == 8192
    assert model.context_index.max_content_chars == 1024


def test_load_config_model_parses_mcp_allowed_roots(tmp_path) -> None:
    config_path = tmp_path / "mcp.toml"
    allowed = tmp_path / "google"
    config_path.write_text(
        "[general]\n"
        f"mcp_allowed_roots = [\"{allowed}\"]\n",
        encoding="utf-8",
    )

    model = load_config_model(config_path=config_path, merge_user=False)
    assert model.general.mcp_allowed_roots == [allowed.resolve()]


def test_load_config_model_merges_env_mcp_allowed_roots(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "mcp_env.toml"
    configured = tmp_path / "configured"
    env_root = tmp_path / "google"
    config_path.write_text(
        "[general]\n"
        f"mcp_allowed_roots = [\"{configured}\"]\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("AFS_MCP_ALLOWED_ROOTS", str(env_root))

    model = load_config_model(config_path=config_path, merge_user=False)
    assert model.general.mcp_allowed_roots == [configured.resolve(), env_root.resolve()]


def test_load_config_model_parses_memory_consolidation_settings(tmp_path) -> None:
    report_output = tmp_path / "reports" / "history_memory.json"
    config_path = tmp_path / "memory_consolidation.toml"
    config_path.write_text(
        "[memory_consolidation]\n"
        "enabled = true\n"
        "auto_start = true\n"
        "interval_seconds = 900\n"
        f"report_output = \"{report_output}\"\n"
        "entries_filename = \"durable.jsonl\"\n"
        "summary_dir_name = \"history_notes\"\n"
        "checkpoint_filename = \"cursor.json\"\n"
        "max_events_per_run = 42\n"
        "max_events_per_entry = 7\n"
        "include_event_types = [\"fs\", \"context\"]\n"
        "write_markdown = false\n",
        encoding="utf-8",
    )

    model = load_config_model(config_path=config_path, merge_user=False)

    assert model.memory_consolidation.enabled is True
    assert model.memory_consolidation.auto_start is True
    assert model.memory_consolidation.interval_seconds == 900
    assert model.memory_consolidation.report_output == report_output.resolve()
    assert model.memory_consolidation.entries_filename == "durable.jsonl"
    assert model.memory_consolidation.summary_dir_name == "history_notes"
    assert model.memory_consolidation.checkpoint_filename == "cursor.json"
    assert model.memory_consolidation.max_events_per_run == 42
    assert model.memory_consolidation.max_events_per_entry == 7
    assert model.memory_consolidation.include_event_types == ["fs", "context"]
    assert model.memory_consolidation.write_markdown is False


def test_load_config_model_parses_service_context_filters(tmp_path) -> None:
    config_path = tmp_path / "services.toml"
    config_path.write_text(
        "[services]\n"
        "enabled = true\n\n"
        "[services.services.context-watch]\n"
        f"context_filters = [\"{tmp_path / 'lab'}\"]\n",
        encoding="utf-8",
    )

    model = load_config_model(config_path=config_path, merge_user=False)

    service = model.services.services["context-watch"]
    assert service.context_filters == [(tmp_path / "lab").resolve()]
