from __future__ import annotations

from pathlib import Path

from afs.cli._utils import write_config
from afs.config import load_config, load_config_model, load_runtime_config_model
from afs.schema import AFSConfig


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


def test_load_runtime_config_model_uses_nearest_repo_config(tmp_path, monkeypatch) -> None:
    repo_root = tmp_path / "repo"
    nested = repo_root / "a" / "b"
    nested.mkdir(parents=True)
    context_root = repo_root / ".context"
    config_path = repo_root / "afs.toml"
    config_path.write_text(
        f"[general]\ncontext_root = \"{context_root}\"\n",
        encoding="utf-8",
    )

    monkeypatch.chdir(nested)
    model, resolved_path = load_runtime_config_model(
        merge_user=False,
        start_dir=nested,
    )

    assert resolved_path == config_path.resolve()
    assert model.general.context_root == context_root.resolve()


def test_load_config_model_parses_profiles_extensions_hooks(tmp_path) -> None:
    config_path = tmp_path / "profiles.toml"
    config_path.write_text(
        "[extensions]\n"
        "enabled_extensions = [\"workspace_adapter\"]\n"
        f"extension_dirs = [\"{tmp_path / 'extensions'}\"]\n\n"
        "[profiles]\n"
        "active_profile = \"work\"\n"
        "auto_apply = true\n\n"
        "[profiles.work]\n"
        "memory_mounts = [\"~/memory\"]\n"
        "knowledge_mounts = [\"~/work/logs\"]\n"
        "skill_roots = [\"~/skills\"]\n"
        "model_registries = [\"~/registry/chat_registry.toml\"]\n"
        "enabled_extensions = [\"workspace_adapter\"]\n"
        "policies = [\"no_zelda\"]\n\n"
        "[hooks]\n"
        "before_context_read = [\"scripts/hooks/read.sh\"]\n"
        "session_start = [\"scripts/hooks/session_start.sh\"]\n"
        "session_end = [\"scripts/hooks/session_end.sh\"]\n"
        "user_prompt_submit = [\"scripts/hooks/prompt.sh\"]\n"
        "task_completed = [\"scripts/hooks/task_completed.sh\"]\n",
        encoding="utf-8",
    )

    model = load_config_model(config_path=config_path, merge_user=False)
    assert model.extensions.enabled_extensions == ["workspace_adapter"]
    assert model.profiles.active_profile == "work"
    assert "work" in model.profiles.profiles
    work = model.profiles.profiles["work"]
    assert work.memory_mounts == [(Path.home() / "memory").resolve()]
    assert work.policies == ["no_zelda"]
    assert model.hooks.before_context_read == ["scripts/hooks/read.sh"]
    assert model.hooks.session_start == ["scripts/hooks/session_start.sh"]
    assert model.hooks.session_end == ["scripts/hooks/session_end.sh"]
    assert model.hooks.user_prompt_submit == ["scripts/hooks/prompt.sh"]
    assert model.hooks.task_completed == ["scripts/hooks/task_completed.sh"]


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


def test_load_config_model_parses_verification_profiles(tmp_path) -> None:
    config_path = tmp_path / "verification.toml"
    config_path.write_text(
        "[verification]\n"
        "default_profile = \"repo\"\n\n"
        "[verification.profiles.repo]\n"
        "description = \"Repo checks\"\n\n"
        "[[verification.profiles.repo.checks]]\n"
        "name = \"python\"\n"
        "paths = [\"**/*.py\", \"pyproject.toml\"]\n"
        "commands = [\"ruff check .\", \"pytest -q\"]\n"
        "skills = [\"python-quality\"]\n"
        "workflows = [\"edit_fast\"]\n"
        "tool_profiles = [\"edit_and_verify\"]\n",
        encoding="utf-8",
    )

    model = load_config_model(config_path=config_path, merge_user=False)
    assert model.verification.default_profile == "repo"
    assert "repo" in model.verification.profiles
    profile = model.verification.profiles["repo"]
    assert profile.description == "Repo checks"
    assert len(profile.checks) == 1
    check = profile.checks[0]
    assert check.name == "python"
    assert check.paths == ["**/*.py", "pyproject.toml"]
    assert check.commands == ["ruff check .", "pytest -q"]
    assert check.skills == ["python-quality"]
    assert check.workflows == ["edit_fast"]
    assert check.tool_profiles == ["edit_and_verify"]


def test_load_config_model_parses_mcp_allowed_roots(tmp_path) -> None:
    config_path = tmp_path / "mcp.toml"
    allowed = tmp_path / "workspace-root"
    config_path.write_text(
        "[general]\n"
        f"mcp_allowed_roots = [\"{allowed}\"]\n",
        encoding="utf-8",
    )

    model = load_config_model(config_path=config_path, merge_user=False)
    assert model.general.mcp_allowed_roots == [allowed.resolve()]


def test_load_config_model_parses_sensitivity_rules(tmp_path) -> None:
    config_path = tmp_path / "sensitivity.toml"
    config_path.write_text(
        "[sensitivity]\n"
        "never_index = [\"knowledge/private/*\"]\n"
        "never_embed = [\"**/*.secret.md\"]\n"
        "never_export = [\"memory/raw/*\"]\n",
        encoding="utf-8",
    )

    model = load_config_model(config_path=config_path, merge_user=False)

    assert model.sensitivity.never_index == ["knowledge/private/*"]
    assert model.sensitivity.never_embed == ["**/*.secret.md"]
    assert model.sensitivity.never_export == ["memory/raw/*"]


def test_load_config_model_merges_env_mcp_allowed_roots(tmp_path, monkeypatch) -> None:
    config_path = tmp_path / "mcp_env.toml"
    configured = tmp_path / "configured"
    env_root = tmp_path / "workspace-root"
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


def test_write_config_round_trips_extended_sections(tmp_path) -> None:
    config_path = tmp_path / "afs.toml"
    config = AFSConfig.from_dict(
        {
            "general": {
                "context_root": str(tmp_path / "context"),
                "python_executable": str(tmp_path / "venv" / "bin" / "python"),
                "workspace_directories": [
                    {"path": str(tmp_path / "workspace"), "description": "Lab"}
                ],
                "mcp_allowed_roots": [str(tmp_path / "workspace-root")],
                "discovery_ignore": ["archive", "vendor"],
            },
            "directories": [
                {"name": "memory", "policy": "read_only", "role": "memory"},
                {"name": "scratchpad", "policy": "writable", "role": "scratchpad"},
            ],
            "orchestrator": {
                "enabled": True,
                "max_agents": 7,
                "auto_routing": False,
                "default_agents": [
                    {
                        "name": "planner",
                        "role": "planner",
                        "backend": "local",
                        "allowed_tools": ["context.status"],
                    }
                ],
            },
            "services": {
                "enabled": True,
                "services": {
                    "context-watch": {
                        "enabled": True,
                        "auto_start": True,
                        "context_filters": [str(tmp_path / "workspace")],
                        "environment": {"AFS_TEST": "1"},
                    }
                },
            },
            "history": {
                "enabled": True,
                "include_payloads": True,
                "max_inline_chars": 2048,
                "payload_dir_name": "payload_archive",
                "redact_sensitive": False,
            },
            "memory_export": {
                "interval_seconds": 60,
                "dataset_output": str(tmp_path / "datasets" / "memory.jsonl"),
                "report_output": str(tmp_path / "reports" / "memory.json"),
                "allow_raw": True,
                "allow_raw_tags": ["allow_raw", "trusted"],
                "default_instruction": "Summarize this memory",
                "limit": 12,
                "require_quality": False,
                "min_quality_score": 0.25,
                "score_profile": "gemini",
                "enable_asar": True,
                "auto_start": True,
                "routes": [
                    {
                        "tags": ["scribe"],
                        "output": str(tmp_path / "datasets" / "scribe.jsonl"),
                        "domain": "scribe",
                    }
                ],
            },
            "memory_consolidation": {
                "enabled": True,
                "auto_start": True,
                "interval_seconds": 900,
                "report_output": str(tmp_path / "reports" / "history_memory.json"),
                "entries_filename": "durable.jsonl",
                "summary_dir_name": "history_notes",
                "checkpoint_filename": "cursor.json",
                "max_events_per_run": 42,
                "max_events_per_entry": 7,
                "include_event_types": ["fs", "context"],
                "write_markdown": False,
            },
            "context_index": {
                "enabled": True,
                "db_filename": "sqlite/context.db",
                "auto_index": False,
                "auto_refresh": False,
                "include_content": False,
                "max_file_size_bytes": 8192,
                "max_content_chars": 1024,
                "decay_hours": 72.0,
            },
            "sensitivity": {
                "never_index": ["knowledge/private/*"],
                "never_embed": ["**/*.secret.md"],
                "never_export": ["memory/raw/*"],
            },
            "hivemind": {
                "default_ttl_hours": 8,
                "reaper_enabled": False,
            },
        }
    )

    write_config(config_path, config)
    roundtrip = load_config_model(config_path=config_path, merge_user=False)
    text = config_path.read_text(encoding="utf-8")

    assert "[services]" in text
    assert "[history]" in text
    assert "[memory_export]" in text
    assert "[memory_consolidation]" in text
    assert "[context_index]" in text
    assert "[sensitivity]" in text
    assert "[hivemind]" in text
    assert roundtrip.general.python_executable == (tmp_path / "venv" / "bin" / "python").resolve()
    assert roundtrip.general.discovery_ignore == ["archive", "vendor"]
    assert roundtrip.orchestrator.enabled is True
    assert roundtrip.orchestrator.max_agents == 7
    assert roundtrip.orchestrator.default_agents[0].name == "planner"
    assert roundtrip.services.enabled is True
    assert roundtrip.services.services["context-watch"].context_filters == [
        (tmp_path / "workspace").resolve()
    ]
    assert roundtrip.services.services["context-watch"].environment == {"AFS_TEST": "1"}
    assert roundtrip.history.include_payloads is True
    assert roundtrip.history.payload_dir_name == "payload_archive"
    assert roundtrip.memory_export.allow_raw is True
    assert roundtrip.memory_export.routes[0].domain == "scribe"
    assert roundtrip.memory_consolidation.entries_filename == "durable.jsonl"
    assert roundtrip.context_index.decay_hours == 72.0
    assert roundtrip.sensitivity.never_embed == ["**/*.secret.md"]
    assert roundtrip.hivemind.default_ttl_hours == 8
