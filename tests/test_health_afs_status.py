from __future__ import annotations

import json
from pathlib import Path

import pytest

import afs.config as config_module
import afs.health.afs_status as afs_status_module
from afs.health.afs_status import _looks_like_afs_mcp_command, collect_afs_health
from afs.manager import AFSManager
from afs.models import MountType
from afs.schema import AFSConfig, DirectoryConfig, GeneralConfig, default_directory_configs


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


def _clear_profile_env(monkeypatch) -> None:  # noqa: ANN001
    for name in (
        "AFS_PROFILE",
        "AFS_ENABLED_EXTENSIONS",
        "AFS_KNOWLEDGE_MOUNTS",
        "AFS_SKILL_ROOTS",
        "AFS_MODEL_REGISTRIES",
        "AFS_POLICIES",
    ):
        monkeypatch.delenv(name, raising=False)


def test_collect_afs_health_snapshot(tmp_path: Path, monkeypatch) -> None:
    _clear_profile_env(monkeypatch)
    context_root = tmp_path / "context"
    context_root.mkdir(parents=True)
    for mount_type in MountType:
        (context_root / mount_type.value).mkdir(exist_ok=True)
    (context_root / "monorepo").mkdir(parents=True, exist_ok=True)
    (context_root / "monorepo" / "active_workspace.toml").write_text(
        'active_workspace = "/tmp/workspace"\n',
        encoding="utf-8",
    )

    config_path = tmp_path / "afs.toml"
    config_path.write_text(
        "[general]\n"
        f"context_root = \"{context_root}\"\n"
        "\n"
        "[profiles]\n"
        "active_profile = \"work\"\n"
        "\n"
        "[profiles.work]\n"
        "knowledge_mounts = []\n"
        "skill_roots = []\n"
        "model_registries = []\n"
        "enabled_extensions = []\n"
        "policies = []\n",
        encoding="utf-8",
    )
    config = config_module.load_config_model(config_path=config_path, merge_user=False)
    monkeypatch.setattr(
        afs_status_module,
        "load_runtime_config_model",
        lambda **_kwargs: (config, config_path.resolve()),
    )

    snapshot = collect_afs_health(config_path=config_path)
    assert snapshot["profile"]["active"] == "work"
    assert snapshot["context"]["path"] == str(context_root.resolve())
    assert "mcp" in snapshot
    assert "extensions" in snapshot
    assert "maintenance" in snapshot
    assert snapshot["context"]["mount_health"]["healthy"] is True
    assert "registered_clients" in snapshot["mcp"]
    assert "registered_with_claude" in snapshot["mcp"]
    assert "registered_with_codex" in snapshot["mcp"]
    assert "workflow_usage" in snapshot["mcp"]
    assert "supervisor" in snapshot["maintenance"]
    assert "agent_supervisor" in snapshot["maintenance"]["reports"]
    assert "history_memory" in snapshot["maintenance"]["reports"]
    assert "doctor_snapshot" in snapshot["maintenance"]["reports"]


def test_detect_mcp_running_matches_cli_variants() -> None:
    assert _looks_like_afs_mcp_command("/usr/bin/python3 -m afs.mcp_server")
    assert _looks_like_afs_mcp_command("/usr/bin/python3 -m afs mcp serve")
    assert _looks_like_afs_mcp_command("$AFS_ROOT/scripts/afs mcp serve")
    assert not _looks_like_afs_mcp_command("/usr/bin/python3 -m afs health")


def test_collect_afs_health_uses_remapped_history_and_scratchpad(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _clear_profile_env(monkeypatch)
    context_root = tmp_path / "context"
    config = AFSConfig(
        general=GeneralConfig(
            context_root=context_root,
        ),
        directories=_remap_directories(history="ledger", scratchpad="notes"),
    )
    manager = AFSManager(config=config)
    project_path = tmp_path / "project"
    project_path.mkdir()
    manager.ensure(path=project_path, context_root=context_root)
    report_dir = context_root / "notes" / "afs_agents"
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "context_warm.json").write_text(
        json.dumps({"status": "ok", "payload": {}}),
        encoding="utf-8",
    )
    (report_dir / "doctor_snapshot.json").write_text(
        json.dumps({"status": "warn", "payload": {"checks": []}}),
        encoding="utf-8",
    )
    (context_root / "ledger" / "events_20260317.jsonl").write_text(
        json.dumps(
            {
                "type": "hook",
                "op": "before_context_read",
                "timestamp": "2026-03-17T12:00:00+00:00",
            }
        )
        + "\n"
        + json.dumps(
            {
                "type": "mcp_tool",
                "timestamp": "2026-03-17T12:05:00+00:00",
                "source": "afs.mcp",
                "metadata": {"tool_name": "afs.session.bootstrap"},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr(
        afs_status_module,
        "load_runtime_config_model",
        lambda **_kwargs: (config, None),
    )
    monkeypatch.setattr(afs_status_module, "find_root", lambda _start_dir=None: context_root)
    monkeypatch.setattr(
        afs_status_module,
        "resolve_context_root",
        lambda _config, linked_root: linked_root or context_root,
    )

    snapshot = collect_afs_health()

    assert snapshot["hooks"]["events"]["before_context_read"]["last_run"] == "2026-03-17T12:00:00+00:00"
    assert snapshot["maintenance"]["reports"]["context_warm"]["path"] == str(
        report_dir / "context_warm.json"
    )
    assert snapshot["maintenance"]["reports"]["history_memory"]["path"] == str(
        report_dir / "history_memory.json"
    )
    assert snapshot["maintenance"]["reports"]["doctor_snapshot"]["path"] == str(
        report_dir / "doctor_snapshot.json"
    )
    assert snapshot["mcp"]["workflow_usage"]["tools"]["afs.session.bootstrap"]["count"] == 1


@pytest.mark.parametrize("linked_component", ["root", "leaf"])
def test_v2_maintenance_health_skips_linked_agent_reports(
    tmp_path: Path,
    monkeypatch,
    linked_component: str,
) -> None:
    context_root = tmp_path / "context"
    project = tmp_path / "project"
    project.mkdir()
    config = AFSConfig(general=GeneralConfig(context_root=context_root))
    manager = AFSManager(config=config)
    manager.ensure(path=project, context_root=context_root, layout_version=2)
    from afs.context_paths import resolve_agent_output_root

    output_root = resolve_agent_output_root(context_root, config=config, scope_id="common")
    output_root.parent.mkdir(parents=True, exist_ok=True)
    outside = tmp_path / "outside"
    outside.mkdir()
    poison = outside / "context_warm.json"
    poison.write_text(
        json.dumps({"status": "ok", "notes": ["poison-canary"]}),
        encoding="utf-8",
    )

    try:
        if linked_component == "root":
            (outside / "context_warm.json").write_text(
                poison.read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            output_root.symlink_to(outside, target_is_directory=True)
        else:
            output_root.mkdir()
            (output_root / "context_warm.json").symlink_to(poison)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    class _NoServices:
        def get_definition(self, _name):  # noqa: ANN001
            return None

    class _UnavailableSupervisor:
        def __init__(self, **_kwargs):  # noqa: ANN003
            raise RuntimeError("disabled for report-read test")

    monkeypatch.setattr(
        afs_status_module,
        "ServiceManager",
        lambda **_kwargs: _NoServices(),
    )
    monkeypatch.setattr(
        "afs.agents.supervisor.AgentSupervisor",
        _UnavailableSupervisor,
    )

    maintenance = afs_status_module._maintenance_health(config, context_root)

    assert maintenance["reports"]["context_warm"]["available"] is False
    assert "poison-canary" not in json.dumps(maintenance)
