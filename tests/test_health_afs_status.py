from __future__ import annotations

from pathlib import Path

import afs.config as config_module
import afs.health.afs_status as afs_status_module
from afs.health.afs_status import _looks_like_afs_mcp_command, collect_afs_health
from afs.models import MountType


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
    monkeypatch.setattr(afs_status_module, "load_config_model", lambda **_kwargs: config)

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


def test_detect_mcp_running_matches_cli_variants() -> None:
    assert _looks_like_afs_mcp_command("/usr/bin/python3 -m afs.mcp_server")
    assert _looks_like_afs_mcp_command("/usr/bin/python3 -m afs mcp serve")
    assert _looks_like_afs_mcp_command("/Users/scawful/src/lab/afs/scripts/afs mcp serve")
    assert not _looks_like_afs_mcp_command("/usr/bin/python3 -m afs health")
