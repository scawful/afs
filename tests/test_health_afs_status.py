from __future__ import annotations

from pathlib import Path

from afs.health.afs_status import collect_afs_health


def test_collect_afs_health_snapshot(tmp_path: Path) -> None:
    context_root = tmp_path / "context"
    context_root.mkdir(parents=True)
    (context_root / "monorepo").mkdir(parents=True)
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

    snapshot = collect_afs_health(config_path=config_path)
    assert snapshot["profile"]["active"] == "work"
    assert snapshot["context"]["path"] == str(context_root.resolve())
    assert "mcp" in snapshot
    assert "extensions" in snapshot
