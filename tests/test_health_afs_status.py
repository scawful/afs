from __future__ import annotations

from pathlib import Path

from afs.health.afs_status import _looks_like_afs_mcp_command, collect_afs_health


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
    assert "registered_clients" in snapshot["mcp"]
    assert "registered_with_claude" in snapshot["mcp"]
    assert "registered_with_codex" in snapshot["mcp"]


def test_detect_mcp_running_matches_cli_variants() -> None:
    assert _looks_like_afs_mcp_command("/usr/bin/python3 -m afs.mcp_server")
    assert _looks_like_afs_mcp_command("/usr/bin/python3 -m afs mcp serve")
    assert _looks_like_afs_mcp_command("/Users/scawful/src/lab/afs/scripts/afs mcp serve")
    assert not _looks_like_afs_mcp_command("/usr/bin/python3 -m afs health")
