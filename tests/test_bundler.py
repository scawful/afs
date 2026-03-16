"""Tests for bundle pack/install/inspect operations."""

from __future__ import annotations

from pathlib import Path

from afs.bundler import inspect_bundle, install_bundle, pack_bundle
from afs.schema import (
    AFSConfig,
    GeneralConfig,
    ProfileConfig,
    ProfilesConfig,
)


def _make_config(tmp_path: Path) -> AFSConfig:
    knowledge_dir = tmp_path / "knowledge-src"
    knowledge_dir.mkdir()
    (knowledge_dir / "notes.md").write_text("# Notes", encoding="utf-8")

    skill_dir = tmp_path / "skills-src"
    skill_dir.mkdir()
    (skill_dir / "skill.py").write_text("# skill", encoding="utf-8")

    return AFSConfig(
        general=GeneralConfig(
            context_root=tmp_path / "context",
            agent_workspaces_dir=tmp_path / "context" / "workspaces",
        ),
        profiles=ProfilesConfig(
            active_profile="demo",
            profiles={
                "demo": ProfileConfig(
                    knowledge_mounts=[knowledge_dir],
                    skill_roots=[skill_dir],
                )
            },
        ),
    )


def test_pack_bundle(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    output = tmp_path / "output"
    output.mkdir()

    result = pack_bundle("demo", config=config, output_path=output)
    assert result.path.exists()
    assert (result.path / "bundle.toml").exists()
    assert result.file_count > 0
    assert result.size_bytes > 0


def test_inspect_bundle(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    output = tmp_path / "output"
    output.mkdir()

    pack_result = pack_bundle("demo", config=config, output_path=output)
    result = inspect_bundle(pack_result.path)
    assert result.manifest.name == "demo"
    assert "knowledge" in result.resource_counts or "skills" in result.resource_counts


def test_install_bundle(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    output = tmp_path / "output"
    output.mkdir()

    pack_result = pack_bundle("demo", config=config, output_path=output)
    install_result = install_bundle(pack_result.path, name_override="test-ext")
    assert install_result.extension_path.exists()
    assert (install_result.extension_path / "extension.toml").exists()
    assert (install_result.extension_path / "bundle.toml").exists()


def test_round_trip_pack_install(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    output = tmp_path / "output"
    output.mkdir()

    pack_result = pack_bundle("demo", config=config, output_path=output)
    install_result = install_bundle(pack_result.path, name_override="roundtrip")

    # Verify the installed extension has the knowledge files
    knowledge_dir = install_result.extension_path / "knowledge"
    assert knowledge_dir.exists()
    assert any(knowledge_dir.rglob("*.md"))


def test_inspect_missing_bundle(tmp_path: Path) -> None:
    import pytest

    with pytest.raises(FileNotFoundError):
        inspect_bundle(tmp_path / "nonexistent")
