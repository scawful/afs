"""Tests for bundle pack/install/inspect operations."""

from __future__ import annotations

from pathlib import Path

from afs.agents import get_agent
from afs.bundler import inspect_bundle, install_bundle, pack_bundle
from afs.cli import build_parser
from afs.manager import AFSManager
from afs.mcp_server import build_mcp_registry
from afs.schema import (
    AFSConfig,
    AgentConfig,
    ExtensionsConfig,
    GeneralConfig,
    ProfileConfig,
    ProfilesConfig,
)


def _command_choices(parser) -> set[str]:
    for action in parser._actions:
        choices = getattr(action, "choices", None)
        if choices:
            return set(choices.keys())
    return set()


def _make_profile_package(tmp_path: Path) -> None:
    package_root = tmp_path / "bundle_mods"
    package_root.mkdir()
    (package_root / "__init__.py").write_text("", encoding="utf-8")
    (package_root / "cli.py").write_text(
        "def register_parsers(subparsers):\n"
        "    parser = subparsers.add_parser('bundle-demo', help='bundle demo command')\n"
        "    parser.set_defaults(func=lambda _args: 0)\n",
        encoding="utf-8",
    )
    (package_root / "mcp.py").write_text(
        "def register_mcp_server(_manager):\n"
        "    return {\n"
        "        'tools': [\n"
        "            {\n"
        "                'name': 'bundle.echo',\n"
        "                'description': 'bundle tool',\n"
        "                'inputSchema': {\n"
        "                    'type': 'object',\n"
        "                    'properties': {},\n"
        "                    'additionalProperties': False,\n"
        "                },\n"
        "                'handler': lambda _arguments: {'ok': True},\n"
        "            }\n"
        "        ]\n"
        "    }\n",
        encoding="utf-8",
    )
    (package_root / "agent.py").write_text(
        "def main(_argv=None):\n"
        "    return 0\n",
        encoding="utf-8",
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
        extensions=ExtensionsConfig(
            auto_discover=False,
            extension_dirs=[tmp_path / "extensions"],
        ),
        profiles=ProfilesConfig(
            active_profile="demo",
            profiles={
                "demo": ProfileConfig(
                    knowledge_mounts=[knowledge_dir],
                    skill_roots=[skill_dir],
                    cli_modules=["bundle_mods.cli"],
                    mcp_tools=["bundle_mods.mcp"],
                    agent_configs=[
                        AgentConfig(
                            name="bundle-agent",
                            module="bundle_mods.agent",
                            schedule="5m",
                        )
                    ],
                )
            },
        ),
    )


def test_pack_bundle(monkeypatch, tmp_path: Path) -> None:
    _make_profile_package(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    config = _make_config(tmp_path)
    output = tmp_path / "output"
    output.mkdir()

    result = pack_bundle("demo", config=config, output_path=output)
    assert result.path.exists()
    assert (result.path / "bundle.toml").exists()
    assert result.file_count > 0
    assert result.size_bytes > 0
    assert "bundle_mods" in result.bundled_modules


def test_inspect_bundle(monkeypatch, tmp_path: Path) -> None:
    _make_profile_package(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    config = _make_config(tmp_path)
    output = tmp_path / "output"
    output.mkdir()

    pack_result = pack_bundle("demo", config=config, output_path=output)
    result = inspect_bundle(pack_result.path)
    assert result.manifest.name == "demo"
    assert "knowledge" in result.resource_counts or "skills" in result.resource_counts
    assert "bundle_mods" in result.bundled_modules
    assert result.manifest.profile.agent_configs[0].name == "bundle-agent"


def test_install_bundle(monkeypatch, tmp_path: Path) -> None:
    _make_profile_package(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    config = _make_config(tmp_path)
    output = tmp_path / "output"
    output.mkdir()

    pack_result = pack_bundle("demo", config=config, output_path=output)
    install_result = install_bundle(pack_result.path, config=config, name_override="test-ext")
    assert install_result.extension_path.exists()
    assert (install_result.extension_path / "extension.toml").exists()
    assert (install_result.extension_path / "bundle.toml").exists()
    assert install_result.profile_snippet_path is not None
    assert install_result.profile_snippet_path.exists()


def test_round_trip_pack_install_exposes_cli_agent_and_mcp(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _make_profile_package(tmp_path)
    monkeypatch.syspath_prepend(str(tmp_path))
    config = _make_config(tmp_path)
    output = tmp_path / "output"
    output.mkdir()

    pack_result = pack_bundle("demo", config=config, output_path=output)
    install_result = install_bundle(pack_result.path, config=config, name_override="roundtrip")

    config_path = tmp_path / "afs.toml"
    config_path.write_text("[extensions]\nauto_discover = false\n", encoding="utf-8")
    monkeypatch.setenv("AFS_CONFIG_PATH", str(config_path))
    monkeypatch.setenv("AFS_EXTENSION_DIRS", str(config.extensions.extension_dirs[0]))
    monkeypatch.setenv("AFS_ENABLED_EXTENSIONS", "roundtrip")

    parser = build_parser()
    assert "bundle-demo" in _command_choices(parser)
    assert get_agent("bundle-agent") is not None

    manager = AFSManager(config=config)
    registry = build_mcp_registry(manager)
    names = set(registry.tools)
    assert "bundle.echo" in names

    knowledge_dir = install_result.extension_path / "knowledge"
    assert knowledge_dir.exists()
    assert any(knowledge_dir.rglob("*.md"))


def test_inspect_missing_bundle(tmp_path: Path) -> None:
    import pytest

    with pytest.raises(FileNotFoundError):
        inspect_bundle(tmp_path / "nonexistent")


def test_pack_bundle_dereferences_symlinked_mount_content(tmp_path: Path) -> None:
    real_knowledge = tmp_path / "real-knowledge"
    real_knowledge.mkdir()
    (real_knowledge / "notes.md").write_text("# Notes", encoding="utf-8")

    real_skills = tmp_path / "real-skills"
    real_skills.mkdir()
    (real_skills / "SKILL.md").write_text("---\nname: linked-skill\n---\n", encoding="utf-8")

    knowledge_link = tmp_path / "knowledge-link"
    knowledge_link.symlink_to(real_knowledge, target_is_directory=True)
    skills_link = tmp_path / "skills-link"
    skills_link.symlink_to(real_skills, target_is_directory=True)

    config = AFSConfig(
        general=GeneralConfig(
            context_root=tmp_path / "context",
            agent_workspaces_dir=tmp_path / "context" / "workspaces",
        ),
        extensions=ExtensionsConfig(
            auto_discover=False,
            extension_dirs=[tmp_path / "extensions"],
        ),
        profiles=ProfilesConfig(
            active_profile="demo",
            profiles={
                "demo": ProfileConfig(
                    knowledge_mounts=[knowledge_link],
                    skill_roots=[skills_link],
                )
            },
        ),
    )

    output = tmp_path / "output"
    output.mkdir()

    result = pack_bundle("demo", config=config, output_path=output)

    bundled_knowledge = result.path / "knowledge" / knowledge_link.name
    bundled_skills = result.path / "skills" / skills_link.name

    assert bundled_knowledge.exists()
    assert bundled_skills.exists()
    assert not bundled_knowledge.is_symlink()
    assert not bundled_skills.is_symlink()
    assert (bundled_knowledge / "notes.md").read_text(encoding="utf-8") == "# Notes"
    assert (bundled_skills / "SKILL.md").read_text(encoding="utf-8").startswith("---")
