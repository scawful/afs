from __future__ import annotations

from pathlib import Path

from afs.manager import AFSManager
from afs.models import MountType
from afs.profiles import apply_profile_mounts, resolve_active_profile
from afs.schema import (
    AFSConfig,
    AgentConfig,
    DirectoryConfig,
    ExtensionsConfig,
    GeneralConfig,
    ProfileConfig,
    ProfilesConfig,
    default_directory_configs,
)


def test_resolve_profile_with_extension(tmp_path: Path) -> None:
    ext_root = tmp_path / "extensions"
    extension_dir = ext_root / "afs_google_test"
    extension_dir.mkdir(parents=True)

    work_knowledge = extension_dir / "knowledge"
    work_skills = extension_dir / "skills"
    work_registry = extension_dir / "registry"
    work_knowledge.mkdir()
    work_skills.mkdir()
    work_registry.mkdir()

    (extension_dir / "extension.toml").write_text(
        "name = \"afs_google_test\"\n"
        "knowledge_mounts = [\"knowledge\"]\n"
        "skill_roots = [\"skills\"]\n"
        "model_registries = [\"registry\"]\n"
        "policies = [\"no_zelda\"]\n",
        encoding="utf-8",
    )

    context_root = tmp_path / "context"
    general = GeneralConfig(
        context_root=context_root,
        agent_workspaces_dir=context_root / "workspaces",
    )
    profiles = ProfilesConfig(
        active_profile="work",
        auto_apply=True,
        profiles={
            "work": ProfileConfig(
                enabled_extensions=["afs_google_test"],
            )
        },
    )
    config = AFSConfig(
        general=general,
        profiles=profiles,
        extensions=ExtensionsConfig(
            enabled_extensions=["afs_google_test"],
            extension_dirs=[ext_root],
        ),
    )

    resolved = resolve_active_profile(config)
    assert resolved.name == "work"
    assert "no_zelda" in resolved.policies
    assert work_knowledge.resolve() in resolved.knowledge_mounts
    assert work_skills.resolve() in resolved.skill_roots
    assert work_registry.resolve() in resolved.model_registries


def test_manager_applies_profile_mounts(tmp_path: Path) -> None:
    knowledge_src = tmp_path / "knowledge-src"
    skill_src = tmp_path / "skills-src"
    registry_src = tmp_path / "registry-src"
    knowledge_src.mkdir()
    skill_src.mkdir()
    registry_src.mkdir()

    context_root = tmp_path / "context"
    general = GeneralConfig(
        context_root=context_root,
        agent_workspaces_dir=context_root / "workspaces",
    )

    profiles = ProfilesConfig(
        active_profile="work",
        auto_apply=True,
        profiles={
            "work": ProfileConfig(
                knowledge_mounts=[knowledge_src],
                skill_roots=[skill_src],
                model_registries=[registry_src],
            )
        },
    )

    manager = AFSManager(config=AFSConfig(general=general, profiles=profiles))

    project = tmp_path / "project"
    project.mkdir()
    context = manager.ensure(path=project, context_root=context_root, profile="work")

    knowledge_mounts = context.get_mounts(MountType.KNOWLEDGE)
    tool_mounts = context.get_mounts(MountType.TOOLS)
    global_mounts = context.get_mounts(MountType.GLOBAL)

    assert any(m.name.startswith("profile-knowledge-work") for m in knowledge_mounts)
    assert any(m.name.startswith("profile-skill-work") for m in tool_mounts)
    assert any(m.name.startswith("profile-registry-work") for m in global_mounts)


def test_resolve_profile_does_not_auto_load_unrequested_extensions(tmp_path: Path) -> None:
    ext_root = tmp_path / "extensions"
    extension_dir = ext_root / "afs_google_test"
    extension_dir.mkdir(parents=True)
    (extension_dir / "knowledge").mkdir()
    (extension_dir / "extension.toml").write_text(
        "name = \"afs_google_test\"\n"
        "knowledge_mounts = [\"knowledge\"]\n",
        encoding="utf-8",
    )

    config = AFSConfig(
        general=GeneralConfig(context_root=tmp_path / "context"),
        extensions=ExtensionsConfig(extension_dirs=[ext_root], auto_discover=True),
    )

    resolved = resolve_active_profile(config)

    assert resolved.enabled_extensions == []
    assert resolved.knowledge_mounts == []


def test_profile_merges_mcp_tools_through_inheritance() -> None:
    profiles = ProfilesConfig(
        active_profile="child",
        profiles={
            "base": ProfileConfig(
                mcp_tools=["base.tools"],
                cli_modules=["base.cli"],
                agent_configs=[AgentConfig(name="base-agent", module="base.mod")],
            ),
            "child": ProfileConfig(
                inherits=["base"],
                mcp_tools=["child.tools"],
                cli_modules=["child.cli"],
                agent_configs=[AgentConfig(name="child-agent", module="child.mod")],
            ),
        },
    )
    config = AFSConfig(profiles=profiles)
    resolved = resolve_active_profile(config)

    assert "base.tools" in resolved.mcp_tools
    assert "child.tools" in resolved.mcp_tools
    assert "base.cli" in resolved.cli_modules
    assert "child.cli" in resolved.cli_modules
    assert len(resolved.agent_configs) == 2
    agent_names = {a.name for a in resolved.agent_configs}
    assert "base-agent" in agent_names
    assert "child-agent" in agent_names


def test_profile_agent_configs_child_wins() -> None:
    profiles = ProfilesConfig(
        active_profile="child",
        profiles={
            "base": ProfileConfig(
                agent_configs=[AgentConfig(name="shared", module="base.mod", role="observer")],
            ),
            "child": ProfileConfig(
                inherits=["base"],
                agent_configs=[AgentConfig(name="shared", module="child.mod", role="worker")],
            ),
        },
    )
    config = AFSConfig(profiles=profiles)
    resolved = resolve_active_profile(config)

    assert len(resolved.agent_configs) == 1
    assert resolved.agent_configs[0].module == "child.mod"
    assert resolved.agent_configs[0].role == "worker"


def test_resolved_profile_new_fields_default_empty() -> None:
    config = AFSConfig()
    resolved = resolve_active_profile(config)
    assert resolved.mcp_tools == []
    assert resolved.cli_modules == []
    assert resolved.agent_configs == []


def test_profile_mount_guard_uses_resolved_mount_directory_names(tmp_path: Path) -> None:
    custom_dirs = [
        DirectoryConfig(
            name="docs" if directory.role == MountType.KNOWLEDGE else directory.name,
            policy=directory.policy,
            description=directory.description,
            role=directory.role,
        )
        for directory in default_directory_configs()
    ]

    source = tmp_path / "context" / "docs" / "nested-source"
    source.mkdir(parents=True)

    config = AFSConfig(
        general=GeneralConfig(context_root=tmp_path / "context"),
        directories=custom_dirs,
        profiles=ProfilesConfig(
            active_profile="work",
            profiles={
                "work": ProfileConfig(knowledge_mounts=[source]),
            },
        ),
    )
    manager = AFSManager(config=config)
    project = tmp_path / "project"
    project.mkdir()
    context_path = manager.ensure(path=project, context_root=tmp_path / "context").path

    profile = resolve_active_profile(config, profile_name="work")
    result = apply_profile_mounts(manager, context_path, profile)

    assert result.mounted["knowledge"] == 0
    assert str(source.resolve()) in result.skipped_missing
