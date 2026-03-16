from __future__ import annotations

import json
from pathlib import Path

from afs.manager import AFSManager
from afs.models import MountType
from afs.schema import (
    AFSConfig,
    GeneralConfig,
    ProfileConfig,
    ProfilesConfig,
    WorkspaceDirectory,
)


def _make_manager(tmp_path: Path) -> AFSManager:
    context_root = tmp_path / "context"
    general = GeneralConfig(
        context_root=context_root,
        agent_workspaces_dir=context_root / "workspaces",
    )
    return AFSManager(config=AFSConfig(general=general))


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


def test_ensure_creates_context_and_metadata(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    project_path = tmp_path / "project"
    project_path.mkdir()

    context = manager.ensure(path=project_path, context_root=tmp_path / "context")

    assert context.path.exists()
    assert (context.path / "metadata.json").exists()
    assert (context.path / "memory").exists()
    assert (context.path / "knowledge").exists()


def test_ensure_with_link_creates_symlink(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    project_path = tmp_path / "project"
    project_path.mkdir()
    context_root = tmp_path / "context"

    manager.ensure(path=project_path, context_root=context_root, link_context=True)

    link_path = project_path / ".context"
    assert link_path.is_symlink()
    assert link_path.resolve() == context_root.resolve()


def test_mount_and_unmount(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    project_path = tmp_path / "project"
    project_path.mkdir()
    context_root = tmp_path / "context"

    context = manager.ensure(path=project_path, context_root=context_root)

    source_dir = tmp_path / "source"
    source_dir.mkdir()

    mount = manager.mount(
        source_dir,
        MountType.KNOWLEDGE,
        context_path=context.path,
    )

    mount_path = context.path / "knowledge" / mount.name
    assert mount_path.is_symlink()
    assert mount_path.resolve() == source_dir.resolve()
    listed = manager.list_context(context_path=context.path)
    listed_mount = listed.get_mounts(MountType.KNOWLEDGE)[0]
    assert listed_mount.provenance is not None
    assert listed_mount.provenance.managed_by == "manual"

    removed = manager.unmount(mount.name, MountType.KNOWLEDGE, context_path=context.path)
    assert removed
    assert not mount_path.exists()
    assert manager.list_context(context_path=context.path).metadata.iter_mount_provenance() == []


def test_mount_rejects_nested_alias_and_duplicate_source(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    project_path = tmp_path / "project"
    project_path.mkdir()
    context = manager.ensure(path=project_path, context_root=tmp_path / "context")

    source_dir = tmp_path / "source"
    source_dir.mkdir()

    manager.mount(source_dir, MountType.KNOWLEDGE, alias="docs", context_path=context.path)

    try:
        manager.mount(source_dir, MountType.KNOWLEDGE, alias="docs-copy", context_path=context.path)
    except FileExistsError:
        pass
    else:
        raise AssertionError("expected duplicate source mount to fail")

    try:
        manager.mount(
            tmp_path,
            MountType.KNOWLEDGE,
            alias="nested/docs",
            context_path=context.path,
        )
    except ValueError:
        pass
    else:
        raise AssertionError("expected nested alias mount to fail")


def test_context_health_reports_broken_and_profile_mount_issues(
    tmp_path: Path,
    monkeypatch,
) -> None:
    _clear_profile_env(monkeypatch)
    knowledge_src = tmp_path / "knowledge-src"
    knowledge_src.mkdir()

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
                knowledge_mounts=[knowledge_src, tmp_path / "missing-src"],
            )
        },
    )
    manager = AFSManager(config=AFSConfig(general=general, profiles=profiles))

    project_path = tmp_path / "project"
    project_path.mkdir()
    context = manager.ensure(path=project_path, context_root=context_root, profile="work")

    broken_mount = manager.mount(
        knowledge_src,
        MountType.TOOLS,
        alias="temp-tool",
        context_path=context.path,
    )
    mount_path = context.path / "tools" / broken_mount.name
    knowledge_src.rmdir()

    health = manager.context_health(context.path, profile_name="work")

    assert health["healthy"] is False
    assert any(entry["name"] == "temp-tool" for entry in health["broken_mounts"])
    assert len(health["profile"]["missing_sources"]) == 2
    assert "restore or update missing profile source paths" in health["suggested_actions"]
    assert mount_path.is_symlink()


def test_repair_context_seeds_provenance_and_remaps_missing_mount(
    tmp_path: Path,
) -> None:
    workspace_root = tmp_path / "workspaces" / "current"
    workspace_root.mkdir(parents=True)
    old_source = tmp_path / "legacy-docs"
    old_source.mkdir()
    new_source = workspace_root / old_source.name
    new_source.mkdir()
    (new_source / "README.md").write_text("relocated docs", encoding="utf-8")

    context_root = tmp_path / "context"
    manager = AFSManager(
        config=AFSConfig(
            general=GeneralConfig(
                context_root=context_root,
                agent_workspaces_dir=context_root / "workspaces",
                workspace_directories=[WorkspaceDirectory(path=workspace_root)],
            )
        )
    )
    project = tmp_path / "project"
    project.mkdir()
    context = manager.ensure(path=project, context_root=context_root)
    manager.mount(old_source, MountType.KNOWLEDGE, alias="docs", context_path=context.path)

    metadata_path = context.path / "metadata.json"
    metadata = manager.list_context(context_path=context.path).metadata.to_dict()
    metadata["mount_provenance"] = {}
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    old_source.rmdir()

    repair = manager.repair_context(context.path)

    assert repair["provenance_seeded"] == 1
    assert len(repair["remapped_mounts"]) == 1
    repaired = manager.list_context(context_path=context.path)
    repaired_mount = repaired.get_mounts(MountType.KNOWLEDGE)[0]
    assert repaired_mount.source == new_source.resolve()
    assert repaired_mount.provenance is not None
    assert repaired_mount.provenance.remapped_from == old_source.resolve(strict=False)
    assert repair["health_after"]["healthy"] is True


def test_repair_context_prunes_stale_provenance(tmp_path: Path) -> None:
    manager = _make_manager(tmp_path)
    project_path = tmp_path / "project"
    project_path.mkdir()
    context = manager.ensure(path=project_path, context_root=tmp_path / "context")
    source_dir = tmp_path / "source"
    source_dir.mkdir()
    mount = manager.mount(source_dir, MountType.KNOWLEDGE, context_path=context.path)
    manager.unmount(
        mount.name,
        MountType.KNOWLEDGE,
        context_path=context.path,
        keep_provenance=True,
    )

    repair = manager.repair_context(context.path)

    assert repair["provenance_pruned"] == 1
    assert repair["health_after"]["provenance"]["stale_records"] == []
