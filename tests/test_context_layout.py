from __future__ import annotations

import argparse
import json
import shutil
import stat
from pathlib import Path

import pytest

from afs.cli._utils import AFS_DIRS, ensure_context_root
from afs.cli.core import init_command
from afs.cli.layout import register_parsers
from afs.context_layout import (
    LAYOUT_VERSION,
    LayoutMetadata,
    LayoutStateError,
    audit_layout,
    build_migration_plan,
    build_rollback_manifest,
    detect_layout_version,
    resolve_system_path,
    scaffold_v2,
    write_manifest,
)
from afs.context_paths import load_context_metadata, resolve_mount_root
from afs.manager import AFSManager
from afs.models import ContextCategory, MountType
from afs.project_registry import ProjectRegistry
from afs.schema import (
    AFSConfig,
    CognitiveConfig,
    GeneralConfig,
    ProfileConfig,
    ProfilesConfig,
)


def test_context_categories_are_paper_aligned_and_legacy_safe() -> None:
    assert [category.value for category in ContextCategory] == [
        "history",
        "memory",
        "scratchpad",
        "knowledge",
        "tools",
        "human",
    ]
    assert ContextCategory.from_mount_type(MountType.KNOWLEDGE) is ContextCategory.KNOWLEDGE
    assert ContextCategory.from_mount_type(MountType.HIVEMIND) is None


def test_scaffold_v2_creates_only_simple_top_level_and_private_metadata(tmp_path: Path) -> None:
    root = tmp_path / ".context"

    metadata = scaffold_v2(root)

    assert metadata.layout_version == LAYOUT_VERSION
    assert LayoutMetadata.load(root) == metadata
    assert {path.name for path in root.iterdir()} == {
        ".afs",
        "README.md",
        "history",
        "memory",
        "scratchpad",
        "knowledge",
        "tools",
        "human",
    }
    assert resolve_system_path(root, "messages") == root / ".afs" / "queue" / "messages"
    assert resolve_system_path(root, "messages").is_dir()
    assert resolve_system_path(root, "search").is_dir()
    assert stat.S_IMODE(root.stat().st_mode) == 0o700
    assert stat.S_IMODE((root / ".afs" / "layout.toml").stat().st_mode) == 0o600
    assert audit_layout(root).valid is True


def test_top_level_init_preserves_existing_v2_layout(
    tmp_path: Path,
) -> None:
    root = tmp_path / ".context"
    scaffold_v2(root)
    before = {path.name for path in root.iterdir()}
    args = argparse.Namespace(
        config=None,
        no_config=True,
        workspace_path=None,
        workspace_name=None,
        force=False,
        context_root=str(root),
        link_context=False,
    )

    assert init_command(args) == 0

    assert {path.name for path in root.iterdir()} == before
    assert not {"hivemind", "global", "items"}.intersection(before)
    assert audit_layout(root).valid is True


def test_top_level_init_fails_closed_for_damaged_v2_layout(tmp_path: Path) -> None:
    root = tmp_path / ".context"
    scaffold_v2(root)
    (root / ".afs" / "layout.toml").write_text(
        "layout_version = 99\n",
        encoding="utf-8",
    )
    args = argparse.Namespace(
        config=None,
        no_config=True,
        workspace_path=None,
        workspace_name=None,
        force=False,
        context_root=str(root),
        link_context=False,
    )

    with pytest.raises(LayoutStateError, match="marker is missing or invalid"):
        init_command(args)

    assert not any((root / name).exists() for name in ("hivemind", "global", "items"))
    assert audit_layout(root).valid is False


def test_top_level_init_retains_fresh_v1_directory_contract(tmp_path: Path) -> None:
    root = tmp_path / ".context"

    ensure_context_root(root)

    assert {path.name for path in root.iterdir()} == set(AFS_DIRS)
    assert detect_layout_version(root) == 1


def test_manager_explicit_v2_uses_central_root_and_compat_resolution(tmp_path: Path) -> None:
    root = tmp_path / "home" / ".context"
    project = tmp_path / "src" / "demo"
    project.mkdir(parents=True)
    manager = AFSManager(config=AFSConfig(general=GeneralConfig(context_root=root)))

    context = manager.ensure(path=project, layout_version=2)

    assert context.path == root.resolve()
    assert context.layout_version == 2
    assert context.is_valid is True
    assert context.total_mounts == 0
    assert not (root / "metadata.json").exists()
    assert (root / ".afs" / "compat" / "metadata.json").is_file()
    assert ProjectRegistry(root).resolve(project).name == "demo"  # type: ignore[union-attr]
    assert manager.resolve_mount_root(root, MountType.HIVEMIND) == root / ".afs" / "queue" / "messages"
    # Exercise the non-manager helper used by history, handoffs, and agents.
    assert resolve_mount_root(root, MountType.HIVEMIND) == root / ".afs" / "queue" / "messages"


def test_v2_context_root_rejects_linked_category_and_does_not_call_scopes_mounts(
    tmp_path: Path,
) -> None:
    root = tmp_path / ".context"
    outside = tmp_path / "outside"
    scaffold_v2(root)
    outside.mkdir()
    shutil.rmtree(root / "knowledge")
    try:
        (root / "knowledge").symlink_to(outside, target_is_directory=True)
    except OSError as exc:  # pragma: no cover - Windows without symlink privilege
        pytest.skip(f"directory symlinks unavailable: {exc}")
    manager = AFSManager(config=AFSConfig(general=GeneralConfig(context_root=root)))

    context = manager.list_context(context_path=root)

    assert context.layout_version == 2
    assert context.is_valid is False
    assert context.total_mounts == 0


def test_manager_v2_ensure_rejects_linked_internal_parent(tmp_path: Path) -> None:
    root = tmp_path / ".context"
    project = tmp_path / "project"
    outside = tmp_path / "outside"
    project.mkdir()
    outside.mkdir()
    scaffold_v2(root)
    shutil.rmtree(root / ".afs" / "queue")
    try:
        (root / ".afs" / "queue").symlink_to(outside, target_is_directory=True)
    except OSError as exc:  # pragma: no cover - Windows without symlink privilege
        pytest.skip(f"directory symlinks unavailable: {exc}")
    manager = AFSManager(config=AFSConfig(general=GeneralConfig(context_root=root)))

    with pytest.raises(ValueError, match="symbolic link or reparse point"):
        manager.ensure(path=project, context_root=root, layout_version=2)
    assert list(outside.iterdir()) == []


def test_manager_v2_cognitive_scaffold_rejects_linked_leaf(tmp_path: Path) -> None:
    root = tmp_path / ".context"
    project = tmp_path / "project"
    outside = tmp_path / "outside-state.md"
    project.mkdir()
    manager = AFSManager(
        config=AFSConfig(
            general=GeneralConfig(context_root=root),
            cognitive=CognitiveConfig(enabled=True),
        )
    )
    context = manager.ensure(path=project, context_root=root, layout_version=2)
    record = ProjectRegistry(root).resolve(project)
    assert record is not None
    state = root / "scratchpad" / "projects" / record.project_id / manager.STATE_FILE
    state.unlink()
    try:
        state.symlink_to(outside)
    except OSError as exc:  # pragma: no cover - Windows without symlink privilege
        pytest.skip(f"file symlinks unavailable: {exc}")

    with pytest.raises(ValueError, match="symbolic link or reparse point"):
        manager.ensure(path=project, context_root=context.path, layout_version=2)
    assert not outside.exists()


def test_v2_mount_routing_ignores_compat_metadata_overrides(tmp_path: Path) -> None:
    root = tmp_path / ".context"
    outside = tmp_path / "outside"
    scaffold_v2(root)
    outside.mkdir()
    metadata_path = root / ".afs" / "compat" / "metadata.json"
    metadata_path.write_text(
        json.dumps({"directories": {"items": str(outside)}}),
        encoding="utf-8",
    )

    assert resolve_mount_root(root, MountType.ITEMS) == root / ".afs" / "compat" / "items"


def test_v2_metadata_loader_rejects_linked_compat_leaf(tmp_path: Path) -> None:
    root = tmp_path / ".context"
    outside = tmp_path / "outside.json"
    scaffold_v2(root)
    outside.write_text('{"description":"outside-private"}', encoding="utf-8")
    metadata_path = root / ".afs" / "compat" / "metadata.json"
    try:
        metadata_path.symlink_to(outside)
    except OSError as exc:  # pragma: no cover - Windows without symlink privilege
        pytest.skip(f"file symlinks unavailable: {exc}")

    with pytest.raises(ValueError, match="symbolic link or reparse point"):
        load_context_metadata(root)


def test_v2_mount_routing_rejects_linklike_fixed_root(tmp_path: Path) -> None:
    root = tmp_path / ".context"
    outside = tmp_path / "outside"
    scaffold_v2(root)
    outside.mkdir()
    items = root / ".afs" / "compat" / "items"
    shutil.rmtree(items)
    items.symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="symbolic link or reparse point"):
        resolve_mount_root(root, MountType.ITEMS)


@pytest.mark.parametrize("operation", ["ensure", "init"])
def test_manager_v2_scopes_cognitive_files_and_skips_legacy_profile_mounts(
    tmp_path: Path,
    operation: str,
) -> None:
    root = tmp_path / "home" / ".context"
    project = tmp_path / "src" / "demo"
    profile_source = tmp_path / "shared-knowledge"
    project.mkdir(parents=True)
    profile_source.mkdir()
    manager = AFSManager(
        config=AFSConfig(
            general=GeneralConfig(context_root=root),
            profiles=ProfilesConfig(
                active_profile="work",
                auto_apply=True,
                profiles={
                    "work": ProfileConfig(knowledge_mounts=[profile_source]),
                },
            ),
            cognitive=CognitiveConfig(
                enabled=True,
                record_emotions=True,
                record_metacognition=True,
                record_goals=True,
                record_epistemic=True,
            ),
        )
    )

    context = getattr(manager, operation)(path=project, layout_version=2, profile="work")
    record = ProjectRegistry(root).resolve(project)

    assert context.layout_version == 2
    assert record is not None
    for category in ContextCategory:
        category_root = root / category.value
        direct_entries = list(category_root.iterdir())
        assert {entry.name for entry in direct_entries} <= {"common", "projects"}
        assert all(entry.is_dir() and not entry.is_symlink() for entry in direct_entries)

    project_scratchpad = root / "scratchpad" / "projects" / record.project_id
    assert {path.name for path in project_scratchpad.iterdir()} == {
        "deferred.md",
        "emotions.json",
        "epistemic.json",
        "goals.json",
        "metacognition.json",
        "state.md",
    }
    assert (root / "memory" / "projects" / record.project_id).is_dir()
    assert not (root / "scratchpad" / "state.md").exists()
    assert not any(path.is_symlink() for path in root.joinpath("knowledge").rglob("*"))
    health = manager.context_health(root, profile_name="work")
    assert health["healthy"] is True
    assert health["profile"]["mounts_supported"] is False
    assert len(health["profile"]["unsupported_mounts"]) == 1
    assert "replace v1 profile mounts with scoped context sources" in health[
        "suggested_actions"
    ]

    with pytest.raises(ValueError, match="not supported for layout v2"):
        manager.apply_profile(root, profile_name="work")


def test_manager_v1_keeps_legacy_cognitive_and_profile_mount_layout(tmp_path: Path) -> None:
    root = tmp_path / "project" / ".context"
    project = root.parent
    profile_source = tmp_path / "shared-knowledge"
    project.mkdir()
    profile_source.mkdir()
    manager = AFSManager(
        config=AFSConfig(
            profiles=ProfilesConfig(
                active_profile="work",
                profiles={
                    "work": ProfileConfig(knowledge_mounts=[profile_source]),
                },
            ),
            cognitive=CognitiveConfig(enabled=True),
        )
    )

    context = manager.ensure(path=project, context_root=root, profile="work")

    assert context.layout_version == 1
    assert (root / "scratchpad" / "state.md").is_file()
    assert (root / "scratchpad" / "deferred.md").is_file()
    mounts = [
        path
        for path in (root / "knowledge").iterdir()
        if path.name.startswith("profile-knowledge-work")
    ]
    assert len(mounts) == 1
    assert mounts[0].is_symlink()
    assert mounts[0].resolve() == profile_source.resolve()


def test_manager_v2_manual_mount_fails_closed_but_legacy_alias_can_be_removed(
    tmp_path: Path,
) -> None:
    root = tmp_path / "home" / ".context"
    project = tmp_path / "src" / "demo"
    source = tmp_path / "shared-knowledge"
    project.mkdir(parents=True)
    source.mkdir()
    manager = AFSManager(config=AFSConfig(general=GeneralConfig(context_root=root)))
    manager.ensure(path=project, layout_version=2)

    with pytest.raises(ValueError, match="manual filesystem mounts.*layout v2"):
        manager.mount(
            source,
            MountType.KNOWLEDGE,
            alias="docs",
            context_path=root,
        )
    assert not (root / "knowledge" / "docs").exists()

    legacy_alias = root / "knowledge" / "legacy-docs"
    legacy_alias.symlink_to(source)
    assert manager.unmount(
        "legacy-docs",
        MountType.KNOWLEDGE,
        context_path=root,
    )
    assert not legacy_alias.exists()


def test_audit_and_plan_are_read_only_and_route_legacy_messages(tmp_path: Path) -> None:
    root = tmp_path / ".context"
    (root / "hivemind").mkdir(parents=True)
    (root / "hivemind" / "message.json").write_text("{}\n", encoding="utf-8")
    (root / "memory").mkdir()
    (root / "memory" / "note.md").write_text("remember\n", encoding="utf-8")
    before = sorted(path.relative_to(root).as_posix() for path in root.rglob("*"))

    audit = audit_layout(root)
    plan = build_migration_plan(root)
    rollback = build_rollback_manifest(plan)

    after = sorted(path.relative_to(root).as_posix() for path in root.rglob("*"))
    assert before == after
    assert audit.migration_ready is True
    assert plan.ready is True
    assert plan.source_file_count == 2
    message_operation = next(operation for operation in plan.operations if operation.source.endswith("/hivemind"))
    assert message_operation.destination.endswith("/.afs/queue/messages")
    assert rollback.operations[-1].destination.endswith("/hivemind")


def test_migration_plan_scopes_category_imports_for_in_place_and_separate_roots(
    tmp_path: Path,
) -> None:
    root = tmp_path / ".context"
    (root / "memory").mkdir(parents=True)
    (root / "memory" / "note.md").write_text("legacy\n", encoding="utf-8")
    before = sorted(path.relative_to(root).as_posix() for path in root.rglob("*"))

    in_place = build_migration_plan(root)
    separate_root = tmp_path / "context-v2"
    separate = build_migration_plan(root, separate_root)

    in_place_memory = next(op for op in in_place.operations if op.source.endswith("/memory"))
    separate_memory = next(op for op in separate.operations if op.source.endswith("/memory"))
    assert in_place_memory.destination == str(root / "memory" / "common")
    assert in_place_memory.operation == "copy_verify_staged"
    assert separate_memory.destination == str(separate_root / "memory" / "common")
    assert separate_memory.operation == "copy_verify"
    assert not separate_root.exists()
    assert before == sorted(path.relative_to(root).as_posix() for path in root.rglob("*"))


def test_migration_plan_blocks_linked_top_level_source_without_reading_target(
    tmp_path: Path,
) -> None:
    root = tmp_path / ".context"
    outside = tmp_path / "outside"
    root.mkdir()
    outside.mkdir()
    (outside / "private.md").write_text("outside migration canary", encoding="utf-8")
    try:
        (root / "knowledge").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    audit = audit_layout(root)
    plan = build_migration_plan(root, tmp_path / "v2")

    assert audit.migration_ready is False
    assert any(
        issue.code == "linklike_migration_source"
        and issue.path == "knowledge"
        and issue.blocking
        for issue in audit.issues
    )
    assert plan.ready is False
    assert plan.blocking_entries == ("knowledge",)
    assert plan.source_file_count == 0
    assert not any(operation.source.endswith("/knowledge") for operation in plan.operations)


@pytest.mark.parametrize("damage", ["truncate", "delete"])
def test_v2_marker_damage_fails_closed_but_layout_audit_remains_available(
    tmp_path: Path,
    damage: str,
) -> None:
    root = tmp_path / ".context"
    scaffold_v2(root)
    marker = root / ".afs" / "layout.toml"
    if damage == "truncate":
        marker.write_text("layout_version = ", encoding="utf-8")
    else:
        marker.unlink()

    with pytest.raises(LayoutStateError, match="repair the marker before accessing"):
        detect_layout_version(root)

    audit = audit_layout(root)
    assert audit.layout_version == LAYOUT_VERSION
    assert audit.valid is False
    assert audit.migration_ready is False
    assert any(issue.code == "invalid_layout_marker" for issue in audit.issues)


def test_v2_whole_state_directory_deletion_fails_closed_from_scaffold_readme(
    tmp_path: Path,
) -> None:
    root = tmp_path / ".context"
    scaffold_v2(root)
    shutil.rmtree(root / ".afs")

    with pytest.raises(LayoutStateError, match="repair the marker before accessing"):
        detect_layout_version(root)

    audit = audit_layout(root)
    assert audit.layout_version == LAYOUT_VERSION
    assert audit.valid is False
    assert any(issue.path == ".afs" and issue.blocking for issue in audit.issues)


@pytest.mark.parametrize("readme_damage", ["delete", "edit"])
def test_v2_state_and_readme_damage_fails_closed_from_structure(
    tmp_path: Path,
    readme_damage: str,
) -> None:
    root = tmp_path / ".context"
    scaffold_v2(root)
    shutil.rmtree(root / ".afs")
    readme = root / "README.md"
    if readme_damage == "delete":
        readme.unlink()
    else:
        readme.write_text("# user-edited context notes\n", encoding="utf-8")

    with pytest.raises(LayoutStateError, match="repair the marker before accessing"):
        detect_layout_version(root)

    audit = audit_layout(root)
    assert audit.layout_version == LAYOUT_VERSION
    assert audit.valid is False
    assert any(issue.path == ".afs" and issue.blocking for issue in audit.issues)


@pytest.mark.parametrize(
    "relative",
    [
        "knowledge",
        ".afs/projects",
        ".afs/search",
        ".afs/compat/items",
    ],
)
def test_v2_audit_rejects_linklike_required_directories(
    tmp_path: Path,
    relative: str,
) -> None:
    root = tmp_path / ".context"
    outside = tmp_path / "outside"
    scaffold_v2(root)
    outside.mkdir()
    target = root / relative
    shutil.rmtree(target)
    target.symlink_to(outside, target_is_directory=True)

    audit = audit_layout(root)

    assert audit.valid is False
    assert any(
        issue.code == "linklike_required_directory"
        and issue.path == relative
        and issue.blocking
        for issue in audit.issues
    )


def test_genuine_v1_root_without_v2_sentinels_remains_supported(tmp_path: Path) -> None:
    root = tmp_path / ".context"
    (root / "memory").mkdir(parents=True)
    # ``projects`` remains a valid v1 mount alias; only the central registry
    # or a canonical prj_* scoped child proves v2 state.
    (root / "knowledge" / "projects").mkdir(parents=True)

    assert detect_layout_version(root) == 1


def test_genuine_v1_readme_is_not_a_v2_sentinel(tmp_path: Path) -> None:
    root = tmp_path / ".context"
    root.mkdir()
    (root / "README.md").write_text("# My legacy context\n", encoding="utf-8")

    assert detect_layout_version(root) == 1


def test_v1_hybrid_search_index_is_not_mistaken_for_v2_state(tmp_path: Path) -> None:
    from afs.hybrid_search import HybridSearchEngine, HybridSource

    root = tmp_path / ".context"
    source = tmp_path / "project"
    (root / "memory").mkdir(parents=True)
    source.mkdir()
    (source / "guide.md").write_text("local search", encoding="utf-8")
    HybridSearchEngine(root / ".afs" / "search").build(
        [HybridSource(source, scope_id="common")]
    )

    assert detect_layout_version(root) == 1


def test_migration_plan_blocks_unknown_entries_and_manifests_are_atomic(tmp_path: Path) -> None:
    root = tmp_path / ".context"
    root.mkdir()
    (root / "mystery").mkdir()
    (root / "mystery" / "data.txt").write_text("unknown", encoding="utf-8")

    plan = build_migration_plan(root)
    output = tmp_path / "plans" / "migration.json"
    write_manifest(output, plan)

    assert plan.ready is False
    assert plan.blocking_entries == ("mystery",)
    assert json.loads(output.read_text(encoding="utf-8"))["transaction_id"] == plan.transaction_id
    assert stat.S_IMODE(output.stat().st_mode) == 0o600


def test_layout_cli_audit_and_plan_register(tmp_path: Path) -> None:
    import argparse

    root = tmp_path / ".context"
    scaffold_v2(root)
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    register_parsers(subparsers)

    audit_args = parser.parse_args(["layout", "audit", "--context-root", str(root), "--json"])
    assert audit_args.func(audit_args) == 0
