from __future__ import annotations

import json
import stat
from pathlib import Path

import pytest

from afs.cli.layout import register_parsers
from afs.context_layout import (
    LAYOUT_VERSION,
    LayoutMetadata,
    audit_layout,
    build_migration_plan,
    build_rollback_manifest,
    resolve_system_path,
    scaffold_v2,
    write_manifest,
)
from afs.context_paths import resolve_mount_root
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


def test_manager_explicit_v2_uses_central_root_and_compat_resolution(tmp_path: Path) -> None:
    root = tmp_path / "home" / ".context"
    project = tmp_path / "src" / "demo"
    project.mkdir(parents=True)
    manager = AFSManager(config=AFSConfig(general=GeneralConfig(context_root=root)))

    context = manager.ensure(path=project, layout_version=2)

    assert context.path == root.resolve()
    assert context.layout_version == 2
    assert context.is_valid is True
    assert not (root / "metadata.json").exists()
    assert (root / ".afs" / "compat" / "metadata.json").is_file()
    assert ProjectRegistry(root).resolve(project).name == "demo"  # type: ignore[union-attr]
    assert manager.resolve_mount_root(root, MountType.HIVEMIND) == root / ".afs" / "queue" / "messages"
    # Exercise the non-manager helper used by history, handoffs, and agents.
    assert resolve_mount_root(root, MountType.HIVEMIND) == root / ".afs" / "queue" / "messages"


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
