from __future__ import annotations

from pathlib import Path

import pytest

from afs.context_fs import ContextFileSystem
from afs.manager import AFSManager
from afs.models import MountType
from afs.project_registry import ProjectRegistry
from afs.schema import AFSConfig, GeneralConfig


def test_v2_context_filesystem_rejects_unscoped_category_write(tmp_path: Path) -> None:
    context_root = tmp_path / ".context"
    project = tmp_path / "project"
    project.mkdir()
    manager = AFSManager(
        config=AFSConfig(general=GeneralConfig(context_root=context_root))
    )
    manager.ensure(
        path=project,
        context_root=context_root,
        layout_version=2,
    )

    with pytest.raises(PermissionError, match="unscoped v2 category writes"):
        ContextFileSystem(manager, context_root).write_text(
            MountType.SCRATCHPAD,
            "raw.md",
            "unsafe",
            mkdirs=True,
        )

    assert not (context_root / "scratchpad" / "raw.md").exists()


def test_v2_context_filesystem_allows_explicit_project_scope_write(tmp_path: Path) -> None:
    context_root = tmp_path / ".context"
    project = tmp_path / "project"
    project.mkdir()
    manager = AFSManager(
        config=AFSConfig(general=GeneralConfig(context_root=context_root))
    )
    manager.ensure(
        path=project,
        context_root=context_root,
        layout_version=2,
    )
    record = ProjectRegistry(context_root).resolve(project)
    assert record is not None
    scope_root = context_root / "scratchpad" / "projects" / record.project_id
    fs = ContextFileSystem(
        manager,
        context_root,
        scoped_mount_roots={MountType.SCRATCHPAD: scope_root},
    )

    target = fs.write_text(MountType.SCRATCHPAD, "note.md", "scoped", mkdirs=True)

    assert target == scope_root / "note.md"
    assert target.read_text(encoding="utf-8") == "scoped"


def test_v2_context_filesystem_normalizes_trusted_parent_alias(tmp_path: Path) -> None:
    real_parent = tmp_path / "real"
    real_parent.mkdir()
    alias_parent = tmp_path / "alias"
    try:
        alias_parent.symlink_to(real_parent, target_is_directory=True)
    except (NotImplementedError, OSError) as exc:
        pytest.skip(f"directory symlinks unavailable: {exc}")

    context_root = alias_parent / ".context"
    project = tmp_path / "project"
    project.mkdir()
    manager = AFSManager(
        config=AFSConfig(general=GeneralConfig(context_root=context_root))
    )
    manager.ensure(
        path=project,
        context_root=context_root,
        layout_version=2,
    )
    record = ProjectRegistry(context_root).resolve(project)
    assert record is not None
    aliased_scope_root = (
        context_root / "scratchpad" / "projects" / record.project_id
    )

    fs = ContextFileSystem(
        manager,
        context_root,
        scoped_mount_roots={MountType.SCRATCHPAD: aliased_scope_root},
    )

    assert fs.resolve_mount_root(MountType.SCRATCHPAD) == (
        real_parent
        / ".context"
        / "scratchpad"
        / "projects"
        / record.project_id
    )
