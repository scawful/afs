from __future__ import annotations

from pathlib import Path

from afs.manager import AFSManager
from afs.models import MountType
from afs.schema import AFSConfig, GeneralConfig


def _make_manager(tmp_path: Path) -> AFSManager:
    context_root = tmp_path / "context"
    general = GeneralConfig(
        context_root=context_root,
        agent_workspaces_dir=context_root / "workspaces",
    )
    return AFSManager(config=AFSConfig(general=general))


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

    removed = manager.unmount(mount.name, MountType.KNOWLEDGE, context_path=context.path)
    assert removed
    assert not mount_path.exists()
