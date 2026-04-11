from __future__ import annotations

from pathlib import Path

from afs.discovery import discover_contexts, get_project_stats
from afs.manager import AFSManager
from afs.schema import AFSConfig, GeneralConfig


def test_discover_contexts_ignores_names(tmp_path: Path) -> None:
    root = tmp_path / "root"
    root.mkdir()

    alpha = root / "alpha"
    alpha.mkdir()

    legacy = root / "legacy"
    legacy.mkdir()
    beta = legacy / "beta"
    beta.mkdir()

    config = AFSConfig(
        general=GeneralConfig(context_root=tmp_path / "context"),
    )
    manager = AFSManager(config=config)

    manager.ensure(path=alpha)
    manager.ensure(path=beta)

    contexts = discover_contexts(search_paths=[root], config=config, max_depth=2)
    names = [context.project_name for context in contexts]

    assert "alpha" in names
    assert "beta" not in names

    stats = get_project_stats(contexts)
    assert stats["total_projects"] == 1


def test_discover_contexts_stops_at_context_boundary_by_default(tmp_path: Path) -> None:
    root = tmp_path / "root"
    parent = root / "parent"
    child = parent / "docs" / "dev"
    child.mkdir(parents=True)

    config = AFSConfig(
        general=GeneralConfig(context_root=tmp_path / "context"),
    )
    manager = AFSManager(config=config)

    manager.ensure(path=parent)
    manager.ensure(path=child)

    contexts = discover_contexts(search_paths=[root], config=config, max_depth=4)

    assert [context.project_name for context in contexts] == ["parent"]


def test_discover_contexts_can_include_nested_contexts(tmp_path: Path) -> None:
    root = tmp_path / "root"
    parent = root / "parent"
    child = parent / "docs" / "dev"
    child.mkdir(parents=True)

    config = AFSConfig(
        general=GeneralConfig(context_root=tmp_path / "context"),
    )
    manager = AFSManager(config=config)

    manager.ensure(path=parent)
    manager.ensure(path=child)

    contexts = discover_contexts(
        search_paths=[root],
        config=config,
        max_depth=4,
        include_nested=True,
    )

    assert [context.project_name for context in contexts] == ["dev", "parent"]
