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
