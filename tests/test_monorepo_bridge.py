from __future__ import annotations

import os
import time
from pathlib import Path

from afs.monorepo_bridge import get_workspace_bridge_status


def test_workspace_bridge_staleness_detection(tmp_path: Path) -> None:
    context_root = tmp_path / ".context"
    bridge_dir = context_root / "monorepo"
    bridge_dir.mkdir(parents=True)
    bridge_file = bridge_dir / "active_workspace.toml"
    bridge_file.write_text(
        'active_workspace = "/tmp/workspace"\n',
        encoding="utf-8",
    )

    old = time.time() - 7200
    os.utime(bridge_file, (old, old))

    status = get_workspace_bridge_status(context_root)
    assert status.exists
    assert status.stale is True
    assert status.age_seconds is not None
    assert status.age_seconds > 3600


def test_workspace_bridge_non_stale_when_recent(tmp_path: Path) -> None:
    context_root = tmp_path / ".context"
    bridge_dir = context_root / "monorepo"
    bridge_dir.mkdir(parents=True)
    bridge_file = bridge_dir / "active_workspace.toml"
    bridge_file.write_text(
        'active_workspace = "/tmp/workspace"\n',
        encoding="utf-8",
    )

    status = get_workspace_bridge_status(context_root)
    assert status.exists
    assert status.stale is False
