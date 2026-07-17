from __future__ import annotations

from pathlib import Path

import pytest

from afs.hivemind import HivemindBus
from afs.messages import MessageBus


def _context(tmp_path: Path) -> Path:
    root = tmp_path / ".context"
    (root / "hivemind").mkdir(parents=True)
    return root


def test_message_bus_reads_current_and_common_scopes_only(tmp_path: Path) -> None:
    root = _context(tmp_path)
    project = MessageBus(root, scope_id="project:alpha")
    project.send("agent", "status", {"value": "project"})
    project.send("agent", "status", {"value": "shared"}, scope_id="common")
    HivemindBus(root).send(
        "agent",
        "status",
        {"value": "other"},
        scope_id="project:beta",
    )
    HivemindBus(root).send("agent", "status", {"value": "legacy"})

    assert [item.payload["value"] for item in project.read(limit=20)] == [
        "project",
        "shared",
    ]


def test_message_bus_all_projects_is_explicit_and_legacy_is_separate(
    tmp_path: Path,
) -> None:
    root = _context(tmp_path)
    legacy = HivemindBus(root)
    legacy.send("agent", "status", {"value": "legacy"})
    legacy.send("agent", "status", {"value": "beta"}, scope_id="project:beta")

    all_scoped = MessageBus(root, scope_id="project:alpha", all_projects=True)
    assert [item.payload["value"] for item in all_scoped.read()] == ["beta"]

    with_legacy = MessageBus(
        root,
        scope_id="project:alpha",
        all_projects=True,
        include_legacy=True,
    )
    assert [item.payload["value"] for item in with_legacy.read()] == [
        "legacy",
        "beta",
    ]


def test_message_bus_rejects_cross_scope_send(tmp_path: Path) -> None:
    bus = MessageBus(_context(tmp_path), scope_id="project:alpha")

    with pytest.raises(PermissionError, match="outside"):
        bus.send("agent", "status", scope_id="project:beta")


@pytest.mark.parametrize("value", ["", "has space", "../escape", "a/b"])
def test_message_bus_rejects_invalid_scope_ids(tmp_path: Path, value: str) -> None:
    with pytest.raises(ValueError):
        MessageBus(_context(tmp_path), scope_id=value)
