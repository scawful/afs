"""Tests for hivemind message bus."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

from afs.context_layout import scaffold_v2
from afs.hivemind import HivemindBus


def _symlink_or_skip(link: Path, target: Path, *, directory: bool = False) -> None:
    try:
        link.symlink_to(target, target_is_directory=directory)
    except OSError as exc:  # pragma: no cover - Windows without symlink privilege
        pytest.skip(f"symlinks unavailable: {exc}")


def test_hivemind_send_creates_file(tmp_path: Path) -> None:
    ctx = tmp_path / ".context"
    ctx.mkdir()
    (ctx / "hivemind").mkdir()
    bus = HivemindBus(ctx)
    msg = bus.send("agent-a", "finding", {"key": "value"})
    assert msg.from_agent == "agent-a"
    assert msg.msg_type == "finding"
    assert msg.payload == {"key": "value"}
    # File should exist
    agent_dir = ctx / "hivemind" / "agent-a"
    assert agent_dir.exists()
    files = list(agent_dir.glob("*.json"))
    assert len(files) == 1


def test_hivemind_send_publishes_final_file_with_atomic_replace(
    tmp_path: Path,
    monkeypatch,
) -> None:
    ctx = tmp_path / ".context"
    ctx.mkdir()
    (ctx / "hivemind").mkdir()
    original_replace = os.replace
    publishes: list[tuple[Path, Path]] = []

    def _track_replace(source, target) -> None:
        source_path = Path(source)
        target_path = Path(target)
        if target_path.parent.name == "agent-a" and target_path.suffix == ".json":
            assert source_path.exists()
            assert source_path.suffix == ".tmp"
            assert not target_path.exists()
            publishes.append((source_path, target_path))
        original_replace(source, target)

    monkeypatch.setattr(os, "replace", _track_replace)
    HivemindBus(ctx).send("agent-a", "status", {})

    assert len(publishes) == 1
    source, target = publishes[0]
    assert not source.exists()
    assert target.exists()


def test_hivemind_read_filters_by_agent(tmp_path: Path) -> None:
    ctx = tmp_path / ".context"
    ctx.mkdir()
    (ctx / "hivemind").mkdir()
    bus = HivemindBus(ctx)
    bus.send("agent-a", "status", {"msg": "hello"})
    bus.send("agent-b", "status", {"msg": "world"})

    a_msgs = bus.read(agent_name="agent-a")
    assert len(a_msgs) == 1
    assert a_msgs[0].from_agent == "agent-a"

    all_msgs = bus.read()
    assert len(all_msgs) == 2


def test_hivemind_read_filters_by_type(tmp_path: Path) -> None:
    ctx = tmp_path / ".context"
    ctx.mkdir()
    (ctx / "hivemind").mkdir()
    bus = HivemindBus(ctx)
    bus.send("agent-a", "finding", {"x": 1})
    bus.send("agent-a", "request", {"x": 2})

    findings = bus.read(msg_type="finding")
    assert len(findings) == 1
    assert findings[0].msg_type == "finding"


def test_hivemind_read_for_recipient(tmp_path: Path) -> None:
    ctx = tmp_path / ".context"
    ctx.mkdir()
    (ctx / "hivemind").mkdir()
    bus = HivemindBus(ctx)
    bus.send("agent-a", "request", {"task": "build"}, to="agent-b")
    bus.send("agent-a", "status", {"done": True})  # broadcast

    b_msgs = bus.read_for("agent-b")
    assert len(b_msgs) == 2  # targeted + broadcast


def test_hivemind_cleanup(tmp_path: Path) -> None:
    ctx = tmp_path / ".context"
    ctx.mkdir()
    (ctx / "hivemind").mkdir()
    bus = HivemindBus(ctx)
    bus.send("agent-a", "status", {})
    # With max_age_hours=0, everything should be removed
    removed = bus.cleanup(max_age_hours=0)
    assert removed >= 1


def test_hivemind_empty_read(tmp_path: Path) -> None:
    ctx = tmp_path / ".context"
    ctx.mkdir()
    (ctx / "hivemind").mkdir()
    bus = HivemindBus(ctx)
    assert bus.read() == []


def test_hivemind_respects_allowed_mounts(tmp_path: Path, monkeypatch) -> None:
    ctx = tmp_path / ".context"
    ctx.mkdir()
    (ctx / "hivemind").mkdir()
    monkeypatch.setenv("AFS_AGENT_NAME", "hive-agent")
    monkeypatch.setenv("AFS_ALLOWED_MOUNTS", "scratchpad")

    with pytest.raises(PermissionError, match="not allowed to access hivemind"):
        HivemindBus(ctx)


def test_hivemind_uses_remapped_mount(tmp_path: Path) -> None:
    ctx = tmp_path / ".context"
    hive_root = ctx / "bus"
    hive_root.mkdir(parents=True)
    (ctx / "metadata.json").write_text(
        json.dumps({"directories": {"hivemind": "bus"}}),
        encoding="utf-8",
    )

    bus = HivemindBus(ctx)
    msg = bus.send("agent-a", "status", {"state": "ok"})

    assert (hive_root / "agent-a" / f"{msg.id}.json").exists()


def test_hivemind_rejects_unsafe_agent_path_names(tmp_path: Path) -> None:
    ctx = tmp_path / ".context"
    hive_root = ctx / "hivemind"
    hive_root.mkdir(parents=True)
    bus = HivemindBus(ctx)
    absolute_target = tmp_path / "absolute-agent"
    unsafe_names = [
        str(absolute_target),
        "../outside-agent",
        "nested/agent",
        r"nested\agent",
    ]
    operations = [
        lambda name: bus.send(name, "status", {}),
        lambda name: bus.read(agent_name=name),
        lambda name: bus.subscribe(name, ["topic"]),
        lambda name: bus.unsubscribe(name, ["topic"]),
        bus.get_subscriptions,
    ]

    for unsafe_name in unsafe_names:
        for operation in operations:
            with pytest.raises(ValueError, match="single path component"):
                operation(unsafe_name)

    assert not absolute_target.exists()
    assert not (ctx / "outside-agent").exists()
    assert not (hive_root / "nested").exists()
    assert not (hive_root / r"nested\agent").exists()
    assert not (hive_root / ".subscriptions").exists()


def test_hivemind_read_cannot_escape_queue(tmp_path: Path) -> None:
    ctx = tmp_path / ".context"
    (ctx / "hivemind").mkdir(parents=True)
    outside_agent = tmp_path / "outside-agent"
    outside_agent.mkdir()
    (outside_agent / "message.json").write_text(
        json.dumps(
            {
                "id": "outside",
                "from": "outside-agent",
                "to": None,
                "type": "status",
                "payload": {"secret": True},
                "timestamp": "2026-01-01T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    bus = HivemindBus(ctx)

    with pytest.raises(ValueError, match="single path component"):
        bus.read(agent_name=str(outside_agent))
    with pytest.raises(ValueError, match="single path component"):
        bus.read(agent_name="../../outside-agent")


def test_v2_hivemind_rejects_replaced_queue_root_for_read_and_write(
    tmp_path: Path,
) -> None:
    ctx = tmp_path / ".context"
    scaffold_v2(ctx)
    bus = HivemindBus(ctx)
    queue_root = ctx / ".afs" / "queue" / "messages"
    queue_root.rmdir()
    outside = tmp_path / "outside"
    outside.mkdir()
    _symlink_or_skip(queue_root, outside, directory=True)

    with pytest.raises(ValueError, match="symbolic link or reparse point"):
        bus.read()
    with pytest.raises(ValueError, match="symbolic link or reparse point"):
        bus.send("agent-a", "status", {"secret": "must stay local"})

    assert list(outside.iterdir()) == []


def test_v2_hivemind_rejects_linked_agent_directory_for_read_and_write(
    tmp_path: Path,
) -> None:
    ctx = tmp_path / ".context"
    scaffold_v2(ctx)
    outside = tmp_path / "outside-agent"
    outside.mkdir()
    outside_message = outside / "outside.json"
    outside_message.write_text(
        json.dumps(
            {
                "id": "outside",
                "from": "agent-a",
                "to": None,
                "type": "status",
                "payload": {"secret": True},
                "timestamp": "2026-01-01T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    agent_dir = ctx / ".afs" / "queue" / "messages" / "agent-a"
    _symlink_or_skip(agent_dir, outside, directory=True)
    bus = HivemindBus(ctx)

    with pytest.raises(ValueError, match="symbolic link or reparse point"):
        bus.read(agent_name="agent-a")
    with pytest.raises(ValueError, match="symbolic link or reparse point"):
        bus.send("agent-a", "status", {"overwrite": True})

    assert json.loads(outside_message.read_text(encoding="utf-8"))["payload"] == {
        "secret": True
    }


def test_v2_hivemind_rejects_linked_message_leaf_for_read_and_cleanup(
    tmp_path: Path,
) -> None:
    ctx = tmp_path / ".context"
    scaffold_v2(ctx)
    agent_dir = ctx / ".afs" / "queue" / "messages" / "agent-a"
    agent_dir.mkdir()
    outside_message = tmp_path / "outside-message.json"
    outside_message.write_text(
        json.dumps(
            {
                "id": "outside",
                "from": "agent-a",
                "to": None,
                "type": "status",
                "payload": {"secret": True},
                "timestamp": "2026-01-01T00:00:00+00:00",
            }
        ),
        encoding="utf-8",
    )
    _symlink_or_skip(agent_dir / "linked.json", outside_message)
    bus = HivemindBus(ctx)

    with pytest.raises(ValueError, match="symbolic link or reparse point"):
        bus.read(agent_name="agent-a")
    with pytest.raises(ValueError, match="symbolic link or reparse point"):
        bus.cleanup_stats(max_age_hours=0)

    assert outside_message.exists()
    assert json.loads(outside_message.read_text(encoding="utf-8"))["payload"] == {
        "secret": True
    }
