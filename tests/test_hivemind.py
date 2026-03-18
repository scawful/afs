"""Tests for hivemind message bus."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from afs.hivemind import HivemindBus


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
