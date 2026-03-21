"""Tests for conversation handoff protocol (Feature 4)."""

from __future__ import annotations

import json
from pathlib import Path

from afs.handoff import HandoffPacket, HandoffStore


def test_handoff_packet_roundtrip() -> None:
    packet = HandoffPacket(
        session_id="abc123",
        agent_name="test-agent",
        timestamp="2026-03-21T12:00:00+00:00",
        accomplished=["built feature X"],
        blocked=["needs review"],
        next_steps=["deploy to staging"],
        context_snapshot={"open_files": ["main.py"]},
        open_tasks=[{"id": "t1", "title": "review"}],
        metadata={"version": "1.0"},
    )
    d = packet.to_dict()
    restored = HandoffPacket.from_dict(d)
    assert restored.session_id == "abc123"
    assert restored.agent_name == "test-agent"
    assert restored.accomplished == ["built feature X"]
    assert restored.blocked == ["needs review"]
    assert restored.next_steps == ["deploy to staging"]


def test_handoff_store_create_and_read(tmp_path: Path) -> None:
    ctx = tmp_path / ".context"
    ctx.mkdir()
    (ctx / "scratchpad").mkdir()
    store = HandoffStore(ctx)

    packet = store.create(
        agent_name="agent-a",
        accomplished=["task 1"],
        blocked=["blocked on review"],
        next_steps=["deploy"],
    )
    assert packet.agent_name == "agent-a"
    assert len(packet.session_id) > 0

    # Read latest
    latest = store.read()
    assert latest is not None
    assert latest.session_id == packet.session_id
    assert latest.accomplished == ["task 1"]


def test_handoff_store_read_by_session_id(tmp_path: Path) -> None:
    ctx = tmp_path / ".context"
    ctx.mkdir()
    (ctx / "scratchpad").mkdir()
    store = HandoffStore(ctx)

    p1 = store.create(agent_name="a", accomplished=["first"])
    p2 = store.create(agent_name="a", accomplished=["second"])

    read1 = store.read(session_id=p1.session_id)
    assert read1 is not None
    assert read1.accomplished == ["first"]

    read2 = store.read(session_id=p2.session_id)
    assert read2 is not None
    assert read2.accomplished == ["second"]


def test_handoff_store_list(tmp_path: Path) -> None:
    ctx = tmp_path / ".context"
    ctx.mkdir()
    (ctx / "scratchpad").mkdir()
    store = HandoffStore(ctx)

    store.create(agent_name="a", accomplished=["first"])
    store.create(agent_name="b", accomplished=["second"])
    store.create(agent_name="c", accomplished=["third"])

    packets = store.list(limit=2)
    assert len(packets) == 2
    # Most recent first
    assert packets[0].accomplished == ["third"]
    assert packets[1].accomplished == ["second"]


def test_handoff_store_read_empty(tmp_path: Path) -> None:
    ctx = tmp_path / ".context"
    ctx.mkdir()
    (ctx / "scratchpad").mkdir()
    store = HandoffStore(ctx)
    assert store.read() is None


def test_handoff_manifest_persistence(tmp_path: Path) -> None:
    ctx = tmp_path / ".context"
    ctx.mkdir()
    (ctx / "scratchpad").mkdir()
    store = HandoffStore(ctx)

    store.create(agent_name="a", session_id="s1")
    store.create(agent_name="a", session_id="s2")

    manifest_path = ctx / "scratchpad" / "handoffs" / "_manifest.json"
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest == ["s1", "s2"]
