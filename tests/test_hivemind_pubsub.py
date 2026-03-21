"""Tests for hivemind pub/sub (Feature 1)."""

from __future__ import annotations

import json
from pathlib import Path

from afs.hivemind import HivemindBus, HivemindMessage, HivemindSubscription


def test_message_topic_field(tmp_path: Path) -> None:
    ctx = tmp_path / ".context"
    ctx.mkdir()
    (ctx / "hivemind").mkdir()
    bus = HivemindBus(ctx)
    msg = bus.send("agent-a", "finding", {"key": "value"}, topic="context:repair")
    assert msg.topic == "context:repair"
    # Verify persisted
    agent_dir = ctx / "hivemind" / "agent-a"
    files = list(agent_dir.glob("*.json"))
    data = json.loads(files[0].read_text(encoding="utf-8"))
    assert data["topic"] == "context:repair"


def test_message_topic_none_by_default(tmp_path: Path) -> None:
    ctx = tmp_path / ".context"
    ctx.mkdir()
    (ctx / "hivemind").mkdir()
    bus = HivemindBus(ctx)
    msg = bus.send("agent-a", "status", {})
    assert msg.topic is None
    d = msg.to_dict()
    assert "topic" not in d


def test_message_from_dict_backward_compat() -> None:
    data = {"id": "x", "from": "a", "to": None, "type": "status", "payload": {}, "timestamp": "t"}
    msg = HivemindMessage.from_dict(data)
    assert msg.topic is None


def test_read_filters_by_topic(tmp_path: Path) -> None:
    ctx = tmp_path / ".context"
    ctx.mkdir()
    (ctx / "hivemind").mkdir()
    bus = HivemindBus(ctx)
    bus.send("agent-a", "finding", {}, topic="context:repair")
    bus.send("agent-a", "status", {}, topic="agent:lifecycle")
    bus.send("agent-a", "status", {})

    repair = bus.read(topic="context:repair")
    assert len(repair) == 1
    assert repair[0].topic == "context:repair"

    all_msgs = bus.read()
    assert len(all_msgs) == 3


def test_subscribe_creates_file(tmp_path: Path) -> None:
    ctx = tmp_path / ".context"
    ctx.mkdir()
    (ctx / "hivemind").mkdir()
    bus = HivemindBus(ctx)
    sub = bus.subscribe("agent-a", ["context:repair", "task:completed"])
    assert set(sub.topics) == {"context:repair", "task:completed"}
    assert sub.agent_name == "agent-a"
    sub_path = ctx / "hivemind" / ".subscriptions" / "agent-a.json"
    assert sub_path.exists()


def test_subscribe_merges_topics(tmp_path: Path) -> None:
    ctx = tmp_path / ".context"
    ctx.mkdir()
    (ctx / "hivemind").mkdir()
    bus = HivemindBus(ctx)
    bus.subscribe("agent-a", ["topic-a"])
    sub = bus.subscribe("agent-a", ["topic-b"])
    assert set(sub.topics) == {"topic-a", "topic-b"}


def test_unsubscribe_removes_topics(tmp_path: Path) -> None:
    ctx = tmp_path / ".context"
    ctx.mkdir()
    (ctx / "hivemind").mkdir()
    bus = HivemindBus(ctx)
    bus.subscribe("agent-a", ["topic-a", "topic-b"])
    sub = bus.unsubscribe("agent-a", ["topic-a"])
    assert sub.topics == ["topic-b"]


def test_get_subscriptions(tmp_path: Path) -> None:
    ctx = tmp_path / ".context"
    ctx.mkdir()
    (ctx / "hivemind").mkdir()
    bus = HivemindBus(ctx)
    assert bus.get_subscriptions("agent-a") is None
    bus.subscribe("agent-a", ["topic-a"])
    sub = bus.get_subscriptions("agent-a")
    assert sub is not None
    assert sub.topics == ["topic-a"]


def test_read_subscribed(tmp_path: Path) -> None:
    ctx = tmp_path / ".context"
    ctx.mkdir()
    (ctx / "hivemind").mkdir()
    bus = HivemindBus(ctx)
    bus.subscribe("agent-b", ["context:repair"])

    bus.send("agent-a", "finding", {"x": 1}, topic="context:repair")
    bus.send("agent-a", "status", {"x": 2}, topic="agent:lifecycle")
    bus.send("agent-a", "status", {"x": 3})  # topicless broadcast
    bus.send("agent-a", "request", {"x": 4}, to="agent-b")  # direct

    msgs = bus.read_subscribed("agent-b")
    assert len(msgs) == 3  # context:repair + broadcast + direct


def test_subscription_to_dict_from_dict() -> None:
    sub = HivemindSubscription(
        agent_name="a", topics=["t1", "t2"], created_at="c", updated_at="u"
    )
    d = sub.to_dict()
    restored = HivemindSubscription.from_dict(d)
    assert restored.agent_name == "a"
    assert restored.topics == ["t1", "t2"]
