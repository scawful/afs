"""Tests for hivemind TTL configuration and reaper integration."""

from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from afs.hivemind import HivemindBus, HivemindSubscription
from afs.schema import AFSConfig, HivemindConfig


@pytest.fixture
def hivemind_context(tmp_path, monkeypatch):
    """Set up a minimal context with hivemind directory."""
    monkeypatch.setenv("AFS_ALLOWED_MOUNTS", "hivemind")
    context = tmp_path / ".context"
    context.mkdir()
    (context / "hivemind").mkdir()
    return context


def _write_message_field(msg_path: Path, **updates: str) -> None:
    data = json.loads(msg_path.read_text(encoding="utf-8"))
    data.update(updates)
    msg_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _create_old_message(bus: HivemindBus, agent: str, *, age_hours: int = 48) -> None:
    bus.send(agent, "status", {"info": "old"}, ttl_hours=age_hours * 2)
    agent_dir = bus._root / agent
    for f in agent_dir.glob("*.json"):
        old_time = time.time() - (age_hours * 3600)
        os.utime(f, (old_time, old_time))


def test_send_sets_expires_at_when_ttl_enabled(hivemind_context):
    bus = HivemindBus(
        hivemind_context,
        config=AFSConfig(hivemind=HivemindConfig(default_ttl_hours=12)),
    )
    msg = bus.send("agent-a", "status", {"info": "fresh"})

    assert msg.expires_at is not None
    assert msg.topic is None


def test_read_skips_expired_messages(hivemind_context):
    bus = HivemindBus(
        hivemind_context,
        config=AFSConfig(hivemind=HivemindConfig(default_ttl_hours=12)),
    )
    msg = bus.send("agent-a", "status", {"info": "old"})
    msg_path = bus._root / "agent-a" / f"{msg.id}.json"
    expired_at = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    _write_message_field(msg_path, expires_at=expired_at)

    messages = bus.read(limit=10)
    assert messages == []


def test_read_subscribed_uses_subscription_ttl_window(hivemind_context):
    bus = HivemindBus(
        hivemind_context,
        config=AFSConfig(hivemind=HivemindConfig(default_ttl_hours=72)),
    )
    msg = bus.send("agent-a", "status", {"info": "old"}, topic="repair")
    msg_path = bus._root / "agent-a" / f"{msg.id}.json"
    old_timestamp = (datetime.now(timezone.utc) - timedelta(hours=24)).isoformat()
    future_expiry = (datetime.now(timezone.utc) + timedelta(hours=24)).isoformat()
    _write_message_field(msg_path, timestamp=old_timestamp, expires_at=future_expiry)

    sub = bus.subscribe("agent-b", ["repair"], ttl_hours=12)
    assert sub.ttl_hours == 12

    messages = bus.read_subscribed("agent-b", limit=10)
    assert messages == []


def test_cleanup_stats_counts_expired_and_aged_out(hivemind_context):
    bus = HivemindBus(
        hivemind_context,
        config=AFSConfig(hivemind=HivemindConfig(default_ttl_hours=24)),
    )
    expired = bus.send("agent-a", "status", {"info": "expired"}, ttl_hours=24)
    expired_path = bus._root / "agent-a" / f"{expired.id}.json"
    _write_message_field(
        expired_path,
        expires_at=(datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),
    )

    _create_old_message(bus, "agent-b", age_hours=48)
    bus.send("agent-a", "status", {"info": "fresh"}, ttl_hours=72)

    stats = bus.cleanup_stats(max_age_hours=24)
    assert stats["removed_count"] == 2
    assert stats["expired_count"] >= 1
    assert stats["aged_out_count"] >= 1
    assert stats["remaining_count"] == 1


def test_reap_dry_run_preserves_files(hivemind_context):
    bus = HivemindBus(
        hivemind_context,
        config=AFSConfig(hivemind=HivemindConfig(default_ttl_hours=24)),
    )
    _create_old_message(bus, "agent-a", age_hours=48)

    stats = bus.reap(max_age_hours=24, dry_run=True)
    assert stats["removed_count"] == 1
    assert len(list((bus._root / "agent-a").glob("*.json"))) == 1


def test_subscription_ttl_hours_roundtrip():
    sub = HivemindSubscription(
        agent_name="test",
        topics=["a"],
        created_at="2024-01-01T00:00:00",
        updated_at="2024-01-01T00:00:00",
        ttl_hours=12,
    )
    data = sub.to_dict()
    assert data["ttl_hours"] == 12
    restored = HivemindSubscription.from_dict(data)
    assert restored.ttl_hours == 12

    sub_no_ttl = HivemindSubscription(
        agent_name="test",
        topics=["b"],
        created_at="2024-01-01T00:00:00",
        updated_at="2024-01-01T00:00:00",
    )
    data2 = sub_no_ttl.to_dict()
    assert "ttl_hours" not in data2
    restored2 = HivemindSubscription.from_dict(data2)
    assert restored2.ttl_hours is None


def test_hivemind_config_from_toml():
    config = AFSConfig.from_dict({
        "hivemind": {
            "default_ttl_hours": 48,
            "reaper_enabled": False,
        }
    })
    assert config.hivemind.default_ttl_hours == 48
    assert config.hivemind.reaper_enabled is False

    default_config = AFSConfig.from_dict({})
    assert default_config.hivemind.default_ttl_hours == 24
    assert default_config.hivemind.reaper_enabled is True
