from __future__ import annotations

import os
import time
from pathlib import Path

import pytest

from afs.hivemind import HivemindBus, normalize_agent_name, recipient_matches
from afs.messages import MessageBus
from afs.schema import AFSConfig, HivemindConfig


def _context(tmp_path: Path) -> Path:
    root = tmp_path / ".context"
    (root / "hivemind").mkdir(parents=True)
    return root


@pytest.mark.parametrize(
    ("to", "recipient", "expected"),
    [
        (None, "cursor", True),
        ("cursor", "cursor", True),
        ("Cursor", "cursor", True),
        ("claude/cursor", "cursor", True),
        ("codex, claude", "claude", True),
        ("all", "codex", True),
        ("claude-kydrog", "claude", True),
        ("claude", "claude-kydrog", True),
        ("Codex RC Planning", "codex-rc-planning", True),
        ("codex-lyra", "codex-docks", False),
        ("codex", "cursor", False),
        ("claude/cursor", "codex", False),
    ],
)
def test_recipient_matches(to: str | None, recipient: str, expected: bool) -> None:
    assert recipient_matches(to, recipient) is expected


def test_normalize_agent_name() -> None:
    assert normalize_agent_name("  Codex RC_Planning ") == "codex-rc-planning"


def test_directed_messages_do_not_get_default_ttl(tmp_path: Path) -> None:
    bus = HivemindBus(
        _context(tmp_path), config=AFSConfig(hivemind=HivemindConfig(default_ttl_hours=24))
    )
    assert bus.send("codex", "request", {}, to="cursor").expires_at is None
    assert bus.send("codex", "status", {}).expires_at is not None
    assert bus.send("codex", "request", {}, to="cursor", ttl_hours=2).expires_at is not None


def test_inbox_tracks_read_and_done_per_recipient(tmp_path: Path) -> None:
    bus = MessageBus(_context(tmp_path), scope_id="project:alpha")
    ask = bus.send("codex", "request", {"text": "add test flag"}, to="cursor/claude")
    bus.send("cursor", "status", {"text": "own message"})
    bus.send("codex", "request", {"text": "not for cursor"}, to="claude")

    inbox = bus.inbox("cursor")
    assert [(entry["id"], entry["state"]) for entry in inbox] == [(ask.id, "unread")]

    bus.ack(ask.id, "Cursor")
    assert bus.inbox("cursor")[0]["state"] == "read"

    bus.ack(ask.id, "cursor", status="done", note="landed")
    assert bus.inbox("cursor") == []
    assert bus.inbox("cursor", include_done=True)[0]["state"] == "done"
    assert bus.inbox("claude")[0]["state"] == "unread"


def test_reply_to_round_trips(tmp_path: Path) -> None:
    bus = MessageBus(_context(tmp_path), scope_id="project:alpha")
    ask = bus.send("codex", "request", {"text": "?"}, to="cursor")
    reply = bus.send("cursor", "status", {"text": "done"}, to="codex", reply_to=ask.id)
    assert bus.inbox("codex")[0]["reply_to"] == ask.id
    assert reply.to_dict()["reply_to"] == ask.id


def test_ack_rejects_unknown_and_out_of_scope_messages(tmp_path: Path) -> None:
    root = _context(tmp_path)
    other = MessageBus(root, scope_id="project:beta").send("codex", "request", {}, to="cursor")
    bus = MessageBus(root, scope_id="project:alpha")
    with pytest.raises(KeyError):
        bus.ack(other.id, "cursor")
    with pytest.raises(KeyError):
        bus.ack("20260101T000000-deadbeef", "cursor")
    with pytest.raises(ValueError):
        bus.ack(other.id, "cursor", status="maybe")


def test_reap_keeps_unfinished_directed_messages(tmp_path: Path) -> None:
    bus = HivemindBus(_context(tmp_path))
    open_ask = bus.send("codex", "request", {}, to="cursor")
    finished = bus.send("codex", "request", {}, to="cursor")
    bus.ack(finished.id, "cursor", status="done")
    old = time.time() - 72 * 3600
    for msg in (open_ask, finished):
        os.utime(bus._root / "codex" / f"{msg.id}.json", (old, old))

    stats = bus.cleanup_stats(max_age_hours=24)

    assert stats["removed_count"] == 1
    assert (bus._root / "codex" / f"{open_ask.id}.json").exists()
    assert not (bus._root / ".acks" / f"{finished.id}.json").exists()
