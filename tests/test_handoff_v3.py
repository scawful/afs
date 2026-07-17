"""Tests for immutable multi-stream handoff revisions."""

from __future__ import annotations

import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from afs.handoff import HANDOFF_SCHEMA_VERSION, HandoffStore


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_v3_revision_is_readable_markdown_in_project_scope(tmp_path: Path) -> None:
    context = tmp_path / ".context"
    context.mkdir()
    store = HandoffStore(context, scope_id="project:prj_example")

    packet = store.create_revision(
        title="Finish the context migration",
        agent_name="codex",
        accomplished=["Built the migration inventory"],
        blocked=["Needs human review"],
        next_steps=["Run the dry-run"],
        target_agent="reviewer",
        project_id="prj_example",
    )

    path = Path(packet.artifact_path)
    assert packet.schema_version == HANDOFF_SCHEMA_VERSION
    assert packet.session_id == packet.revision_id
    assert (
        path.parent.parent
        == (context / "memory" / "projects" / "prj_example" / "handoffs").resolve()
    )
    assert "finish-the-context-migration" in path.name
    assert "# Finish the context migration" in path.read_text(encoding="utf-8")

    restored = store.read(session_id=packet.revision_id)
    assert restored is not None
    assert restored.to_dict() == packet.to_dict()


def test_v3_requires_title_and_rejects_path_identifiers(tmp_path: Path) -> None:
    store = HandoffStore(tmp_path / ".context")
    with pytest.raises(ValueError, match="title is required"):
        store.create_revision(title="  ", agent_name="agent")
    with pytest.raises(ValueError, match="stream_id"):
        store.create_revision(
            title="Unsafe stream",
            agent_name="agent",
            stream_id="../../escape",
        )
    with pytest.raises(ValueError, match="session_id"):
        store.create(agent_name="agent", session_id="../overwrite")
    assert store.read(session_id="../../escape") is None


def test_multiple_streams_and_superseding_dag(tmp_path: Path) -> None:
    store = HandoffStore(tmp_path / ".context", scope_id="project:prj_multi")
    first = store.create_revision(title="Release train", agent_name="a")
    second = store.create_revision(
        title="Release train follow-up",
        agent_name="b",
        supersedes=first.revision_id,
    )
    sibling = store.create_revision(
        title="Independent follow-up",
        agent_name="c",
        stream_id=first.stream_id,
        supersedes=first.revision_id,
    )
    other = store.create_revision(title="Unrelated audit", agent_name="d")

    assert second.stream_id == first.stream_id == sibling.stream_id
    assert second.supersedes == [first.revision_id]
    revisions = store.list_revisions(first.stream_id)
    assert {item.revision_id for item in revisions} == {
        first.revision_id,
        second.revision_id,
        sibling.revision_id,
    }
    streams = store.list_streams()
    assert len(streams) == 2
    summary = next(item for item in streams if item.stream_id == first.stream_id)
    assert summary.revision_count == 3
    assert other.stream_id != first.stream_id


def test_supersedes_must_exist_and_stay_in_one_stream(tmp_path: Path) -> None:
    store = HandoffStore(tmp_path / ".context")
    with pytest.raises(ValueError, match="does not exist"):
        store.create_revision(
            title="Missing parent",
            agent_name="a",
            supersedes="missing",
        )
    first = store.create_revision(title="First", agent_name="a")
    other = store.create_revision(title="Other", agent_name="b")
    with pytest.raises(ValueError, match="same stream"):
        store.create_revision(
            title="Invalid merge",
            agent_name="c",
            stream_id=first.stream_id,
            supersedes=other.revision_id,
        )


def test_acknowledge_and_close_do_not_rewrite_revision(tmp_path: Path) -> None:
    store = HandoffStore(tmp_path / ".context")
    packet = store.create_revision(
        title="Immutable review",
        agent_name="producer",
        target_agent="consumer",
    )
    path = Path(packet.artifact_path)
    before = _digest(path)

    assert store.acknowledge(packet.revision_id, "consumer") is True
    assert _digest(path) == before
    acknowledged = store.read(session_id=packet.revision_id)
    assert acknowledged is not None
    assert acknowledged.acknowledged_by == ["consumer"]
    assert store.pending_for_agent("consumer") == []

    assert store.close(packet.stream_id, actor="human", reason="landed") is True
    assert _digest(path) == before
    closed = store.read(session_id=packet.revision_id)
    assert closed is not None and closed.closed is True
    events = [
        json.loads(line)
        for line in (store._root / "_events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert [event["kind"] for event in events] == ["acknowledged", "closed"]
    with pytest.raises(ValueError, match="stream is closed"):
        store.create_revision(
            title="Too late",
            agent_name="producer",
            stream_id=packet.stream_id,
            supersedes=packet.revision_id,
        )


def test_concurrent_acknowledgements_are_complete_jsonl_events(tmp_path: Path) -> None:
    store = HandoffStore(tmp_path / ".context")
    packet = store.create_revision(
        title="Concurrent readers",
        agent_name="producer",
        target_agent="consumer-0",
    )
    path = Path(packet.artifact_path)
    before = _digest(path)

    with ThreadPoolExecutor(max_workers=12) as pool:
        results = list(
            pool.map(
                lambda index: store.acknowledge(packet.revision_id, f"consumer-{index}"),
                range(32),
            )
        )

    assert all(results)
    restored = store.read(session_id=packet.revision_id)
    assert restored is not None
    assert set(restored.acknowledged_by) == {f"consumer-{index}" for index in range(32)}
    event_lines = (store._root / "_events.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(event_lines) == 32
    assert all(json.loads(line)["kind"] == "acknowledged" for line in event_lines)
    assert _digest(path) == before


def test_legacy_json_and_wrapped_manifest_remain_readable(tmp_path: Path) -> None:
    context = tmp_path / ".context"
    legacy_root = context / "scratchpad" / "handoffs"
    legacy_root.mkdir(parents=True)
    (legacy_root / "_manifest.json").write_text(
        json.dumps({"schema_version": "2", "sessions": ["legacy-id"]}),
        encoding="utf-8",
    )
    (legacy_root / "legacy-id.json").write_text(
        json.dumps(
            {
                "session_id": "legacy-id",
                "agent_name": "old-agent",
                "timestamp": "2026-01-01T00:00:00Z",
                "accomplished": ["old work"],
                "target_agent": "new-agent",
                "schema_version": "1",
                "acknowledged_by": ["historical-reader"],
            }
        ),
        encoding="utf-8",
    )

    store = HandoffStore(context)
    packet = store.read(session_id="legacy-id")
    assert packet is not None
    assert packet.schema_version == "1"
    assert packet.accomplished == ["old work"]
    assert packet.stream_id == packet.revision_id == "legacy-id"
    assert packet.acknowledged_by == ["historical-reader"]
    assert store.list()[0].session_id == "legacy-id"

    assert store.acknowledge("legacy-id", "new-agent") is True
    assert "new-agent" in store.read(session_id="legacy-id").acknowledged_by
    # Legacy source packets are imports and remain untouched too.
    raw = json.loads((legacy_root / "legacy-id.json").read_text(encoding="utf-8"))
    assert raw["acknowledged_by"] == ["historical-reader"]


def test_concurrent_revisions_are_unique_and_manifest_is_complete(tmp_path: Path) -> None:
    store = HandoffStore(tmp_path / ".context")

    def create(index: int) -> str:
        return store.create_revision(
            title="Concurrent stream",
            agent_name=f"agent-{index}",
        ).revision_id

    with ThreadPoolExecutor(max_workers=12) as pool:
        revision_ids = list(pool.map(create, range(48)))

    assert len(set(revision_ids)) == 48
    assert {packet.revision_id for packet in store.list(limit=100)} == set(revision_ids)
    manifest = json.loads(
        (tmp_path / ".context" / "scratchpad" / "handoffs" / "_manifest.json").read_text(
            encoding="utf-8"
        )
    )
    assert set(manifest) == set(revision_ids)


def test_duplicate_explicit_revision_is_never_overwritten(tmp_path: Path) -> None:
    store = HandoffStore(tmp_path / ".context")

    def create(title: str) -> str:
        packet = store.create_revision(
            title=title,
            agent_name="agent",
            revision_id="fixed-revision",
        )
        return packet.artifact_path

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(create, "Original"), pool.submit(create, "Replacement")]
    paths: list[str] = []
    failures = 0
    for future in futures:
        try:
            paths.append(future.result())
        except FileExistsError:
            failures += 1

    assert len(paths) == 1
    assert failures == 1
    before = _digest(Path(paths[0]))
    assert len(store.list()) == 1
    assert _digest(Path(paths[0])) == before
