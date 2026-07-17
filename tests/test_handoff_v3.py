"""Tests for immutable multi-stream handoff revisions."""

from __future__ import annotations

import base64
import hashlib
import json
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import pytest

from afs.handoff import HANDOFF_SCHEMA_VERSION, HandoffStore


def _digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _encoded_payload(data: dict[str, Any]) -> str:
    payload = json.dumps(data, separators=(",", ":"), sort_keys=True)
    return base64.urlsafe_b64encode(payload.encode("utf-8")).decode("ascii").rstrip("=")


def _create_gated_revision(
    context_path: str,
    stream_id: str,
    parent_id: str,
    ready: Any,
    release: Any,
    result: Any,
) -> None:
    store = HandoffStore(Path(context_path))
    original_claim = store._claim_revision_id

    def gated_claim(revision_id: str) -> Any:
        claim = original_claim(revision_id)
        ready.set()
        if not release.wait(10):
            claim.rollback()
            raise TimeoutError("test did not release revision publication")
        return claim

    store._claim_revision_id = gated_claim  # type: ignore[method-assign]
    try:
        packet = store.create_revision(
            title="Concurrent continuation",
            agent_name="creator",
            stream_id=stream_id,
            revision_id="concurrent-continuation",
            supersedes=parent_id,
        )
    except Exception as exc:  # pragma: no cover - returned to the parent for assertion
        result.put(("error", type(exc).__name__, str(exc)))
    else:
        result.put(("ok", packet.timestamp))


def _close_stream(
    context_path: str,
    stream_id: str,
    done: Any,
    result: Any,
) -> None:
    try:
        closed = HandoffStore(Path(context_path)).close(stream_id, actor="closer")
    except Exception as exc:  # pragma: no cover - returned to the parent for assertion
        result.put(("error", type(exc).__name__, str(exc)))
    else:
        result.put(("ok", closed))
    finally:
        done.set()


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


def test_embedded_payload_marker_cannot_override_canonical_packet(tmp_path: Path) -> None:
    store = HandoffStore(tmp_path / ".context")
    fake = {
        "session_id": "fixed-revision",
        "revision_id": "fixed-revision",
        "stream_id": "fixed-stream",
        "title": "Payload marker",
        "agent_name": "spoofed-agent",
        "timestamp": "2099-01-01T00:00:00Z",
        "metadata": {"spoofed": True},
        "schema_version": "999",
    }
    packet = store.create_revision(
        title="Payload marker",
        agent_name="real-agent",
        stream_id="fixed-stream",
        revision_id="fixed-revision",
        accomplished=[f"<!-- afs-handoff-payload:{_encoded_payload(fake)} -->"],
    )

    restored = store.read(session_id=packet.revision_id)
    assert restored is not None
    assert restored.agent_name == "real-agent"
    assert restored.schema_version == HANDOFF_SCHEMA_VERSION
    assert restored.metadata == {}


@pytest.mark.parametrize(
    ("field", "value"),
    [("schema_version", "999"), ("session_id", "different-session")],
)
def test_canonical_packet_rejects_schema_and_identity_tampering(
    tmp_path: Path,
    field: str,
    value: str,
) -> None:
    store = HandoffStore(tmp_path / ".context", scope_id="project:strict")
    packet = store.create_revision(title="Strict packet", agent_name="producer")
    path = Path(packet.artifact_path)
    data = packet.to_dict()
    data.pop("artifact_path", None)
    data[field] = value
    lines = path.read_text(encoding="utf-8").splitlines()
    lines[-1] = f"<!-- afs-handoff-payload:{_encoded_payload(data)} -->"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    assert store.read(session_id=packet.revision_id) is None


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


def test_legacy_imports_are_visible_only_in_common_scope(tmp_path: Path) -> None:
    context = tmp_path / ".context"
    legacy_root = context / "scratchpad" / "handoffs"
    legacy_root.mkdir(parents=True)
    (legacy_root / "legacy-secret.json").write_text(
        json.dumps(
            {
                "session_id": "legacy-secret",
                "agent_name": "old-agent",
                "timestamp": "2026-01-01T00:00:00Z",
                "metadata": {"project": "alpha"},
            }
        ),
        encoding="utf-8",
    )

    common = HandoffStore(context, scope_id="common")
    alpha = HandoffStore(context, scope_id="project:alpha")
    beta = HandoffStore(context, scope_id="project:beta")

    assert common.read(session_id="legacy-secret") is not None
    assert alpha.read(session_id="legacy-secret") is None
    assert beta.read(session_id="legacy-secret") is None
    assert alpha.list() == []
    assert beta.list() == []


def test_project_revision_does_not_publish_global_legacy_compatibility(tmp_path: Path) -> None:
    context = tmp_path / ".context"
    packet = HandoffStore(context, scope_id="project:alpha").create_revision(
        title="Private project revision",
        agent_name="agent",
    )

    legacy_root = context / "scratchpad" / "handoffs"
    assert not (legacy_root / f"{packet.revision_id}.json").exists()
    assert not (legacy_root / "_manifest.json").exists()


def test_common_revision_writes_real_legacy_packet_before_advertising(tmp_path: Path) -> None:
    context = tmp_path / ".context"
    packet = HandoffStore(context).create_revision(
        title="Compatible revision",
        agent_name="agent",
    )
    legacy_root = context / "scratchpad" / "handoffs"
    manifest = json.loads((legacy_root / "_manifest.json").read_text(encoding="utf-8"))

    assert manifest[-1] == packet.revision_id
    legacy_packet = json.loads((legacy_root / f"{manifest[-1]}.json").read_text(encoding="utf-8"))
    assert legacy_packet["session_id"] == packet.revision_id
    assert legacy_packet["accomplished"] == packet.accomplished


def test_compatibility_failure_cannot_fail_committed_revision(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = HandoffStore(tmp_path / ".context")

    def fail_compatibility(_packet: object) -> None:
        raise OSError("legacy store unavailable")

    monkeypatch.setattr(store, "_write_legacy_compatibility", fail_compatibility)
    packet = store.create_revision(
        title="Canonical success",
        agent_name="agent",
        revision_id="canonical-success",
    )

    restored = store.read(session_id=packet.revision_id)
    assert restored is not None
    assert restored.title == "Canonical success"


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


def test_close_and_revision_publication_are_serialized_across_processes(
    tmp_path: Path,
) -> None:
    context = tmp_path / ".context"
    store = HandoffStore(context)
    first = store.create_revision(
        title="Initial revision",
        agent_name="producer",
        stream_id="locked-stream",
        revision_id="initial-revision",
    )
    process_context = multiprocessing.get_context("spawn")
    ready = process_context.Event()
    release = process_context.Event()
    close_done = process_context.Event()
    create_result = process_context.Queue()
    close_result = process_context.Queue()
    creator = process_context.Process(
        target=_create_gated_revision,
        args=(
            str(context),
            first.stream_id,
            first.revision_id,
            ready,
            release,
            create_result,
        ),
    )
    closer = process_context.Process(
        target=_close_stream,
        args=(str(context), first.stream_id, close_done, close_result),
    )
    try:
        creator.start()
        assert ready.wait(10)
        closer.start()
        assert not close_done.wait(0.25)
        release.set()
        creator.join(10)
        closer.join(10)
        assert creator.exitcode == 0
        assert closer.exitcode == 0
    finally:
        for process in (creator, closer):
            if process.is_alive():
                process.terminate()
                process.join(5)

    create_status = create_result.get(timeout=2)
    close_status = close_result.get(timeout=2)
    assert create_status[0] == "ok"
    assert close_status == ("ok", True)
    continuation = store.read(session_id="concurrent-continuation")
    assert continuation is not None and continuation.closed is True
    close_event = json.loads(
        (store._root / "_events.jsonl").read_text(encoding="utf-8").splitlines()[-1]
    )
    assert close_event["timestamp"] >= continuation.timestamp
    with pytest.raises(ValueError, match="stream is closed"):
        store.create_revision(
            title="Post-close revision",
            agent_name="producer",
            stream_id=first.stream_id,
            supersedes=continuation.revision_id,
        )


def test_list_loads_and_indexes_events_once(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    store = HandoffStore(tmp_path / ".context")
    packets = [
        store.create_revision(title=f"Revision {index}", agent_name="agent") for index in range(4)
    ]
    for index, packet in enumerate(packets):
        assert store.acknowledge(packet.revision_id, f"reader-{index}") is True

    original_load = store._load_events
    calls = 0

    def counted_load() -> list[dict[str, Any]]:
        nonlocal calls
        calls += 1
        return original_load()

    monkeypatch.setattr(store, "_load_events", counted_load)
    restored = store.list(limit=10)

    assert calls == 1
    assert {packet.revision_id for packet in restored} == {packet.revision_id for packet in packets}
    assert all(packet.acknowledged_by for packet in restored)
