from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from afs.atomic_io import atomic_write_text
from afs.missions import Mission, MissionNotFoundError, MissionStore


def _store(tmp_path: Path) -> MissionStore:
    context_root = tmp_path / ".context"
    context_root.mkdir()
    return MissionStore(context_root)


def test_create_sets_active_status_and_fields(tmp_path: Path) -> None:
    store = _store(tmp_path)
    mission = store.create(
        title="Triage incident 4821",
        owner="gemini",
        next_steps=["pull logs"],
        tags=["incident"],
    )
    assert mission.mission_id.startswith("mission_")
    assert mission.status == "active"
    assert mission.owner == "gemini"
    assert mission.next_steps == ["pull logs"]
    assert mission.tags == ["incident"]
    # Round-trips from disk.
    loaded = store.get(mission.mission_id)
    assert loaded is not None
    assert loaded.title == "Triage incident 4821"


def test_create_requires_title(tmp_path: Path) -> None:
    store = _store(tmp_path)
    with pytest.raises(ValueError):
        store.create(title="   ")


def test_update_transitions_status_and_appends_log(tmp_path: Path) -> None:
    store = _store(tmp_path)
    mission = store.create(title="Ship fusion")
    updated = store.update(
        mission.mission_id,
        status="blocked",
        blockers=["waiting on review"],
        note="pinged reviewer",
        actor="claude",
        link_session="sess-1",
        link_handoff="handoff-1",
    )
    assert updated.status == "blocked"
    assert updated.blockers == ["waiting on review"]
    assert updated.linked_sessions == ["sess-1"]
    assert updated.linked_handoffs == ["handoff-1"]
    assert len(updated.log) == 1
    assert updated.log[0]["note"] == "pinged reviewer"
    assert updated.log[0]["actor"] == "claude"


def test_update_rejects_invalid_status(tmp_path: Path) -> None:
    store = _store(tmp_path)
    mission = store.create(title="X")
    with pytest.raises(ValueError, match="invalid mission status"):
        store.update(mission.mission_id, status="nope")


def test_update_unknown_mission_raises(tmp_path: Path) -> None:
    store = _store(tmp_path)
    with pytest.raises(MissionNotFoundError):
        store.update("mission_does_not_exist", status="done")


def test_active_returns_only_in_flight_newest_first(tmp_path: Path) -> None:
    store = _store(tmp_path)
    first = store.create(title="first")
    second = store.create(title="second")
    third = store.create(title="third")
    store.update(second.mission_id, status="done")  # completed -> excluded
    store.update(third.mission_id, status="blocked")  # still in flight

    active = store.active()
    ids = [m.mission_id for m in active]
    assert second.mission_id not in ids
    assert first.mission_id in ids
    assert third.mission_id in ids
    # Newest first (manifest is reverse-ordered).
    assert ids[0] == third.mission_id


def test_list_filters_by_status(tmp_path: Path) -> None:
    store = _store(tmp_path)
    a = store.create(title="a")
    store.create(title="b")
    store.update(a.mission_id, status="done")
    done = store.list(status="done")
    assert [m.mission_id for m in done] == [a.mission_id]


def test_construction_is_read_only_no_dir_created(tmp_path: Path) -> None:
    # Session bootstrap constructs a store just to read active missions; that read path
    # must not create items/missions on disk (would dirty a repo / external mount).
    context_root = tmp_path / ".context"
    context_root.mkdir()
    store = MissionStore(context_root)
    assert not store._root.exists()  # type: ignore[attr-defined]
    # Read APIs tolerate the absent directory and return empty.
    assert store.list() == []
    assert store.active() == []
    assert store.get("mission_missing") is None
    assert not store._root.exists()  # type: ignore[attr-defined]
    # The first write lazily creates it.
    store.create(title="first real mission")
    assert store._root.exists()  # type: ignore[attr-defined]


def test_mission_survives_manifest_loss(tmp_path: Path) -> None:
    # Simulate a lost concurrent-create race: the mission file exists but the manifest
    # never recorded its id. list()/active() must still surface it via disk reconciliation.
    store = _store(tmp_path)
    kept = store.create(title="kept")
    raced = store.create(title="raced")
    # Rewrite the manifest to drop the raced mission (as a lost read-modify-write would).
    atomic_write_text(
        store._manifest_path,  # type: ignore[attr-defined]
        __import__("json").dumps([kept.mission_id]) + "\n",
    )
    listed_ids = {m.mission_id for m in store.list()}
    assert raced.mission_id in listed_ids
    assert kept.mission_id in listed_ids
    assert raced.mission_id in {m.mission_id for m in store.active()}


def test_mission_dataclass_round_trip() -> None:
    mission = Mission(
        mission_id="mission_1",
        title="t",
        status="active",
        created_at="2026-07-01T00:00:00Z",
        updated_at="2026-07-01T00:00:00Z",
        next_steps=["s"],
    )
    assert Mission.from_dict(mission.to_dict()) == mission


def test_programmatic_acceptance_is_only_a_suggestion(tmp_path: Path) -> None:
    store = _store(tmp_path)
    mission = store.create(
        title="Ship the reactor",
        acceptance="reactor starts agents from history events with tests",
        acceptance_set_by="human",
    )
    loaded = store.get(mission.mission_id)
    assert loaded is not None
    assert loaded.acceptance == ""
    assert (
        loaded.acceptance_suggestion
        == "reactor starts agents from history events with tests"
    )
    assert loaded.acceptance_human_confirmed is False

    store.update(mission.mission_id, acceptance="also covers hivemind topics")
    updated = store.get(mission.mission_id)
    assert updated is not None
    assert updated.acceptance == ""
    assert updated.acceptance_suggestion == "also covers hivemind topics"

    # Records written before the field existed load with an empty acceptance.
    assert Mission.from_dict({"mission_id": "mission_old", "title": "Old"}).acceptance == ""


def test_string_acceptance_confirmation_fails_closed() -> None:
    loaded = Mission.from_dict(
        {
            "mission_id": "mission_untrusted",
            "title": "Untrusted",
            "acceptance": "fabricated done criteria",
            "acceptance_human_confirmed": "false",
            "acceptance_identity_authenticated": "true",
        }
    )
    assert loaded.acceptance == ""
    assert loaded.acceptance_suggestion == "fabricated done criteria"
    assert loaded.acceptance_human_confirmed is False
    assert loaded.acceptance_identity_authenticated is False


def test_broker_authorized_acceptance_round_trips(tmp_path: Path) -> None:
    from afs.human_provenance import _broker_for_reader

    acceptance = "reactor starts agents with tests"
    store = _store(tmp_path)
    authorization = _broker_for_reader(lambda _prompt: "human").confirm_token(
        "human",
        "prompt",
        scope=store.human_acceptance_scope(
            "create", "Ship the reactor", acceptance
        ),
    )
    assert authorization is not None
    mission = store.create(
        title="Ship the reactor",
        acceptance=acceptance,
        acceptance_authorization=authorization,
    )
    loaded = store.get(mission.mission_id)
    assert loaded is not None
    assert loaded.acceptance == "reactor starts agents with tests"
    assert loaded.acceptance_human_confirmed is True
    with pytest.raises(ValueError, match="fresh HumanDecisionBroker"):
        store.create(
            title="Ship the reactor",
            acceptance=acceptance,
            acceptance_authorization=authorization,
        )


def test_concurrent_updates_preserve_human_acceptance_and_agent_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from afs.human_provenance import _broker_for_reader

    store = _store(tmp_path)
    mission = store.create(title="Ship the reactor", summary="v1")
    acceptance = "v2 passes the reactor integration suite"
    authorization = _broker_for_reader(lambda _prompt: "human").confirm_token(
        "human",
        "prompt",
        scope=store.human_acceptance_scope("update", mission.mission_id, acceptance),
    )
    assert authorization is not None

    original_get = MissionStore.get

    def slow_get(self: MissionStore, mission_id: str):
        loaded = original_get(self, mission_id)
        if threading.current_thread() is not threading.main_thread():
            time.sleep(0.15)
        return loaded

    monkeypatch.setattr(MissionStore, "get", slow_get)
    start = threading.Barrier(2)
    acceptance_store = MissionStore(store._context_path)  # type: ignore[attr-defined]
    summary_store = MissionStore(store._context_path)  # type: ignore[attr-defined]

    def update_acceptance() -> None:
        start.wait(timeout=5)
        acceptance_store.update(
            mission.mission_id,
            acceptance=acceptance,
            acceptance_authorization=authorization,
        )

    def update_summary() -> None:
        start.wait(timeout=5)
        summary_store.update(mission.mission_id, summary="agent summary v2")

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(update_acceptance), pool.submit(update_summary)]
        for future in futures:
            future.result(timeout=10)

    final = original_get(store, mission.mission_id)
    assert final is not None
    assert final.summary == "agent summary v2"
    assert final.acceptance == acceptance
    assert final.acceptance_human_confirmed is True
    assert any(
        entry.get("note") == "human-confirmed acceptance updated"
        for entry in final.log
    )


def test_agent_acceptance_suggestion_is_not_attributed_to_prior_human(
    tmp_path: Path,
) -> None:
    from afs.human_provenance import _broker_for_reader

    store = _store(tmp_path)
    title = "Ship the reactor"
    acceptance = "human definition of done"
    authorization = _broker_for_reader(lambda _prompt: "human").confirm_token(
        "human",
        "prompt",
        scope=store.human_acceptance_scope("create", title, acceptance),
    )
    assert authorization is not None
    mission = store.create(
        title=title,
        acceptance=acceptance,
        acceptance_authorization=authorization,
    )

    updated = store.update(
        mission.mission_id,
        acceptance="agent suggestion",
        actor="planner-agent",
    )

    assert updated.acceptance == acceptance
    assert updated.acceptance_suggestion == "agent suggestion"
    assert updated.log[-1]["actor"] == "planner-agent"
    assert updated.log[-1]["note"] == "unauthenticated acceptance suggestion updated"
