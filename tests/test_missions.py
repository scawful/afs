from __future__ import annotations

from pathlib import Path

import pytest

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
    store._atomic_write(  # type: ignore[attr-defined]
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


def test_acceptance_round_trips_and_updates(tmp_path: Path) -> None:
    store = _store(tmp_path)
    mission = store.create(
        title="Ship the reactor",
        acceptance="reactor starts agents from history events with tests",
    )
    loaded = store.get(mission.mission_id)
    assert loaded is not None
    assert loaded.acceptance == "reactor starts agents from history events with tests"

    store.update(mission.mission_id, acceptance="also covers hivemind topics")
    assert store.get(mission.mission_id).acceptance == "also covers hivemind topics"

    # Records written before the field existed load with an empty acceptance.
    assert Mission.from_dict({"mission_id": "mission_old", "title": "Old"}).acceptance == ""
