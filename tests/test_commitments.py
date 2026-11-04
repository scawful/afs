from __future__ import annotations

import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from afs.cli import main
from afs.commitments import CommitmentStore
from afs.context_layout import scaffold_v2
from afs.handoff import HandoffStore
from afs.project_registry import ProjectRegistry
from afs.tasks import TaskQueue
from afs.work_assistant import WorkAssistantStore


@pytest.fixture
def store(tmp_path, monkeypatch):
    for key in ("AFS_CONTEXT_ROOT", "AFS_ALLOWED_MOUNTS", "AFS_CONFIG_PATH"):
        monkeypatch.delenv(key, raising=False)
    monkeypatch.setenv("AFS_CONFIG_HOME", str(tmp_path / "config"))
    root = tmp_path / "context"
    root.mkdir()
    (root / "metadata.json").write_text(json.dumps({"directories": {"global": "global", "items": "items"}}))
    return CommitmentStore(root)


def proposal(store, *, owner="worker", scopes=None, dependencies=None, **extra):
    task = TaskQueue(store.context_root).create("Bounded outcome", context={"scope_id": store.scope.scope_id})
    request = {"event_id": f"offer-{task.id}", "task_id": task.id,
        "actor": "coordinator", "op": "propose", "expected_version": 0,
        "owner": owner, "inputs": {"a": "a1", "b": "b1", "tool": "t1", "environment": "e1"},
        "conditions": {"case-a": ["a"], "case-b": ["b"]},
        "writer_scopes": scopes or [], "dependencies": dependencies or {}, **extra}
    store.apply(request)
    return task.id, request


def event(store, task_id, op, *, actor="worker", revision=1, **extra):
    # Tests deliberately read the persisted stream version, not board status.
    with store._connect() as connection:
        task = store._load(connection)[store.system][task_id]
    request = {"event_id": f"{task_id}-{task['version']}-{op}", "task_id": task_id,
        "expected_version": task["version"], "revision": revision,
        "actor": actor, "op": op, **extra}
    return store.apply(request), request


def evidence(store, task_id, condition, *, passed=True):
    with store._connect() as connection:
        tasks = store._load(connection)[store.system]
    task = tasks[task_id]
    revision = store._current(task)
    bindings = store._bindings(tasks, revision, condition)
    return event(store, task_id, "evidence", revision=revision["revision"],
                 condition=condition, bindings=bindings, passed=passed,
                 reference=f"receipt:{task_id}/{condition}")


def finish(store, task_id):
    for condition in ("case-a", "case-b"):
        evidence(store, task_id, condition)
    event(store, task_id, "complete")
    return event(store, task_id, "confirm", actor="coordinator")


def test_real_lifecycle_completion_is_not_acceptance_and_board_replays(store):
    key, request = proposal(store)
    assert store.board()["accepted"] == []
    assert len(store.board()["proposals"]) == 1
    event(store, key, "accept")
    with pytest.raises(ValueError, match="conditions"):
        event(store, key, "complete")
    evidence(store, key, "case-a")
    evidence(store, key, "case-b")
    completed, _ = event(store, key, "complete")
    assert completed["board"]["accepted"][0]["status"] == "completion_proposed"
    confirmed, _ = event(store, key, "confirm", actor="coordinator")
    assert confirmed["board"]["accepted"][0]["status"] == "confirmed"
    assert CommitmentStore(store.context_root).board() == confirmed["board"]
    # The legacy task queue has not acquired a competing status writer.
    assert TaskQueue(store.context_root).get(key).status == "pending"
    assert store.db_path == WorkAssistantStore(store.context_root).db_path


def test_dedupe_is_stable_across_restart_and_rejects_changed_payload(store):
    key, request = proposal(store)
    event(store, key, "accept")
    restarted = CommitmentStore(store.context_root)
    result = restarted.apply(request)
    assert result["applied"] is False
    assert result["board"] == store.board()
    with pytest.raises(ValueError, match="different content"):
        restarted.apply({**request, "owner": "other"})
    with restarted._connect() as connection:
        assert connection.execute("SELECT count(*) FROM activity").fetchone()[0] == 2


def test_reordered_and_wrong_actor_events_do_not_change_board(store):
    key, _ = proposal(store)
    before = store.board()
    with pytest.raises(ValueError, match="proposed owner"):
        event(store, key, "accept", actor="intruder")
    with pytest.raises(ValueError, match="reordered"):
        store.apply({"event_id": "early", "task_id": key, "op": "accept",
                     "actor": "worker", "revision": 1, "expected_version": 2})
    assert store.board() == before
    event(store, key, "accept")
    with pytest.raises(ValueError, match="owner/reviewer"):
        event(store, key, "release", actor="coordinator", reason="takeover")
    with pytest.raises(ValueError, match="propose completion"):
        event(store, key, "confirm", actor="coordinator")


def test_scope_overlap_blocks_even_same_owner_but_allows_disjoint_work(store, tmp_path):
    scope = str(tmp_path / "code")
    one, _ = proposal(store, scopes=[scope])
    event(store, one, "accept")
    two, _ = proposal(store, scopes=[str(tmp_path / "code" / "child.py")])
    with pytest.raises(ValueError, match="writer scope conflicts"):
        event(store, two, "accept")
    other, _ = proposal(store, scopes=[str(tmp_path / "other")])
    event(store, other, "accept")
    event(store, one, "release", reason="handoff recorded")
    event(store, two, "accept")


def test_parallel_accepts_have_exactly_one_scope_writer(store, tmp_path):
    keys = [proposal(store, scopes=[str(tmp_path / "shared")])[0] for _ in range(2)]
    def accept(key):
        try:
            event(CommitmentStore(store.context_root), key, "accept")
            return "accepted"
        except ValueError as exc:
            return str(exc)
    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(accept, keys))
    assert results.count("accepted") == 1
    assert sum("conflicts" in r for r in results) == 1
    assert len(store.board()["accepted"]) == 1


def test_new_revision_keeps_accepted_owner_until_explicit_release_and_acceptance(store):
    key, request = proposal(store)
    event(store, key, "accept")
    revision = {**request, "event_id": "revision2", "expected_version": 2,
                "owner": "successor"}
    store.apply(revision)
    assert store.board()["accepted"][0]["owner"] == "worker"
    with pytest.raises(ValueError, match="previous owner must release"):
        event(store, key, "accept", actor="successor", revision=2)
    event(store, key, "release", reason="saved handoff")
    result, _ = event(store, key, "accept", actor="successor", revision=2)
    assert result["board"]["accepted"][0]["revision"] == 2
    with store._connect() as connection:
        task = store._load(connection)[store.system][key]
    assert task["revisions"]["1"]["status"] == "superseded"
    with pytest.raises(ValueError, match="active accepted"):
        event(store, key, "inputs", inputs={"a": "a2"})


def test_scoped_input_change_retains_only_unaffected_conditions_and_receipts(store):
    key, _ = proposal(store)
    event(store, key, "accept")
    finish(store, key)
    result, _ = event(store, key, "inputs", inputs={"a": "a2"})
    row = result["board"]["accepted"][0]
    assert row["valid_conditions"] == ["case-b"]
    assert row["status"] == "needs_review"
    with pytest.raises(ValueError, match="conditions"):
        event(store, key, "complete")
    evidence(store, key, "case-a")
    event(store, key, "complete")
    event(store, key, "confirm", actor="coordinator")
    with store._connect() as connection:
        task = store._load(connection)[store.system][key]
    assert len(task["revisions"]["1"]["evidence"]["case-a"]) == 2
    assert len(task["revisions"]["1"]["evidence"]["case-b"]) == 1


@pytest.mark.parametrize("extra", [{"inputs": {"tool": "t", "environment": None}},
                                   {"volatile": ["environment"]}])
def test_unknown_or_volatile_inputs_never_claim_cached_validity(store, extra):
    extra.setdefault("conditions", {"case-a": ["tool"], "case-b": ["tool"]})
    key, _ = proposal(store, **extra)
    event(store, key, "accept")
    evidence(store, key, "case-a")
    evidence(store, key, "case-b")
    assert store.board()["accepted"][0]["valid_conditions"] == []
    with pytest.raises(ValueError, match="conditions"):
        event(store, key, "complete")


def test_tool_change_invalidates_every_bound_case_and_failed_attempt_is_retained(store):
    key, _ = proposal(store)
    event(store, key, "accept")
    evidence(store, key, "case-a", passed=False)
    assert store.board()["accepted"][0]["valid_conditions"] == []
    finish(store, key)
    event(store, key, "inputs", inputs={"tool": "new-tool"})
    assert store.board()["accepted"][0]["valid_conditions"] == []


def test_dependency_change_invalidates_transitively_but_not_unrelated_tasks(store):
    a, _ = proposal(store)
    event(store, a, "accept")
    finish(store, a)
    b, _ = proposal(store, dependencies={a: 1})
    event(store, b, "accept")
    finish(store, b)
    c, _ = proposal(store, dependencies={b: 1})
    event(store, c, "accept")
    finish(store, c)
    d, _ = proposal(store)
    event(store, d, "accept")
    finish(store, d)
    event(store, a, "inputs", inputs={"a": "new"})
    rows = {r["task_id"]: r for r in store.board()["accepted"]}
    assert rows[a]["valid_conditions"] == ["case-b"]
    assert rows[b]["status"] == rows[c]["status"] == "needs_review"
    assert rows[b]["valid_conditions"] == rows[c]["valid_conditions"] == []
    assert rows[d]["status"] == "confirmed"


def test_release_after_confirm_preserves_dependency_evidence(store):
    a, _ = proposal(store)
    event(store, a, "accept")
    finish(store, a)
    b, _ = proposal(store, dependencies={a: 1})
    event(store, b, "accept")
    finish(store, b)
    event(store, a, "release", reason="verified outcome retained")
    rows = {r["task_id"]: r for r in store.board()["accepted"]}
    assert rows[a]["status"] == rows[b]["status"] == "confirmed"
    assert rows[a]["writer_scopes"] == []


def test_cyclic_revision_is_refused(store):
    a, request = proposal(store)
    event(store, a, "accept")
    b, _ = proposal(store, dependencies={a: 1})
    event(store, b, "accept")
    with pytest.raises(ValueError, match="cycle"):
        store.apply({**request, "event_id": "cycle", "expected_version": 2,
                     "dependencies": {b: 1}})


def test_atomic_insert_failure_leaves_no_accepted_board_and_retry_recovers(store):
    key, _ = proposal(store)
    before = store.board()
    with store._connect() as connection:
        connection.execute("CREATE TRIGGER fail_event BEFORE INSERT ON activity "
                           "BEGIN SELECT RAISE(ABORT, 'injected disk failure'); END")
    with pytest.raises(sqlite3.IntegrityError, match="injected"):
        event(store, key, "accept")
    assert CommitmentStore(store.context_root).board() == before
    with store._connect() as connection:
        connection.execute("DROP TRIGGER fail_event")
    result, _ = event(store, key, "accept")
    assert len(result["board"]["accepted"]) == 1


def test_v2_projects_are_scoped_and_cannot_overlap_accepted_writer(store, tmp_path):
    root = tmp_path / "central"
    scaffold_v2(root)
    projects = [tmp_path / "alpha", tmp_path / "beta"]
    for project in projects:
        project.mkdir()
        ProjectRegistry(root).register(project, name=project.name)
    a, b = [CommitmentStore(root, requester_path=project) for project in projects]
    key, _ = proposal(a, scopes=[str(tmp_path / "resource")])
    event(a, key, "accept")
    assert b.board() == {"accepted": [], "proposals": []}
    second, _ = proposal(b, scopes=[str(tmp_path / "resource")])
    with pytest.raises(ValueError, match="conflicts"):
        event(b, second, "accept")
    with pytest.raises(PermissionError, match="not registered"):
        CommitmentStore(root, requester_path=tmp_path / "unregistered")


def test_symlink_database_is_refused_before_sqlite_can_modify_target(tmp_path):
    root = tmp_path / "context"
    (root / "global").mkdir(parents=True)
    (root / "metadata.json").write_text(json.dumps({"directories": {"global": "global", "items": "items"}}))
    outside = tmp_path / "outside"
    outside.write_text("original")
    (root / "global" / "work_assistant.sqlite3").symlink_to(outside)
    with pytest.raises(ValueError, match="symbolic link"):
        CommitmentStore(root)
    assert outside.read_text() == "original"


def test_cli_uses_existing_handoff_and_board_surfaces_without_touching_manual_board(store, tmp_path, capsys):
    key, request = proposal(store)
    manual = store.context_root / "scratchpad" / "agent_board.md"
    manual.parent.mkdir()
    manual.write_text("# User board\n")
    config = tmp_path / "afs.toml"
    config.write_text(f'[general]\ncontext_root = "{store.context_root}"\n'
                      '[extensions]\nauto_discover = false\n[plugins]\nauto_discover = false\n')
    args = ["--config", str(config), "--context-root", str(store.context_root)]
    event_file = tmp_path / "accept.json"
    event_file.write_text(json.dumps({"event_id": "cli-accept", "op": "accept", "actor": "worker",
        "task_id": key, "revision": 1, "expected_version": 1}))
    assert main(["handoff", "agree", *args, "--event-file", str(event_file)]) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["board"]["accepted"][0]["owner"] == "worker"
    assert main(["board", "accepted", *args, "--json"]) == 0
    assert json.loads(capsys.readouterr().out) == result["board"]
    assert main(["handoff", "agreement", key, *args]) == 0
    inspected = json.loads(capsys.readouterr().out)
    assert inspected["version"] == 2
    assert inspected["evidence_bindings"]["case-a"]["a"] == "a1"
    assert manual.read_text() == "# User board\n"


def test_handoff_reference_links_actual_immutable_revision(store):
    packet = HandoffStore(store.context_root).create_revision(
        title="Bounded work", agent_name="coordinator", target_agent="worker")
    key, _ = proposal(store, handoff_revision=packet.revision_id)
    event(store, key, "accept")
    assert store.board()["accepted"][0]["handoff_revision"] == packet.revision_id


def test_legacy_claim_and_completion_are_refused_for_opted_in_task_until_retirement(store):
    key, _ = proposal(store)
    queue = TaskQueue(store.context_root)
    with pytest.raises(ValueError, match="accepted agreements"):
        queue.claim(key, "uninvited-worker")
    event(store, key, "accept")
    with pytest.raises(ValueError, match="accepted agreements"):
        queue.complete(key)
    with pytest.raises(ValueError, match="release owners"):
        event(store, key, "retire", actor="coordinator")
    event(store, key, "release", reason="returning to legacy queue")
    event(store, key, "retire", actor="coordinator")
    assert queue.claim(key, "new-legacy-owner").assigned_to == "new-legacy-owner"


def test_opt_in_never_takes_over_existing_legacy_claim(store):
    queue = TaskQueue(store.context_root)
    task = queue.create("Existing claimed work")
    queue.claim(task.id, "existing-owner")
    with pytest.raises(ValueError, match="legacy task ownership"):
        store.apply({"task_id": task.id, "event_id": "takeover", "op": "propose",
                     "actor": "coordinator", "expected_version": 0, "owner": "new-owner"})
    assert queue.get(task.id).assigned_to == "existing-owner"
    assert store.board() == {"accepted": [], "proposals": []}


def test_released_result_still_accepts_input_invalidation_without_reclaim(store):
    key, _ = proposal(store)
    event(store, key, "accept")
    finish(store, key)
    event(store, key, "release", reason="work finished")
    result, _ = event(store, key, "inputs", actor="coordinator", inputs={"a": "new"})
    row = result["board"]["accepted"][0]
    assert row["status"] == "needs_review"
    assert row["valid_conditions"] == ["case-b"]
    assert row["writer_scopes"] == []
    with pytest.raises(ValueError, match="active accepted"):
        evidence(store, key, "case-a")


def test_supported_inspect_exposes_dependency_bindings_and_historical_revisions(store):
    a, _ = proposal(store)
    event(store, a, "accept")
    finish(store, a)
    b, _ = proposal(store, dependencies={a: 1})
    event(store, b, "accept")
    inspected = store.inspect(b)
    assert f"dependency:{a}" in inspected["evidence_bindings"]["case-a"]
    event(store, b, "evidence", condition="case-a", passed=True,
          reference="receipt:external", bindings=inspected["evidence_bindings"]["case-a"])
    assert "case-a" in store.board()["accepted"][1 if a < b else 0]["valid_conditions"]
    assert store.inspect(b, revision=1)["agreement"]["owner"] == "worker"


def test_task_opt_in_and_legacy_claim_are_serialized(store):
    key = TaskQueue(store.context_root).create("Race for one task").id
    def opt_in():
        try:
            store.apply({"task_id": key, "event_id": "race-offer", "op": "propose",
                "actor": "coordinator", "expected_version": 0, "owner": "worker",
                "inputs": {"tool": "t", "environment": "e"}, "conditions": {"proof": []}})
            return "proposed"
        except ValueError:
            return "proposal refused"
    def claim():
        try:
            TaskQueue(store.context_root).claim(key, "legacy-owner")
            return "claimed"
        except ValueError:
            return "claim refused"
    with ThreadPoolExecutor(max_workers=2) as executor:
        a, b = executor.submit(opt_in), executor.submit(claim)
        result = [a.result(), b.result()]
    assert result in (["proposed", "claim refused"], ["proposal refused", "claimed"])


def test_retirement_invalidates_dependents_before_legacy_work_can_change(store):
    a, _ = proposal(store)
    event(store, a, "accept")
    finish(store, a)
    b, _ = proposal(store, dependencies={a: 1})
    event(store, b, "accept")
    finish(store, b)
    event(store, a, "release", reason="legacy migration")
    event(store, a, "retire", actor="coordinator")
    TaskQueue(store.context_root).claim(a, "legacy-owner")
    rows = {r["task_id"]: r for r in store.board()["accepted"]}
    assert rows[a]["status"] == "retired"
    assert rows[b]["status"] == "needs_review"
    assert rows[b]["valid_conditions"] == []


def test_custom_routing_cannot_bypass_queue_guard_without_persisted_metadata(tmp_path):
    from afs.models import MountType
    from afs.schema import AFSConfig, DirectoryConfig, PolicyType

    root = tmp_path / "context"
    root.mkdir()
    config = AFSConfig(directories=[DirectoryConfig(
        "custom-global", PolicyType.WRITABLE, role=MountType.GLOBAL)])
    with pytest.raises(ValueError, match="persisted global/items"):
        CommitmentStore(root, config=config)
    (root / "metadata.json").write_text(json.dumps({"directories": {
        "global": "custom-global", "items": "custom-items"}}))
    store = CommitmentStore(root, config=config)
    key, _ = proposal(store)
    event(store, key, "accept")
    with pytest.raises(ValueError, match="accepted agreements"):
        TaskQueue(root).claim(key, "legacy-owner")
