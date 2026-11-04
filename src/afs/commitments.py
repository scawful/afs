"""Opt-in task agreements replayed from the existing work-assistant activity log.

This is bookkeeping, not authentication or execution authority. Hosts must bind
actor labels to authorized callers. No conversation or embedding changes state.
"""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any

from .agent_scope import assert_mount_allowed
from .atomic_io import secure_mkdir
from .context_paths import load_context_metadata, resolve_mount_root
from .handoff import HandoffStore
from .models import MountType
from .path_safety import assert_no_linklike_components
from .scopes import resolve_scope
from .tasks import TaskQueue
from .work_assistant import DEFAULT_DB_FILENAME, WorkAssistantStore

_EVENT_TYPE = "commitment.v1"
_HELD = {"accepted", "completion_proposed", "confirmed"}


def _json(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _text(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 1024:
        raise ValueError(f"{field} must be a nonempty string of at most 1024 characters")
    if any(ord(c) < 32 for c in value):
        raise ValueError(f"{field} cannot contain control characters")
    return value


def _identifier(value: Any, field: str) -> str:
    normalized = _text(value, field)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}", normalized):
        raise ValueError(f"{field} must be a safe identifier")
    return normalized


def _holds(revision: dict[str, Any]) -> bool:
    return revision["status"] in _HELD and not revision.get("released", False)


def _overlaps(left: str, right: str) -> bool:
    a, b = Path(left), Path(right)
    return a == b or a in b.parents or b in a.parents


def assert_legacy_task_mutation_allowed(context_root: Path, task_id: str, *, scope_id: str) -> None:
    """Guard legacy queue writers without creating state for legacy-only users."""
    root = resolve_mount_root(context_root, MountType.GLOBAL)
    database = root / DEFAULT_DB_FILENAME
    for suffix in ("", "-journal", "-wal", "-shm"):
        assert_no_linklike_components(Path(f"{database}{suffix}"))
    if not database.exists():
        return
    with closing(sqlite3.connect(f"{database.as_uri()}?mode=ro", uri=True)) as connection:
        if not connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='activity'"
        ).fetchone():
            return
        row = connection.execute(
            "SELECT metadata_json FROM activity WHERE activity_type=? AND target_system=? "
            "AND target_id=? ORDER BY rowid DESC LIMIT 1",
            (_EVENT_TYPE, f"afs.commitments:{scope_id}", task_id),
        ).fetchone()
    if row and not json.loads(row[0])["state"].get("retired", False):
        raise ValueError("task uses accepted agreements; use afs handoff agree (retire before legacy writes)")


class CommitmentStore(WorkAssistantStore):
    """One append-only, scoped agreement stream per existing task ID.

    Reuses the SQLite activity table and its connection/containment lifecycle.
    No second database, identity registry, mutable projection, or board file.
    """

    def __init__(self, context_root: Path, *, requester_path: Path | None = None,
                 config: Any = None) -> None:
        assert_mount_allowed(MountType.ITEMS, operation="access commitments")
        assert_mount_allowed(MountType.GLOBAL, operation="access commitment events")
        self.scope = resolve_scope(context_root, requester_path=requester_path)
        self.system = f"afs.commitments:{self.scope.scope_id}"
        if self.scope.layout_version == 1:
            metadata = load_context_metadata(context_root)
            if not metadata or not {"global", "items"} <= metadata.directories.keys():
                raise ValueError("agreements require initialized v1 context with persisted global/items routing")
        for mount in (MountType.GLOBAL, MountType.ITEMS):
            routed = resolve_mount_root(context_root, mount, config=config)
            assert_no_linklike_components(routed)
            if not routed.resolve().is_relative_to(self.scope.context_root):
                raise ValueError("agreement storage must remain within its context root")
        # Reuse the existing database; create private context-local directories.
        root = resolve_mount_root(context_root, MountType.GLOBAL, config=config)
        assert_no_linklike_components(root)
        secure_mkdir(root)
        self._commitment_db = root / DEFAULT_DB_FILENAME
        for suffix in ("", "-journal", "-wal", "-shm"):
            assert_no_linklike_components(Path(f"{self._commitment_db}{suffix}"))
        super().__init__(context_root, config=config)
        assert_no_linklike_components(self.db_path)
        self.db_path.chmod(0o600)

    def _assert_managed_db_safe(self, *, require_parent: bool = False) -> None:
        super()._assert_managed_db_safe(require_parent=require_parent)
        for suffix in ("", "-journal", "-wal", "-shm"):
            assert_no_linklike_components(Path(f"{self._commitment_db}{suffix}"))

    @staticmethod
    def _load(connection: sqlite3.Connection) -> dict[str, dict[str, Any]]:
        states: dict[str, dict[str, Any]] = {}
        rows = connection.execute(
            "SELECT target_system, target_id, metadata_json FROM activity "
            "WHERE activity_type = ? ORDER BY rowid", (_EVENT_TYPE,),
        )
        for row in rows:
            metadata = json.loads(row["metadata_json"])
            states.setdefault(row["target_system"], {})[row["target_id"]] = metadata["state"]
        return states

    @staticmethod
    def _current(task: dict[str, Any]) -> dict[str, Any] | None:
        revisions: dict[str, dict[str, Any]] = task["revisions"]
        return revisions.get(str(task["active"]))

    @classmethod
    def _signature(cls, tasks: dict[str, Any], task_id: str,
                   seen: frozenset[str] = frozenset()) -> str:
        if task_id in seen:
            raise ValueError("dependency cycle")
        task = tasks.get(task_id)
        current = cls._current(task) if task else None
        if not current:
            return "unaccepted"
        assert task is not None
        deps = {key: cls._signature(tasks, key, seen | {task_id})
                for key in current["dependencies"]}
        return hashlib.sha256(_json({"revision": current["revision"],
            "inputs": current["inputs"], "status": current["status"],
            "dependencies": deps, "retired": task.get("retired", False)}).encode()).hexdigest()

    @classmethod
    def _bindings(cls, tasks: dict[str, Any], revision: dict[str, Any],
                  condition: str) -> dict[str, Any]:
        roles = set(revision["conditions"][condition]) | {"tool", "environment"}
        return {**{role: revision["inputs"].get(role) for role in sorted(roles)},
                **{f"dependency:{key}": cls._signature(tasks, key)
                   for key in revision["dependencies"]}}

    @classmethod
    def _valid_conditions(cls, tasks: dict[str, Any], revision: dict[str, Any]) -> list[str]:
        valid = []
        for condition in revision["conditions"]:
            expected = cls._bindings(tasks, revision, condition)
            if any(value is None for value in expected.values()):
                continue
            if any(role in revision["volatile"] for role in expected):
                continue
            receipts = revision["evidence"].get(condition, [])
            if receipts and receipts[-1]["passed"] and receipts[-1]["bindings"] == expected:
                valid.append(condition)
        return valid

    @classmethod
    def _dependencies_ready(cls, tasks: dict[str, Any], revision: dict[str, Any],
                            seen: frozenset[str] = frozenset()) -> bool:
        for key, version in revision["dependencies"].items():
            if key in seen or key not in tasks or tasks[key].get("retired", False):
                return False
            current = cls._current(tasks[key])
            if (not current or current["revision"] != version or current["status"] != "confirmed"
                    or len(cls._valid_conditions(tasks, current)) != len(current["conditions"])
                    or not cls._dependencies_ready(tasks, current, seen | {key})):
                return False
        return True

    @classmethod
    def _board(cls, tasks: dict[str, Any]) -> dict[str, Any]:
        accepted, proposals = [], []
        for key, task in sorted(tasks.items()):
            current = cls._current(task)
            pending = task["revisions"][str(task["latest"])]
            if pending["status"] == "proposed":
                proposals.append({"task_id": key, "revision": pending["revision"],
                                  "owner": pending["owner"]})
            if not current:
                continue
            valid = cls._valid_conditions(tasks, current)
            ready = cls._dependencies_ready(tasks, current)
            status = current["status"]
            if status in {"completion_proposed", "confirmed"} and (
                    not ready or len(valid) != len(current["conditions"])):
                status = "needs_review"
            if task.get("retired", False):
                status, valid = "retired", []
            accepted.append({"task_id": key, "title": task["title"],
                "retired": task.get("retired", False),
                "version": task["version"], "revision": current["revision"],
                "owner": current["owner"], "status": status,
                "writer_scopes": current["writer_scopes"] if _holds(current) else [],
                "valid_conditions": valid, "dependencies_ready": ready,
                "handoff_revision": current["handoff_revision"]})
        return {"accepted": accepted, "proposals": proposals}

    def board(self) -> dict[str, Any]:
        with self._connect() as connection:
            return self._board(self._load(connection).get(self.system, {}))

    def inspect(self, task_id: str, *, revision: int | None = None) -> dict[str, Any]:
        _identifier(task_id, "task_id")
        with self._connect() as connection:
            tasks = self._load(connection).get(self.system, {})
            task = tasks.get(task_id)
            if not task:
                raise ValueError("agreement not found in this scope")
            number = revision if revision is not None else task["latest"]
            agreement = task["revisions"].get(str(number))
            if not agreement:
                raise ValueError("unknown revision")
            return {"task_id": task_id, "version": task["version"],
                    "active_revision": task["active"], "retired": task.get("retired", False),
                    "agreement": agreement,
                    "evidence_bindings": {condition: self._bindings(tasks, agreement, condition)
                                          for condition in agreement["conditions"]}}

    def apply(self, request: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(request, dict):
            raise ValueError("event must be an object")
        task_id = _identifier(request.get("task_id"), "task_id")
        with TaskQueue(self.context_root).transaction(task_id):
            return self._apply(request)

    def _apply(self, request: dict[str, Any]) -> dict[str, Any]:
        """Apply one version-checked event and return its board in one transaction.

        Duplicate IDs with identical requests are harmless; changed payloads and
        out-of-order transitions fail closed. A failed insert rolls back entirely.
        """
        if not isinstance(request, dict) or len(_json(request).encode()) > 32768:
            raise ValueError("event must be an object of at most 32768 bytes")
        _identifier(request.get("event_id"), "event_id")
        task_id = _identifier(request.get("task_id"), "task_id")
        actor = _text(request.get("actor"), "actor")
        op = _text(request.get("op"), "op")
        version = request.get("expected_version")
        if type(version) is not int or version < 0:
            raise ValueError("expected_version must be a nonnegative integer")
        event_key = "commitment_" + hashlib.sha256(
            f"{self.system}:{request['event_id']}".encode()).hexdigest()
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            states = self._load(connection)
            tasks = states.setdefault(self.system, {})
            duplicate = connection.execute(
                "SELECT metadata_json FROM activity WHERE activity_id = ?", (event_key,),
            ).fetchone()
            if duplicate:
                metadata = json.loads(duplicate[0])
                if metadata["request"] != request:
                    raise ValueError("event_id already used with different content")
                return {"applied": False, "version": metadata["state"]["version"],
                        "board": self._board(tasks)}
            task = tasks.get(task_id)
            if (task["version"] if task else 0) != version:
                raise ValueError("stale or reordered event: expected_version does not match")
            if op == "propose":
                task = self._propose(request, task, tasks)
                tasks[task_id] = task
            else:
                if not task:
                    raise ValueError("propose an agreement first")
                revision_number = request.get("revision")
                if type(revision_number) is not int:
                    raise ValueError("revision must be an integer")
                revision = task["revisions"].get(str(revision_number))
                if not revision:
                    raise ValueError("unknown revision")
                self._transition(request, task, revision, states, tasks)
            assert task is not None
            task["version"] = version + 1
            board = self._board(tasks)
            connection.execute(
                "INSERT INTO activity (activity_id,timestamp,event_id,activity_type,summary,"
                "target_system,target_id,actor,metadata_json) VALUES (?,datetime('now'),?,?,?,?,?,?,?)",
                (event_key, request["event_id"], _EVENT_TYPE, op, self.system, task_id,
                 actor, _json({"request": request, "state": task})),
            )
            return {"applied": True, "version": task["version"], "board": board}

    def _propose(self, request: dict[str, Any], task: dict[str, Any] | None,
                 tasks: dict[str, Any]) -> dict[str, Any]:
        actor = request["actor"]
        source = TaskQueue(self.context_root).get(request["task_id"])
        if source and (source.status != "pending" or source.assigned_to):
            raise ValueError("existing legacy task ownership is retained; opt in only an unassigned pending task")
        if task:
            latest = task["revisions"][str(task["latest"])]
            if actor != latest["reviewer"] or latest["status"] == "proposed":
                raise ValueError("only the agreed reviewer can revise; resolve pending proposal first")
        else:
            if not source or source.context.get("scope_id", "common") != self.scope.scope_id:
                raise ValueError("task_id must name an existing task in this scope")
            task = {"title": source.title, "version": 0, "active": None,
                    "latest": 0, "revisions": {}}
        owner = _text(request.get("owner"), "owner")
        if request.get("reviewer", actor) != actor:
            raise ValueError("the proposer must be the reviewer")
        inputs = request.get("inputs")
        if not isinstance(inputs, dict) or not {"tool", "environment"} <= inputs.keys():
            raise ValueError("inputs must declare tool and environment fingerprints (null if unknown)")
        for role, digest in inputs.items():
            _text(role, "input role")
            if role.startswith("dependency:"):
                raise ValueError("dependency: is a reserved role prefix")
            if digest is not None:
                _text(digest, "fingerprint")
        conditions = request.get("conditions")
        if not isinstance(conditions, dict) or not conditions:
            raise ValueError("conditions must map agreed predicate IDs to input role lists")
        for condition, roles in conditions.items():
            _identifier(condition, "condition")
            if not isinstance(roles, list) or any(not isinstance(r, str) or r not in inputs for r in roles):
                raise ValueError("condition roles must be declared inputs")
        scopes = request.get("writer_scopes", [])
        if not isinstance(scopes, list):
            raise ValueError("writer_scopes must be a list of absolute paths")
        canonical = []
        for path in scopes:
            _text(path, "writer scope")
            if not Path(path).is_absolute():
                raise ValueError("writer scopes must be absolute paths")
            canonical.append(str(Path(path).resolve()))
        dependencies = request.get("dependencies", {})
        if not isinstance(dependencies, dict):
            raise ValueError("dependencies must map task IDs to accepted revisions")
        for key, revision in dependencies.items():
            if key == request["task_id"] or key not in tasks or type(revision) is not int:
                raise ValueError("dependency must name another accepted task revision")
            current = self._current(tasks[key])
            if not current or current["revision"] != revision:
                raise ValueError("dependency must name the current accepted revision")
            self._check_cycle(tasks, key, request["task_id"], frozenset())
        volatile = request.get("volatile", [])
        if not isinstance(volatile, list) or any(not isinstance(r, str) or r not in inputs for r in volatile):
            raise ValueError("volatile must list declared input roles")
        handoff_id = request.get("handoff_revision", "")
        if handoff_id:
            _identifier(handoff_id, "handoff_revision")
            if not HandoffStore(self.context_root, scope_id=self.scope.scope_id).read(session_id=handoff_id):
                raise ValueError("handoff revision not found in this scope")
        task["retired"] = False
        number = task["latest"] + 1
        task["latest"] = number
        task["revisions"][str(number)] = {"revision": number, "owner": owner,
            "reviewer": actor, "status": "proposed", "inputs": inputs,
            "conditions": conditions, "writer_scopes": sorted(set(canonical)),
            "dependencies": dependencies, "volatile": volatile,
            "handoff_revision": handoff_id, "evidence": {}}
        return task

    @classmethod
    def _check_cycle(cls, tasks: dict[str, Any], key: str, target: str,
                     seen: frozenset[str]) -> None:
        if key == target or key in seen:
            raise ValueError("dependency cycle")
        current = cls._current(tasks[key])
        if current:
            for child in current["dependencies"]:
                cls._check_cycle(tasks, child, target, seen | {key})

    def _transition(self, request: dict[str, Any], task: dict[str, Any],
                    revision: dict[str, Any], states: dict[str, Any],
                    tasks: dict[str, Any]) -> None:
        op, actor = request["op"], request["actor"]
        if op == "retire":
            if actor != revision["reviewer"] or task["latest"] != revision["revision"]:
                raise ValueError("only the current reviewer can retire agreements")
            if revision["status"] == "proposed" or any(_holds(r) for r in task["revisions"].values()):
                raise ValueError("resolve proposals and release owners before retirement")
            task["retired"] = True
            return
        if task.get("retired", False):
            raise ValueError("agreement is retired; propose a new revision to opt in again")
        if op == "accept":
            if actor != revision["owner"] or revision["status"] != "proposed":
                raise ValueError("only the proposed owner can accept a pending revision")
            previous = self._current(task)
            if previous and _holds(previous) and previous["owner"] != actor:
                raise ValueError("previous owner must release before transfer")
            for system, scoped_tasks in states.items():
                for key, other in scoped_tasks.items():
                    if system == self.system and key == request["task_id"]:
                        continue
                    current = self._current(other)
                    if current and _holds(current) and any(
                        _overlaps(a, b) for a in current["writer_scopes"]
                        for b in revision["writer_scopes"]
                    ):
                        raise ValueError("writer scope conflicts with an accepted commitment")
            for key in revision["dependencies"]:
                self._check_cycle(tasks, key, request["task_id"], frozenset())
            if previous:
                previous["status"] = "superseded"
            revision["status"] = "accepted"
            task["active"] = revision["revision"]
            return
        if op == "withdraw":
            if actor != revision["reviewer"] or revision["status"] != "proposed":
                raise ValueError("only reviewer can withdraw a proposal")
            revision["status"] = "withdrawn"
            return
        if (op == "inputs" and task["active"] == revision["revision"]
                and actor in {revision["owner"], revision["reviewer"]}):
            # Input facts can invalidate released evidence without reacquiring scope.
            self._update_inputs(request, revision)
            return
        if task["active"] != revision["revision"] or not _holds(revision):
            raise ValueError("transition requires the active accepted revision")
        expected_actor = revision["reviewer"] if op == "confirm" else revision["owner"]
        if actor != expected_actor:
            raise ValueError("actor is not the agreed owner/reviewer for this transition")
        if op == "release":
            _text(request.get("reason"), "release reason")
            revision["released"] = True
            if revision["status"] != "confirmed":
                revision["status"] = "released"
        elif op == "evidence":
            condition = _identifier(request.get("condition"), "condition")
            if condition not in revision["conditions"]:
                raise ValueError("unknown acceptance condition")
            expected = self._bindings(tasks, revision, condition)
            if request.get("bindings") != expected or type(request.get("passed")) is not bool:
                raise ValueError("receipt must bind exact current inputs and a boolean result")
            _text(request.get("reference"), "evidence reference")
            revision["evidence"].setdefault(condition, []).append({
                "bindings": expected, "passed": request["passed"],
                "reference": request["reference"], "event_id": request["event_id"]})
            if revision["status"] == "confirmed":
                revision["status"] = "completion_proposed"
        elif op in {"complete", "confirm"}:
            if op == "confirm" and revision["status"] != "completion_proposed":
                raise ValueError("owner must propose completion before reviewer confirmation")
            if (len(self._valid_conditions(tasks, revision)) != len(revision["conditions"])
                    or not self._dependencies_ready(tasks, revision)):
                raise ValueError("acceptance conditions or dependencies lack valid evidence")
            revision["status"] = "confirmed" if op == "confirm" else "completion_proposed"
        else:
            raise ValueError(f"unknown commitment operation: {op}")

    @staticmethod
    def _update_inputs(request: dict[str, Any], revision: dict[str, Any]) -> None:
        changes = request.get("inputs")
        if not isinstance(changes, dict) or not changes or any(k not in revision["inputs"] for k in changes):
            raise ValueError("inputs update must name existing roles")
        for value in changes.values():
            if value is not None:
                _text(value, "fingerprint")
        revision["inputs"].update(changes)
