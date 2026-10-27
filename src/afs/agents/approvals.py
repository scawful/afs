"""SQLite-backed approval gate for guarded agent actions.

One database holds every agent-gate request and its whole lifecycle:

* ``pending`` waits for a person. Only a person at a terminal moves it to
  ``approved`` or ``rejected`` with ``human_confirmed=True`` (see
  :class:`afs.human_provenance.HumanDecisionBroker`).
* The system closes requests nobody needs any more: ``expired`` (TTL ran out),
  ``superseded`` (a newer run or config from the same source asked again) and
  ``withdrawn`` (the requester no longer needs it). System states are never
  human judgments, so calibration — which reads only human-confirmed
  approved/rejected rows — ignores them.
* An approved request's ``scope`` is ``standing`` (authorizes every matching
  check) or ``once`` (consumed by the first run that uses it). A standing
  approval can be revoked without a terminal, because revoking only removes
  permission.

The legacy ``approvals.json`` / ``approvals.history.jsonl`` pair is imported
into the database (idempotently, by request id) and never written again.
Malformed legacy rows are skipped and counted, never fatal.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import sqlite3
import time
import uuid
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Actions that always require approval
ALWAYS_APPROVE = frozenset(
    {
        "git_push",
        "git_force_push",
        "file_delete",
        "rom_edit",
        "deploy",
        "send_email",
        "create_pr",
        "merge_pr",
    }
)

# Actions that can be auto-approved for background agents
AUTO_APPROVE = frozenset(
    {
        "file_read",
        "file_write_scratchpad",
        "git_status",
        "git_diff",
        "git_log",
        "context_read",
        "context_write",
        "run_tests",
        "run_build",
        "embedding_update",
    }
)

DEFAULT_TTL_HOURS = 72.0
SYSTEM_RETENTION_DAYS = 30.0
MAX_PAYLOAD_CHARS = 64 * 1024
SCHEMA_VERSION = 1

PENDING = "pending"
HUMAN_DECISIONS = ("approved", "rejected")
SYSTEM_CLOSED = ("expired", "superseded", "withdrawn")
SCOPES = ("once", "standing")

_DB_SUFFIXES = (".sqlite3", ".sqlite", ".db")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS approval_requests (
    request_id TEXT PRIMARY KEY,
    agent TEXT NOT NULL,
    action TEXT NOT NULL,
    detail TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL DEFAULT '',
    run_id TEXT NOT NULL DEFAULT '',
    scope TEXT NOT NULL DEFAULT 'standing',
    payload TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL DEFAULT '',
    reviewed_by TEXT NOT NULL DEFAULT '',
    reviewed_at TEXT NOT NULL DEFAULT '',
    reviewed_via TEXT NOT NULL DEFAULT '',
    reviewer_subject TEXT NOT NULL DEFAULT '',
    identity_authenticated INTEGER NOT NULL DEFAULT 0,
    human_confirmed INTEGER NOT NULL DEFAULT 0,
    rationale TEXT NOT NULL DEFAULT '',
    superseded_by TEXT NOT NULL DEFAULT '',
    consumed_at TEXT NOT NULL DEFAULT '',
    consumed_by_run TEXT NOT NULL DEFAULT '',
    revoked_at TEXT NOT NULL DEFAULT '',
    revoked_reason TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_approval_status
    ON approval_requests(status, expires_at);
CREATE INDEX IF NOT EXISTS idx_approval_key
    ON approval_requests(agent, action, source);
CREATE TABLE IF NOT EXISTS approval_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

_COLUMNS = (
    "request_id",
    "agent",
    "action",
    "detail",
    "source",
    "run_id",
    "scope",
    "payload",
    "status",
    "created_at",
    "expires_at",
    "reviewed_by",
    "reviewed_at",
    "reviewed_via",
    "reviewer_subject",
    "identity_authenticated",
    "human_confirmed",
    "rationale",
    "superseded_by",
    "consumed_at",
    "consumed_by_run",
    "revoked_at",
    "revoked_reason",
)


def _new_request_id() -> str:
    return f"gate_{uuid.uuid4().hex[:12]}"


def _content_request_id(item: dict[str, Any]) -> str:
    digest = hashlib.sha256(
        json.dumps(item, sort_keys=True, ensure_ascii=False, default=str).encode("utf-8")
    ).hexdigest()
    return f"gate_{digest[:12]}"


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _iso(moment: datetime) -> str:
    """Canonical UTC timestamp; lexicographic order equals time order."""
    return moment.astimezone(timezone.utc).isoformat(timespec="microseconds")


def _parse_time(value: str) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value.strip().replace("Z", "+00:00"))
    except (AttributeError, ValueError):
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _encode_payload(payload: Any) -> str:
    """Serialize the context shown to the reviewer; informational, not parsed."""
    if payload is None or payload == "":
        return ""
    text = (
        payload
        if isinstance(payload, str)
        else json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)
    )
    if len(text) > MAX_PAYLOAD_CHARS:
        text = text[: MAX_PAYLOAD_CHARS - 32] + "\n...[payload truncated]"
    return text


def db_path_for(path: Path) -> Path:
    """Map a configured store path to the database file.

    A ``.sqlite3``/``.sqlite``/``.db`` path is the database itself. Any other
    path (the legacy ``approvals.json``) maps to a sibling ``.sqlite3`` file,
    and the JSON file next to it becomes the one-time legacy import source.
    """
    resolved = path.expanduser().resolve()
    if resolved.suffix in _DB_SUFFIXES:
        return resolved
    return resolved.with_suffix(".sqlite3")


def default_db_path() -> Path:
    """Resolve the global gate database.

    Order: ``AFS_AGENT_APPROVALS_DB``, then the legacy
    ``AFS_AGENT_APPROVALS_PATH`` (mapped through :func:`db_path_for`), then
    the shared agents state directory.
    """
    from .guardrails import _prefer_writable_state_path

    db_env = os.getenv("AFS_AGENT_APPROVALS_DB", "").strip()
    if db_env:
        return Path(db_env).expanduser().resolve()
    legacy_env = os.getenv("AFS_AGENT_APPROVALS_PATH", "").strip()
    if legacy_env:
        return db_path_for(Path(legacy_env))
    return _prefer_writable_state_path(
        "approvals.sqlite3", env_var="AFS_AGENT_APPROVALS_DB"
    )


@dataclass
class ApprovalRequest:
    agent: str
    action: str
    detail: str
    timestamp: str
    # pending, approved, rejected (human); expired, superseded, withdrawn (system)
    status: str = "pending"
    reviewed_by: str = ""
    reviewed_at: str = ""
    rationale: str = ""
    request_id: str = ""
    reviewed_via: str = ""
    reviewer_subject: str = ""
    identity_authenticated: bool = False
    human_confirmed: bool = False
    source: str = ""
    run_id: str = ""
    scope: str = "standing"
    payload: str = ""
    expires_at: str = ""
    superseded_by: str = ""
    consumed_at: str = ""
    consumed_by_run: str = ""
    revoked_at: str = ""
    revoked_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "agent": self.agent,
            "action": self.action,
            "detail": self.detail,
            "timestamp": self.timestamp,
            "status": self.status,
            "reviewed_by": self.reviewed_by,
            "reviewed_at": self.reviewed_at,
            "rationale": self.rationale,
            "request_id": self.request_id,
            "reviewed_via": self.reviewed_via,
            "reviewer_subject": self.reviewer_subject,
            "identity_authenticated": self.identity_authenticated,
            "human_confirmed": self.human_confirmed,
            "source": self.source,
            "run_id": self.run_id,
            "scope": self.scope,
            "payload": self.payload,
            "expires_at": self.expires_at,
            "superseded_by": self.superseded_by,
            "consumed_at": self.consumed_at,
            "consumed_by_run": self.consumed_by_run,
            "revoked_at": self.revoked_at,
            "revoked_reason": self.revoked_reason,
        }


class ApprovalStateError(RuntimeError):
    """The approval store is unreadable and must not be overwritten."""


def _row_to_request(row: sqlite3.Row) -> ApprovalRequest:
    return ApprovalRequest(
        agent=row["agent"],
        action=row["action"],
        detail=row["detail"],
        timestamp=row["created_at"],
        status=row["status"],
        reviewed_by=row["reviewed_by"],
        reviewed_at=row["reviewed_at"],
        rationale=row["rationale"],
        request_id=row["request_id"],
        reviewed_via=row["reviewed_via"],
        reviewer_subject=row["reviewer_subject"],
        identity_authenticated=row["identity_authenticated"] == 1,
        human_confirmed=row["human_confirmed"] == 1,
        source=row["source"],
        run_id=row["run_id"],
        scope=row["scope"],
        payload=row["payload"],
        expires_at=row["expires_at"],
        superseded_by=row["superseded_by"],
        consumed_at=row["consumed_at"],
        consumed_by_run=row["consumed_by_run"],
        revoked_at=row["revoked_at"],
        revoked_reason=row["revoked_reason"],
    )


def _legacy_row(item: Any, ttl: timedelta, now: datetime) -> dict[str, Any] | None:
    """Validate one legacy JSON record; ``None`` means skip it.

    Persisted JSON is an untrusted boundary: provenance flags count only when
    they are literally ``true`` (the string ``"false"`` is truthy in Python).
    """
    if not isinstance(item, dict):
        return None
    text_fields = (
        "agent",
        "action",
        "detail",
        "timestamp",
        "status",
        "reviewed_by",
        "reviewed_at",
        "rationale",
        "request_id",
        "reviewed_via",
        "reviewer_subject",
    )
    values: dict[str, str] = {}
    for name in text_fields:
        value = item.get(name, "")
        if value is None:
            value = ""
        if not isinstance(value, str):
            return None
        values[name] = value
    if not values["agent"] or not values["action"]:
        return None
    status = values["status"] or PENDING
    created = values["timestamp"] or _iso(now)
    expires_at = ""
    if status == PENDING:
        created_at = _parse_time(created) or now
        expires_at = _iso(created_at + ttl)
    return {
        # Legacy rows without an id get one derived from their content, so a
        # re-import never duplicates them (calibration refs must stay stable).
        "request_id": values["request_id"] or _content_request_id(item),
        "agent": values["agent"],
        "action": values["action"],
        "detail": values["detail"],
        "source": "",
        "run_id": "",
        "scope": "standing",
        "payload": "",
        "status": status,
        "created_at": created,
        "expires_at": expires_at,
        "reviewed_by": values["reviewed_by"],
        "reviewed_at": values["reviewed_at"],
        "reviewed_via": values["reviewed_via"],
        "reviewer_subject": values["reviewer_subject"],
        "identity_authenticated": 1 if item.get("identity_authenticated") is True else 0,
        "human_confirmed": 1 if item.get("human_confirmed") is True else 0,
        "rationale": values["rationale"],
        "superseded_by": "",
        "consumed_at": "",
        "consumed_by_run": "",
        "revoked_at": "",
        "revoked_reason": "",
    }


def pending_snapshot(path: Path) -> list[ApprovalRequest] | None:
    """Pending, unexpired requests, read without writing to the store.

    For status surfaces (dashboards) that must not create, migrate, import, or
    reap the database. ``None`` means no readable store at ``path``; the
    caller decides the fallback. :class:`ApprovalGate` stays the reader of
    record, and ``afs approvals list --json`` is the CLI form of it.
    """
    db_path = db_path_for(path)
    if not db_path.is_file():
        return None
    try:
        conn = sqlite3.connect(f"{db_path.as_uri()}?mode=ro", uri=True, timeout=1.0)
    except sqlite3.Error:
        return None
    try:
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT * FROM approval_requests WHERE status=? "
            "AND (expires_at='' OR expires_at>?) ORDER BY created_at, rowid",
            (PENDING, _iso(_now())),
        ).fetchall()
    except sqlite3.Error:
        return None
    finally:
        conn.close()
    return [_row_to_request(row) for row in rows]


class ApprovalGate:
    """Gate that blocks dangerous actions until human review."""

    def __init__(
        self,
        path: Path | None = None,
        *,
        ttl_hours: float = DEFAULT_TTL_HOURS,
    ):
        self._path = db_path_for(path) if path is not None else default_db_path()
        self._legacy_json = self._path.with_suffix(".json")
        self._legacy_history = self._path.with_name(f"{self._path.stem}.history.jsonl")
        self._ttl = timedelta(hours=ttl_hours)
        self._schema_ready = False

    @property
    def path(self) -> Path:
        return self._path

    def human_authorization_scope(self, decision: str, request_id: str, rationale: str) -> str:
        """Return the broker scope for one decision in this exact store."""
        from ..human_provenance import decision_scope_parts

        return decision_scope_parts(
            "approval-gate",
            decision,
            str(self._path),
            request_id,
            rationale.strip(),
        )

    def batch_authorization_scope(
        self, decision: str, request_ids: Iterable[str], rationale: str
    ) -> str:
        """Return the broker scope for one decision over an exact id set."""
        from ..human_provenance import decision_scope_parts

        return decision_scope_parts(
            "approval-gate",
            f"{decision}-batch",
            str(self._path),
            *sorted(set(request_ids)),
            rationale.strip(),
        )

    # -- storage -----------------------------------------------------------

    def _store_exists(self) -> bool:
        return (
            self._path.exists()
            or self._legacy_json.exists()
            or self._legacy_history.exists()
        )

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        try:
            conn = sqlite3.connect(str(self._path), timeout=5.0, isolation_level=None)
        except sqlite3.Error as exc:
            raise ApprovalStateError(
                f"approval store cannot be opened at {self._path}: {exc}"
            ) from exc
        try:
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA busy_timeout=5000")
            if not self._schema_ready:
                self._ensure_schema(conn)
            yield conn
        except sqlite3.OperationalError as exc:
            # Busy past the timeout, I/O failure: the store may be fine.
            raise ApprovalStateError(
                f"approval store at {self._path} is unavailable: {exc}"
            ) from exc
        except sqlite3.DatabaseError as exc:
            raise ApprovalStateError(
                f"approval store is unreadable at {self._path}; refusing to use it: {exc}"
            ) from exc
        finally:
            conn.close()

    def _ensure_schema(self, conn: sqlite3.Connection) -> None:
        version = conn.execute("PRAGMA user_version").fetchone()[0]
        if version > SCHEMA_VERSION:
            raise ApprovalStateError(
                f"approval store at {self._path} has schema v{version}; "
                f"this AFS understands v{SCHEMA_VERSION}"
            )
        if version < SCHEMA_VERSION:
            # One IMMEDIATE transaction (it waits on busy_timeout), so
            # processes opening a new store together create it exactly once.
            conn.executescript(
                f"BEGIN IMMEDIATE;{_SCHEMA}PRAGMA user_version={SCHEMA_VERSION};COMMIT;"
            )
        # Switching to WAL needs an exclusive lock, and SQLite reports
        # "database is locked" at once instead of waiting on busy_timeout
        # when another process holds one, so a first open retries briefly.
        # The mode persists in the file; later opens make this a no-op.
        deadline = time.monotonic() + 5.0
        while True:
            try:
                conn.execute("PRAGMA journal_mode=WAL")
                break
            except sqlite3.OperationalError:
                if time.monotonic() >= deadline:
                    raise
                time.sleep(0.02)
        self._schema_ready = True

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        """One write transaction: legacy sync and TTL reaping run first."""
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            try:
                self._sync_legacy(conn)
                self._reap(conn, _now())
                yield conn
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            conn.execute("COMMIT")

    def _meta(self, conn: sqlite3.Connection, key: str) -> str:
        row = conn.execute("SELECT value FROM approval_meta WHERE key=?", (key,)).fetchone()
        return str(row["value"]) if row else ""

    def _set_meta(self, conn: sqlite3.Connection, key: str, value: str) -> None:
        conn.execute(
            "INSERT INTO approval_meta(key, value) VALUES(?, ?) "
            "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, value),
        )

    def _sync_legacy(self, conn: sqlite3.Connection) -> None:
        """Import the legacy JSON stores when they changed since the last import.

        Active rows import first so they win over archived snapshots of the
        same id. A decision recorded in JSON by an older AFS process updates
        a row that is still pending here; any other existing row is left as
        is. The JSON files are only read.
        """
        for key, source, reader in (
            ("legacy_active", self._legacy_json, self._read_legacy_active),
            ("legacy_history", self._legacy_history, self._read_legacy_history),
        ):
            try:
                stat = source.stat()
            except OSError:
                continue
            signature = f"{stat.st_mtime_ns}:{stat.st_size}"
            if self._meta(conn, f"{key}_signature") == signature:
                continue
            items, unreadable = reader(source)
            now = _now()
            skipped = unreadable
            for item in items:
                row = _legacy_row(item, self._ttl, now)
                if row is None:
                    skipped += 1
                    continue
                self._upsert_legacy(conn, row)
            if skipped:
                logger.warning(
                    "Skipped %d malformed legacy approval record(s) in %s", skipped, source
                )
            self._set_meta(conn, f"{key}_signature", signature)
            self._set_meta(conn, f"{key}_skipped", str(skipped))

    @staticmethod
    def _read_legacy_active(source: Path) -> tuple[list[Any], int]:
        try:
            data = json.loads(source.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            logger.warning("Legacy approvals file %s is unreadable; skipping it", source)
            return [], 1
        if not isinstance(data, list):
            logger.warning("Legacy approvals file %s is not a JSON list; skipping it", source)
            return [], 1
        return data, 0

    @staticmethod
    def _read_legacy_history(source: Path) -> tuple[list[Any], int]:
        try:
            lines = source.read_text(encoding="utf-8").splitlines()
        except (OSError, UnicodeDecodeError):
            logger.warning("Legacy approvals history %s is unreadable; skipping it", source)
            return [], 1
        items: list[Any] = []
        unreadable = 0
        for line in lines:
            if not line.strip():
                continue
            try:
                items.append(json.loads(line))
            except json.JSONDecodeError:
                unreadable += 1
        return items, unreadable

    @staticmethod
    def _upsert_legacy(conn: sqlite3.Connection, row: dict[str, Any]) -> None:
        placeholders = ", ".join("?" for _ in _COLUMNS)
        decision_columns = (
            "status",
            "reviewed_by",
            "reviewed_at",
            "reviewed_via",
            "reviewer_subject",
            "identity_authenticated",
            "human_confirmed",
            "rationale",
        )
        updates = ", ".join(f"{name}=excluded.{name}" for name in decision_columns)
        conn.execute(
            f"INSERT INTO approval_requests ({', '.join(_COLUMNS)}) VALUES ({placeholders}) "
            f"ON CONFLICT(request_id) DO UPDATE SET {updates}, expires_at='' "
            "WHERE approval_requests.status='pending' AND excluded.status!='pending'",
            tuple(row[name] for name in _COLUMNS),
        )

    def _reap(self, conn: sqlite3.Connection, now: datetime) -> int:
        """Expire pending requests whose TTL ran out (a system state)."""
        stamp = _iso(now)
        cursor = conn.execute(
            "UPDATE approval_requests SET status='expired', reviewed_by='system', "
            "reviewed_at=?, reviewed_via='ttl', "
            "rationale='expired: no decision before the TTL' "
            "WHERE status='pending' AND expires_at!='' AND expires_at<=?",
            (stamp, stamp),
        )
        if cursor.rowcount:
            logger.info("Expired %d approval request(s) past their TTL", cursor.rowcount)
        return cursor.rowcount

    def _select(self, sql: str, params: tuple[Any, ...] = ()) -> list[ApprovalRequest]:
        if not self._store_exists():
            return []
        with self._transaction() as conn:
            rows = conn.execute(sql, params).fetchall()
        return [_row_to_request(row) for row in rows]

    # -- reads -------------------------------------------------------------

    def all_requests(self) -> list[ApprovalRequest]:
        """Return every request, oldest first, including decided ones."""
        return self._select("SELECT * FROM approval_requests ORDER BY created_at, rowid")

    def pending_requests(
        self, agent: str | None = None, source: str | None = None
    ) -> list[ApprovalRequest]:
        sql = "SELECT * FROM approval_requests WHERE status='pending'"
        params: list[Any] = []
        if agent is not None:
            sql += " AND agent=?"
            params.append(agent)
        if source is not None:
            sql += " AND source=?"
            params.append(source)
        return self._select(sql + " ORDER BY created_at, rowid", tuple(params))

    def get(self, request_id: str) -> ApprovalRequest | None:
        rows = self._select(
            "SELECT * FROM approval_requests WHERE request_id=?", (request_id,)
        )
        return rows[0] if rows else None

    def find_pending(
        self,
        agent: str = "",
        action: str = "",
        request_id: str | None = None,
    ) -> ApprovalRequest | None:
        """Find one pending request.

        With ``request_id`` the match is exact. Without it, the oldest pending
        request for ``agent``/``action`` wins, which is ambiguous when several
        requests share the pair (one per mission, say) — surfaces that let a
        person pick a specific row should pass the id.
        """
        if request_id:
            rows = self._select(
                "SELECT * FROM approval_requests WHERE request_id=? AND status='pending'",
                (request_id,),
            )
        else:
            rows = self._select(
                "SELECT * FROM approval_requests WHERE agent=? AND action=? "
                "AND status='pending' ORDER BY created_at, rowid LIMIT 1",
                (agent, action),
            )
        return rows[0] if rows else None

    def stale_pending(self, older_than_hours: float) -> list[ApprovalRequest]:
        """Pending requests created at least ``older_than_hours`` ago."""
        cutoff = _now() - timedelta(hours=older_than_hours)
        stale: list[ApprovalRequest] = []
        for request in self.pending_requests():
            created = _parse_time(request.timestamp)
            if created is None or created <= cutoff:
                stale.append(request)
        return stale

    def health(self) -> dict[str, Any]:
        """Store location, counts, and skipped legacy rows for status surfaces."""
        report: dict[str, Any] = {
            "db": str(self._path),
            "exists": self._path.exists(),
            "schema_version": SCHEMA_VERSION,
            "pending": 0,
            "legacy_skipped": 0,
        }
        if not self._store_exists():
            return report
        with self._transaction() as conn:
            report["pending"] = conn.execute(
                "SELECT COUNT(*) FROM approval_requests WHERE status='pending'"
            ).fetchone()[0]
            report["legacy_skipped"] = sum(
                int(self._meta(conn, f"{key}_skipped") or 0)
                for key in ("legacy_active", "legacy_history")
            )
        report["exists"] = True
        return report

    # -- requests ----------------------------------------------------------

    def check(
        self,
        agent: str,
        action: str,
        detail: str = "",
        *,
        source: str = "",
        run_id: str = "",
        scope: str = "standing",
        payload: Any = None,
        force: bool = False,
    ) -> bool:
        """Return True when the action is allowed; otherwise queue a request.

        ``source`` names the requester (``mission:<name>``) so a newer request
        from it supersedes its older pending ones; ``run_id`` ties a ``once``
        approval to the run that consumed it. ``force`` gates an action even
        when it is in :data:`AUTO_APPROVE`.
        """
        if action in AUTO_APPROVE and not force:
            return True
        if scope not in SCOPES:
            raise ValueError(f"scope must be one of {SCOPES}, got {scope!r}")
        now = _now()
        key = (agent, action, detail, source)
        with self._transaction() as conn:
            approved = conn.execute(
                "SELECT * FROM approval_requests WHERE agent=? AND action=? AND detail=? "
                "AND source=? AND status='approved' AND human_confirmed=1 "
                "AND revoked_at='' ORDER BY reviewed_at DESC, rowid DESC",
                key,
            ).fetchall()
            for row in approved:
                if row["scope"] != "once":
                    return True
                if not row["consumed_at"]:
                    conn.execute(
                        "UPDATE approval_requests SET consumed_at=?, consumed_by_run=? "
                        "WHERE request_id=?",
                        (_iso(now), run_id, row["request_id"]),
                    )
                    return True
                if run_id and row["consumed_by_run"] == run_id:
                    return True
            if run_id and conn.execute(
                "SELECT 1 FROM approval_requests WHERE agent=? AND action=? AND detail=? "
                "AND source=? AND run_id=? AND status='rejected' LIMIT 1",
                (*key, run_id),
            ).fetchone():
                return False
            if conn.execute(
                "SELECT 1 FROM approval_requests WHERE agent=? AND action=? AND detail=? "
                "AND source=? AND run_id=? AND status='pending' LIMIT 1",
                (*key, run_id),
            ).fetchone():
                return False
            request_id = _new_request_id()
            superseded = conn.execute(
                "UPDATE approval_requests SET status='superseded', superseded_by=?, "
                "reviewed_by='system', reviewed_at=?, reviewed_via='supersede', "
                "rationale='superseded by a newer request from the same source' "
                "WHERE status='pending' AND agent=? AND action=? AND source=? "
                "AND (?!='' OR detail=?)",
                (request_id, _iso(now), agent, action, source, source, detail),
            ).rowcount
            row = {
                "request_id": request_id,
                "agent": agent,
                "action": action,
                "detail": detail,
                "source": source,
                "run_id": run_id,
                "scope": scope,
                "payload": _encode_payload(payload),
                "status": PENDING,
                "created_at": _iso(now),
                "expires_at": _iso(now + self._ttl),
            }
            conn.execute(
                f"INSERT INTO approval_requests ({', '.join(row)}) "
                f"VALUES ({', '.join('?' for _ in row)})",
                tuple(row.values()),
            )
        logger.info(
            "Queued approval request %s: agent=%s action=%s source=%s superseded=%d",
            request_id,
            agent,
            action,
            source,
            superseded,
        )
        return False

    def withdraw(
        self,
        agent: str,
        *,
        action: str = "",
        source: str = "",
        reason: str = "",
    ) -> int:
        """Close pending requests the requester no longer needs.

        A system state, not a human decision. Returns the number withdrawn.
        """
        if not action and not source:
            raise ValueError("withdraw needs an action or a source")
        sql = (
            "UPDATE approval_requests SET status='withdrawn', reviewed_by='system', "
            "reviewed_at=?, reviewed_via='withdraw', rationale=? "
            "WHERE status='pending' AND agent=? AND source=?"
        )
        params: list[Any] = [
            _iso(_now()),
            reason.strip() or "withdrawn by the requester",
            agent,
            source,
        ]
        if action:
            sql += " AND action=?"
            params.append(action)
        if not self._store_exists():
            return 0
        with self._transaction() as conn:
            return conn.execute(sql, tuple(params)).rowcount

    def revoke(self, request_id: str, *, reason: str) -> bool:
        """Stop an approval from authorizing further checks.

        Revoking only removes permission, so it needs no terminal
        confirmation. The human decision stays in history for calibration.
        """
        if not reason.strip():
            raise ValueError("a reason is required to revoke an approval")
        if not self._store_exists():
            return False
        with self._transaction() as conn:
            return (
                conn.execute(
                    "UPDATE approval_requests SET revoked_at=?, revoked_reason=? "
                    "WHERE request_id=? AND status='approved' AND revoked_at=''",
                    (_iso(_now()), reason.strip(), request_id),
                ).rowcount
                == 1
            )

    def clear_completed(
        self, older_than_days: float = SYSTEM_RETENTION_DAYS
    ) -> tuple[int, int]:
        """Prune old system-closed rows; human decisions are kept forever.

        Returns ``(pruned, pending)``.
        """
        if not self._store_exists():
            return 0, 0
        cutoff = _iso(_now() - timedelta(days=older_than_days))
        placeholders = ", ".join("?" for _ in SYSTEM_CLOSED)
        with self._transaction() as conn:
            pruned = conn.execute(
                f"DELETE FROM approval_requests WHERE status IN ({placeholders}) "
                "AND reviewed_at<=?",
                (*SYSTEM_CLOSED, cutoff),
            ).rowcount
            pending = conn.execute(
                "SELECT COUNT(*) FROM approval_requests WHERE status='pending'"
            ).fetchone()[0]
        return pruned, pending

    # -- decisions ---------------------------------------------------------

    def approve(
        self,
        agent: str,
        action: str,
        reviewer: str = "human",
        *,
        rationale: str = "",
        reviewed_via: str = "",
    ) -> bool:
        """Record a non-authoritative programmatic approval decision.

        Compatibility callers may still record the result, but caller-supplied
        reviewer/via strings are untrusted and this record never authorizes
        :meth:`check`. Use :meth:`approve_human` with a broker capability for
        an authoritative approval.
        """
        return self._resolve(
            agent,
            action,
            "approved",
            "unauthenticated",
            rationale,
            "programmatic",
            reviewer_subject="",
            identity_authenticated=False,
            human_confirmed=False,
            expected_request_id=None,
        )

    def approve_human(
        self,
        agent: str = "",
        action: str = "",
        *,
        rationale: str,
        authorization: Any,
        request_id: str | None = None,
    ) -> bool:
        """Authorize a pending request using a broker-minted capability."""
        return self._decide_human(
            "approve", "approved", agent, action, rationale, authorization, request_id
        )

    def reject(
        self,
        agent: str,
        action: str,
        reviewer: str = "human",
        *,
        rationale: str = "",
        reviewed_via: str = "",
    ) -> bool:
        """Record a fail-safe, non-authoritative programmatic rejection."""
        return self._resolve(
            agent,
            action,
            "rejected",
            "unauthenticated",
            rationale,
            "programmatic",
            reviewer_subject="",
            identity_authenticated=False,
            human_confirmed=False,
            expected_request_id=None,
        )

    def reject_human(
        self,
        agent: str = "",
        action: str = "",
        *,
        rationale: str,
        authorization: Any,
        request_id: str | None = None,
    ) -> bool:
        """Record a human-confirmed rejection using a broker capability."""
        return self._decide_human(
            "reject", "rejected", agent, action, rationale, authorization, request_id
        )

    def reject_human_batch(
        self,
        request_ids: Iterable[str],
        *,
        rationale: str,
        authorization: Any,
    ) -> list[str]:
        """Reject an exact set of pending requests under one human confirmation.

        The authorization scope covers the sorted id set and rationale, so it
        cannot be replayed against a different set. Returns the ids rejected;
        ids that stopped being pending in the meantime are skipped.
        """
        from ..human_provenance import consume_human_authorization

        if not rationale.strip():
            raise ValueError("a rationale is required for a human rejection")
        ids = sorted(set(request_ids))
        if not ids:
            return []
        scope = self.batch_authorization_scope("reject", ids, rationale)
        if not consume_human_authorization(authorization, scope=scope):
            raise ValueError("a HumanDecisionBroker authorization is required")
        identity = authorization.identity
        rejected: list[str] = []
        with self._transaction() as conn:
            for request_id in ids:
                if self._mark_decided(
                    conn,
                    request_id,
                    "rejected",
                    identity.reviewer,
                    rationale,
                    authorization.confirmed_via,
                    reviewer_subject=identity.subject,
                    identity_authenticated=identity.authenticated,
                    human_confirmed=True,
                ):
                    rejected.append(request_id)
        return rejected

    def _decide_human(
        self,
        decision: str,
        status: str,
        agent: str,
        action: str,
        rationale: str,
        authorization: Any,
        request_id: str | None,
    ) -> bool:
        from ..human_provenance import consume_human_authorization

        if not rationale.strip():
            noun = "approval" if decision == "approve" else "rejection"
            raise ValueError(f"a rationale is required for a human {noun}")
        request = self.find_pending(agent, action, request_id=request_id)
        if request is None:
            return False
        scope = self.human_authorization_scope(decision, request.request_id, rationale)
        if not consume_human_authorization(authorization, scope=scope):
            raise ValueError("a HumanDecisionBroker authorization is required")
        identity = authorization.identity
        return self._resolve(
            request.agent,
            request.action,
            status,
            identity.reviewer,
            rationale,
            authorization.confirmed_via,
            reviewer_subject=identity.subject,
            identity_authenticated=identity.authenticated,
            human_confirmed=True,
            expected_request_id=request.request_id,
        )

    def _resolve(
        self,
        agent: str,
        action: str,
        status: str,
        reviewer: str,
        rationale: str,
        reviewed_via: str,
        *,
        reviewer_subject: str,
        identity_authenticated: bool,
        human_confirmed: bool,
        expected_request_id: str | None,
    ) -> bool:
        """Resolve one pending request inside a single write transaction."""
        if not self._store_exists():
            return False
        with self._transaction() as conn:
            if expected_request_id is not None:
                row = conn.execute(
                    "SELECT request_id FROM approval_requests WHERE request_id=? "
                    "AND agent=? AND action=? AND status='pending'",
                    (expected_request_id, agent, action),
                ).fetchone()
            else:
                row = conn.execute(
                    "SELECT request_id FROM approval_requests WHERE agent=? AND action=? "
                    "AND status='pending' ORDER BY created_at, rowid LIMIT 1",
                    (agent, action),
                ).fetchone()
            if row is None:
                return False
            return self._mark_decided(
                conn,
                row["request_id"],
                status,
                reviewer,
                rationale,
                reviewed_via,
                reviewer_subject=reviewer_subject,
                identity_authenticated=identity_authenticated,
                human_confirmed=human_confirmed,
            )

    @staticmethod
    def _mark_decided(
        conn: sqlite3.Connection,
        request_id: str,
        status: str,
        reviewer: str,
        rationale: str,
        reviewed_via: str,
        *,
        reviewer_subject: str,
        identity_authenticated: bool,
        human_confirmed: bool,
    ) -> bool:
        cursor = conn.execute(
            "UPDATE approval_requests SET status=?, reviewed_by=?, reviewed_at=?, "
            "rationale=?, reviewed_via=?, reviewer_subject=?, "
            "identity_authenticated=?, human_confirmed=? "
            "WHERE request_id=? AND status='pending'",
            (
                status,
                reviewer,
                _iso(_now()),
                rationale.strip(),
                reviewed_via,
                reviewer_subject,
                1 if identity_authenticated else 0,
                1 if human_confirmed else 0,
                request_id,
            ),
        )
        return cursor.rowcount == 1
