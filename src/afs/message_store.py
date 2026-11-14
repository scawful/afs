"""SQLite store for inter-agent messages.

Scratchpad and handoff documents stay files: agents read those paths
directly. Mail does not. A message is a row with a recipient index, an
ack label (``unread`` / ``read`` / ``done``), and ``reply_to`` for the
thread. Listing an inbox is a query, not a scan of one JSON file per
message.

Legacy ``<agent>/<id>.json`` and ``.acks/<id>.json`` files are imported
once. New messages are not written back out as JSON.
"""

from __future__ import annotations

import json
import re
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .path_safety import assert_no_linklike_components

SCHEMA_VERSION = 2
DB_FILENAME = "messages.sqlite3"
_BROADCAST_TOKENS = frozenset({"all", "*", "everyone"})
_RECIPIENT_SEPARATORS = re.compile(r"[,/;]+")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    from_agent TEXT NOT NULL,
    from_norm TEXT NOT NULL,
    to_raw TEXT,
    broadcast INTEGER NOT NULL DEFAULT 0,
    msg_type TEXT NOT NULL,
    topic TEXT,
    payload_json TEXT NOT NULL,
    timestamp TEXT NOT NULL,
    expires_at TEXT,
    scope_id TEXT NOT NULL DEFAULT '',
    reply_to TEXT,
    revision INTEGER NOT NULL DEFAULT 1
);
CREATE INDEX IF NOT EXISTS idx_messages_from_ts ON messages(from_agent, timestamp);
CREATE INDEX IF NOT EXISTS idx_messages_scope_ts ON messages(scope_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_messages_ts ON messages(timestamp);

CREATE TABLE IF NOT EXISTS message_recipients (
    message_id TEXT NOT NULL,
    token TEXT NOT NULL,
    family TEXT NOT NULL,
    PRIMARY KEY (message_id, token)
);
CREATE INDEX IF NOT EXISTS idx_recipients_token ON message_recipients(token);
CREATE INDEX IF NOT EXISTS idx_recipients_family ON message_recipients(family);

CREATE TABLE IF NOT EXISTS message_acks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    message_id TEXT NOT NULL,
    by_norm TEXT NOT NULL,
    status TEXT NOT NULL,
    note TEXT NOT NULL DEFAULT '',
    at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_acks_message ON message_acks(message_id, by_norm);

CREATE TABLE IF NOT EXISTS subscriptions (
    agent_name TEXT PRIMARY KEY,
    topics_json TEXT NOT NULL,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    ttl_hours INTEGER
);

CREATE TABLE IF NOT EXISTS sessions (
    host TEXT NOT NULL,
    session_id TEXT NOT NULL,
    scope_id TEXT NOT NULL DEFAULT '',
    cwd TEXT NOT NULL DEFAULT '',
    source TEXT NOT NULL DEFAULT '',
    registered_at TEXT NOT NULL,
    extra_json TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (host, session_id)
);
CREATE INDEX IF NOT EXISTS idx_sessions_host_reg ON sessions(host, registered_at);

CREATE TABLE IF NOT EXISTS delivery_claims (
    host TEXT NOT NULL,
    session_id TEXT NOT NULL,
    message_id TEXT NOT NULL,
    claimed_at TEXT NOT NULL,
    PRIMARY KEY (host, session_id, message_id)
);

CREATE TABLE IF NOT EXISTS store_meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def normalize_agent_name(name: str) -> str:
    """Comparable agent label. Kept aligned with ``hivemind.normalize_agent_name``."""
    return re.sub(r"[\s_]+", "-", str(name).strip().lower()).strip("-")


def _family(name: str) -> str:
    return name.split("-", 1)[0]


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class MessageStoreError(RuntimeError):
    """The message database cannot be read or written."""


class MessageStore:
    """One database per message-mount root."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.path = root / DB_FILENAME

    def _connect(self) -> sqlite3.Connection:
        self.root.mkdir(parents=True, exist_ok=True)
        assert_no_linklike_components(self.root, allow_missing=False)
        if self.path.exists() or self.path.is_symlink():
            assert_no_linklike_components(self.path, boundary=self.root, allow_missing=False)
        try:
            connection = sqlite3.connect(str(self.path), timeout=5.0, isolation_level=None)
        except sqlite3.Error as exc:
            raise MessageStoreError(f"message store cannot be opened at {self.path}: {exc}") from exc
        connection.row_factory = sqlite3.Row
        try:
            connection.execute("PRAGMA busy_timeout=5000")
            self._ensure_schema(connection)
            try:
                self.path.chmod(0o600)
            except OSError:
                pass
        except Exception:
            connection.close()
            raise
        return connection

    def _ensure_schema(self, connection: sqlite3.Connection) -> None:
        version = int(connection.execute("PRAGMA user_version").fetchone()[0])
        if version > SCHEMA_VERSION:
            raise MessageStoreError(
                f"message store at {self.path} has schema v{version}; "
                f"this AFS understands v{SCHEMA_VERSION}"
            )
        if version < SCHEMA_VERSION:
            connection.executescript(
                f"BEGIN IMMEDIATE;{_SCHEMA}PRAGMA user_version={SCHEMA_VERSION};COMMIT;"
            )
        connection.execute("PRAGMA journal_mode=WAL")

    def ensure_imported(self) -> None:
        """Import legacy JSON once. Link-like queue entries are refused."""
        with closing(self._connect()) as connection:
            self._import_legacy(connection)

    def insert(self, message: Any) -> None:
        """Insert one message. ``message`` is a ``HivemindMessage``."""
        payload = message.payload if isinstance(message.payload, dict) else {}
        to_raw = message.to if isinstance(message.to, str) else None
        broadcast = 1 if to_raw is None else 0
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                connection.execute(
                    """
                    INSERT INTO messages (
                        id, from_agent, from_norm, to_raw, broadcast, msg_type,
                        topic, payload_json, timestamp, expires_at, scope_id, reply_to
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        message.id,
                        message.from_agent,
                        normalize_agent_name(message.from_agent),
                        to_raw,
                        broadcast,
                        message.msg_type,
                        message.topic,
                        json.dumps(payload, ensure_ascii=False),
                        message.timestamp,
                        message.expires_at,
                        message.scope_id or "",
                        message.reply_to,
                    ),
                )
                self._insert_recipients(connection, message.id, to_raw)
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise

    def get(self, message_id: str) -> dict[str, Any] | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM messages WHERE id = ?",
                (message_id,),
            ).fetchone()
        return _row_to_dict(row) if row is not None else None

    def update_fields(
        self,
        message_id: str,
        *,
        timestamp: str | None = None,
        expires_at: str | None = None,
    ) -> None:
        """Adjust stored clocks. Used by retention tests and repairs."""
        assignments: list[str] = ["revision = revision + 1"]
        params: list[Any] = []
        if timestamp is not None:
            assignments.append("timestamp = ?")
            params.append(timestamp)
        if expires_at is not None:
            assignments.append("expires_at = ?")
            params.append(expires_at)
        params.append(message_id)
        with closing(self._connect()) as connection:
            connection.execute(
                f"UPDATE messages SET {', '.join(assignments)} WHERE id = ?",
                params,
            )

    def query(
        self,
        *,
        from_agent: str | None = None,
        msg_type: str | None = None,
        topic: str | None = None,
        since_iso: str | None = None,
        now_iso: str | None = None,
        scope_ids: set[str] | None = None,
        include_unscoped: bool = True,
        require_scope: bool = False,
        recipient_norm: str | None = None,
        recipient_family: str | None = None,
        exclude_from_norm: str | None = None,
        include_broadcasts: bool = True,
        exclude_done_for: str | None = None,
        limit: int = 0,
    ) -> list[dict[str, Any]]:
        clauses = ["(expires_at IS NULL OR expires_at > ?)"]
        params: list[Any] = [now_iso or _utcnow().isoformat()]
        if from_agent is not None:
            clauses.append("from_agent = ?")
            params.append(from_agent)
        if msg_type is not None:
            clauses.append("msg_type = ?")
            params.append(msg_type)
        if topic is not None:
            clauses.append("topic = ?")
            params.append(topic)
        if since_iso is not None:
            clauses.append("timestamp >= ?")
            params.append(since_iso)
        if require_scope:
            clauses.append("scope_id != ''")
        if scope_ids is not None:
            if not scope_ids and not include_unscoped:
                return []
            marks = ", ".join("?" for _ in scope_ids)
            scope_params = sorted(scope_ids)
            if include_unscoped:
                clauses.append(f"(scope_id IN ({marks}) OR scope_id = '')" if marks else "scope_id = ''")
            else:
                clauses.append(f"scope_id IN ({marks})")
            params.extend(scope_params)
        if recipient_norm is not None:
            family = recipient_family if recipient_family is not None else _family(recipient_norm)
            clauses.append(
                """
                (
                    broadcast = 1
                    OR EXISTS (
                        SELECT 1 FROM message_recipients AS r
                        WHERE r.message_id = messages.id
                          AND (
                            r.token = ?
                            OR r.family = ?
                            OR r.token = ?
                            OR r.token IN ('all', '*', 'everyone')
                          )
                    )
                )
                """
            )
            params.extend((recipient_norm, recipient_norm, family))
        if not include_broadcasts:
            clauses.append("to_raw IS NOT NULL")
        if exclude_from_norm:
            clauses.append("from_norm != ?")
            params.append(exclude_from_norm)
        if exclude_done_for:
            clauses.append(
                """
                NOT EXISTS (
                    SELECT 1 FROM message_acks AS a
                    WHERE a.message_id = messages.id
                      AND a.by_norm = ?
                      AND a.status = 'done'
                )
                """
            )
            params.append(exclude_done_for)
        where = " AND ".join(clauses)
        if limit > 0:
            sql = f"""
                SELECT * FROM (
                    SELECT * FROM messages
                    WHERE {where}
                    ORDER BY timestamp DESC, id DESC
                    LIMIT ?
                )
                ORDER BY timestamp ASC, id ASC
            """
            params.append(limit)
        else:
            sql = f"SELECT * FROM messages WHERE {where} ORDER BY timestamp ASC, id ASC"
        with closing(self._connect()) as connection:
            rows = connection.execute(sql, params).fetchall()
        return [_row_to_dict(row) for row in rows]

    def acks(self, message_id: str) -> list[dict[str, str]]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT by_norm, status, note, at
                FROM message_acks
                WHERE message_id = ?
                ORDER BY id ASC
                """,
                (message_id,),
            ).fetchall()
        records: list[dict[str, str]] = []
        for row in rows:
            record = {"by": row["by_norm"], "status": row["status"], "at": row["at"]}
            if row["note"]:
                record["note"] = row["note"]
            records.append(record)
        return records

    def states_for(self, message_ids: list[str], recipient_norm: str) -> dict[str, str]:
        if not message_ids:
            return {}
        marks = ", ".join("?" for _ in message_ids)
        with closing(self._connect()) as connection:
            rows = connection.execute(
                f"""
                SELECT message_id, status FROM message_acks
                WHERE by_norm = ? AND message_id IN ({marks})
                """,
                (recipient_norm, *message_ids),
            ).fetchall()
        states: dict[str, str] = {}
        for row in rows:
            current = states.get(row["message_id"], "unread")
            if row["status"] == "done" or current == "done":
                states[row["message_id"]] = "done"
            else:
                states[row["message_id"]] = "read"
        return states

    def ack_state(self, message_id: str, recipient_norm: str) -> str:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT status FROM message_acks
                WHERE message_id = ? AND by_norm = ?
                """,
                (message_id, recipient_norm),
            ).fetchall()
        state = "unread"
        for row in rows:
            if row["status"] == "done":
                return "done"
            state = "read"
        return state

    def add_ack(
        self,
        message_id: str,
        by_norm: str,
        *,
        status: str,
        note: str,
        at: str,
    ) -> list[dict[str, str]]:
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                found = connection.execute(
                    "SELECT 1 FROM messages WHERE id = ?",
                    (message_id,),
                ).fetchone()
                if found is None:
                    connection.execute("ROLLBACK")
                    raise KeyError(f"message not found: {message_id}")
                connection.execute(
                    """
                    INSERT INTO message_acks (message_id, by_norm, status, note, at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (message_id, by_norm, status, note, at),
                )
                connection.execute(
                    "UPDATE messages SET revision = revision + 1 WHERE id = ?",
                    (message_id,),
                )
                connection.execute("COMMIT")
            except KeyError:
                raise
            except Exception:
                connection.execute("ROLLBACK")
                raise
        return self.acks(message_id)

    def cleanup_stats(
        self,
        *,
        max_age_hours: int,
        now: datetime | None = None,
        dry_run: bool = False,
    ) -> dict[str, Any]:
        current = now or _utcnow()
        now_iso = current.isoformat()
        cutoff_iso = datetime.fromtimestamp(
            current.timestamp() - (max_age_hours * 3600),
            tz=timezone.utc,
        ).isoformat()
        with closing(self._connect()) as connection:
            rows = connection.execute("SELECT * FROM messages").fetchall()
            removed_count = 0
            remaining_count = 0
            expired_count = 0
            aged_out_count = 0
            oldest_remaining: str | None = None
            per_agent: dict[str, dict[str, int]] = {}
            delete_ids: list[str] = []
            for row in rows:
                agent = str(row["from_agent"])
                stats = per_agent.setdefault(
                    agent,
                    {"removed": 0, "remaining": 0, "expired": 0, "aged_out": 0},
                )
                expired = bool(row["expires_at"]) and str(row["expires_at"]) <= now_iso
                awaiting = row["to_raw"] is not None and not _has_done_ack(connection, row["id"])
                aged_out = str(row["timestamp"]) < cutoff_iso and not awaiting
                if expired or aged_out:
                    delete_ids.append(str(row["id"]))
                    stats["removed"] += 1
                    removed_count += 1
                    if expired:
                        stats["expired"] += 1
                        expired_count += 1
                    if aged_out:
                        stats["aged_out"] += 1
                        aged_out_count += 1
                else:
                    stats["remaining"] += 1
                    remaining_count += 1
                    stamp = str(row["timestamp"])
                    if oldest_remaining is None or stamp < oldest_remaining:
                        oldest_remaining = stamp
            if delete_ids and not dry_run:
                connection.execute("BEGIN IMMEDIATE")
                try:
                    for message_id in delete_ids:
                        connection.execute("DELETE FROM message_acks WHERE message_id = ?", (message_id,))
                        connection.execute(
                            "DELETE FROM message_recipients WHERE message_id = ?",
                            (message_id,),
                        )
                        connection.execute("DELETE FROM messages WHERE id = ?", (message_id,))
                    connection.execute("COMMIT")
                except Exception:
                    connection.execute("ROLLBACK")
                    raise
        return {
            "removed_count": removed_count,
            "remaining_count": remaining_count,
            "expired_count": expired_count,
            "aged_out_count": aged_out_count,
            "oldest_remaining": oldest_remaining,
            "per_agent": per_agent,
        }

    def upsert_subscription(
        self,
        agent_name: str,
        topics: list[str],
        *,
        created_at: str,
        updated_at: str,
        ttl_hours: int | None = None,
    ) -> dict[str, Any]:
        with closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT INTO subscriptions (agent_name, topics_json, created_at, updated_at, ttl_hours)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT(agent_name) DO UPDATE SET
                    topics_json = excluded.topics_json,
                    updated_at = excluded.updated_at,
                    ttl_hours = excluded.ttl_hours
                """,
                (
                    agent_name,
                    json.dumps(list(topics), ensure_ascii=False),
                    created_at,
                    updated_at,
                    ttl_hours,
                ),
            )
        return {
            "agent_name": agent_name,
            "topics": list(topics),
            "created_at": created_at,
            "updated_at": updated_at,
            "ttl_hours": ttl_hours,
        }

    def get_subscription(self, agent_name: str) -> dict[str, Any] | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM subscriptions WHERE agent_name = ?",
                (agent_name,),
            ).fetchone()
        if row is None:
            return None
        topics = json.loads(row["topics_json"] or "[]")
        if not isinstance(topics, list):
            topics = []
        return {
            "agent_name": row["agent_name"],
            "topics": [str(item) for item in topics],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "ttl_hours": row["ttl_hours"],
        }

    def upsert_session(self, record: dict[str, Any]) -> dict[str, Any]:
        host = str(record["host"])
        session_id = str(record["session_id"])
        known = {"host", "session_id", "scope_id", "cwd", "source", "registered_at"}
        extra = {key: value for key, value in record.items() if key not in known}
        with closing(self._connect()) as connection:
            connection.execute(
                """
                INSERT INTO sessions (
                    host, session_id, scope_id, cwd, source, registered_at, extra_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(host, session_id) DO UPDATE SET
                    scope_id = excluded.scope_id,
                    cwd = excluded.cwd,
                    source = excluded.source,
                    registered_at = excluded.registered_at,
                    extra_json = excluded.extra_json
                """,
                (
                    host,
                    session_id,
                    str(record.get("scope_id", "") or ""),
                    str(record.get("cwd", "") or ""),
                    str(record.get("source", "") or ""),
                    str(record["registered_at"]),
                    json.dumps(extra, ensure_ascii=False),
                ),
            )
        return self.get_session(host, session_id) or dict(record)

    def get_session(self, host: str, session_id: str) -> dict[str, Any] | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM sessions WHERE host = ? AND session_id = ?",
                (host, session_id),
            ).fetchone()
        return _session_row(row) if row is not None else None

    def list_sessions(self, host: str, *, scope_id: str | None = None) -> list[dict[str, Any]]:
        with closing(self._connect()) as connection:
            if scope_id is None:
                rows = connection.execute(
                    """
                    SELECT * FROM sessions
                    WHERE host = ?
                    ORDER BY registered_at DESC, session_id DESC
                    """,
                    (host,),
                ).fetchall()
            else:
                rows = connection.execute(
                    """
                    SELECT * FROM sessions
                    WHERE host = ? AND scope_id = ?
                    ORDER BY registered_at DESC, session_id DESC
                    """,
                    (host, scope_id),
                ).fetchall()
        return [_session_row(row) for row in rows if row is not None]

    def prune_sessions(self, *, retention_days: int) -> int:
        cutoff = datetime.fromtimestamp(
            _utcnow().timestamp() - retention_days * 86400,
            tz=timezone.utc,
        ).isoformat()
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                stale = connection.execute(
                    "SELECT host, session_id FROM sessions WHERE registered_at < ?",
                    (cutoff,),
                ).fetchall()
                for row in stale:
                    connection.execute(
                        "DELETE FROM delivery_claims WHERE host = ? AND session_id = ?",
                        (row["host"], row["session_id"]),
                    )
                    connection.execute(
                        "DELETE FROM sessions WHERE host = ? AND session_id = ?",
                        (row["host"], row["session_id"]),
                    )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
            return len(stale)

    def delivered_ids(self, host: str, session_id: str) -> set[str]:
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT message_id FROM delivery_claims
                WHERE host = ? AND session_id = ?
                """,
                (host, session_id),
            ).fetchall()
        return {str(row["message_id"]) for row in rows}

    def claim_delivery(self, host: str, session_id: str, message_id: str, *, claimed_at: str) -> bool:
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            try:
                existing = connection.execute(
                    """
                    SELECT 1 FROM delivery_claims
                    WHERE host = ? AND session_id = ? AND message_id = ?
                    """,
                    (host, session_id, message_id),
                ).fetchone()
                if existing is not None:
                    connection.execute("ROLLBACK")
                    return False
                connection.execute(
                    """
                    INSERT INTO delivery_claims (host, session_id, message_id, claimed_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (host, session_id, message_id, claimed_at),
                )
                connection.execute("COMMIT")
            except Exception:
                connection.execute("ROLLBACK")
                raise
            return True

    def iter_inventory(self) -> list[dict[str, Any]]:
        """Lightweight rows for watchers that used to stat JSON files."""
        if not self.path.exists():
            return []
        with closing(self._connect()) as connection:
            rows = connection.execute(
                """
                SELECT id, from_agent, timestamp, revision,
                       length(payload_json) AS nbytes
                FROM messages
                """
            ).fetchall()
        return [
            {
                "id": row["id"],
                "from_agent": row["from_agent"],
                "timestamp": row["timestamp"],
                "revision": int(row["revision"]),
                "nbytes": int(row["nbytes"] or 0),
            }
            for row in rows
        ]

    def message_json_bytes(self, message_id: str) -> bytes | None:
        row = self.get(message_id)
        if row is None:
            return None
        return json.dumps(_public_dict(row), ensure_ascii=False).encode("utf-8")

    def _import_legacy(self, connection: sqlite3.Connection) -> None:
        if (
            _meta(connection, "legacy_import") == "1"
            and _meta(connection, "mailbox_sidecars_import") == "1"
        ):
            return
        if not self.root.exists():
            _set_meta(connection, "legacy_import", "1")
            _set_meta(connection, "mailbox_sidecars_import", "1")
            return
        connection.execute("BEGIN IMMEDIATE")
        try:
            if _meta(connection, "legacy_import") != "1":
                for child in self.root.iterdir():
                    if child.name.startswith(".") or child.name == DB_FILENAME:
                        continue
                    if child.name.endswith("-wal") or child.name.endswith("-shm"):
                        continue
                    assert_no_linklike_components(child, boundary=self.root, allow_missing=False)
                    if not child.is_dir():
                        continue
                    for msg_file in child.glob("*.json"):
                        assert_no_linklike_components(
                            msg_file,
                            boundary=self.root,
                            allow_missing=False,
                        )
                        self._import_message_file(connection, msg_file)
                ack_dir = self.root / ".acks"
                if ack_dir.exists():
                    assert_no_linklike_components(
                        ack_dir,
                        boundary=self.root,
                        allow_missing=False,
                    )
                    if ack_dir.is_dir():
                        for ack_file in ack_dir.glob("*.json"):
                            assert_no_linklike_components(
                                ack_file,
                                boundary=self.root,
                                allow_missing=False,
                            )
                            self._import_ack_file(connection, ack_file)
                _set_meta(connection, "legacy_import", "1")
            if _meta(connection, "mailbox_sidecars_import") != "1":
                self._import_mailbox_sidecars(connection)
                _set_meta(connection, "mailbox_sidecars_import", "1")
            connection.execute("COMMIT")
        except Exception:
            connection.execute("ROLLBACK")
            raise

    def _import_mailbox_sidecars(self, connection: sqlite3.Connection) -> None:
        subscriptions = self.root / ".subscriptions"
        if subscriptions.exists():
            assert_no_linklike_components(subscriptions, boundary=self.root, allow_missing=False)
            if subscriptions.is_dir():
                for path in subscriptions.glob("*.json"):
                    assert_no_linklike_components(
                        path,
                        boundary=self.root,
                        allow_missing=False,
                    )
                    self._import_subscription_file(connection, path)
        sessions_root = self.root / ".sessions"
        if sessions_root.exists():
            assert_no_linklike_components(sessions_root, boundary=self.root, allow_missing=False)
            if sessions_root.is_dir():
                for host_dir in sessions_root.iterdir():
                    if host_dir.name.startswith("."):
                        continue
                    assert_no_linklike_components(
                        host_dir,
                        boundary=self.root,
                        allow_missing=False,
                    )
                    if not host_dir.is_dir():
                        continue
                    for path in host_dir.glob("*.json"):
                        assert_no_linklike_components(
                            path,
                            boundary=self.root,
                            allow_missing=False,
                        )
                        self._import_session_file(connection, path)
                    for claims_dir in host_dir.glob("*.delivered"):
                        assert_no_linklike_components(
                            claims_dir,
                            boundary=self.root,
                            allow_missing=False,
                        )
                        if not claims_dir.is_dir():
                            continue
                        session_id = claims_dir.name[: -len(".delivered")]
                        for claim in claims_dir.iterdir():
                            assert_no_linklike_components(
                                claim,
                                boundary=self.root,
                                allow_missing=False,
                            )
                            if not claim.is_file():
                                continue
                            try:
                                claimed_at = claim.read_text(encoding="utf-8").strip()
                            except OSError:
                                claimed_at = ""
                            connection.execute(
                                """
                                INSERT OR IGNORE INTO delivery_claims
                                    (host, session_id, message_id, claimed_at)
                                VALUES (?, ?, ?, ?)
                                """,
                                (
                                    host_dir.name,
                                    session_id,
                                    claim.name,
                                    claimed_at or _utcnow().isoformat(),
                                ),
                            )

    def _import_subscription_file(
        self, connection: sqlite3.Connection, path: Path
    ) -> None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        if not isinstance(data, dict):
            return
        agent_name = str(data.get("agent_name", "") or path.stem)
        topics = data.get("topics")
        if not agent_name or not isinstance(topics, list):
            return
        connection.execute(
            """
            INSERT OR IGNORE INTO subscriptions
                (agent_name, topics_json, created_at, updated_at, ttl_hours)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                agent_name,
                json.dumps([str(item) for item in topics], ensure_ascii=False),
                str(data.get("created_at", "")),
                str(data.get("updated_at", "")),
                data.get("ttl_hours") if isinstance(data.get("ttl_hours"), int) else None,
            ),
        )

    def _import_session_file(self, connection: sqlite3.Connection, path: Path) -> None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        if not isinstance(data, dict) or not data.get("session_id"):
            return
        known = {"host", "session_id", "scope_id", "cwd", "source", "registered_at"}
        extra = {key: value for key, value in data.items() if key not in known}
        connection.execute(
            """
            INSERT OR IGNORE INTO sessions (
                host, session_id, scope_id, cwd, source, registered_at, extra_json
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(data.get("host", "") or path.parent.name),
                str(data["session_id"]),
                str(data.get("scope_id", "") or ""),
                str(data.get("cwd", "") or ""),
                str(data.get("source", "") or ""),
                str(data.get("registered_at", "")),
                json.dumps(extra, ensure_ascii=False),
            ),
        )

    def _import_message_file(self, connection: sqlite3.Connection, path: Path) -> None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        if not isinstance(data, dict) or not str(data.get("id", "")).strip():
            return
        payload = data.get("payload") if isinstance(data.get("payload"), dict) else {}
        to_raw = data.get("to") if isinstance(data.get("to"), str) else None
        message_id = str(data["id"])
        connection.execute(
            """
            INSERT OR IGNORE INTO messages (
                id, from_agent, from_norm, to_raw, broadcast, msg_type,
                topic, payload_json, timestamp, expires_at, scope_id, reply_to
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                message_id,
                str(data.get("from", "")),
                normalize_agent_name(str(data.get("from", ""))),
                to_raw,
                1 if to_raw is None else 0,
                str(data.get("type", "status")),
                data.get("topic") if isinstance(data.get("topic"), str) else None,
                json.dumps(payload, ensure_ascii=False),
                str(data.get("timestamp", "")),
                data.get("expires_at") if isinstance(data.get("expires_at"), str) else None,
                str(data.get("scope_id", "") or ""),
                data.get("reply_to") if isinstance(data.get("reply_to"), str) else None,
            ),
        )
        self._insert_recipients(connection, message_id, to_raw)

    def _import_ack_file(self, connection: sqlite3.Connection, path: Path) -> None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            return
        if not isinstance(data, dict):
            return
        message_id = str(data.get("msg_id") or path.stem)
        records = data.get("acks")
        if not isinstance(records, list):
            return
        exists = connection.execute(
            "SELECT 1 FROM messages WHERE id = ?",
            (message_id,),
        ).fetchone()
        if exists is None:
            return
        for record in records:
            if not isinstance(record, dict):
                continue
            by_norm = normalize_agent_name(str(record.get("by", "")))
            status = str(record.get("status", ""))
            if not by_norm or status not in {"read", "done"}:
                continue
            connection.execute(
                """
                INSERT INTO message_acks (message_id, by_norm, status, note, at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    message_id,
                    by_norm,
                    status,
                    str(record.get("note", "") or ""),
                    str(record.get("at", "")),
                ),
            )

    @staticmethod
    def _insert_recipients(connection: sqlite3.Connection, message_id: str, to_raw: str | None) -> None:
        if not to_raw:
            return
        for raw in _RECIPIENT_SEPARATORS.split(to_raw):
            token = normalize_agent_name(raw)
            if not token:
                continue
            connection.execute(
                """
                INSERT OR IGNORE INTO message_recipients (message_id, token, family)
                VALUES (?, ?, ?)
                """,
                (message_id, token, _family(token)),
            )


def _meta(connection: sqlite3.Connection, key: str) -> str:
    row = connection.execute("SELECT value FROM store_meta WHERE key = ?", (key,)).fetchone()
    return "" if row is None else str(row["value"])


def _set_meta(connection: sqlite3.Connection, key: str, value: str) -> None:
    connection.execute(
        "INSERT INTO store_meta (key, value) VALUES (?, ?) "
        "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
        (key, value),
    )


def _has_done_ack(connection: sqlite3.Connection, message_id: str) -> bool:
    row = connection.execute(
        "SELECT 1 FROM message_acks WHERE message_id = ? AND status = 'done' LIMIT 1",
        (message_id,),
    ).fetchone()
    return row is not None


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    payload = json.loads(row["payload_json"]) if row["payload_json"] else {}
    if not isinstance(payload, dict):
        payload = {}
    return {
        "id": row["id"],
        "from": row["from_agent"],
        "to": row["to_raw"],
        "type": row["msg_type"],
        "payload": payload,
        "timestamp": row["timestamp"],
        "topic": row["topic"],
        "expires_at": row["expires_at"],
        "scope_id": row["scope_id"] or "",
        "reply_to": row["reply_to"],
    }


def _public_dict(row: dict[str, Any]) -> dict[str, Any]:
    published = {
        "id": row["id"],
        "from": row["from"],
        "to": row["to"],
        "type": row["type"],
        "payload": row["payload"],
        "timestamp": row["timestamp"],
    }
    if row.get("topic") is not None:
        published["topic"] = row["topic"]
    if row.get("expires_at") is not None:
        published["expires_at"] = row["expires_at"]
    if row.get("scope_id"):
        published["scope_id"] = row["scope_id"]
    if row.get("reply_to") is not None:
        published["reply_to"] = row["reply_to"]
    return published


def _session_row(row: sqlite3.Row) -> dict[str, Any]:
    extra_raw = row["extra_json"] or "{}"
    try:
        extra = json.loads(extra_raw)
    except json.JSONDecodeError:
        extra = {}
    if not isinstance(extra, dict):
        extra = {}
    record = {
        "host": row["host"],
        "session_id": row["session_id"],
        "scope_id": row["scope_id"] or "",
        "cwd": row["cwd"] or "",
        "source": row["source"] or "",
        "registered_at": row["registered_at"],
    }
    record.update(extra)
    return record
