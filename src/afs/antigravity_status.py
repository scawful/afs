"""Lightweight Antigravity capture status helpers."""

from __future__ import annotations

import base64
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

DEFAULT_ANTIGRAVITY_DB = (
    Path.home()
    / "Library"
    / "Application Support"
    / "Antigravity"
    / "User"
    / "globalStorage"
    / "state.vscdb"
)

DEFAULT_ANTIGRAVITY_STATE_KEYS = (
    "antigravityUnifiedStateSync.trajectorySummaries",
    "unifiedStateSync.trajectorySummaries",
)


def antigravity_status(
    *,
    db_path: Path | None = None,
    state_keys: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Return lightweight status for the local Antigravity capture database."""
    resolved_db = (db_path or DEFAULT_ANTIGRAVITY_DB).expanduser().resolve()
    keys = list(state_keys or DEFAULT_ANTIGRAVITY_STATE_KEYS)
    status: dict[str, Any] = {
        "db_path": str(resolved_db),
        "db_exists": resolved_db.exists(),
        "payload_count": 0,
        "state_keys": keys,
        "last_sync": None,
        "error": None,
    }

    if status["db_exists"]:
        status["last_sync"] = datetime.fromtimestamp(
            resolved_db.stat().st_mtime,
            tz=timezone.utc,
        ).isoformat()
    else:
        status["error"] = "database not found"
        return status

    try:
        # sqlite3's context manager only commits/rolls back — it does not
        # close the connection; closing() releases the fd as well.
        with closing(sqlite3.connect(resolved_db)) as connection:
            cursor = connection.cursor()
            payloads = 0
            for key in keys:
                cursor.execute("SELECT value FROM ItemTable WHERE key = ?", (key,))
                row = cursor.fetchone()
                if not row or row[0] is None:
                    continue
                payload = row[0]
                if isinstance(payload, bytes):
                    try:
                        payload = payload.decode("utf-8")
                    except UnicodeDecodeError:
                        continue
                if not isinstance(payload, str) or not payload.strip():
                    continue
                try:
                    base64.b64decode(payload, validate=True)
                except (ValueError, TypeError):
                    continue
                payloads += 1
            status["payload_count"] = payloads
    except sqlite3.Error as exc:
        status["error"] = f"sqlite error: {exc}"

    return status
