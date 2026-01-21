"""Usage logging with SQLite database.

Captures all model queries, responses, and user feedback in a structured database.
Supports quality scoring, deduplication, and efficient querying.
"""

import hashlib
import sqlite3
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Iterator


@dataclass
class UsageRecord:
    """A single model usage record."""

    id: str
    timestamp: str
    query: str
    response: str
    model: str
    expert: Optional[str] = None
    latency_ms: float = 0
    token_count: int = 0
    quality_score: float = 0.0
    user_feedback: Optional[int] = None  # -1, 0, 1
    feedback_text: Optional[str] = None
    context_hash: Optional[str] = None
    dedupe_hash: Optional[str] = None

    @classmethod
    def from_dict(cls, data: dict) -> "UsageRecord":
        return cls(**data)

    def to_dict(self) -> dict:
        return asdict(self)

    def compute_dedupe_hash(self) -> str:
        """Compute hash for deduplication."""
        content = f"{self.query}||{self.response}"
        return hashlib.sha256(content.encode()).hexdigest()


class UsageDatabase:
    """SQLite database for usage logging."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS usage (
                    id TEXT PRIMARY KEY,
                    timestamp TEXT NOT NULL,
                    query TEXT NOT NULL,
                    response TEXT NOT NULL,
                    model TEXT NOT NULL,
                    expert TEXT,
                    latency_ms REAL,
                    token_count INTEGER,
                    quality_score REAL DEFAULT 0.0,
                    user_feedback INTEGER,
                    feedback_text TEXT,
                    context_hash TEXT,
                    dedupe_hash TEXT,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
                """
            )
            # Indexes for common queries
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON usage(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_model ON usage(model)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_quality_score ON usage(quality_score)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_user_feedback ON usage(user_feedback)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_dedupe_hash ON usage(dedupe_hash)")
            conn.commit()

    def insert(self, record: UsageRecord) -> bool:
        """Insert a usage record."""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute(
                    """
                    INSERT INTO usage (
                        id, timestamp, query, response, model, expert,
                        latency_ms, token_count, quality_score, user_feedback,
                        feedback_text, context_hash, dedupe_hash
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        record.id,
                        record.timestamp,
                        record.query,
                        record.response,
                        record.model,
                        record.expert,
                        record.latency_ms,
                        record.token_count,
                        record.quality_score,
                        record.user_feedback,
                        record.feedback_text,
                        record.context_hash,
                        record.dedupe_hash,
                    ),
                )
                conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False  # Duplicate ID

    def update_feedback(
        self,
        record_id: str,
        feedback: int,
        feedback_text: Optional[str] = None,
    ) -> bool:
        """Update user feedback for a record."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                """
                UPDATE usage
                SET user_feedback = ?, feedback_text = ?
                WHERE id = ?
                """,
                (feedback, feedback_text, record_id),
            )
            conn.commit()
            return conn.total_changes > 0

    def update_quality_score(self, record_id: str, score: float) -> bool:
        """Update quality score for a record."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "UPDATE usage SET quality_score = ? WHERE id = ?",
                (score, record_id),
            )
            conn.commit()
            return conn.total_changes > 0

    def query(
        self,
        since: Optional[datetime] = None,
        model: Optional[str] = None,
        min_quality: Optional[float] = None,
        with_feedback_only: bool = False,
        limit: Optional[int] = None,
    ) -> Iterator[UsageRecord]:
        """Query usage records with filters."""
        conditions = []
        params = []

        if since:
            conditions.append("timestamp >= ?")
            params.append(since.isoformat())
        if model:
            conditions.append("model = ?")
            params.append(model)
        if min_quality is not None:
            conditions.append("quality_score >= ?")
            params.append(min_quality)
        if with_feedback_only:
            conditions.append("user_feedback IS NOT NULL")

        where_clause = " AND ".join(conditions) if conditions else "1=1"
        query = f"SELECT * FROM usage WHERE {where_clause} ORDER BY timestamp DESC"

        if limit:
            query += f" LIMIT {limit}"

        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute(query, params)

            for row in cursor:
                yield UsageRecord(
                    id=row["id"],
                    timestamp=row["timestamp"],
                    query=row["query"],
                    response=row["response"],
                    model=row["model"],
                    expert=row["expert"],
                    latency_ms=row["latency_ms"],
                    token_count=row["token_count"],
                    quality_score=row["quality_score"],
                    user_feedback=row["user_feedback"],
                    feedback_text=row["feedback_text"],
                    context_hash=row["context_hash"],
                    dedupe_hash=row["dedupe_hash"],
                )

    def get_statistics(self, since: Optional[datetime] = None) -> dict:
        """Get usage statistics."""
        conditions = []
        params = []

        if since:
            conditions.append("timestamp >= ?")
            params.append(since.isoformat())

        where_clause = " AND ".join(conditions) if conditions else "1=1"

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                f"""
                SELECT
                    COUNT(*) as total,
                    COUNT(user_feedback) as with_feedback,
                    SUM(CASE WHEN user_feedback > 0 THEN 1 ELSE 0 END) as positive,
                    SUM(CASE WHEN user_feedback < 0 THEN 1 ELSE 0 END) as negative,
                    AVG(quality_score) as avg_quality,
                    AVG(latency_ms) as avg_latency,
                    SUM(token_count) as total_tokens
                FROM usage
                WHERE {where_clause}
                """,
                params,
            )
            row = cursor.fetchone()

            return {
                "total": row[0] or 0,
                "with_feedback": row[1] or 0,
                "positive_feedback": row[2] or 0,
                "negative_feedback": row[3] or 0,
                "avg_quality_score": row[4] or 0.0,
                "avg_latency_ms": row[5] or 0.0,
                "total_tokens": row[6] or 0,
                "feedback_rate": (row[1] or 0) / (row[0] or 1),
            }

    def count_new_samples(self, since: datetime, min_quality: float = 0.5) -> int:
        """Count new quality samples since timestamp."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute(
                """
                SELECT COUNT(*)
                FROM usage
                WHERE timestamp >= ? AND quality_score >= ?
                """,
                (since.isoformat(), min_quality),
            )
            return cursor.fetchone()[0]

    def deduplicate(self, dry_run: bool = False) -> int:
        """Remove duplicate records based on dedupe_hash.

        Returns count of duplicates found.
        """
        with sqlite3.connect(self.db_path) as conn:
            # Find duplicates
            cursor = conn.execute(
                """
                SELECT dedupe_hash, COUNT(*) as cnt
                FROM usage
                WHERE dedupe_hash IS NOT NULL
                GROUP BY dedupe_hash
                HAVING cnt > 1
                """
            )
            dupes = cursor.fetchall()

            if not dry_run:
                # Keep newest record for each duplicate
                for dedupe_hash, count in dupes:
                    conn.execute(
                        """
                        DELETE FROM usage
                        WHERE dedupe_hash = ?
                        AND id NOT IN (
                            SELECT id FROM usage
                            WHERE dedupe_hash = ?
                            ORDER BY timestamp DESC
                            LIMIT 1
                        )
                        """,
                        (dedupe_hash, dedupe_hash),
                    )
                conn.commit()

            return len(dupes)


class UsageLogger:
    """High-level usage logger with database backend."""

    def __init__(self, db_path: Path):
        self.db = UsageDatabase(db_path)

    def log(
        self,
        query: str,
        response: str,
        model: str,
        expert: Optional[str] = None,
        latency_ms: float = 0,
        quality_score: float = 0.0,
    ) -> str:
        """Log a model usage.

        Returns the record ID.
        """
        # Generate ID
        record_id = hashlib.md5(
            f"{query}{response}{time.time()}".encode()
        ).hexdigest()[:12]

        # Create record
        record = UsageRecord(
            id=record_id,
            timestamp=datetime.now().isoformat(),
            query=query,
            response=response,
            model=model,
            expert=expert,
            latency_ms=latency_ms,
            token_count=len(response.split()),
            quality_score=quality_score,
            dedupe_hash=UsageRecord(
                id="", timestamp="", query=query, response=response, model=model
            ).compute_dedupe_hash(),
        )

        self.db.insert(record)
        return record_id

    def record_feedback(
        self,
        record_id: str,
        feedback: int,
        feedback_text: Optional[str] = None,
    ) -> bool:
        """Record user feedback."""
        return self.db.update_feedback(record_id, feedback, feedback_text)

    def get_records(
        self,
        since: Optional[datetime] = None,
        model: Optional[str] = None,
        min_quality: Optional[float] = None,
        with_feedback_only: bool = False,
        limit: Optional[int] = None,
    ) -> Iterator[UsageRecord]:
        """Query usage records."""
        return self.db.query(since, model, min_quality, with_feedback_only, limit)

    def get_statistics(self, since: Optional[datetime] = None) -> dict:
        """Get usage statistics."""
        return self.db.get_statistics(since)
