"""Inference logging and feedback collection.

Logs all inference requests and responses for continuous learning.
"""

import hashlib
import json
import logging
import time
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class InferenceRecord:
    """A single inference request/response pair."""
    id: str
    timestamp: str
    prompt: str
    response: str
    model: str
    expert: str | None = None
    latency_ms: float = 0
    token_count: int = 0
    feedback_score: float | None = None  # User feedback: -1, 0, or 1
    feedback_text: str | None = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "InferenceRecord":
        return cls(**data)

    @property
    def has_feedback(self) -> bool:
        return self.feedback_score is not None


class InferenceLogger:
    """Logs inference requests and responses."""

    def __init__(
        self,
        log_dir: Path,
        buffer_size: int = 100,
        rotate_size_mb: int = 100,
    ):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.buffer_size = buffer_size
        self.rotate_size_mb = rotate_size_mb

        self._buffer: list[InferenceRecord] = []
        self._current_file: Path | None = None
        self._file_size = 0

    def _get_log_file(self) -> Path:
        """Get current log file, rotating if needed."""
        if self._current_file is None:
            self._current_file = self.log_dir / f"inference_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
            self._file_size = 0

        # Check rotation
        if self._current_file.exists():
            size_mb = self._current_file.stat().st_size / (1024 * 1024)
            if size_mb >= self.rotate_size_mb:
                self._current_file = self.log_dir / f"inference_{datetime.now():%Y%m%d_%H%M%S}.jsonl"
                self._file_size = 0

        return self._current_file

    def log(
        self,
        prompt: str,
        response: str,
        model: str,
        expert: str | None = None,
        latency_ms: float = 0,
        metadata: dict | None = None,
    ) -> str:
        """Log an inference request/response.

        Returns the record ID.
        """
        record_id = hashlib.md5(
            f"{prompt}{response}{time.time()}".encode()
        ).hexdigest()[:12]

        record = InferenceRecord(
            id=record_id,
            timestamp=datetime.now().isoformat(),
            prompt=prompt,
            response=response,
            model=model,
            expert=expert,
            latency_ms=latency_ms,
            token_count=len(response.split()),
            metadata=metadata or {},
        )

        self._buffer.append(record)

        if len(self._buffer) >= self.buffer_size:
            self.flush()

        return record_id

    def flush(self) -> int:
        """Flush buffer to disk."""
        if not self._buffer:
            return 0

        log_file = self._get_log_file()

        with open(log_file, "a") as f:
            for record in self._buffer:
                f.write(json.dumps(record.to_dict()) + "\n")

        count = len(self._buffer)
        self._buffer.clear()

        logger.debug(f"Flushed {count} records to {log_file}")
        return count

    def get_records(
        self,
        since: datetime | None = None,
        model: str | None = None,
        expert: str | None = None,
        with_feedback_only: bool = False,
    ):
        """Iterate over logged records."""
        for log_file in sorted(self.log_dir.glob("inference_*.jsonl")):
            with open(log_file) as f:
                for line in f:
                    record = InferenceRecord.from_dict(json.loads(line))

                    # Filter
                    if since:
                        record_time = datetime.fromisoformat(record.timestamp)
                        if record_time < since:
                            continue
                    if model and record.model != model:
                        continue
                    if expert and record.expert != expert:
                        continue
                    if with_feedback_only and not record.has_feedback:
                        continue

                    yield record

    def get_statistics(self) -> dict:
        """Get logging statistics."""
        total = 0
        with_feedback = 0
        positive = 0
        negative = 0

        for record in self.get_records():
            total += 1
            if record.has_feedback:
                with_feedback += 1
                if record.feedback_score > 0:
                    positive += 1
                elif record.feedback_score < 0:
                    negative += 1

        return {
            "total_records": total,
            "with_feedback": with_feedback,
            "positive_feedback": positive,
            "negative_feedback": negative,
            "feedback_rate": with_feedback / total if total > 0 else 0,
        }


class FeedbackCollector:
    """Collects user feedback on inference results."""

    def __init__(self, logger: InferenceLogger):
        self.logger = logger
        self._feedback_file = logger.log_dir / "feedback.jsonl"

    def record_feedback(
        self,
        record_id: str,
        score: float,  # -1, 0, or 1
        text: str | None = None,
    ) -> bool:
        """Record feedback for an inference record."""
        feedback = {
            "record_id": record_id,
            "score": score,
            "text": text,
            "timestamp": datetime.now().isoformat(),
        }

        with open(self._feedback_file, "a") as f:
            f.write(json.dumps(feedback) + "\n")

        logger.info(f"Recorded feedback for {record_id}: score={score}")
        return True

    def get_feedback(self, record_id: str | None = None):
        """Get feedback records."""
        if not self._feedback_file.exists():
            return

        with open(self._feedback_file) as f:
            for line in f:
                feedback = json.loads(line)
                if record_id is None or feedback["record_id"] == record_id:
                    yield feedback

    def get_positive_examples(self, min_score: float = 0.5):
        """Get inference records with positive feedback."""
        feedback_map = {}
        for fb in self.get_feedback():
            if fb["score"] >= min_score:
                feedback_map[fb["record_id"]] = fb

        for record in self.logger.get_records():
            if record.id in feedback_map:
                yield record, feedback_map[record.id]

    def export_training_data(
        self,
        output_path: Path,
        min_score: float = 0.5,
        format_type: str = "chatml",
    ) -> int:
        """Export positive examples as training data."""
        count = 0

        with open(output_path, "w") as f:
            for record, _feedback in self.get_positive_examples(min_score):
                if format_type == "chatml":
                    sample = {
                        "messages": [
                            {"role": "user", "content": record.prompt},
                            {"role": "assistant", "content": record.response},
                        ]
                    }
                else:
                    sample = {
                        "instruction": record.prompt,
                        "input": "",
                        "output": record.response,
                    }

                f.write(json.dumps(sample) + "\n")
                count += 1

        logger.info(f"Exported {count} positive examples to {output_path}")
        return count
