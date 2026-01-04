"""Periodic retraining with quality gates.

Manages continuous learning by periodically retraining on new feedback data.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Callable

from .logger import InferenceLogger, FeedbackCollector
from .sampler import FeedbackSampler

logger = logging.getLogger(__name__)


@dataclass
class RetrainConfig:
    """Configuration for periodic retraining."""
    min_new_samples: int = 100
    retrain_interval_hours: int = 168  # 1 week
    min_positive_ratio: float = 0.7
    validation_split: float = 0.1
    output_dir: Path = Path("retrain_data")
    quality_threshold: float = 0.6


@dataclass
class QualityGate:
    """Quality gate for approving retraining data."""
    min_samples: int = 100
    min_positive_ratio: float = 0.7
    max_negative_ratio: float = 0.1
    require_human_review: bool = True

    def check(self, stats: dict) -> tuple[bool, str]:
        """Check if data passes quality gate.

        Returns (passed, reason)
        """
        total = stats.get("total", 0)
        positive = stats.get("positive", 0)
        negative = stats.get("negative", 0)

        if total < self.min_samples:
            return False, f"Insufficient samples: {total} < {self.min_samples}"

        positive_ratio = positive / total if total > 0 else 0
        if positive_ratio < self.min_positive_ratio:
            return False, f"Low positive ratio: {positive_ratio:.2f} < {self.min_positive_ratio}"

        negative_ratio = negative / total if total > 0 else 0
        if negative_ratio > self.max_negative_ratio:
            return False, f"High negative ratio: {negative_ratio:.2f} > {self.max_negative_ratio}"

        return True, "Quality gate passed"


@dataclass
class RetrainJob:
    """A retraining job record."""
    id: str
    created_at: str
    samples_count: int
    status: str  # pending, running, completed, failed
    data_path: Optional[str] = None
    model_path: Optional[str] = None
    metrics: dict = field(default_factory=dict)


class PeriodicRetrainer:
    """Manages periodic retraining cycles."""

    def __init__(
        self,
        feedback_logger: InferenceLogger,
        config: Optional[RetrainConfig] = None,
    ):
        self.logger = feedback_logger
        self.config = config or RetrainConfig()
        self.collector = FeedbackCollector(feedback_logger)
        self.sampler = FeedbackSampler(feedback_logger)
        self.quality_gate = QualityGate()

        self._jobs_file = self.config.output_dir / "retrain_jobs.jsonl"
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

    def check_retrain_needed(self) -> tuple[bool, str]:
        """Check if retraining is needed."""
        # Check last retrain time
        last_job = self._get_last_job()
        if last_job:
            last_time = datetime.fromisoformat(last_job.created_at)
            interval = timedelta(hours=self.config.retrain_interval_hours)
            if datetime.now() - last_time < interval:
                return False, f"Too soon since last retrain ({last_time})"

        # Check new sample count
        stats = self.logger.get_statistics()
        new_samples = stats.get("with_feedback", 0)

        if new_samples < self.config.min_new_samples:
            return False, f"Insufficient new samples: {new_samples}"

        return True, f"Retrain needed: {new_samples} new samples"

    def prepare_training_data(self) -> tuple[Path, dict]:
        """Prepare training data from feedback.

        Returns (data_path, stats)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        data_path = self.config.output_dir / f"retrain_{timestamp}.jsonl"

        # Collect positive examples
        count = self.collector.export_training_data(
            data_path,
            min_score=self.config.quality_threshold,
        )

        # Compute stats
        stats = {
            "total": count,
            "positive": count,  # All exported are positive
            "negative": 0,
            "timestamp": timestamp,
        }

        return data_path, stats

    def run_retrain_cycle(
        self,
        train_fn: Optional[Callable[[Path], dict]] = None,
    ) -> Optional[RetrainJob]:
        """Run a full retrain cycle.

        Args:
            train_fn: Optional function (data_path) -> metrics
                      If None, only prepares data

        Returns:
            RetrainJob if successful, None otherwise
        """
        # Check if needed
        needed, reason = self.check_retrain_needed()
        if not needed:
            logger.info(f"Retrain not needed: {reason}")
            return None

        logger.info(f"Starting retrain cycle: {reason}")

        # Prepare data
        data_path, stats = self.prepare_training_data()

        # Quality gate
        passed, gate_reason = self.quality_gate.check(stats)
        if not passed:
            logger.warning(f"Quality gate failed: {gate_reason}")
            return None

        # Create job
        job = RetrainJob(
            id=f"retrain_{datetime.now():%Y%m%d_%H%M%S}",
            created_at=datetime.now().isoformat(),
            samples_count=stats["total"],
            status="pending",
            data_path=str(data_path),
        )

        # Run training if function provided
        if train_fn:
            try:
                job.status = "running"
                metrics = train_fn(data_path)
                job.metrics = metrics
                job.status = "completed"
                logger.info(f"Retrain completed: {metrics}")
            except Exception as e:
                job.status = "failed"
                job.metrics = {"error": str(e)}
                logger.error(f"Retrain failed: {e}")
        else:
            job.status = "data_ready"
            logger.info(f"Training data prepared: {data_path}")

        # Save job record
        self._save_job(job)

        return job

    def _get_last_job(self) -> Optional[RetrainJob]:
        """Get the most recent retrain job."""
        if not self._jobs_file.exists():
            return None

        last = None
        with open(self._jobs_file) as f:
            for line in f:
                data = json.loads(line)
                last = RetrainJob(**data)

        return last

    def _save_job(self, job: RetrainJob) -> None:
        """Save job record."""
        with open(self._jobs_file, "a") as f:
            f.write(json.dumps({
                "id": job.id,
                "created_at": job.created_at,
                "samples_count": job.samples_count,
                "status": job.status,
                "data_path": job.data_path,
                "model_path": job.model_path,
                "metrics": job.metrics,
            }) + "\n")

    def get_retrain_history(self) -> list[RetrainJob]:
        """Get all retrain jobs."""
        if not self._jobs_file.exists():
            return []

        jobs = []
        with open(self._jobs_file) as f:
            for line in f:
                jobs.append(RetrainJob(**json.loads(line)))

        return jobs
