"""Automatic retraining triggers.

Monitors usage metrics and triggers retraining based on:
- Sample count thresholds
- Time-based schedules
- Quality score drops
- Error rate increases
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Optional, Callable

from .logger import UsageLogger
from .generator import TrainingDataGenerator, DataGeneratorConfig

logger = logging.getLogger(__name__)


class TriggerType(Enum):
    """Types of retraining triggers."""

    SAMPLE_COUNT = "sample_count"
    SCHEDULED = "scheduled"
    QUALITY_DROP = "quality_drop"
    ERROR_RATE = "error_rate"
    MANUAL = "manual"


@dataclass
class TriggerConfig:
    """Configuration for retraining triggers."""

    # Sample count trigger
    enable_sample_count: bool = True
    min_new_samples: int = 1000
    min_quality_score: float = 0.7

    # Scheduled trigger
    enable_scheduled: bool = True
    schedule_interval_hours: int = 168  # 1 week

    # Quality drop trigger
    enable_quality_drop: bool = True
    quality_drop_threshold: float = 0.1  # 10% drop
    quality_window_hours: int = 24

    # Error rate trigger
    enable_error_rate: bool = False
    error_rate_threshold: float = 0.2  # 20% errors
    error_window_hours: int = 24

    # General
    min_samples_for_retrain: int = 100
    cooldown_hours: int = 24  # Minimum time between retrains


@dataclass
class TriggerResult:
    """Result of a trigger check."""

    triggered: bool
    trigger_type: Optional[TriggerType] = None
    reason: str = ""
    metrics: dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class RetrainTrigger:
    """Monitors conditions and triggers retraining."""

    def __init__(
        self,
        usage_logger: UsageLogger,
        config: Optional[TriggerConfig] = None,
    ):
        self.logger = usage_logger
        self.config = config or TriggerConfig()
        self._last_retrain_time: Optional[datetime] = None
        self._baseline_quality: Optional[float] = None

    def check_triggers(self) -> TriggerResult:
        """Check all enabled triggers.

        Returns the first trigger that fires, or a non-triggered result.
        """
        # Check cooldown
        if self._last_retrain_time:
            cooldown = timedelta(hours=self.config.cooldown_hours)
            if datetime.now() - self._last_retrain_time < cooldown:
                return TriggerResult(
                    triggered=False,
                    reason=f"In cooldown period (last retrain: {self._last_retrain_time})",
                )

        # Check each trigger type
        if self.config.enable_sample_count:
            result = self._check_sample_count()
            if result.triggered:
                return result

        if self.config.enable_scheduled:
            result = self._check_scheduled()
            if result.triggered:
                return result

        if self.config.enable_quality_drop:
            result = self._check_quality_drop()
            if result.triggered:
                return result

        if self.config.enable_error_rate:
            result = self._check_error_rate()
            if result.triggered:
                return result

        return TriggerResult(triggered=False, reason="No triggers fired")

    def _check_sample_count(self) -> TriggerResult:
        """Check if enough new quality samples have accumulated."""
        # Get stats since last retrain
        since = self._last_retrain_time
        count = self.logger.db.count_new_samples(
            since or datetime.now() - timedelta(days=365),
            min_quality=self.config.min_quality_score,
        )

        if count >= self.config.min_new_samples:
            return TriggerResult(
                triggered=True,
                trigger_type=TriggerType.SAMPLE_COUNT,
                reason=f"Accumulated {count} quality samples (threshold: {self.config.min_new_samples})",
                metrics={"new_samples": count},
            )

        return TriggerResult(
            triggered=False,
            reason=f"Insufficient samples: {count}/{self.config.min_new_samples}",
        )

    def _check_scheduled(self) -> TriggerResult:
        """Check if scheduled retrain is due."""
        if not self._last_retrain_time:
            return TriggerResult(
                triggered=True,
                trigger_type=TriggerType.SCHEDULED,
                reason="No previous retrain, triggering initial training",
            )

        interval = timedelta(hours=self.config.schedule_interval_hours)
        elapsed = datetime.now() - self._last_retrain_time

        if elapsed >= interval:
            return TriggerResult(
                triggered=True,
                trigger_type=TriggerType.SCHEDULED,
                reason=f"Scheduled interval reached ({elapsed.days} days since last retrain)",
                metrics={"days_since_last": elapsed.days},
            )

        return TriggerResult(
            triggered=False,
            reason=f"Schedule not due ({elapsed.days}/{interval.days} days)",
        )

    def _check_quality_drop(self) -> TriggerResult:
        """Check if quality has dropped significantly."""
        # Get recent quality
        window = datetime.now() - timedelta(hours=self.config.quality_window_hours)
        stats = self.logger.get_statistics(since=window)
        current_quality = stats.get("avg_quality_score", 0.0)

        # Establish baseline if needed
        if self._baseline_quality is None:
            overall_stats = self.logger.get_statistics()
            self._baseline_quality = overall_stats.get("avg_quality_score", 0.0)
            logger.info(f"Established quality baseline: {self._baseline_quality:.3f}")

        # Check drop
        if self._baseline_quality > 0:
            drop = self._baseline_quality - current_quality
            drop_ratio = drop / self._baseline_quality

            if drop_ratio >= self.config.quality_drop_threshold:
                return TriggerResult(
                    triggered=True,
                    trigger_type=TriggerType.QUALITY_DROP,
                    reason=f"Quality dropped {drop_ratio*100:.1f}% (from {self._baseline_quality:.3f} to {current_quality:.3f})",
                    metrics={
                        "baseline_quality": self._baseline_quality,
                        "current_quality": current_quality,
                        "drop_ratio": drop_ratio,
                    },
                )

        return TriggerResult(
            triggered=False,
            reason=f"Quality stable (current: {current_quality:.3f}, baseline: {self._baseline_quality or 0:.3f})",
        )

    def _check_error_rate(self) -> TriggerResult:
        """Check if error rate has increased."""
        window = datetime.now() - timedelta(hours=self.config.error_window_hours)
        stats = self.logger.get_statistics(since=window)

        total = stats.get("total", 0)
        negative = stats.get("negative_feedback", 0)
        error_rate = negative / total if total > 0 else 0.0

        if error_rate >= self.config.error_rate_threshold:
            return TriggerResult(
                triggered=True,
                trigger_type=TriggerType.ERROR_RATE,
                reason=f"Error rate {error_rate*100:.1f}% exceeds threshold {self.config.error_rate_threshold*100:.1f}%",
                metrics={"error_rate": error_rate, "negative_count": negative},
            )

        return TriggerResult(
            triggered=False,
            reason=f"Error rate acceptable ({error_rate*100:.1f}%)",
        )

    def mark_retrain_completed(self) -> None:
        """Mark that a retrain has been completed."""
        self._last_retrain_time = datetime.now()
        logger.info(f"Marked retrain completed at {self._last_retrain_time}")

    def force_trigger(self, reason: str = "Manual trigger") -> TriggerResult:
        """Force a manual trigger."""
        return TriggerResult(
            triggered=True,
            trigger_type=TriggerType.MANUAL,
            reason=reason,
        )


class AutoRetrainer:
    """Automatic retraining orchestrator.

    Combines trigger checking, data generation, and training execution.
    """

    def __init__(
        self,
        usage_logger: UsageLogger,
        trigger_config: Optional[TriggerConfig] = None,
        generator_config: Optional[DataGeneratorConfig] = None,
        output_dir: Path = Path("~/.context/training/continuous").expanduser(),
    ):
        self.usage_logger = usage_logger
        self.trigger = RetrainTrigger(usage_logger, trigger_config)
        self.generator = TrainingDataGenerator(usage_logger, generator_config)
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def check_and_retrain(
        self,
        train_fn: Optional[Callable[[Path], dict]] = None,
    ) -> Optional[dict]:
        """Check triggers and retrain if needed.

        Args:
            train_fn: Optional training function (data_path) -> metrics

        Returns:
            Result dict if retrain was triggered, None otherwise
        """
        # Check triggers
        trigger_result = self.trigger.check_triggers()

        if not trigger_result.triggered:
            logger.info(f"No retrain needed: {trigger_result.reason}")
            return None

        logger.info(
            f"Retrain triggered: {trigger_result.trigger_type.value} - {trigger_result.reason}"
        )

        # Generate training data
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        data_path = self.output_dir / f"retrain_{timestamp}.jsonl"

        logger.info("Generating training data...")
        gen_result = self.generator.generate(
            output_path=data_path,
            since=self.trigger._last_retrain_time,
        )

        # Check if we have enough samples
        if gen_result.final_count < self.trigger.config.min_samples_for_retrain:
            logger.warning(
                f"Insufficient samples for retrain: {gen_result.final_count} < {self.trigger.config.min_samples_for_retrain}"
            )
            return None

        result = {
            "trigger": {
                "type": trigger_result.trigger_type.value,
                "reason": trigger_result.reason,
                "metrics": trigger_result.metrics,
            },
            "generation": {
                "total_candidates": gen_result.total_candidates,
                "final_count": gen_result.final_count,
                "quality_stats": gen_result.quality_stats,
                "data_path": str(data_path),
            },
            "timestamp": timestamp,
        }

        # Execute training if function provided
        if train_fn:
            logger.info("Starting training...")
            try:
                training_metrics = train_fn(data_path)
                result["training"] = training_metrics
                result["status"] = "completed"
                logger.info(f"Training completed: {training_metrics}")
            except Exception as e:
                result["status"] = "failed"
                result["error"] = str(e)
                logger.error(f"Training failed: {e}")
                return result
        else:
            result["status"] = "data_ready"
            logger.info(f"Training data prepared: {data_path}")

        # Mark retrain completed
        self.trigger.mark_retrain_completed()

        return result
