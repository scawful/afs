"""A/B testing framework for model versions.

Deploys multiple model versions in parallel, routes traffic proportionally,
and compares performance metrics to automatically promote winning models.
"""

import json
import logging
import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from pathlib import Path
from typing import Any

from .logger import UsageLogger

logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """Status of a model version."""

    CHAMPION = "champion"  # Current production model
    CHALLENGER = "challenger"  # Model under test
    RETIRED = "retired"  # No longer active


@dataclass
class ModelVersion:
    """A model version in A/B testing."""

    id: str
    name: str
    path: Path
    status: ModelStatus
    traffic_weight: float = 0.0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    metrics: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "path": str(self.path),
            "status": self.status.value,
            "traffic_weight": self.traffic_weight,
            "created_at": self.created_at,
            "metrics": self.metrics,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "ModelVersion":
        return cls(
            id=data["id"],
            name=data["name"],
            path=Path(data["path"]),
            status=ModelStatus(data["status"]),
            traffic_weight=data.get("traffic_weight", 0.0),
            created_at=data.get("created_at", datetime.now().isoformat()),
            metrics=data.get("metrics", {}),
        )


@dataclass
class TrafficSplit:
    """Traffic split configuration."""

    champion_weight: float = 0.9  # 90% to champion
    challenger_weight: float = 0.1  # 10% to challenger

    def normalize(self) -> "TrafficSplit":
        """Normalize weights to sum to 1.0."""
        total = self.champion_weight + self.challenger_weight
        if total > 0:
            self.champion_weight /= total
            self.challenger_weight /= total
        return self


@dataclass
class ABTestConfig:
    """Configuration for A/B testing."""

    # Traffic
    initial_challenger_traffic: float = 0.1  # 10%
    min_samples_per_version: int = 100
    model_a_version: str | None = None
    model_b_version: str | None = None
    traffic_split_percent: float | None = None
    duration_hours: int | None = None

    # Evaluation
    evaluation_window_hours: int = 24
    min_improvement_threshold: float = 0.05  # 5% improvement

    # Metrics to compare
    metrics: list[str] = field(
        default_factory=lambda: [
            "avg_quality_score",
            "positive_feedback_rate",
            "avg_latency_ms",
        ]
    )
    metric_weights: dict[str, float] = field(
        default_factory=lambda: {
            "avg_quality_score": 0.5,
            "positive_feedback_rate": 0.3,
            "avg_latency_ms": -0.2,  # Negative weight (lower is better)
        }
    )

    # Auto-promotion
    enable_auto_promotion: bool = True
    promotion_threshold: float = 0.05  # 5% better
    min_duration_hours: int = 48  # Test for at least 48 hours

    def __post_init__(self) -> None:
        if self.traffic_split_percent is not None:
            self.initial_challenger_traffic = max(
                0.0, min(1.0, self.traffic_split_percent / 100.0)
            )
        if self.duration_hours is not None:
            self.evaluation_window_hours = self.duration_hours
            self.min_duration_hours = self.duration_hours


@dataclass
class ABTestResult:
    """Result of an A/B test comparison."""

    champion_metrics: dict
    challenger_metrics: dict
    improvement: float  # Positive = challenger better
    winner: ModelStatus
    reason: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class ABTestManager:
    """Manages A/B testing of model versions."""

    def __init__(
        self,
        usage_logger: UsageLogger | None = None,
        config: ABTestConfig | None = None,
        *,
        ab_config: ABTestConfig | None = None,
        registry: Any | None = None,
        state_file: Path = Path("~/.context/training/continuous/ab_test_state.json").expanduser(),
    ):
        self.usage_logger = usage_logger or UsageLogger(
            Path("~/.context/training/continuous/usage.db").expanduser()
        )
        self.config = config or ab_config or ABTestConfig()
        self.registry = registry
        self.state_file = state_file
        self.state_file.parent.mkdir(parents=True, exist_ok=True)

        self.champion: ModelVersion | None = None
        self.challenger: ModelVersion | None = None
        self.traffic_split = TrafficSplit(
            champion_weight=1.0 - self.config.initial_challenger_traffic,
            challenger_weight=self.config.initial_challenger_traffic,
        )

        self._load_state()

    def deploy_challenger(
        self,
        model_name: str,
        model_path: Path,
        traffic_weight: float | None = None,
    ) -> ModelVersion:
        """Deploy a new challenger model.

        Args:
            model_name: Name of the model
            model_path: Path to model weights
            traffic_weight: Optional traffic weight (default: config.initial_challenger_traffic)

        Returns:
            The deployed ModelVersion
        """
        # Retire old challenger if exists
        if self.challenger:
            self.challenger.status = ModelStatus.RETIRED
            self.challenger.traffic_weight = 0.0
            logger.info(f"Retired old challenger: {self.challenger.name}")

        # Create new challenger
        model_id = f"{model_name}_{datetime.now():%Y%m%d_%H%M%S}"
        self.challenger = ModelVersion(
            id=model_id,
            name=model_name,
            path=model_path,
            status=ModelStatus.CHALLENGER,
            traffic_weight=traffic_weight or self.config.initial_challenger_traffic,
        )

        # Adjust traffic split
        self.traffic_split = TrafficSplit(
            champion_weight=1.0 - self.challenger.traffic_weight,
            challenger_weight=self.challenger.traffic_weight,
        )

        self._save_state()
        logger.info(
            f"Deployed challenger: {model_name} with {self.challenger.traffic_weight*100:.1f}% traffic"
        )

        return self.challenger

    def route_request(self) -> ModelVersion | None:
        """Route a request to a model based on traffic split.

        Returns:
            The selected ModelVersion, or None if no champion
        """
        if not self.champion:
            return None

        # No challenger, route to champion
        if not self.challenger or self.challenger.status != ModelStatus.CHALLENGER:
            return self.champion

        # Random selection based on weights
        rand = random.random()
        if rand < self.traffic_split.challenger_weight:
            return self.challenger
        else:
            return self.champion

    def get_model_metrics(
        self,
        model_version: ModelVersion,
        window_hours: int | None = None,
    ) -> dict:
        """Get metrics for a specific model version.

        Args:
            model_version: The model version
            window_hours: Time window for metrics (default: config.evaluation_window_hours)

        Returns:
            Dict of metrics
        """
        window = window_hours or self.config.evaluation_window_hours
        since = datetime.now() - timedelta(hours=window)

        # Get stats from usage logger
        stats = self.usage_logger.get_statistics(since=since)

        # Filter for this model (would need model tracking in usage logs)
        # For now, return overall stats
        # TODO: Add model_id tracking to UsageRecord

        return {
            "avg_quality_score": stats.get("avg_quality_score", 0.0),
            "positive_feedback_rate": stats.get("positive_feedback", 0)
            / max(stats.get("with_feedback", 1), 1),
            "avg_latency_ms": stats.get("avg_latency_ms", 0.0),
            "total_requests": stats.get("total", 0),
        }

    def compare_models(self) -> ABTestResult | None:
        """Compare champion and challenger performance.

        Returns:
            ABTestResult if comparison is possible, None otherwise
        """
        if not self.champion or not self.challenger:
            logger.warning("Cannot compare: missing champion or challenger")
            return None

        # Get metrics for both
        champion_metrics = self.get_model_metrics(self.champion)
        challenger_metrics = self.get_model_metrics(self.challenger)

        # Check minimum samples
        if champion_metrics["total_requests"] < self.config.min_samples_per_version:
            logger.info(
                f"Champion has insufficient samples: {champion_metrics['total_requests']}"
            )
            return None

        if challenger_metrics["total_requests"] < self.config.min_samples_per_version:
            logger.info(
                f"Challenger has insufficient samples: {challenger_metrics['total_requests']}"
            )
            return None

        # Compute weighted improvement
        improvement = 0.0
        for metric in self.config.metrics:
            if metric not in self.config.metric_weights:
                continue

            weight = self.config.metric_weights[metric]
            champion_val = champion_metrics.get(metric, 0.0)
            challenger_val = challenger_metrics.get(metric, 0.0)

            if champion_val > 0:
                diff_ratio = (challenger_val - champion_val) / champion_val
                improvement += weight * diff_ratio

        # Determine winner
        if improvement >= self.config.min_improvement_threshold:
            winner = ModelStatus.CHALLENGER
            reason = f"Challenger {improvement*100:.2f}% better than champion"
        elif improvement <= -self.config.min_improvement_threshold:
            winner = ModelStatus.CHAMPION
            reason = f"Champion {-improvement*100:.2f}% better than challenger"
        else:
            winner = ModelStatus.CHAMPION  # Default to champion if too close
            reason = f"Inconclusive ({improvement*100:.2f}% difference, below threshold)"

        result = ABTestResult(
            champion_metrics=champion_metrics,
            challenger_metrics=challenger_metrics,
            improvement=improvement,
            winner=winner,
            reason=reason,
        )

        logger.info(f"A/B test comparison: {reason}")
        return result

    def promote_challenger(self) -> bool:
        """Promote challenger to champion.

        Returns:
            True if promotion successful
        """
        if not self.challenger:
            logger.warning("No challenger to promote")
            return False

        # Retire old champion
        if self.champion:
            self.champion.status = ModelStatus.RETIRED
            self.champion.traffic_weight = 0.0
            logger.info(f"Retired old champion: {self.champion.name}")

        # Promote challenger
        self.champion = self.challenger
        self.champion.status = ModelStatus.CHAMPION
        self.champion.traffic_weight = 1.0

        self.challenger = None

        # Reset traffic split
        self.traffic_split = TrafficSplit(champion_weight=1.0, challenger_weight=0.0)

        self._save_state()
        logger.info(f"Promoted challenger to champion: {self.champion.name}")

        return True

    def auto_promote_if_ready(self) -> bool:
        """Check if challenger should be auto-promoted.

        Returns:
            True if promotion occurred
        """
        if not self.config.enable_auto_promotion:
            return False

        if not self.challenger:
            return False

        # Check minimum duration
        created = datetime.fromisoformat(self.challenger.created_at)
        age = datetime.now() - created
        min_age = timedelta(hours=self.config.min_duration_hours)

        if age < min_age:
            logger.debug(
                f"Challenger too young for promotion: {age.total_seconds() / 3600:.1f} hours"
            )
            return False

        # Compare models
        result = self.compare_models()
        if not result:
            return False

        # Check if challenger wins and improvement is sufficient
        if (
            result.winner == ModelStatus.CHALLENGER
            and result.improvement >= self.config.promotion_threshold
        ):
            logger.info(
                f"Auto-promoting challenger (improvement: {result.improvement*100:.2f}%)"
            )
            return self.promote_challenger()

        return False

    def _save_state(self) -> None:
        """Save current state to disk."""
        state = {
            "champion": self.champion.to_dict() if self.champion else None,
            "challenger": self.challenger.to_dict() if self.challenger else None,
            "traffic_split": {
                "champion_weight": self.traffic_split.champion_weight,
                "challenger_weight": self.traffic_split.challenger_weight,
            },
        }

        with open(self.state_file, "w") as f:
            json.dump(state, f, indent=2)

    def _load_state(self) -> None:
        """Load state from disk."""
        if not self.state_file.exists():
            return

        try:
            with open(self.state_file) as f:
                state = json.load(f)

            if state.get("champion"):
                self.champion = ModelVersion.from_dict(state["champion"])
            if state.get("challenger"):
                self.challenger = ModelVersion.from_dict(state["challenger"])

            if state.get("traffic_split"):
                split = state["traffic_split"]
                self.traffic_split = TrafficSplit(
                    champion_weight=split["champion_weight"],
                    challenger_weight=split["challenger_weight"],
                )

            logger.info("Loaded A/B test state from disk")
        except Exception as e:
            logger.error(f"Failed to load A/B test state: {e}")
