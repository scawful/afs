"""Cost optimization recommendations for training.

Provides recommendations for batch size, epochs, early stopping, and dataset size.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _ensure_utc(timestamp: datetime) -> datetime:
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


@dataclass
class OptimizationRecommendation:
    """A recommendation for cost optimization."""

    category: str  # 'batch_size', 'epochs', 'early_stopping', 'dataset', 'compute'
    title: str
    description: str
    estimated_savings: float  # Estimated cost savings in dollars
    estimated_time_saved: float | None = None  # Hours saved
    confidence: float = 0.8  # 0-1, confidence in recommendation
    action: str = ""
    risks: list[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=_utc_now)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data


class CostOptimizer:
    """Optimizes training costs through recommendations."""

    def __init__(self, data_dir: Path | None = None):
        """Initialize cost optimizer.

        Args:
            data_dir: Directory to store optimization history. Defaults to ~/.context/training
        """
        if data_dir is None:
            data_dir = Path.home() / ".context" / "training"

        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.recommendations: list[OptimizationRecommendation] = []
        self._load_recommendations()

    def _load_recommendations(self) -> None:
        """Load past recommendations from disk."""
        rec_file = self.data_dir / "recommendations.json"
        if not rec_file.exists():
            return

        try:
            with open(rec_file) as f:
                data = json.load(f)

            for rec_data in data:
                rec_data["timestamp"] = _ensure_utc(
                    datetime.fromisoformat(rec_data["timestamp"])
                )
                self.recommendations.append(OptimizationRecommendation(**rec_data))

            logger.info(f"Loaded {len(self.recommendations)} past recommendations")
        except Exception as e:
            logger.error(f"Failed to load recommendations: {e}")

    def _save_recommendations(self) -> None:
        """Save recommendations to disk."""
        rec_file = self.data_dir / "recommendations.json"

        data = [r.to_dict() for r in self.recommendations]
        with open(rec_file, "w") as f:
            json.dump(data, f, indent=2)

        logger.debug(f"Saved {len(self.recommendations)} recommendations")

    def recommend_batch_size(
        self,
        current_batch_size: int,
        gpu_vram_gb: int,
        model_param_count: int,
        current_throughput: float,
        gpu_price_per_hour: float,
        epoch_hours: float,
    ) -> OptimizationRecommendation | None:
        """Recommend batch size optimization.

        Args:
            current_batch_size: Current batch size
            gpu_vram_gb: GPU VRAM in GB
            model_param_count: Model parameter count
            current_throughput: Current samples per second
            gpu_price_per_hour: GPU cost per hour
            epoch_hours: Hours per epoch at current settings

        Returns:
            Recommendation or None
        """
        # Rough estimate: 4 bytes per parameter for FP32
        param_bytes = model_param_count * 4 / (1024**3)

        # Estimate optimal batch size (general heuristic)
        # Assume ~2GB for model + optimizer states, rest for batch
        available_vram = max(1.0, gpu_vram_gb - param_bytes - 2.0)
        bytes_per_sample = 1024 / 1000  # ~1KB per sample (rough estimate)

        max_batch_size = int(available_vram * 1024 / bytes_per_sample)
        optimal_batch_size = min(max_batch_size, 256)  # Cap at 256

        if optimal_batch_size <= current_batch_size:
            return None  # Can't increase further

        # Calculate speedup from batch size increase
        speedup_factor = (optimal_batch_size / current_batch_size) ** 0.5  # Sublinear scaling

        new_throughput = current_throughput * speedup_factor
        time_savings = epoch_hours * (1 - (current_throughput / new_throughput))
        cost_savings = time_savings * gpu_price_per_hour

        if cost_savings < 0.1:  # Less than $0.10 savings
            return None

        rec = OptimizationRecommendation(
            category="batch_size",
            title=f"Increase batch size from {current_batch_size} to {optimal_batch_size}",
            description=(
                f"Increasing batch size from {current_batch_size} to {optimal_batch_size} "
                f"can improve throughput ({speedup_factor:.2f}x) and reduce training time."
            ),
            estimated_savings=cost_savings,
            estimated_time_saved=time_savings,
            action=f"Set batch_size={optimal_batch_size} in training config",
            risks=["May reduce model quality if batch size is too large"],
            confidence=0.7,
        )

        self.recommendations.append(rec)
        self._save_recommendations()
        return rec

    def recommend_early_stopping(
        self,
        validation_loss_history: list[float],
        gpu_price_per_hour: float,
        hours_per_epoch: float,
        improvement_threshold: float = 0.001,
        patience: int = 3,
    ) -> OptimizationRecommendation | None:
        """Recommend early stopping.

        Args:
            validation_loss_history: Validation loss over epochs
            gpu_price_per_hour: GPU cost per hour
            hours_per_epoch: Training hours per epoch
            improvement_threshold: Minimum improvement to continue
            patience: Epochs without improvement before stopping

        Returns:
            Recommendation or None
        """
        if len(validation_loss_history) < patience + 2:
            return None

        # Check if recent epochs show improvement
        recent_epochs = validation_loss_history[-patience:]
        best_loss = min(validation_loss_history[:-patience])

        improvements = []
        for loss in recent_epochs:
            improvement = best_loss - loss
            improvements.append(improvement)

        # If no significant improvement in last N epochs
        if all(imp < improvement_threshold for imp in improvements):
            current_epoch = len(validation_loss_history)
            saved_epochs = patience
            time_saved = saved_epochs * hours_per_epoch
            cost_saved = time_saved * gpu_price_per_hour

            rec = OptimizationRecommendation(
                category="early_stopping",
                title=f"Stop training at epoch {current_epoch} with early stopping",
                description=(
                    f"Validation loss has plateaued (no improvement > {improvement_threshold:.3f} "
                    f"in last {patience} epochs). Training can stop now to save ${cost_saved:.2f}."
                ),
                estimated_savings=cost_saved,
                estimated_time_saved=time_saved,
                action=f"Enable early_stopping with patience={patience}",
                risks=["May miss late convergence improvements"],
                confidence=0.85,
            )

            self.recommendations.append(rec)
            self._save_recommendations()
            return rec

        return None

    def recommend_epoch_count(
        self,
        validation_scores: list[float],
        gpu_price_per_hour: float,
        hours_per_epoch: float,
        score_type: str = "accuracy",
    ) -> OptimizationRecommendation | None:
        """Recommend optimal epoch count based on diminishing returns.

        Args:
            validation_scores: Validation scores over epochs
            gpu_price_per_hour: GPU cost per hour
            hours_per_epoch: Training hours per epoch
            score_type: Type of score (accuracy, loss, etc.)

        Returns:
            Recommendation or None
        """
        if len(validation_scores) < 3:
            return None

        # Calculate improvement per epoch
        improvements = []
        for i in range(1, len(validation_scores)):
            improvement = abs(validation_scores[i] - validation_scores[i - 1])
            improvements.append(improvement)

        # Find knee point (where improvements start diminishing significantly)
        avg_improvement = sum(improvements) / len(improvements)
        knee_epoch = None

        for i in range(len(improvements)):
            if improvements[i] < avg_improvement * 0.25:  # 75% drop in improvement rate
                knee_epoch = i + 1
                break

        if knee_epoch is None or knee_epoch >= len(validation_scores) - 1:
            return None  # No clear diminishing returns

        current_epochs = len(validation_scores)
        saved_epochs = current_epochs - knee_epoch
        time_saved = saved_epochs * hours_per_epoch
        cost_saved = time_saved * gpu_price_per_hour

        if cost_saved < 0.1:  # Less than $0.10
            return None

        rec = OptimizationRecommendation(
            category="epochs",
            title=f"Reduce epochs from {current_epochs} to {knee_epoch}",
            description=(
                f"Validation {score_type} shows diminishing returns after epoch {knee_epoch}. "
                f"Training {saved_epochs} additional epochs only provides marginal gains. "
                f"Stopping at epoch {knee_epoch} saves ${cost_saved:.2f}."
            ),
            estimated_savings=cost_saved,
            estimated_time_saved=time_saved,
            action=f"Set num_epochs={knee_epoch} in training config",
            risks=["May leave some performance on the table"],
            confidence=0.75,
        )

        self.recommendations.append(rec)
        self._save_recommendations()
        return rec

    def recommend_dataset_size(
        self,
        dataset_size: int,
        validation_accuracy_curve: dict[int, float],
        gpu_price_per_hour: float,
        hours_per_epoch: float,
        num_epochs: int,
    ) -> OptimizationRecommendation | None:
        """Recommend dataset size based on accuracy vs data size curve.

        Args:
            dataset_size: Current dataset size
            validation_accuracy_curve: Dict mapping dataset size to accuracy
            gpu_price_per_hour: GPU cost per hour
            hours_per_epoch: Training hours per epoch
            num_epochs: Number of epochs to train

        Returns:
            Recommendation or None
        """
        if len(validation_accuracy_curve) < 2:
            return None

        # Sort by dataset size
        sorted_points = sorted(validation_accuracy_curve.items())

        # Check for diminishing returns in accuracy gain
        best_accuracy = max(v for _, v in sorted_points)
        threshold_accuracy = best_accuracy * 0.95  # 95% of best

        # Find smallest dataset achieving 95% of best accuracy
        smallest_dataset = None
        for size, accuracy in sorted_points:
            if accuracy >= threshold_accuracy:
                smallest_dataset = size
                break

        if smallest_dataset is None or smallest_dataset >= dataset_size * 0.9:
            return None  # Already near optimal

        reduction_factor = smallest_dataset / dataset_size
        time_saved = (1 - reduction_factor) * hours_per_epoch * num_epochs
        cost_saved = time_saved * gpu_price_per_hour

        if cost_saved < 1.0:  # Less than $1
            return None

        rec = OptimizationRecommendation(
            category="dataset",
            title=f"Reduce dataset from {dataset_size:,} to {smallest_dataset:,} samples",
            description=(
                f"Analysis shows {smallest_dataset:,} samples achieves 95% of best accuracy. "
                f"Training on full dataset ({dataset_size:,}) has diminishing returns. "
                f"Reducing dataset saves ${cost_saved:.2f}."
            ),
            estimated_savings=cost_saved,
            estimated_time_saved=time_saved,
            action=f"Use data filtering to reduce dataset to {smallest_dataset:,} samples",
            risks=[
                "May lose edge cases or important data",
                "Could impact generalization",
            ],
            confidence=0.6,
        )

        self.recommendations.append(rec)
        self._save_recommendations()
        return rec

    def recommend_compute_tier(
        self,
        current_gpu: str,
        current_price_per_hour: float,
        current_throughput: float,
        alternative_gpus: dict[str, tuple[float, float]],
        daily_training_hours: float = 8.0,
    ) -> OptimizationRecommendation | None:
        """Recommend switching to a different compute tier.

        Args:
            current_gpu: Current GPU name
            current_price_per_hour: Current GPU hourly cost
            current_throughput: Current throughput (samples/sec)
            alternative_gpus: Dict of GPU name -> (price/hr, throughput)
            daily_training_hours: Hours of daily training

        Returns:
            Recommendation or None
        """
        best_ratio = current_throughput / current_price_per_hour
        best_gpu = None
        savings = 0.0

        for gpu_name, (price, throughput) in alternative_gpus.items():
            ratio = throughput / price
            if ratio > best_ratio * 1.2:  # 20% better efficiency
                daily_cost = price * daily_training_hours
                current_daily_cost = current_price_per_hour * daily_training_hours
                daily_savings = current_daily_cost - daily_cost
                monthly_savings = daily_savings * 30

                if monthly_savings > savings:
                    best_gpu = gpu_name
                    savings = monthly_savings

        if best_gpu is None or savings < 10:  # Less than $10/month
            return None

        rec = OptimizationRecommendation(
            category="compute",
            title=f"Switch from {current_gpu} to {best_gpu}",
            description=(
                f"{best_gpu} offers better cost-per-throughput ratio. "
                f"Expected monthly savings: ${savings:.2f}."
            ),
            estimated_savings=savings,
            confidence=0.7,
            action=f"Switch GPU instance to {best_gpu}",
            risks=["May require code adaptation if GPU has different capabilities"],
        )

        self.recommendations.append(rec)
        self._save_recommendations()
        return rec

    def get_all_recommendations(self, hours: int = 24) -> list[OptimizationRecommendation]:
        """Get recent recommendations.

        Args:
            hours: Only return recommendations from last N hours

        Returns:
            List of recent recommendations
        """
        from datetime import timedelta

        cutoff = _utc_now() - timedelta(hours=hours)
        return [r for r in self.recommendations if r.timestamp >= cutoff]

    def get_total_potential_savings(self) -> float:
        """Get total potential savings from all recommendations.

        Returns:
            Total estimated savings in dollars
        """
        return sum(r.estimated_savings for r in self.recommendations)

    def get_high_confidence_recommendations(
        self, confidence_threshold: float = 0.8
    ) -> list[OptimizationRecommendation]:
        """Get recommendations with high confidence.

        Args:
            confidence_threshold: Minimum confidence level (0-1)

        Returns:
            List of high-confidence recommendations
        """
        return [
            r for r in self.recommendations if r.confidence >= confidence_threshold
        ]
