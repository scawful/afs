"""Cost analysis for training runs.

Analyzes training costs and provides detailed cost breakdowns.
"""

import json
import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _ensure_utc(timestamp: datetime) -> datetime:
    if timestamp.tzinfo is None:
        return timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc)


@dataclass
class TrainingMetrics:
    """Metrics from a training run."""

    run_id: str = ""
    model_name: str = ""
    num_samples: int = 0
    num_epochs: int = 0
    batch_size: int = 0
    learning_rate: float = 0.0
    total_duration_hours: float = 0.0
    gpu_name: str = ""
    gpu_price_per_hour: float = 0.0
    validation_loss: Optional[float] = None
    test_accuracy: Optional[float] = None
    tokens_processed: Optional[int] = None
    provider: Optional[str] = None
    hours_used: Optional[float] = None
    total_cost: Optional[float] = None
    throughput: Optional[float] = None
    accuracy: Optional[float] = None
    timestamp: datetime = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        self.timestamp = _ensure_utc(self.timestamp)
        if not self.run_id:
            self.run_id = "unknown"
        if not self.model_name:
            self.model_name = "unknown"
        if self.hours_used is not None and self.total_duration_hours <= 0:
            self.total_duration_hours = self.hours_used
        if self.accuracy is not None and self.test_accuracy is None:
            self.test_accuracy = self.accuracy
        if (
            self.total_cost is None
            and self.total_duration_hours > 0
            and self.gpu_price_per_hour > 0
        ):
            self.total_cost = self.total_duration_hours * self.gpu_price_per_hour

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        data["timestamp"] = self.timestamp.isoformat()
        return data

    @classmethod
    def from_dict(cls, data: Dict) -> "TrainingMetrics":
        """Create from dictionary."""
        data = data.copy()
        if isinstance(data.get("timestamp"), str):
            data["timestamp"] = _ensure_utc(datetime.fromisoformat(data["timestamp"]))
        return cls(**data)


@dataclass
class TrainingCostReport:
    """Detailed cost report for a training run."""

    run_id: str
    model_name: str
    total_cost: float
    gpu_hours: float
    cost_per_sample: float
    cost_per_epoch: float
    cost_per_token: Optional[float] = None
    efficiency_score: float = 0.0  # 0-1, based on accuracy per dollar
    cost_per_accuracy_point: Optional[float] = None
    metrics: Optional[TrainingMetrics] = None
    timestamp: datetime = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        self.timestamp = _ensure_utc(self.timestamp)

    def to_dict(self) -> Dict:
        """Convert to dictionary for JSON serialization."""
        data = asdict(self)
        if isinstance(self.timestamp, datetime):
            data["timestamp"] = self.timestamp.isoformat()
        if self.metrics:
            data["metrics"] = self.metrics.to_dict()
        return data


@dataclass
class BudgetAlert:
    """Alert for budget threshold crossed."""

    model_name: str
    alert_type: str  # 'warning_50', 'warning_75', 'warning_90', 'exceeded'
    current_cost: float
    budget_limit: float
    percent_used: float
    timestamp: datetime = field(default_factory=_utc_now)

    def __post_init__(self) -> None:
        """Validate alert type."""
        self.timestamp = _ensure_utc(self.timestamp)
        valid_types = ["warning_50", "warning_75", "warning_90", "exceeded"]
        if self.alert_type not in valid_types:
            raise ValueError(f"Invalid alert type: {self.alert_type}")


class CostAnalyzer:
    """Analyzes training costs and generates reports."""

    def __init__(self, data_dir: Optional[Path] = None):
        """Initialize cost analyzer.

        Args:
            data_dir: Directory to store cost history. Defaults to ~/.context/training
        """
        if data_dir is None:
            data_dir = Path.home() / ".context" / "training"

        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.reports: Dict[str, TrainingCostReport] = {}
        self.metrics: Dict[str, TrainingMetrics] = {}
        self.budgets: Dict[str, float] = {}
        self.alerts: List[BudgetAlert] = []

        self._load_history()

    def _load_history(self) -> None:
        """Load cost history from disk."""
        cost_file = self.data_dir / "cost_history.json"
        budget_file = self.data_dir / "budgets.json"

        if cost_file.exists():
            try:
                with open(cost_file) as f:
                    data = json.load(f)

                for run_id, report_data in data.items():
                    if "metrics" in report_data and report_data["metrics"]:
                        report_data["metrics"] = TrainingMetrics.from_dict(
                            report_data["metrics"]
                        )
                    if isinstance(report_data.get("timestamp"), str):
                        report_data["timestamp"] = _ensure_utc(
                            datetime.fromisoformat(report_data["timestamp"])
                        )
                    self.reports[run_id] = TrainingCostReport(**report_data)

                logger.info(f"Loaded {len(self.reports)} training cost reports")
            except Exception as e:
                logger.error(f"Failed to load cost history: {e}")

        if budget_file.exists():
            try:
                with open(budget_file) as f:
                    self.budgets = json.load(f)
                logger.info(f"Loaded {len(self.budgets)} budget limits")
            except Exception as e:
                logger.error(f"Failed to load budgets: {e}")

    def _save_history(self) -> None:
        """Save cost history to disk."""
        cost_file = self.data_dir / "cost_history.json"
        budget_file = self.data_dir / "budgets.json"

        data = {run_id: report.to_dict() for run_id, report in self.reports.items()}
        with open(cost_file, "w") as f:
            json.dump(data, f, indent=2)

        with open(budget_file, "w") as f:
            json.dump(self.budgets, f, indent=2)

        logger.debug("Saved cost history and budgets")

    def analyze_training_run(
        self, metrics: TrainingMetrics
    ) -> TrainingCostReport:
        """Analyze a completed training run.

        Args:
            metrics: Training metrics from the run

        Returns:
            Detailed cost report
        """
        # Calculate costs
        total_cost = (
            metrics.total_cost
            if metrics.total_cost is not None
            else metrics.total_duration_hours * metrics.gpu_price_per_hour
        )
        cost_per_sample = total_cost / metrics.num_samples if metrics.num_samples > 0 else 0
        cost_per_epoch = total_cost / metrics.num_epochs if metrics.num_epochs > 0 else 0
        cost_per_token = None
        if metrics.tokens_processed and metrics.tokens_processed > 0:
            cost_per_token = total_cost / (metrics.tokens_processed / 1e6)  # Cost per million tokens

        # Calculate efficiency score (0-1)
        efficiency_score = 0.0
        cost_per_accuracy_point = None

        if metrics.test_accuracy is not None and metrics.test_accuracy > 0:
            # Efficiency: higher accuracy per dollar
            accuracy_per_dollar = metrics.test_accuracy / total_cost if total_cost > 0 else 0
            # Normalize to 0-1 range (assuming max 100% accuracy per $10)
            efficiency_score = min(1.0, accuracy_per_dollar / 10.0)

            cost_per_accuracy_point = total_cost / metrics.test_accuracy

        report = TrainingCostReport(
            run_id=metrics.run_id,
            model_name=metrics.model_name,
            total_cost=total_cost,
            gpu_hours=metrics.total_duration_hours,
            cost_per_sample=cost_per_sample,
            cost_per_epoch=cost_per_epoch,
            cost_per_token=cost_per_token,
            efficiency_score=efficiency_score,
            cost_per_accuracy_point=cost_per_accuracy_point,
            metrics=metrics,
            timestamp=metrics.timestamp,
        )

        self.reports[metrics.run_id] = report
        self.metrics[metrics.run_id] = metrics
        self._save_history()

        logger.info(f"Analyzed training run {metrics.run_id}: ${total_cost:.2f}")
        return report

    def calculate_cost_efficiency(self, metrics: TrainingMetrics) -> float:
        """Calculate a simple cost efficiency score for a training run."""
        total_cost = metrics.total_cost
        if total_cost is None:
            if metrics.total_duration_hours > 0 and metrics.gpu_price_per_hour > 0:
                total_cost = metrics.total_duration_hours * metrics.gpu_price_per_hour
                metrics.total_cost = total_cost
            else:
                total_cost = 0.0

        if total_cost <= 0:
            return 0.0

        accuracy = metrics.test_accuracy
        if accuracy is None:
            accuracy = metrics.accuracy

        if accuracy is not None:
            return accuracy / total_cost
        if metrics.throughput is not None:
            return metrics.throughput / total_cost

        return 0.0

    def set_budget(self, model_name: str, budget_limit: float) -> None:
        """Set a budget limit for a model.

        Args:
            model_name: Name of the model
            budget_limit: Maximum cost in dollars
        """
        self.budgets[model_name] = budget_limit
        self._save_history()
        logger.info(f"Set budget for {model_name}: ${budget_limit:.2f}")

    def check_budget(self, model_name: str, current_cost: float) -> Optional[BudgetAlert]:
        """Check if budget threshold has been crossed.

        Args:
            model_name: Name of the model
            current_cost: Current accumulated cost

        Returns:
            BudgetAlert if threshold crossed, None otherwise
        """
        if model_name not in self.budgets:
            return None

        budget_limit = self.budgets[model_name]
        percent_used = (current_cost / budget_limit) * 100 if budget_limit > 0 else 0

        alert = None
        if percent_used >= 100:
            alert = BudgetAlert(
                model_name=model_name,
                alert_type="exceeded",
                current_cost=current_cost,
                budget_limit=budget_limit,
                percent_used=percent_used,
            )
            logger.error(f"Budget exceeded for {model_name}: ${current_cost:.2f} / ${budget_limit:.2f}")
        elif percent_used >= 90:
            alert = BudgetAlert(
                model_name=model_name,
                alert_type="warning_90",
                current_cost=current_cost,
                budget_limit=budget_limit,
                percent_used=percent_used,
            )
            logger.warning(f"90% of budget used for {model_name}")
        elif percent_used >= 75:
            alert = BudgetAlert(
                model_name=model_name,
                alert_type="warning_75",
                current_cost=current_cost,
                budget_limit=budget_limit,
                percent_used=percent_used,
            )
            logger.warning(f"75% of budget used for {model_name}")
        elif percent_used >= 50:
            alert = BudgetAlert(
                model_name=model_name,
                alert_type="warning_50",
                current_cost=current_cost,
                budget_limit=budget_limit,
                percent_used=percent_used,
            )
            logger.info(f"50% of budget used for {model_name}")

        if alert:
            self.alerts.append(alert)

        return alert

    def get_cost_comparison(self) -> Dict[str, Dict]:
        """Compare costs across all training runs.

        Returns:
            Dictionary with cost statistics by model
        """
        comparison = {}

        for run_id, report in self.reports.items():
            model = report.model_name

            if model not in comparison:
                comparison[model] = {
                    "runs": 0,
                    "total_cost": 0.0,
                    "avg_cost_per_run": 0.0,
                    "avg_cost_per_sample": 0.0,
                    "avg_cost_per_epoch": 0.0,
                    "avg_accuracy": 0.0,
                    "best_efficiency_score": 0.0,
                }

            stats = comparison[model]
            stats["runs"] += 1
            stats["total_cost"] += report.total_cost

            if report.metrics:
                if report.metrics.test_accuracy:
                    stats["avg_accuracy"] = (
                        (stats["avg_accuracy"] * (stats["runs"] - 1) + report.metrics.test_accuracy)
                        / stats["runs"]
                    )

            stats["best_efficiency_score"] = max(
                stats["best_efficiency_score"], report.efficiency_score
            )

        # Calculate averages
        for model, stats in comparison.items():
            if stats["runs"] > 0:
                stats["avg_cost_per_run"] = stats["total_cost"] / stats["runs"]

            # Get average cost per sample/epoch from metrics
            sample_costs = []
            epoch_costs = []
            for run_id, report in self.reports.items():
                if report.model_name == model:
                    sample_costs.append(report.cost_per_sample)
                    epoch_costs.append(report.cost_per_epoch)

            if sample_costs:
                stats["avg_cost_per_sample"] = sum(sample_costs) / len(sample_costs)
            if epoch_costs:
                stats["avg_cost_per_epoch"] = sum(epoch_costs) / len(epoch_costs)

        return comparison

    def forecast_cost(
        self,
        model_name: str,
        planned_runs: int,
        avg_cost_per_run: Optional[float] = None,
    ) -> Dict:
        """Forecast cost for future training.

        Args:
            model_name: Name of the model
            planned_runs: Number of planned training runs
            avg_cost_per_run: Override average cost. If None, uses historical average.

        Returns:
            Forecast dictionary
        """
        if avg_cost_per_run is None:
            # Calculate historical average
            model_runs = [
                r for r in self.reports.values() if r.model_name == model_name
            ]
            if not model_runs:
                avg_cost_per_run = 0.0
            else:
                avg_cost_per_run = sum(r.total_cost for r in model_runs) / len(
                    model_runs
                )

        forecast = {
            "model_name": model_name,
            "planned_runs": planned_runs,
            "avg_cost_per_run": avg_cost_per_run,
            "estimated_total": planned_runs * avg_cost_per_run,
            "budget_status": "OK",
        }

        if model_name in self.budgets:
            budget = self.budgets[model_name]
            if forecast["estimated_total"] > budget:
                forecast["budget_status"] = "OVER"
                forecast["overage"] = forecast["estimated_total"] - budget
            elif forecast["estimated_total"] > budget * 0.9:
                forecast["budget_status"] = "WARNING"

        return forecast

    def get_roi_analysis(self, model_name: str) -> Dict:
        """Calculate ROI of training investments.

        Args:
            model_name: Name of the model

        Returns:
            ROI analysis
        """
        model_reports = [r for r in self.reports.values() if r.model_name == model_name]

        if not model_reports:
            return {
                "model_name": model_name,
                "total_investment": 0.0,
                "improvement_per_dollar": 0.0,
            }

        total_cost = sum(r.total_cost for r in model_reports)

        # Calculate accuracy improvement
        first_run = min(model_reports, key=lambda r: r.timestamp)
        last_run = max(model_reports, key=lambda r: r.timestamp)

        accuracy_improvement = 0.0
        if (
            first_run.metrics
            and last_run.metrics
            and first_run.metrics.test_accuracy
            and last_run.metrics.test_accuracy
        ):
            accuracy_improvement = (
                last_run.metrics.test_accuracy - first_run.metrics.test_accuracy
            )

        roi = {
            "model_name": model_name,
            "runs": len(model_reports),
            "total_investment": total_cost,
            "first_accuracy": (
                first_run.metrics.test_accuracy if first_run.metrics else 0.0
            ),
            "current_accuracy": (
                last_run.metrics.test_accuracy if last_run.metrics else 0.0
            ),
            "accuracy_improvement": accuracy_improvement,
            "improvement_per_dollar": (
                accuracy_improvement / total_cost if total_cost > 0 else 0.0
            ),
        }

        return roi
