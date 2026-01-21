"""Model experiment registry for tracking and A/B testing."""

from __future__ import annotations

import json
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class ExperimentMetrics:
    """Metrics for a training experiment."""

    final_loss: float | None = None
    final_val_loss: float | None = None
    perplexity: float | None = None
    bleu_score: float | None = None
    rouge_scores: dict[str, float] = field(default_factory=dict)
    custom_metrics: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ExperimentMetrics:
        """Create from dictionary."""
        return cls(**data)


@dataclass
class ModelExperiment:
    """A single model training experiment."""

    experiment_id: str
    run_name: str
    base_model: str
    framework: str  # mlx, unsloth, llama_cpp
    status: str = "pending"  # pending, running, completed, failed
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    completed_at: str | None = None

    # Training details
    dataset_path: str | None = None
    output_dir: str | None = None
    config: dict[str, Any] = field(default_factory=dict)

    # Results
    metrics: ExperimentMetrics = field(default_factory=ExperimentMetrics)

    # A/B testing
    ab_group: str | None = None  # e.g., "thinking_models", "coder_models"
    ab_variant: str | None = None  # e.g., "A", "B", "control"

    # Metadata
    tags: list[str] = field(default_factory=list)
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        data = asdict(self)
        data["metrics"] = self.metrics.to_dict()
        return data

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelExperiment:
        """Create from dictionary."""
        metrics_data = data.pop("metrics", {})
        metrics = ExperimentMetrics.from_dict(metrics_data)
        return cls(**data, metrics=metrics)


class ModelRegistry:
    """Registry for tracking model training experiments.

    Persists experiments to a JSON file for tracking across sessions.

    Example:
        ```python
        registry = ModelRegistry()

        # Create experiment
        exp = registry.create_experiment(
            run_name="qwen_coder_v1",
            base_model="Qwen/Qwen2.5-Coder-7B-Instruct",
            framework="mlx",
            ab_group="coder_models",
        )

        # Update status
        registry.update_status(exp.experiment_id, "running")

        # Record metrics
        registry.record_metrics(exp.experiment_id, final_loss=0.15)

        # Compare experiments
        registry.compare(["exp1", "exp2"])
        ```
    """

    DEFAULT_PATH = Path.home() / ".afs" / "model_registry.json"

    def __init__(self, registry_path: Path | None = None):
        """Initialize registry.

        Args:
            registry_path: Path to registry JSON file
        """
        self.registry_path = Path(registry_path or self.DEFAULT_PATH)
        self.experiments: dict[str, ModelExperiment] = {}
        self._load()

    def _load(self) -> None:
        """Load experiments from disk."""
        if self.registry_path.exists():
            with open(self.registry_path) as f:
                data = json.load(f)
                for exp_data in data.get("experiments", []):
                    exp = ModelExperiment.from_dict(exp_data)
                    self.experiments[exp.experiment_id] = exp

    def _save(self) -> None:
        """Save experiments to disk."""
        self.registry_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "version": "1.0",
            "updated_at": datetime.now().isoformat(),
            "experiments": [exp.to_dict() for exp in self.experiments.values()],
        }
        with open(self.registry_path, "w") as f:
            json.dump(data, f, indent=2)

    def create_experiment(
        self,
        run_name: str,
        base_model: str,
        framework: str,
        dataset_path: str | None = None,
        output_dir: str | None = None,
        config: dict[str, Any] | None = None,
        ab_group: str | None = None,
        ab_variant: str | None = None,
        tags: list[str] | None = None,
        notes: str = "",
    ) -> ModelExperiment:
        """Create a new experiment.

        Args:
            run_name: Human-readable name
            base_model: Base model identifier
            framework: Training framework
            dataset_path: Path to training data
            output_dir: Output directory for checkpoints
            config: Training configuration dict
            ab_group: A/B test group name
            ab_variant: A/B test variant
            tags: Tags for categorization
            notes: Free-form notes

        Returns:
            Created experiment
        """
        experiment_id = str(uuid.uuid4())[:8]

        experiment = ModelExperiment(
            experiment_id=experiment_id,
            run_name=run_name,
            base_model=base_model,
            framework=framework,
            dataset_path=dataset_path,
            output_dir=output_dir,
            config=config or {},
            ab_group=ab_group,
            ab_variant=ab_variant,
            tags=tags or [],
            notes=notes,
        )

        self.experiments[experiment_id] = experiment
        self._save()
        return experiment

    def get(self, experiment_id: str) -> ModelExperiment | None:
        """Get experiment by ID."""
        return self.experiments.get(experiment_id)

    def list(
        self,
        status: str | None = None,
        ab_group: str | None = None,
        framework: str | None = None,
        tags: list[str] | None = None,
    ) -> list[ModelExperiment]:
        """List experiments with optional filtering.

        Args:
            status: Filter by status
            ab_group: Filter by A/B test group
            framework: Filter by framework
            tags: Filter by tags (any match)

        Returns:
            List of matching experiments
        """
        results = list(self.experiments.values())

        if status:
            results = [e for e in results if e.status == status]
        if ab_group:
            results = [e for e in results if e.ab_group == ab_group]
        if framework:
            results = [e for e in results if e.framework == framework]
        if tags:
            results = [e for e in results if any(t in e.tags for t in tags)]

        return sorted(results, key=lambda e: e.created_at, reverse=True)

    def update_status(
        self,
        experiment_id: str,
        status: str,
        notes: str | None = None,
    ) -> None:
        """Update experiment status.

        Args:
            experiment_id: Experiment ID
            status: New status
            notes: Optional notes to append
        """
        exp = self.experiments.get(experiment_id)
        if not exp:
            raise ValueError(f"Experiment not found: {experiment_id}")

        exp.status = status
        if status == "completed" or status == "failed":
            exp.completed_at = datetime.now().isoformat()
        if notes:
            exp.notes = f"{exp.notes}\n{notes}" if exp.notes else notes

        self._save()

    def record_metrics(
        self,
        experiment_id: str,
        **metrics: float,
    ) -> None:
        """Record metrics for an experiment.

        Args:
            experiment_id: Experiment ID
            **metrics: Metric name-value pairs
        """
        exp = self.experiments.get(experiment_id)
        if not exp:
            raise ValueError(f"Experiment not found: {experiment_id}")

        # Update known metrics
        for name, value in metrics.items():
            if hasattr(exp.metrics, name):
                setattr(exp.metrics, name, value)
            else:
                exp.metrics.custom_metrics[name] = value

        self._save()

    def compare(
        self,
        experiment_ids: list[str],
        metrics: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Compare experiments side by side.

        Args:
            experiment_ids: List of experiment IDs to compare
            metrics: Specific metrics to compare (default: all)

        Returns:
            List of comparison dicts
        """
        if metrics is None:
            metrics = ["final_loss", "final_val_loss", "perplexity", "bleu_score"]

        results = []
        for exp_id in experiment_ids:
            exp = self.experiments.get(exp_id)
            if not exp:
                continue

            row = {
                "experiment_id": exp.experiment_id,
                "run_name": exp.run_name,
                "base_model": exp.base_model,
                "status": exp.status,
            }

            for metric in metrics:
                if hasattr(exp.metrics, metric):
                    row[metric] = getattr(exp.metrics, metric)
                elif metric in exp.metrics.custom_metrics:
                    row[metric] = exp.metrics.custom_metrics[metric]
                else:
                    row[metric] = None

            results.append(row)

        return results

    def get_ab_group_experiments(self, group_name: str) -> dict[str, list[ModelExperiment]]:
        """Get experiments grouped by A/B variant.

        Args:
            group_name: A/B test group name

        Returns:
            Dict mapping variant name to experiments
        """
        grouped: dict[str, list[ModelExperiment]] = {}

        for exp in self.experiments.values():
            if exp.ab_group == group_name:
                variant = exp.ab_variant or "unassigned"
                if variant not in grouped:
                    grouped[variant] = []
                grouped[variant].append(exp)

        return grouped

    def delete(self, experiment_id: str) -> bool:
        """Delete an experiment.

        Args:
            experiment_id: Experiment ID

        Returns:
            True if deleted, False if not found
        """
        if experiment_id in self.experiments:
            del self.experiments[experiment_id]
            self._save()
            return True
        return False

    def summary(self) -> str:
        """Generate a summary of all experiments.

        Returns:
            Formatted summary string
        """
        lines = ["Model Experiment Registry", "=" * 40]

        by_status: dict[str, list[ModelExperiment]] = {}
        for exp in self.experiments.values():
            if exp.status not in by_status:
                by_status[exp.status] = []
            by_status[exp.status].append(exp)

        for status in ["running", "completed", "failed", "pending"]:
            if status in by_status:
                lines.append(f"\n{status.upper()} ({len(by_status[status])})")
                for exp in by_status[status]:
                    loss = exp.metrics.final_loss
                    loss_str = f"loss={loss:.4f}" if loss else ""
                    lines.append(f"  {exp.experiment_id}: {exp.run_name} ({exp.base_model}) {loss_str}")

        return "\n".join(lines)
