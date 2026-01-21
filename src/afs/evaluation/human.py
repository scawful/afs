"""Human evaluation workflow management.

Provides tools for:
- Creating evaluation batches for human review
- Exporting tasks in various formats (JSON, CSV)
- Importing human ratings
- Updating training data based on feedback
"""

from __future__ import annotations

import csv
import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from afs.generators.base import TrainingSample


class SamplingStrategy(str, Enum):
    """Strategy for selecting samples for human evaluation."""

    RANDOM = "random"  # Random sampling
    UNCERTAINTY = "uncertainty"  # Samples with ELECTRA score near 0.5
    LOW_QUALITY = "low_quality"  # Samples with low quality scores
    HIGH_QUALITY = "high_quality"  # Samples with high quality scores
    STRATIFIED = "stratified"  # Stratified by domain


class TaskStatus(str, Enum):
    """Status of a human evaluation task."""

    PENDING = "pending"
    COMPLETED = "completed"
    SKIPPED = "skipped"


@dataclass
class HumanEvalTask:
    """A single human evaluation task."""

    task_id: str
    sample_id: str
    instruction: str
    output: str
    domain: str

    # Pre-computed scores (for reference)
    quality_score: float = 0.0
    electra_score: float = 0.0

    # Human ratings
    status: TaskStatus = TaskStatus.PENDING
    human_rating: float | None = None  # 0.0 - 1.0
    human_notes: str = ""
    rated_at: str | None = None
    rated_by: str | None = None

    # Metadata
    created_at: str = ""
    batch_id: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "task_id": self.task_id,
            "sample_id": self.sample_id,
            "instruction": self.instruction,
            "output": self.output,
            "domain": self.domain,
            "quality_score": self.quality_score,
            "electra_score": self.electra_score,
            "status": self.status.value,
            "human_rating": self.human_rating,
            "human_notes": self.human_notes,
            "rated_at": self.rated_at,
            "rated_by": self.rated_by,
            "created_at": self.created_at,
            "batch_id": self.batch_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HumanEvalTask:
        return cls(
            task_id=data["task_id"],
            sample_id=data["sample_id"],
            instruction=data["instruction"],
            output=data["output"],
            domain=data.get("domain", "unknown"),
            quality_score=data.get("quality_score", 0.0),
            electra_score=data.get("electra_score", 0.0),
            status=TaskStatus(data.get("status", "pending")),
            human_rating=data.get("human_rating"),
            human_notes=data.get("human_notes", ""),
            rated_at=data.get("rated_at"),
            rated_by=data.get("rated_by"),
            created_at=data.get("created_at", ""),
            batch_id=data.get("batch_id", ""),
        )

    @classmethod
    def from_sample(
        cls,
        sample: TrainingSample,
        batch_id: str,
        quality_score: float = 0.0,
        electra_score: float = 0.0,
    ) -> HumanEvalTask:
        """Create task from a training sample."""
        return cls(
            task_id=str(uuid.uuid4()),
            sample_id=sample.sample_id,
            instruction=sample.instruction,
            output=sample.output,
            domain=sample.domain,
            quality_score=quality_score,
            electra_score=electra_score,
            created_at=datetime.now().isoformat(),
            batch_id=batch_id,
        )


@dataclass
class HumanEvalBatch:
    """A batch of human evaluation tasks."""

    batch_id: str
    name: str
    created_at: str
    strategy: SamplingStrategy
    tasks: list[HumanEvalTask] = field(default_factory=list)

    # Batch metadata
    total_tasks: int = 0
    completed_tasks: int = 0
    skipped_tasks: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "batch_id": self.batch_id,
            "name": self.name,
            "created_at": self.created_at,
            "strategy": self.strategy.value,
            "total_tasks": self.total_tasks,
            "completed_tasks": self.completed_tasks,
            "skipped_tasks": self.skipped_tasks,
            "tasks": [t.to_dict() for t in self.tasks],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> HumanEvalBatch:
        batch = cls(
            batch_id=data["batch_id"],
            name=data["name"],
            created_at=data["created_at"],
            strategy=SamplingStrategy(data.get("strategy", "random")),
            total_tasks=data.get("total_tasks", 0),
            completed_tasks=data.get("completed_tasks", 0),
            skipped_tasks=data.get("skipped_tasks", 0),
        )
        batch.tasks = [HumanEvalTask.from_dict(t) for t in data.get("tasks", [])]
        return batch

    def update_counts(self) -> None:
        """Update task counts from tasks list."""
        self.total_tasks = len(self.tasks)
        self.completed_tasks = sum(1 for t in self.tasks if t.status == TaskStatus.COMPLETED)
        self.skipped_tasks = sum(1 for t in self.tasks if t.status == TaskStatus.SKIPPED)


class HumanEvaluationManager:
    """Manage human evaluation workflows."""

    def __init__(self, storage_dir: Path | None = None):
        """Initialize manager.

        Args:
            storage_dir: Directory for storing batches
        """
        self.storage_dir = storage_dir or Path("./human_eval")
        self.storage_dir.mkdir(parents=True, exist_ok=True)

    def create_batch(
        self,
        samples: list[TrainingSample],
        name: str = "",
        n: int = 50,
        strategy: SamplingStrategy = SamplingStrategy.UNCERTAINTY,
    ) -> HumanEvalBatch:
        """Create a new evaluation batch.

        Args:
            samples: Source samples to select from
            name: Batch name
            n: Number of samples to include
            strategy: Selection strategy

        Returns:
            HumanEvalBatch with selected tasks
        """
        batch_id = str(uuid.uuid4())[:8]
        name = name or f"batch_{batch_id}"

        batch = HumanEvalBatch(
            batch_id=batch_id,
            name=name,
            created_at=datetime.now().isoformat(),
            strategy=strategy,
        )

        # Select samples based on strategy
        selected = self._select_samples(samples, n, strategy)

        # Create tasks
        for sample in selected:
            task = HumanEvalTask.from_sample(
                sample,
                batch_id=batch_id,
                quality_score=sample.quality_score,
                electra_score=sample._metadata.get("quality_components", {}).get("electra", 0.5),
            )
            batch.tasks.append(task)

        batch.update_counts()
        return batch

    def _select_samples(
        self,
        samples: list[TrainingSample],
        n: int,
        strategy: SamplingStrategy,
    ) -> list[TrainingSample]:
        """Select samples using the specified strategy."""
        import random

        if not samples:
            return []

        n = min(n, len(samples))

        if strategy == SamplingStrategy.RANDOM:
            return random.sample(samples, n)

        elif strategy == SamplingStrategy.UNCERTAINTY:
            # Sort by distance from 0.5 ELECTRA score (most uncertain first)
            def uncertainty(s: TrainingSample) -> float:
                electra = s._metadata.get("quality_components", {}).get("electra", 0.5)
                return abs(electra - 0.5)

            sorted_samples = sorted(samples, key=uncertainty)
            return sorted_samples[:n]

        elif strategy == SamplingStrategy.LOW_QUALITY:
            sorted_samples = sorted(samples, key=lambda s: s.quality_score)
            return sorted_samples[:n]

        elif strategy == SamplingStrategy.HIGH_QUALITY:
            sorted_samples = sorted(samples, key=lambda s: -s.quality_score)
            return sorted_samples[:n]

        elif strategy == SamplingStrategy.STRATIFIED:
            # Select proportionally from each domain
            from collections import defaultdict

            by_domain: dict[str, list[TrainingSample]] = defaultdict(list)
            for s in samples:
                by_domain[s.domain].append(s)

            selected: list[TrainingSample] = []
            domains = list(by_domain.keys())
            per_domain = max(1, n // len(domains))

            for domain in domains:
                domain_samples = by_domain[domain]
                count = min(per_domain, len(domain_samples))
                selected.extend(random.sample(domain_samples, count))

            # Fill remaining with random
            remaining = n - len(selected)
            if remaining > 0:
                pool = [s for s in samples if s not in selected]
                if pool:
                    selected.extend(random.sample(pool, min(remaining, len(pool))))

            return selected[:n]

        return samples[:n]

    def save_batch(self, batch: HumanEvalBatch, path: Path | None = None) -> Path:
        """Save batch to JSON file.

        Args:
            batch: Batch to save
            path: Output path (default: storage_dir/batch_{id}.json)

        Returns:
            Path to saved file
        """
        path = path or (self.storage_dir / f"batch_{batch.batch_id}.json")
        with open(path, "w") as f:
            json.dump(batch.to_dict(), f, indent=2)
        return path

    def load_batch(self, path: Path) -> HumanEvalBatch:
        """Load batch from JSON file."""
        with open(path) as f:
            data = json.load(f)
        return HumanEvalBatch.from_dict(data)

    def export_csv(self, batch: HumanEvalBatch, path: Path) -> None:
        """Export batch to CSV for spreadsheet-based review.

        Args:
            batch: Batch to export
            path: Output CSV path
        """
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "task_id",
                "sample_id",
                "domain",
                "instruction",
                "output",
                "quality_score",
                "electra_score",
                "human_rating",
                "human_notes",
            ])

            for task in batch.tasks:
                writer.writerow([
                    task.task_id,
                    task.sample_id,
                    task.domain,
                    task.instruction,
                    task.output[:500],  # Truncate for spreadsheet
                    f"{task.quality_score:.3f}",
                    f"{task.electra_score:.3f}",
                    "",  # human_rating - to be filled
                    "",  # human_notes - to be filled
                ])

    def import_csv(self, batch: HumanEvalBatch, path: Path) -> int:
        """Import ratings from CSV back into batch.

        Args:
            batch: Batch to update
            path: CSV path with ratings

        Returns:
            Number of tasks updated
        """
        task_map = {t.task_id: t for t in batch.tasks}
        updated = 0

        with open(path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                task_id = row.get("task_id", "")
                if task_id in task_map:
                    task = task_map[task_id]

                    rating_str = row.get("human_rating", "").strip()
                    if rating_str:
                        try:
                            task.human_rating = float(rating_str)
                            task.status = TaskStatus.COMPLETED
                            task.rated_at = datetime.now().isoformat()
                            updated += 1
                        except ValueError:
                            pass

                    task.human_notes = row.get("human_notes", "")

        batch.update_counts()
        return updated

    def import_results(self, batch: HumanEvalBatch, results_path: Path) -> int:
        """Import results from JSON file.

        Args:
            batch: Batch to update
            results_path: JSON file with task_id -> rating mapping

        Returns:
            Number of tasks updated
        """
        with open(results_path) as f:
            results = json.load(f)

        task_map = {t.task_id: t for t in batch.tasks}
        updated = 0

        for task_id, rating_data in results.items():
            if task_id in task_map:
                task = task_map[task_id]

                if isinstance(rating_data, (int, float)):
                    task.human_rating = float(rating_data)
                elif isinstance(rating_data, dict):
                    task.human_rating = rating_data.get("rating")
                    task.human_notes = rating_data.get("notes", "")
                    task.rated_by = rating_data.get("rater")

                if task.human_rating is not None:
                    task.status = TaskStatus.COMPLETED
                    task.rated_at = datetime.now().isoformat()
                    updated += 1

        batch.update_counts()
        return updated

    def update_training_data(
        self,
        batch: HumanEvalBatch,
        samples: list[TrainingSample],
        threshold: float = 0.7,
    ) -> tuple[list[TrainingSample], list[TrainingSample]]:
        """Update training samples based on human ratings.

        Samples rated above threshold are kept, below are filtered.

        Args:
            batch: Completed evaluation batch
            samples: Original training samples
            threshold: Minimum human rating to keep

        Returns:
            Tuple of (kept_samples, filtered_samples)
        """
        # Build rating lookup
        ratings = {}
        for task in batch.tasks:
            if task.status == TaskStatus.COMPLETED and task.human_rating is not None:
                ratings[task.sample_id] = task.human_rating

        kept = []
        filtered = []

        for sample in samples:
            if sample.sample_id in ratings:
                rating = ratings[sample.sample_id]
                sample._metadata["human_rating"] = rating
                sample._metadata["human_evaluated"] = True

                if rating >= threshold:
                    kept.append(sample)
                else:
                    filtered.append(sample)
            else:
                # Not evaluated - keep by default
                kept.append(sample)

        return kept, filtered

    def get_batch_summary(self, batch: HumanEvalBatch) -> dict[str, Any]:
        """Get summary statistics for a batch."""
        completed = [t for t in batch.tasks if t.status == TaskStatus.COMPLETED]
        ratings = [t.human_rating for t in completed if t.human_rating is not None]

        return {
            "batch_id": batch.batch_id,
            "name": batch.name,
            "total_tasks": batch.total_tasks,
            "completed": len(completed),
            "pending": batch.total_tasks - len(completed) - batch.skipped_tasks,
            "skipped": batch.skipped_tasks,
            "completion_rate": len(completed) / batch.total_tasks if batch.total_tasks else 0,
            "ratings": {
                "count": len(ratings),
                "mean": sum(ratings) / len(ratings) if ratings else 0,
                "min": min(ratings) if ratings else 0,
                "max": max(ratings) if ratings else 0,
            },
        }


def create_eval_batch(
    samples: list[TrainingSample],
    n: int = 50,
    strategy: str = "uncertainty",
    output_path: Path | None = None,
) -> HumanEvalBatch:
    """Convenience function to create an evaluation batch.

    Args:
        samples: Source samples
        n: Number of samples
        strategy: Selection strategy name
        output_path: Optional path to save batch

    Returns:
        Created batch
    """
    manager = HumanEvaluationManager()
    batch = manager.create_batch(
        samples,
        n=n,
        strategy=SamplingStrategy(strategy),
    )

    if output_path:
        manager.save_batch(batch, output_path)

    return batch


def import_eval_results(
    batch_path: Path,
    results_path: Path,
    output_path: Path | None = None,
) -> HumanEvalBatch:
    """Import evaluation results and optionally save updated batch.

    Args:
        batch_path: Path to batch JSON
        results_path: Path to results (JSON or CSV)
        output_path: Optional path to save updated batch

    Returns:
        Updated batch
    """
    manager = HumanEvaluationManager()
    batch = manager.load_batch(batch_path)

    if results_path.suffix == ".csv":
        manager.import_csv(batch, results_path)
    else:
        manager.import_results(batch, results_path)

    if output_path:
        manager.save_batch(batch, output_path)

    return batch
