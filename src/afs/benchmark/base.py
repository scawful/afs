"""Base classes for the benchmark suite."""

from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator


@dataclass
class BenchmarkItem:
    """A single benchmark test case."""

    id: str
    category: str
    difficulty: int  # 1-4 (basic to expert)
    code: str
    expected_output: str | None = None
    expected_metrics: dict[str, float] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkItem":
        """Create from dictionary."""
        return cls(
            id=data["id"],
            category=data.get("category", "general"),
            difficulty=data.get("difficulty", 1),
            code=data["code"],
            expected_output=data.get("expected_output"),
            expected_metrics=data.get("expected_metrics", {}),
            metadata=data.get("metadata", {}),
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "category": self.category,
            "difficulty": self.difficulty,
            "code": self.code,
            "expected_output": self.expected_output,
            "expected_metrics": self.expected_metrics,
            "metadata": self.metadata,
        }


@dataclass
class BenchmarkResult:
    """Results from running a benchmark."""

    model: str
    domain: str
    category: str
    total_items: int
    passed: int
    failed: int
    metrics: dict[str, float]
    item_results: list[dict[str, Any]]
    timestamp: str
    duration_seconds: float

    @property
    def pass_rate(self) -> float:
        """Calculate pass rate."""
        if self.total_items == 0:
            return 0.0
        return self.passed / self.total_items

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model": self.model,
            "domain": self.domain,
            "category": self.category,
            "total_items": self.total_items,
            "passed": self.passed,
            "failed": self.failed,
            "pass_rate": self.pass_rate,
            "metrics": self.metrics,
            "item_results": self.item_results,
            "timestamp": self.timestamp,
            "duration_seconds": self.duration_seconds,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkResult":
        """Create from dictionary."""
        return cls(
            model=data["model"],
            domain=data["domain"],
            category=data.get("category", "all"),
            total_items=data["total_items"],
            passed=data.get("passed", 0),
            failed=data.get("failed", 0),
            metrics=data.get("metrics", {}),
            item_results=data.get("item_results", []),
            timestamp=data.get("timestamp", datetime.now().isoformat()),
            duration_seconds=data.get("duration_seconds", 0.0),
        )

    def summary(self) -> str:
        """Generate summary string."""
        lines = [
            f"Benchmark: {self.domain}/{self.category}",
            f"Model: {self.model}",
            f"Pass Rate: {self.pass_rate:.1%} ({self.passed}/{self.total_items})",
            f"Duration: {self.duration_seconds:.1f}s",
            "",
            "Metrics:",
        ]
        for name, value in self.metrics.items():
            lines.append(f"  {name}: {value:.3f}")
        return "\n".join(lines)


@dataclass
class BenchmarkConfig:
    """Configuration for running benchmarks."""

    dataset_path: Path
    model_name: str
    model_path: Path | None = None
    model_type: str = "api"  # api, mlx, huggingface
    api_provider: str = "gemini"
    enable_semantic_eval: bool = False
    parallel_workers: int = 1
    timeout_seconds: float = 60.0
    save_outputs: bool = True
    output_dir: Path = field(default_factory=lambda: Path("benchmark_results"))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "dataset_path": str(self.dataset_path),
            "model_name": self.model_name,
            "model_path": str(self.model_path) if self.model_path else None,
            "model_type": self.model_type,
            "api_provider": self.api_provider,
            "enable_semantic_eval": self.enable_semantic_eval,
            "parallel_workers": self.parallel_workers,
            "timeout_seconds": self.timeout_seconds,
            "save_outputs": self.save_outputs,
            "output_dir": str(self.output_dir),
        }


class BenchmarkRunner(ABC):
    """Base class for domain-specific benchmark runners."""

    domain: str = "base"

    def __init__(self, config: BenchmarkConfig):
        self.config = config
        self._generator = None

    def _get_generator(self):
        """Lazy-load generator based on config."""
        if self._generator is None:
            from ..generators.model_generator import create_generator

            self._generator = create_generator(
                model_type=self.config.model_type,
                api_provider=self.config.api_provider,
                model_path=self.config.model_path,
            )
        return self._generator

    def load_dataset(self) -> list[BenchmarkItem]:
        """Load benchmark items from dataset file."""
        items = []
        with open(self.config.dataset_path) as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    items.append(BenchmarkItem.from_dict(data))
        return items

    @abstractmethod
    def evaluate_item(self, item: BenchmarkItem, output: str) -> dict[str, Any]:
        """Evaluate a single benchmark item.

        Args:
            item: The benchmark test case
            output: The model's generated output

        Returns:
            Dictionary with:
                - passed: bool
                - score: float (0-1)
                - metrics: dict of specific metrics
                - details: any additional info
        """
        pass

    def generate_for_item(self, item: BenchmarkItem) -> str:
        """Generate output for a benchmark item."""
        generator = self._get_generator()
        instruction = self._format_instruction(item)
        sample = generator.generate_one(instruction)
        return sample.output if sample else ""

    def _format_instruction(self, item: BenchmarkItem) -> str:
        """Format instruction for the model. Override in subclasses."""
        return item.code

    def run(self) -> BenchmarkResult:
        """Run the benchmark and return results."""
        items = self.load_dataset()
        start_time = time.time()
        item_results = []
        passed = 0
        failed = 0
        all_metrics: dict[str, list[float]] = {}

        for item in items:
            try:
                output = self.generate_for_item(item)
                eval_result = self.evaluate_item(item, output)

                if eval_result.get("passed", False):
                    passed += 1
                else:
                    failed += 1

                # Collect metrics
                for key, value in eval_result.get("metrics", {}).items():
                    if key not in all_metrics:
                        all_metrics[key] = []
                    all_metrics[key].append(value)

                item_results.append({
                    "id": item.id,
                    "passed": eval_result.get("passed", False),
                    "score": eval_result.get("score", 0.0),
                    "metrics": eval_result.get("metrics", {}),
                    "output": output if self.config.save_outputs else None,
                })

            except Exception as e:
                failed += 1
                item_results.append({
                    "id": item.id,
                    "passed": False,
                    "score": 0.0,
                    "error": str(e),
                })

        # Aggregate metrics
        aggregated_metrics = {}
        for key, values in all_metrics.items():
            if values:
                aggregated_metrics[f"{key}_mean"] = sum(values) / len(values)
                aggregated_metrics[f"{key}_min"] = min(values)
                aggregated_metrics[f"{key}_max"] = max(values)

        duration = time.time() - start_time

        return BenchmarkResult(
            model=self.config.model_name,
            domain=self.domain,
            category="all",
            total_items=len(items),
            passed=passed,
            failed=failed,
            metrics=aggregated_metrics,
            item_results=item_results,
            timestamp=datetime.now().isoformat(),
            duration_seconds=duration,
        )


def load_benchmark_items(path: Path) -> Iterator[BenchmarkItem]:
    """Load benchmark items from a JSONL file."""
    with open(path) as f:
        for line in f:
            if line.strip():
                yield BenchmarkItem.from_dict(json.loads(line))


def save_benchmark_items(items: list[BenchmarkItem], path: Path) -> int:
    """Save benchmark items to a JSONL file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for item in items:
            f.write(json.dumps(item.to_dict()) + "\n")
    return len(items)
