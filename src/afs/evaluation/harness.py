"""Evaluation harness for training samples and model outputs.

Provides automated metrics for:
- Syntax validity (asar compilation)
- Quality scores (ELECTRA discriminator)
- Entity coverage and precision
- Output diversity
"""

from __future__ import annotations

import json
import math
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from afs.generators.base import TrainingSample
    from afs.training.scoring import QualityScorer


@dataclass
class EvaluationResult:
    """Results from evaluating a set of samples."""

    # Sample counts
    total_samples: int = 0
    evaluated_samples: int = 0

    # Syntax validation
    syntax_pass_count: int = 0
    syntax_pass_rate: float = 0.0
    syntax_errors: list[str] = field(default_factory=list)

    # Quality scores
    mean_quality_score: float = 0.0
    median_quality_score: float = 0.0
    std_quality_score: float = 0.0
    quality_distribution: dict[str, int] = field(default_factory=dict)

    # ELECTRA scores
    mean_electra_score: float = 0.0
    electra_distribution: dict[str, int] = field(default_factory=dict)

    # Entity metrics
    total_entities: int = 0
    known_entities: int = 0
    entity_coverage: float = 0.0
    entity_precision: float = 0.0  # Known / Total
    unique_entities: int = 0

    # Diversity metrics
    unique_instructions: int = 0
    unique_outputs: int = 0
    instruction_diversity: float = 0.0  # unique / total
    output_diversity: float = 0.0
    domain_distribution: dict[str, int] = field(default_factory=dict)

    # Length stats
    mean_output_length: float = 0.0
    min_output_length: int = 0
    max_output_length: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "total_samples": self.total_samples,
            "evaluated_samples": self.evaluated_samples,
            "syntax": {
                "pass_count": self.syntax_pass_count,
                "pass_rate": self.syntax_pass_rate,
                "error_count": len(self.syntax_errors),
            },
            "quality": {
                "mean": self.mean_quality_score,
                "median": self.median_quality_score,
                "std": self.std_quality_score,
                "distribution": self.quality_distribution,
            },
            "electra": {
                "mean": self.mean_electra_score,
                "distribution": self.electra_distribution,
            },
            "entities": {
                "total": self.total_entities,
                "known": self.known_entities,
                "coverage": self.entity_coverage,
                "precision": self.entity_precision,
                "unique": self.unique_entities,
            },
            "diversity": {
                "unique_instructions": self.unique_instructions,
                "unique_outputs": self.unique_outputs,
                "instruction_diversity": self.instruction_diversity,
                "output_diversity": self.output_diversity,
                "domain_distribution": self.domain_distribution,
            },
            "length": {
                "mean": self.mean_output_length,
                "min": self.min_output_length,
                "max": self.max_output_length,
            },
        }

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Evaluation Summary ({self.evaluated_samples}/{self.total_samples} samples)",
            "=" * 60,
            "",
            "Syntax Validation:",
            f"  Pass rate: {100 * self.syntax_pass_rate:.1f}% ({self.syntax_pass_count}/{self.evaluated_samples})",
            "",
            "Quality Scores:",
            f"  Mean: {self.mean_quality_score:.3f} (std: {self.std_quality_score:.3f})",
            f"  Median: {self.median_quality_score:.3f}",
            "",
            "ELECTRA Discriminator:",
            f"  Mean score: {self.mean_electra_score:.3f}",
            "",
            "Entity Coverage:",
            f"  Total entities: {self.total_entities}",
            f"  Known entities: {self.known_entities} ({100 * self.entity_precision:.1f}%)",
            f"  Unique entities: {self.unique_entities}",
            "",
            "Diversity:",
            f"  Unique instructions: {self.unique_instructions} ({100 * self.instruction_diversity:.1f}%)",
            f"  Unique outputs: {self.unique_outputs} ({100 * self.output_diversity:.1f}%)",
            "",
            "Output Length:",
            f"  Mean: {self.mean_output_length:.0f} chars",
            f"  Range: {self.min_output_length} - {self.max_output_length} chars",
        ]
        return "\n".join(lines)


@dataclass
class ComparisonResult:
    """Results from comparing two datasets."""

    baseline: EvaluationResult
    candidate: EvaluationResult

    # Improvement metrics (positive = candidate better)
    syntax_improvement: float = 0.0
    quality_improvement: float = 0.0
    electra_improvement: float = 0.0
    entity_improvement: float = 0.0
    diversity_improvement: float = 0.0

    # Statistical significance
    is_significant: bool = False
    p_value: float = 1.0

    # Recommendation
    recommendation: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline": self.baseline.to_dict(),
            "candidate": self.candidate.to_dict(),
            "improvements": {
                "syntax": self.syntax_improvement,
                "quality": self.quality_improvement,
                "electra": self.electra_improvement,
                "entity": self.entity_improvement,
                "diversity": self.diversity_improvement,
            },
            "significance": {
                "is_significant": self.is_significant,
                "p_value": self.p_value,
            },
            "recommendation": self.recommendation,
        }

    def summary(self) -> str:
        """Generate comparison summary."""
        lines = [
            "Comparison Summary",
            "=" * 60,
            "",
            f"{'Metric':<25} {'Baseline':>12} {'Candidate':>12} {'Change':>12}",
            "-" * 60,
            f"{'Syntax pass rate':<25} {100*self.baseline.syntax_pass_rate:>11.1f}% {100*self.candidate.syntax_pass_rate:>11.1f}% {100*self.syntax_improvement:>+11.1f}%",
            f"{'Mean quality':<25} {self.baseline.mean_quality_score:>12.3f} {self.candidate.mean_quality_score:>12.3f} {self.quality_improvement:>+12.3f}",
            f"{'Mean ELECTRA':<25} {self.baseline.mean_electra_score:>12.3f} {self.candidate.mean_electra_score:>12.3f} {self.electra_improvement:>+12.3f}",
            f"{'Entity precision':<25} {100*self.baseline.entity_precision:>11.1f}% {100*self.candidate.entity_precision:>11.1f}% {100*self.entity_improvement:>+11.1f}%",
            f"{'Output diversity':<25} {100*self.baseline.output_diversity:>11.1f}% {100*self.candidate.output_diversity:>11.1f}% {100*self.diversity_improvement:>+11.1f}%",
            "",
            f"Recommendation: {self.recommendation}",
        ]
        return "\n".join(lines)


class EvaluationHarness:
    """Harness for evaluating training samples."""

    def __init__(
        self,
        scorer: "QualityScorer | None" = None,
        validate_syntax: bool = True,
        extract_entities: bool = True,
    ):
        """Initialize evaluation harness.

        Args:
            scorer: Quality scorer instance (creates default if None)
            validate_syntax: Whether to validate syntax with asar
            extract_entities: Whether to extract and count entities
        """
        self._scorer = scorer
        self._validate_syntax = validate_syntax
        self._extract_entities = extract_entities

    @property
    def scorer(self) -> "QualityScorer":
        """Lazy load scorer."""
        if self._scorer is None:
            from afs.training.scoring import QualityScorer, ScoringConfig

            self._scorer = QualityScorer(config=ScoringConfig())
        return self._scorer

    def evaluate(self, samples: list["TrainingSample"]) -> EvaluationResult:
        """Evaluate a set of samples.

        Args:
            samples: List of training samples

        Returns:
            EvaluationResult with all metrics
        """
        result = EvaluationResult()
        result.total_samples = len(samples)

        if not samples:
            return result

        # Score all samples
        scores = self.scorer.score_batch(samples, update_samples=True)
        result.evaluated_samples = len(scores)

        # Collect metrics
        quality_scores = []
        electra_scores = []
        output_lengths = []
        all_entities: set[str] = set()
        instructions: set[str] = set()
        outputs: set[str] = set()
        domains: Counter[str] = Counter()

        for sample, score in zip(samples, scores):
            quality_scores.append(score.overall)
            electra_scores.append(score.electra_score)
            output_lengths.append(len(sample.output))

            if score.asar_valid:
                result.syntax_pass_count += 1
            else:
                if score.asar_errors:
                    result.syntax_errors.extend(score.asar_errors[:3])  # Limit errors

            result.total_entities += score.entity_count
            result.known_entities += score.known_entity_count

            # Track unique entities from sample
            for entity in sample.kg_entities:
                all_entities.add(entity)

            instructions.add(sample.instruction)
            outputs.add(sample.output)
            domains[sample.domain] += 1

        # Compute aggregates
        n = len(quality_scores)

        # Syntax
        result.syntax_pass_rate = result.syntax_pass_count / n if n else 0.0

        # Quality
        result.mean_quality_score = sum(quality_scores) / n if n else 0.0
        sorted_scores = sorted(quality_scores)
        result.median_quality_score = sorted_scores[n // 2] if n else 0.0
        if n > 1:
            variance = sum((s - result.mean_quality_score) ** 2 for s in quality_scores) / (n - 1)
            result.std_quality_score = math.sqrt(variance)

        # Quality distribution
        result.quality_distribution = self._compute_histogram(quality_scores)

        # ELECTRA
        result.mean_electra_score = sum(electra_scores) / n if n else 0.0
        result.electra_distribution = self._compute_histogram(electra_scores)

        # Entities
        result.unique_entities = len(all_entities)
        result.entity_precision = result.known_entities / result.total_entities if result.total_entities else 0.0
        result.entity_coverage = result.entity_precision  # Same metric

        # Diversity
        result.unique_instructions = len(instructions)
        result.unique_outputs = len(outputs)
        result.instruction_diversity = len(instructions) / n if n else 0.0
        result.output_diversity = len(outputs) / n if n else 0.0
        result.domain_distribution = dict(domains)

        # Length
        result.mean_output_length = sum(output_lengths) / n if n else 0.0
        result.min_output_length = min(output_lengths) if output_lengths else 0
        result.max_output_length = max(output_lengths) if output_lengths else 0

        return result

    def compare(
        self,
        baseline: list["TrainingSample"],
        candidate: list["TrainingSample"],
    ) -> ComparisonResult:
        """Compare two sets of samples.

        Args:
            baseline: Baseline samples
            candidate: Candidate samples to compare

        Returns:
            ComparisonResult with improvement metrics
        """
        baseline_result = self.evaluate(baseline)
        candidate_result = self.evaluate(candidate)

        result = ComparisonResult(
            baseline=baseline_result,
            candidate=candidate_result,
        )

        # Compute improvements
        result.syntax_improvement = (
            candidate_result.syntax_pass_rate - baseline_result.syntax_pass_rate
        )
        result.quality_improvement = (
            candidate_result.mean_quality_score - baseline_result.mean_quality_score
        )
        result.electra_improvement = (
            candidate_result.mean_electra_score - baseline_result.mean_electra_score
        )
        result.entity_improvement = (
            candidate_result.entity_precision - baseline_result.entity_precision
        )
        result.diversity_improvement = (
            candidate_result.output_diversity - baseline_result.output_diversity
        )

        # Generate recommendation
        improvements = [
            ("syntax", result.syntax_improvement),
            ("quality", result.quality_improvement),
            ("electra", result.electra_improvement),
            ("entity", result.entity_improvement),
            ("diversity", result.diversity_improvement),
        ]

        positive = sum(1 for _, v in improvements if v > 0.01)
        negative = sum(1 for _, v in improvements if v < -0.01)

        if positive >= 4:
            result.recommendation = "Strong improvement - recommend candidate"
        elif positive >= 3 and negative == 0:
            result.recommendation = "Moderate improvement - recommend candidate"
        elif positive > negative:
            result.recommendation = "Mixed results - candidate slightly better"
        elif negative > positive:
            result.recommendation = "Regression - keep baseline"
        else:
            result.recommendation = "No significant difference"

        return result

    def _compute_histogram(self, values: list[float], buckets: int = 10) -> dict[str, int]:
        """Compute histogram of values."""
        hist: dict[str, int] = {}
        for i in range(buckets):
            low = i / buckets
            high = (i + 1) / buckets
            key = f"{low:.1f}-{high:.1f}"
            hist[key] = sum(1 for v in values if low <= v < high)
        return hist


def evaluate_samples(
    samples: list["TrainingSample"],
    electra_path: Path | None = None,
) -> EvaluationResult:
    """Convenience function to evaluate samples.

    Args:
        samples: Samples to evaluate
        electra_path: Optional path to ELECTRA model

    Returns:
        EvaluationResult
    """
    from afs.training.scoring import QualityScorer, ScoringConfig

    config = ScoringConfig(electra_model_path=electra_path)
    scorer = QualityScorer(config=config)
    harness = EvaluationHarness(scorer=scorer)
    return harness.evaluate(samples)


def compare_datasets(
    baseline_path: Path,
    candidate_path: Path,
    electra_path: Path | None = None,
) -> ComparisonResult:
    """Compare two JSONL datasets.

    Args:
        baseline_path: Path to baseline JSONL
        candidate_path: Path to candidate JSONL
        electra_path: Optional path to ELECTRA model

    Returns:
        ComparisonResult
    """
    from afs.generators.base import TrainingSample
    from afs.training.scoring import QualityScorer, ScoringConfig

    # Load samples
    def load_samples(path: Path) -> list[TrainingSample]:
        samples = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    samples.append(TrainingSample.from_dict(json.loads(line)))
        return samples

    baseline = load_samples(baseline_path)
    candidate = load_samples(candidate_path)

    config = ScoringConfig(electra_model_path=electra_path)
    scorer = QualityScorer(config=config)
    harness = EvaluationHarness(scorer=scorer)

    return harness.compare(baseline, candidate)
