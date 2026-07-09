"""Quality scoring for training samples.

Provides minimal scoring primitives used by extension-owned training workflows
and other core modules. Domain-specific scoring logic belongs in a companion
extension repo.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ScoringConfig:
    """Configuration for quality scoring."""

    min_instruction_length: int = 10
    min_output_length: int = 20
    penalize_short: bool = True


@dataclass
class QualityScore:
    """Result of scoring a single sample."""

    overall: float = 0.0
    instruction_clarity: float = 0.0
    output_quality: float = 0.0
    details: dict[str, Any] = field(default_factory=dict)

    def overall_score(self) -> float:
        return self.overall


class QualityScorer:
    """Score training samples for quality.

    This base implementation uses heuristic length/structure checks.
    Domain-specific scorers (LLM-based, embedding-based) belong in a companion extension repo.
    """

    def __init__(self, config: ScoringConfig | None = None):
        self.config = config or ScoringConfig()

    def score(self, sample: Any) -> QualityScore:
        """Score a single training sample."""
        instruction = getattr(sample, "instruction", "") or ""
        output = getattr(sample, "output", "") or ""

        inst_len = len(instruction.strip())
        out_len = len(output.strip())

        # Simple heuristic scoring
        inst_score = min(1.0, inst_len / max(self.config.min_instruction_length * 5, 1))
        out_score = min(1.0, out_len / max(self.config.min_output_length * 5, 1))

        if self.config.penalize_short:
            if inst_len < self.config.min_instruction_length:
                inst_score *= 0.5
            if out_len < self.config.min_output_length:
                out_score *= 0.5

        overall = (inst_score + out_score) / 2.0

        return QualityScore(
            overall=overall,
            instruction_clarity=inst_score,
            output_quality=out_score,
        )

    def score_batch(
        self,
        samples: list[Any],
        update_samples: bool = False,
    ) -> list[QualityScore]:
        """Score a batch of training samples."""
        scores = []
        for sample in samples:
            qs = self.score(sample)
            if update_samples and hasattr(sample, "quality_score"):
                sample.quality_score = qs.overall
            scores.append(qs)
        return scores
