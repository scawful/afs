"""Active learning samplers for training data selection.

Provides:
- UncertaintySampler: Select samples where model is most uncertain
- CurriculumManager: Progressive difficulty training
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from afs.generators.base import TrainingSample
    from afs.training.scoring import QualityScorer


class CurriculumStage(str, Enum):
    """Curriculum learning stages."""

    SIMPLE = "simple"  # Short outputs, common patterns
    MODERATE = "moderate"  # Medium complexity
    COMPLEX = "complex"  # Long outputs, multiple entities
    ADVANCED = "advanced"  # Edge cases, rare patterns


@dataclass
class UncertaintySamplerConfig:
    """Configuration for uncertainty sampling."""

    # Target ELECTRA score for maximum uncertainty
    target_score: float = 0.5

    # Tolerance around target
    tolerance: float = 0.1

    # Whether to prefer samples never seen before
    prefer_unseen: bool = True

    # Minimum quality score to consider
    min_quality: float = 0.3


class UncertaintySampler:
    """Select samples where the discriminator is most uncertain.

    Samples with ELECTRA scores near 0.5 are most informative for
    improving the discriminator, as the model is unsure whether
    they are real or generated.
    """

    def __init__(self, config: UncertaintySamplerConfig | None = None):
        self.config = config or UncertaintySamplerConfig()

    def sample(
        self,
        candidates: list[TrainingSample],
        n: int,
        scorer: QualityScorer | None = None,
    ) -> list[TrainingSample]:
        """Select n most uncertain samples.

        Args:
            candidates: Pool of candidate samples
            n: Number of samples to select
            scorer: Quality scorer (uses cached scores if None)

        Returns:
            List of n most uncertain samples
        """
        if not candidates:
            return []

        # Score if needed
        if scorer:
            scorer.score_batch(candidates, update_samples=True)

        # Compute uncertainty for each sample
        scored = []
        for sample in candidates:
            # Get ELECTRA score from metadata
            electra = sample._metadata.get("quality_components", {}).get("electra", 0.5)

            # Uncertainty = 1 - distance from target
            distance = abs(electra - self.config.target_score)
            uncertainty = 1.0 - distance

            # Apply quality filter
            if sample.quality_score < self.config.min_quality:
                uncertainty *= 0.5  # Penalize low quality

            # Boost unseen samples
            if self.config.prefer_unseen:
                if not sample._metadata.get("seen_count", 0):
                    uncertainty *= 1.2

            scored.append((sample, uncertainty))

        # Sort by uncertainty (highest first)
        scored.sort(key=lambda x: -x[1])

        return [s for s, _ in scored[:n]]

    def get_uncertainty_distribution(
        self,
        samples: list[TrainingSample],
    ) -> dict[str, int]:
        """Get distribution of samples by uncertainty level.

        Returns:
            Dict mapping uncertainty level to count
        """
        distribution = {
            "very_high": 0,  # 0.4-0.6 ELECTRA
            "high": 0,  # 0.3-0.4 or 0.6-0.7
            "medium": 0,  # 0.2-0.3 or 0.7-0.8
            "low": 0,  # <0.2 or >0.8
        }

        for sample in samples:
            electra = sample._metadata.get("quality_components", {}).get("electra", 0.5)
            distance = abs(electra - 0.5)

            if distance <= 0.1:
                distribution["very_high"] += 1
            elif distance <= 0.2:
                distribution["high"] += 1
            elif distance <= 0.3:
                distribution["medium"] += 1
            else:
                distribution["low"] += 1

        return distribution


@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning."""

    # Length thresholds (character count)
    simple_max_length: int = 200
    moderate_max_length: int = 500
    complex_max_length: int = 1000
    # Above complex_max_length = advanced

    # Entity count thresholds
    simple_max_entities: int = 2
    moderate_max_entities: int = 5
    complex_max_entities: int = 10

    # Quality thresholds
    min_quality_for_curriculum: float = 0.4


class CurriculumManager:
    """Manage curriculum learning progression.

    Curriculum learning trains on easy examples first, then
    progressively harder ones. This helps models learn
    fundamental patterns before tackling complex cases.
    """

    def __init__(self, config: CurriculumConfig | None = None):
        self.config = config or CurriculumConfig()
        self._current_stage = CurriculumStage.SIMPLE

    @property
    def current_stage(self) -> CurriculumStage:
        return self._current_stage

    def classify_sample(self, sample: TrainingSample) -> CurriculumStage:
        """Classify a sample into a curriculum stage.

        Args:
            sample: Sample to classify

        Returns:
            CurriculumStage for this sample
        """
        length = len(sample.output)
        entity_count = len(sample.kg_entities)

        # Classify by length first
        if length <= self.config.simple_max_length:
            length_stage = CurriculumStage.SIMPLE
        elif length <= self.config.moderate_max_length:
            length_stage = CurriculumStage.MODERATE
        elif length <= self.config.complex_max_length:
            length_stage = CurriculumStage.COMPLEX
        else:
            length_stage = CurriculumStage.ADVANCED

        # Classify by entity count
        if entity_count <= self.config.simple_max_entities:
            entity_stage = CurriculumStage.SIMPLE
        elif entity_count <= self.config.moderate_max_entities:
            entity_stage = CurriculumStage.MODERATE
        elif entity_count <= self.config.complex_max_entities:
            entity_stage = CurriculumStage.COMPLEX
        else:
            entity_stage = CurriculumStage.ADVANCED

        # Return the harder of the two
        stages = [CurriculumStage.SIMPLE, CurriculumStage.MODERATE,
                  CurriculumStage.COMPLEX, CurriculumStage.ADVANCED]

        return stages[max(stages.index(length_stage), stages.index(entity_stage))]

    def get_samples_for_stage(
        self,
        samples: list[TrainingSample],
        stage: CurriculumStage | None = None,
    ) -> list[TrainingSample]:
        """Get samples for a curriculum stage.

        Args:
            samples: All available samples
            stage: Target stage (uses current if None)

        Returns:
            Samples matching the stage
        """
        stage = stage or self._current_stage

        result = []
        for sample in samples:
            if sample.quality_score < self.config.min_quality_for_curriculum:
                continue

            if self.classify_sample(sample) == stage:
                result.append(sample)

        return result

    def get_stage_distribution(
        self,
        samples: list[TrainingSample],
    ) -> dict[str, int]:
        """Get distribution of samples across stages."""
        distribution = {stage.value: 0 for stage in CurriculumStage}

        for sample in samples:
            stage = self.classify_sample(sample)
            distribution[stage.value] += 1

        return distribution

    def advance_stage(self) -> bool:
        """Advance to next curriculum stage.

        Returns:
            True if advanced, False if already at final stage
        """
        stages = list(CurriculumStage)
        current_idx = stages.index(self._current_stage)

        if current_idx < len(stages) - 1:
            self._current_stage = stages[current_idx + 1]
            return True

        return False

    def reset(self) -> None:
        """Reset to first stage."""
        self._current_stage = CurriculumStage.SIMPLE

    def get_curriculum_plan(
        self,
        samples: list[TrainingSample],
    ) -> dict[str, Any]:
        """Generate a curriculum plan showing samples per stage.

        Args:
            samples: Available training samples

        Returns:
            Dict with stage info and sample counts
        """
        distribution = self.get_stage_distribution(samples)
        total = sum(distribution.values())

        plan = {
            "total_samples": total,
            "stages": [],
        }

        cumulative = 0
        for stage in CurriculumStage:
            count = distribution[stage.value]
            cumulative += count
            plan["stages"].append({
                "stage": stage.value,
                "sample_count": count,
                "percentage": 100 * count / total if total else 0,
                "cumulative_count": cumulative,
                "cumulative_percentage": 100 * cumulative / total if total else 0,
            })

        return plan


def sample_by_uncertainty(
    samples: list[TrainingSample],
    n: int,
    scorer: QualityScorer | None = None,
) -> list[TrainingSample]:
    """Convenience function for uncertainty sampling.

    Args:
        samples: Candidate samples
        n: Number to select
        scorer: Optional scorer

    Returns:
        Most uncertain samples
    """
    sampler = UncertaintySampler()
    return sampler.sample(samples, n, scorer)


    """
    return [s for s in samples if s.metadata.get("stage") == stage]

