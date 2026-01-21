"""Unified quality scoring for training samples.

Combines multiple quality signals:
- ELECTRA discriminator score (real vs fake assembly)
- Asar syntax validation (compile-time correctness)
- Entity coverage (known ALTTP addresses)
- Length/structure heuristics

The unified score populates the quality_score field in TrainingSample.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from afs.discriminator.electra import ASMElectra
    from afs.generators.asar_validator import AsarValidator
    from afs.generators.base import TrainingSample
    from afs.knowledge.entity_extractor import EntityExtractor


@dataclass
class ScoringWeights:
    """Weights for combining quality signals."""

    electra: float = 0.4  # Discriminator score weight
    asar: float = 0.3  # Syntax validation weight
    entity: float = 0.2  # Entity coverage weight
    length: float = 0.1  # Length/structure weight

    def normalize(self) -> ScoringWeights:
        """Normalize weights to sum to 1.0."""
        total = self.electra + self.asar + self.entity + self.length
        if total == 0:
            return ScoringWeights(0.25, 0.25, 0.25, 0.25)
        return ScoringWeights(
            electra=self.electra / total,
            asar=self.asar / total,
            entity=self.entity / total,
            length=self.length / total,
        )

    def to_dict(self) -> dict[str, float]:
        return {
            "electra": self.electra,
            "asar": self.asar,
            "entity": self.entity,
            "length": self.length,
        }

    @classmethod
    def from_dict(cls, data: dict[str, float]) -> ScoringWeights:
        return cls(
            electra=data.get("electra", 0.4),
            asar=data.get("asar", 0.3),
            entity=data.get("entity", 0.2),
            length=data.get("length", 0.1),
        )


@dataclass
class QualityScore:
    """Unified quality score for a training sample."""

    overall: float  # Combined 0.0-1.0 score
    electra_score: float  # ELECTRA probability (0.0-1.0)
    asar_valid: bool  # Syntax validation passed
    entity_coverage: float  # Ratio of known entities (0.0-1.0)
    length_score: float  # Length-based score (0.0-1.0)

    # Component breakdown for debugging
    components: dict[str, float] = field(default_factory=dict)

    # Additional metadata
    asar_errors: list[str] = field(default_factory=list)
    entity_count: int = 0
    known_entity_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "overall": self.overall,
            "electra_score": self.electra_score,
            "asar_valid": self.asar_valid,
            "entity_coverage": self.entity_coverage,
            "length_score": self.length_score,
            "components": self.components,
            "asar_errors": self.asar_errors,
            "entity_count": self.entity_count,
            "known_entity_count": self.known_entity_count,
        }


@dataclass
class ScoringConfig:
    """Configuration for quality scoring."""

    # Weights
    weights: ScoringWeights = field(default_factory=ScoringWeights)

    # Length scoring parameters
    min_output_length: int = 50  # Minimum expected output length
    ideal_output_length: int = 500  # Ideal output length
    max_output_length: int = 5000  # Maximum before penalty

    # ELECTRA settings
    electra_model_path: Path | None = None

    # Asar settings
    asar_timeout: int = 10
    skip_asar_if_no_model: bool = True

    # Entity extraction
    include_hardware_entities: bool = True


class QualityScorer:
    """Compute unified quality scores for training samples."""

    def __init__(
        self,
        config: ScoringConfig | None = None,
        electra: ASMElectra | None = None,
        asar_validator: AsarValidator | None = None,
        entity_extractor: EntityExtractor | None = None,
    ):
        """Initialize scorer.

        Args:
            config: Scoring configuration
            electra: Pre-loaded ELECTRA model (loads from config if None)
            asar_validator: Pre-loaded validator (creates new if None)
            entity_extractor: Pre-loaded extractor (creates new if None)
        """
        self.config = config or ScoringConfig()
        self._electra = electra
        self._asar = asar_validator
        self._entity_extractor = entity_extractor
        self._electra_loaded = False

    @property
    def electra(self) -> ASMElectra | None:
        """Lazy load ELECTRA model."""
        if self._electra is None and not self._electra_loaded:
            self._electra_loaded = True
            if self.config.electra_model_path:
                from afs.discriminator.electra import ASMElectra

                self._electra = ASMElectra(model_path=self.config.electra_model_path)
        return self._electra

    @property
    def asar(self) -> AsarValidator:
        """Lazy load asar validator."""
        if self._asar is None:
            from afs.generators.asar_validator import AsarValidator, AsarValidatorConfig

            self._asar = AsarValidator(
                AsarValidatorConfig(timeout=self.config.asar_timeout)
            )
        return self._asar

    @property
    def entity_extractor(self) -> EntityExtractor:
        """Lazy load entity extractor."""
        if self._entity_extractor is None:
            from afs.knowledge.entity_extractor import EntityExtractor

            self._entity_extractor = EntityExtractor(
                include_hardware=self.config.include_hardware_entities
            )
        return self._entity_extractor

    def score(self, sample: TrainingSample) -> QualityScore:
        """Compute unified quality score for a sample.

        Args:
            sample: Training sample to score

        Returns:
            QualityScore with overall and component scores
        """
        weights = self.config.weights.normalize()
        components: dict[str, float] = {}

        # 1. ELECTRA score
        electra_score = 0.5  # Default neutral score
        if self.electra is not None:
            try:
                electra_score = self.electra.score(sample.output)
            except Exception as e:
                logger.debug("ELECTRA scoring failed, using default 0.5: %s", e)
                electra_score = 0.5
        components["electra"] = electra_score

        # 2. Asar validation
        asar_valid = False
        asar_errors: list[str] = []
        asar_score = 0.0

        if not self.config.skip_asar_if_no_model or self.electra is not None:
            try:
                result = self.asar.validate_sample(sample)
                asar_valid = result.passed
                asar_score = 1.0 if result.passed else 0.0
                if result.error_message:
                    asar_errors.append(result.error_message)
            except Exception as e:
                asar_errors.append(str(e))
        else:
            # Default to passing if we're skipping
            asar_valid = True
            asar_score = 0.5
        components["asar"] = asar_score

        # 3. Entity coverage
        extraction = self.entity_extractor.extract(sample.output)
        entity_count = len(extraction.entities)
        known_count = extraction.known_count
        entity_coverage = extraction.coverage
        components["entity"] = entity_coverage

        # 4. Length score
        length_score = self._compute_length_score(sample.output)
        components["length"] = length_score

        # Compute weighted overall score
        overall = (
            weights.electra * electra_score
            + weights.asar * asar_score
            + weights.entity * entity_coverage
            + weights.length * length_score
        )

        return QualityScore(
            overall=overall,
            electra_score=electra_score,
            asar_valid=asar_valid,
            entity_coverage=entity_coverage,
            length_score=length_score,
            components=components,
            asar_errors=asar_errors,
            entity_count=entity_count,
            known_entity_count=known_count,
        )

    def _compute_length_score(self, text: str) -> float:
        """Compute length-based quality score."""
        length = len(text)

        if length < self.config.min_output_length:
            # Too short - linear penalty
            return length / self.config.min_output_length * 0.5

        if length <= self.config.ideal_output_length:
            # In ideal range
            return 0.5 + 0.5 * (length - self.config.min_output_length) / (
                self.config.ideal_output_length - self.config.min_output_length
            )

        if length <= self.config.max_output_length:
            # Between ideal and max - slight penalty
            excess = length - self.config.ideal_output_length
            max_excess = self.config.max_output_length - self.config.ideal_output_length
            return 1.0 - 0.3 * (excess / max_excess)

        # Too long
        return 0.5

    def score_batch(
        self,
        samples: list[TrainingSample],
        update_samples: bool = True,
    ) -> list[QualityScore]:
        """Score multiple samples efficiently.

        Args:
            samples: List of samples to score
            update_samples: Whether to update sample.quality_score field

        Returns:
            List of QualityScore objects
        """
        scores: list[QualityScore] = []

        # Batch ELECTRA scoring if available
        electra_scores: list[float] = []
        if self.electra is not None:
            try:
                texts = [s.output for s in samples]
                electra_scores = self.electra.score_batch(texts)
            except Exception as e:
                logger.warning(
                    "Batch ELECTRA scoring failed for %d samples, using default 0.5: %s",
                    len(samples),
                    e,
                )
                electra_scores = [0.5] * len(samples)
        else:
            electra_scores = [0.5] * len(samples)

        # Score each sample
        for i, sample in enumerate(samples):
            # Use pre-computed ELECTRA score
            weights = self.config.weights.normalize()
            components: dict[str, float] = {}
            components["electra"] = electra_scores[i]

            # Asar validation
            asar_valid = False
            asar_errors: list[str] = []
            asar_score = 0.0

            if not self.config.skip_asar_if_no_model or self.electra is not None:
                try:
                    result = self.asar.validate_sample(sample)
                    asar_valid = result.passed
                    asar_score = 1.0 if result.passed else 0.0
                    if result.error_message:
                        asar_errors.append(result.error_message)
                except Exception as e:
                    asar_errors.append(str(e))
            else:
                asar_valid = True
                asar_score = 0.5
            components["asar"] = asar_score

            # Entity coverage
            extraction = self.entity_extractor.extract(sample.output)
            entity_coverage = extraction.coverage
            components["entity"] = entity_coverage

            # Length score
            length_score = self._compute_length_score(sample.output)
            components["length"] = length_score

            # Compute overall
            overall = (
                weights.electra * electra_scores[i]
                + weights.asar * asar_score
                + weights.entity * entity_coverage
                + weights.length * length_score
            )

            score = QualityScore(
                overall=overall,
                electra_score=electra_scores[i],
                asar_valid=asar_valid,
                entity_coverage=entity_coverage,
                length_score=length_score,
                components=components,
                asar_errors=asar_errors,
                entity_count=len(extraction.entities),
                known_entity_count=extraction.known_count,
            )
            scores.append(score)

            # Update sample if requested
            if update_samples:
                sample.quality_score = overall
                sample._metadata["quality_components"] = components

        return scores


def build_scoring_config(
    profile: str,
    *,
    electra_model_path: Path | None = None,
    enable_asar: bool = False,
) -> ScoringConfig:
    """Build a scoring config for common profiles."""
    normalized = (profile or "generic").strip().lower()

    if normalized in {"asm", "assembly"}:
        weights = ScoringWeights()
        if not enable_asar:
            weights.asar = 0.0
        config = ScoringConfig(
            electra_model_path=electra_model_path,
            weights=weights,
            include_hardware_entities=True,
        )
        config.skip_asar_if_no_model = not enable_asar
        return config

    if normalized in {"dialogue", "chat", "assistant"}:
        weights = ScoringWeights(
            electra=0.0,
            asar=0.0,
            entity=0.0,
            length=1.0,
        )
        config = ScoringConfig(
            electra_model_path=None,
            weights=weights,
            include_hardware_entities=False,
        )
        config.min_output_length = 20
        config.ideal_output_length = 200
        config.max_output_length = 2000
        config.skip_asar_if_no_model = True
        return config

    weights = ScoringWeights(
        electra=0.0,
        asar=0.0,
        entity=0.0,
        length=1.0,
    )
    config = ScoringConfig(
        electra_model_path=None,
        weights=weights,
        include_hardware_entities=False,
    )
    config.skip_asar_if_no_model = True
    return config


def score_samples(
    samples: list[TrainingSample],
    electra_path: Path | None = None,
    weights: ScoringWeights | None = None,
) -> list[QualityScore]:
    """Convenience function to score samples.

    Args:
        samples: Samples to score
        electra_path: Path to ELECTRA model (optional)
        weights: Scoring weights (optional)

    Returns:
        List of QualityScore objects
    """
    config = ScoringConfig(
        electra_model_path=electra_path,
        weights=weights or ScoringWeights(),
    )
    scorer = QualityScorer(config=config)
    return scorer.score_batch(samples, update_samples=True)


def score_jsonl(
    input_path: Path,
    output_path: Path,
    electra_path: Path | None = None,
    weights: ScoringWeights | None = None,
    min_score: float | None = None,
) -> dict[str, Any]:
    """Score samples in JSONL file and write results.

    Args:
        input_path: Input JSONL file
        output_path: Output JSONL file with scores
        electra_path: Path to ELECTRA model
        weights: Scoring weights
        min_score: If set, only include samples above this score

    Returns:
        Statistics dict
    """
    from afs.generators.base import TrainingSample

    # Load samples
    samples = []
    with open(input_path) as f:
        for line in f:
            if line.strip():
                samples.append(TrainingSample.from_dict(json.loads(line)))

    # Score
    config = ScoringConfig(
        electra_model_path=electra_path,
        weights=weights or ScoringWeights(),
    )
    scorer = QualityScorer(config=config)
    scores = scorer.score_batch(samples, update_samples=True)

    # Filter if threshold set
    output_samples = []
    filtered_count = 0
    for sample, score in zip(samples, scores, strict=False):
        if min_score is None or score.overall >= min_score:
            output_samples.append(sample)
        else:
            filtered_count += 1

    # Write output
    with open(output_path, "w") as f:
        for sample in output_samples:
            f.write(json.dumps(sample.to_dict()) + "\n")

    # Compute stats
    all_scores = [s.overall for s in scores]
    return {
        "input_count": len(samples),
        "output_count": len(output_samples),
        "filtered_count": filtered_count,
        "mean_score": sum(all_scores) / len(all_scores) if all_scores else 0,
        "min_score": min(all_scores) if all_scores else 0,
        "max_score": max(all_scores) if all_scores else 0,
        "electra_available": scorer.electra is not None,
    }


def analyze_scores(scores: list[QualityScore]) -> dict[str, Any]:
    """Analyze score distribution.

    Args:
        scores: List of QualityScore objects

    Returns:
        Analysis dict with distribution info
    """
    if not scores:
        return {"count": 0}

    overall = [s.overall for s in scores]
    electra = [s.electra_score for s in scores]
    entity = [s.entity_coverage for s in scores]

    # Histogram buckets
    def histogram(values: list[float], buckets: int = 10) -> dict[str, int]:
        hist: dict[str, int] = {}
        for i in range(buckets):
            low = i / buckets
            high = (i + 1) / buckets
            key = f"{low:.1f}-{high:.1f}"
            hist[key] = sum(1 for v in values if low <= v < high)
        return hist

    return {
        "count": len(scores),
        "overall": {
            "mean": sum(overall) / len(overall),
            "min": min(overall),
            "max": max(overall),
            "histogram": histogram(overall),
        },
        "electra": {
            "mean": sum(electra) / len(electra),
            "min": min(electra),
            "max": max(electra),
        },
        "entity_coverage": {
            "mean": sum(entity) / len(entity),
            "min": min(entity),
            "max": max(entity),
        },
        "asar_pass_rate": sum(1 for s in scores if s.asar_valid) / len(scores),
        "entity_stats": {
            "total_entities": sum(s.entity_count for s in scores),
            "known_entities": sum(s.known_entity_count for s in scores),
        },
    }
