"""Feedback sampling strategies for continuous learning.

Prioritizes which examples to collect feedback on or use for retraining.
"""

import random
from collections.abc import Iterator
from dataclasses import dataclass
from enum import Enum

from .logger import InferenceLogger, InferenceRecord


class SamplingStrategy(Enum):
    """Available sampling strategies."""
    RANDOM = "random"
    UNCERTAIN = "uncertain"
    NEGATIVE = "negative"
    RECENT = "recent"
    DIVERSE = "diverse"


@dataclass
class SamplerConfig:
    """Configuration for feedback sampler."""
    strategy: SamplingStrategy = SamplingStrategy.UNCERTAIN
    sample_rate: float = 0.1  # 10% of inferences
    max_samples_per_day: int = 100
    uncertainty_threshold: float = 0.5


class FeedbackSampler:
    """Samples inference records for feedback collection."""

    def __init__(
        self,
        logger: InferenceLogger,
        config: SamplerConfig | None = None,
    ):
        self.logger = logger
        self.config = config or SamplerConfig()

    def should_collect_feedback(self, record: InferenceRecord) -> bool:
        """Determine if we should collect feedback for this record."""
        strategy = self.config.strategy

        if strategy == SamplingStrategy.RANDOM:
            return random.random() < self.config.sample_rate

        elif strategy == SamplingStrategy.UNCERTAIN:
            # Check metadata for uncertainty score
            uncertainty = record.metadata.get("uncertainty", 0.5)
            return uncertainty >= self.config.uncertainty_threshold

        elif strategy == SamplingStrategy.NEGATIVE:
            # Sample more from potentially problematic responses
            # Check for error indicators
            error_keywords = ["error", "sorry", "cannot", "failed"]
            has_errors = any(kw in record.response.lower() for kw in error_keywords)
            if has_errors:
                return random.random() < self.config.sample_rate * 3
            return random.random() < self.config.sample_rate

        elif strategy == SamplingStrategy.RECENT:
            return random.random() < self.config.sample_rate

        return random.random() < self.config.sample_rate

    def sample_for_feedback(
        self,
        count: int = 10,
        exclude_with_feedback: bool = True,
    ) -> Iterator[InferenceRecord]:
        """Sample records that need feedback."""
        candidates = []

        for record in self.logger.get_records():
            if exclude_with_feedback and record.has_feedback:
                continue
            if self.should_collect_feedback(record):
                candidates.append(record)

        # Shuffle and limit
        random.shuffle(candidates)
        for record in candidates[:count]:
            yield record

    def sample_for_training(
        self,
        count: int = 1000,
        min_score: float = 0.5,
    ) -> Iterator[InferenceRecord]:
        """Sample high-quality records for training."""
        candidates = []

        for record in self.logger.get_records(with_feedback_only=True):
            if record.feedback_score and record.feedback_score >= min_score:
                candidates.append(record)

        # Sort by feedback score descending
        candidates.sort(key=lambda r: r.feedback_score or 0, reverse=True)

        for record in candidates[:count]:
            yield record

    def get_distribution_stats(self) -> dict:
        """Get distribution of records by various dimensions."""
        by_expert = {}
        by_model = {}
        by_feedback = {"positive": 0, "neutral": 0, "negative": 0, "none": 0}

        for record in self.logger.get_records():
            # By expert
            expert = record.expert or "unknown"
            by_expert[expert] = by_expert.get(expert, 0) + 1

            # By model
            by_model[record.model] = by_model.get(record.model, 0) + 1

            # By feedback
            if record.feedback_score is None:
                by_feedback["none"] += 1
            elif record.feedback_score > 0:
                by_feedback["positive"] += 1
            elif record.feedback_score < 0:
                by_feedback["negative"] += 1
            else:
                by_feedback["neutral"] += 1

        return {
            "by_expert": by_expert,
            "by_model": by_model,
            "by_feedback": by_feedback,
        }
