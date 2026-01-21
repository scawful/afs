"""Training data generator from production usage.

Converts good interactions into training samples with quality filtering,
deduplication, and format conversion.
"""

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Iterator

from afs.generators.base import TrainingSample
from afs.training.scoring import QualityScorer, ScoringConfig, QualityScore

from .logger import UsageLogger, UsageRecord

logger = logging.getLogger(__name__)


@dataclass
class DataGeneratorConfig:
    """Configuration for training data generation."""

    min_quality_score: float = 0.7
    min_user_feedback: int = 1  # Only positive feedback
    deduplicate: bool = True
    include_existing_data: bool = True
    existing_data_path: Optional[Path] = None
    max_samples: Optional[int] = None
    format_type: str = "chatml"  # chatml, alpaca, completion


@dataclass
class GenerationResult:
    """Result of training data generation."""

    total_candidates: int = 0
    filtered_by_quality: int = 0
    filtered_by_feedback: int = 0
    duplicates_removed: int = 0
    final_count: int = 0
    output_path: Optional[Path] = None
    quality_stats: dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class TrainingDataGenerator:
    """Generates training data from production usage."""

    def __init__(
        self,
        usage_logger: UsageLogger,
        config: Optional[DataGeneratorConfig] = None,
    ):
        self.logger = usage_logger
        self.config = config or DataGeneratorConfig()
        self.quality_scorer = None

        if self.config.min_quality_score > 0:
            self.quality_scorer = QualityScorer(config=ScoringConfig())

    def generate(
        self,
        output_path: Path,
        since: Optional[datetime] = None,
    ) -> GenerationResult:
        """Generate training data from usage logs.

        Args:
            output_path: Where to write training data
            since: Only include records after this time

        Returns:
            GenerationResult with statistics
        """
        result = GenerationResult(output_path=output_path)

        logger.info("Starting training data generation")

        # Stage 1: Collect candidates
        logger.info("Stage 1: Collecting candidates from usage logs")
        candidates = list(self._collect_candidates(since))
        result.total_candidates = len(candidates)
        logger.info(f"  Found {result.total_candidates} candidate records")

        if not candidates:
            logger.warning("No candidates found")
            return result

        # Stage 2: Filter by user feedback
        if self.config.min_user_feedback is not None:
            logger.info("Stage 2: Filtering by user feedback")
            before = len(candidates)
            candidates = [
                c
                for c in candidates
                if c.user_feedback
                and c.user_feedback >= self.config.min_user_feedback
            ]
            result.filtered_by_feedback = before - len(candidates)
            logger.info(
                f"  Filtered {result.filtered_by_feedback} records (remaining: {len(candidates)})"
            )

        # Stage 3: Score quality if not already scored
        logger.info("Stage 3: Scoring quality")
        candidates = self._score_quality(candidates)

        # Stage 4: Filter by quality
        if self.config.min_quality_score > 0:
            logger.info("Stage 4: Filtering by quality score")
            before = len(candidates)
            candidates = [
                c for c in candidates if c.quality_score >= self.config.min_quality_score
            ]
            result.filtered_by_quality = before - len(candidates)
            logger.info(
                f"  Filtered {result.filtered_by_quality} records (remaining: {len(candidates)})"
            )

        # Stage 5: Deduplicate
        if self.config.deduplicate:
            logger.info("Stage 5: Deduplicating")
            before = len(candidates)
            candidates = self._deduplicate(candidates)
            result.duplicates_removed = before - len(candidates)
            logger.info(
                f"  Removed {result.duplicates_removed} duplicates (remaining: {len(candidates)})"
            )

        # Stage 6: Merge with existing data
        if self.config.include_existing_data and self.config.existing_data_path:
            logger.info("Stage 6: Merging with existing data")
            candidates = self._merge_with_existing(candidates)
            logger.info(f"  Total after merge: {len(candidates)}")

        # Stage 7: Limit samples
        if self.config.max_samples and len(candidates) > self.config.max_samples:
            logger.info(f"Stage 7: Limiting to {self.config.max_samples} samples")
            # Keep highest quality
            candidates.sort(key=lambda c: c.quality_score, reverse=True)
            candidates = candidates[: self.config.max_samples]

        result.final_count = len(candidates)

        # Compute quality stats
        if candidates:
            scores = [c.quality_score for c in candidates]
            result.quality_stats = {
                "mean": sum(scores) / len(scores),
                "min": min(scores),
                "max": max(scores),
            }

        # Stage 8: Export
        logger.info("Stage 8: Exporting training data")
        self._export(candidates, output_path)
        logger.info(f"  Wrote {result.final_count} samples to {output_path}")

        logger.info("Training data generation complete")
        return result

    def _collect_candidates(
        self, since: Optional[datetime]
    ) -> Iterator[UsageRecord]:
        """Collect candidate records from usage logs."""
        for record in self.logger.get_records(since=since):
            yield record

    def _score_quality(self, records: list[UsageRecord]) -> list[UsageRecord]:
        """Score quality of records if not already scored."""
        if not self.quality_scorer:
            return records

        # Convert to TrainingSample for scoring
        samples = []
        for record in records:
            if record.quality_score > 0:
                # Already scored
                continue

            sample = TrainingSample(
                instruction=record.query,
                input="",
                output=record.response,
            )
            samples.append((record, sample))

        if samples:
            # Score in batch
            logger.info(f"  Scoring {len(samples)} records")
            training_samples = [s for _, s in samples]
            scores = self.quality_scorer.score_batch(training_samples, update_samples=False)

            # Update records
            for (record, _), score in zip(samples, scores):
                record.quality_score = score.overall
                self.logger.db.update_quality_score(record.id, score.overall)

        return records

    def _deduplicate(self, records: list[UsageRecord]) -> list[UsageRecord]:
        """Remove duplicate records based on content."""
        seen = set()
        unique = []

        for record in records:
            dedupe_hash = record.compute_dedupe_hash()
            if dedupe_hash not in seen:
                seen.add(dedupe_hash)
                unique.append(record)

        return unique

    def _merge_with_existing(
        self, new_records: list[UsageRecord]
    ) -> list[UsageRecord]:
        """Merge with existing training data."""
        if not self.config.existing_data_path or not self.config.existing_data_path.exists():
            return new_records

        # Load existing data
        existing_hashes = set()
        with open(self.config.existing_data_path) as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    data = json.loads(line)
                    # Compute hash from instruction/output
                    if "messages" in data:
                        # ChatML format
                        instruction = data["messages"][0]["content"]
                        output = data["messages"][1]["content"]
                    else:
                        # Alpaca format
                        instruction = data.get("instruction", "")
                        output = data.get("output", "")

                    content = f"{instruction}||{output}"
                    existing_hashes.add(
                        hashlib.sha256(content.encode()).hexdigest()
                    )
                except:
                    continue

        # Filter out duplicates with existing data
        unique = []
        for record in new_records:
            if record.compute_dedupe_hash() not in existing_hashes:
                unique.append(record)

        logger.info(
            f"  Filtered {len(new_records) - len(unique)} duplicates with existing data"
        )
        return unique

    def _export(self, records: list[UsageRecord], output_path: Path) -> None:
        """Export records as training data."""
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, "w") as f:
            for record in records:
                sample = self._format_sample(record)
                f.write(json.dumps(sample) + "\n")

    def _format_sample(self, record: UsageRecord) -> dict:
        """Format record as training sample."""
        if self.config.format_type == "chatml":
            return {
                "messages": [
                    {"role": "user", "content": record.query},
                    {"role": "assistant", "content": record.response},
                ],
                "metadata": {
                    "quality_score": record.quality_score,
                    "user_feedback": record.user_feedback,
                    "model": record.model,
                    "expert": record.expert,
                },
            }
        elif self.config.format_type == "alpaca":
            return {
                "instruction": record.query,
                "input": "",
                "output": record.response,
                "quality_score": record.quality_score,
            }
        elif self.config.format_type == "completion":
            return {
                "text": f"{record.query}\n\n{record.response}",
                "quality_score": record.quality_score,
            }
        else:
            raise ValueError(f"Unknown format type: {self.config.format_type}")


import hashlib  # Add missing import
