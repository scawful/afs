"""Filter training samples using ASM-ELECTRA discriminator."""

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

from .data import extract_code_blocks
from .electra import ASMElectra


@dataclass
class FilterConfig:
    """Configuration for sample filtering."""

    # Thresholds
    min_score: float = 0.7  # Minimum score to accept
    max_score: float = 1.0  # Maximum score (for filtering too-simple samples)

    # Batch processing
    batch_size: int = 32

    # Fields to score
    score_field: str = "output"  # Which field contains code to score

    # Output
    include_scores: bool = True  # Add score to output samples


@dataclass
class FilterResult:
    """Result of filtering operation."""

    accepted: int
    rejected: int
    total: int
    mean_score: float
    score_distribution: dict  # Histogram of scores

    def __str__(self) -> str:
        return (
            f"Filtered {self.total} samples:\n"
            f"  Accepted: {self.accepted} ({100*self.accepted/self.total:.1f}%)\n"
            f"  Rejected: {self.rejected} ({100*self.rejected/self.total:.1f}%)\n"
            f"  Mean score: {self.mean_score:.3f}"
        )


class SampleFilter:
    """Filter training samples using ASM-ELECTRA."""

    def __init__(
        self,
        electra: ASMElectra | None = None,
        model_path: Path | str | None = None,
        config: FilterConfig | None = None,
    ):
        """Initialize filter.

        Args:
            electra: Pre-loaded ASMElectra instance
            model_path: Path to load ASMElectra from
            config: Filter configuration
        """
        if electra is None and model_path is None:
            raise ValueError("Must provide either electra or model_path")

        self.electra = electra or ASMElectra(model_path=model_path)
        self.config = config or FilterConfig()

    def score_sample(self, sample: dict) -> float:
        """Score a single training sample."""
        text = sample.get(self.config.score_field, "")

        # Extract code blocks if present
        blocks = extract_code_blocks(text)
        if blocks:
            # Score all blocks, return minimum (most suspicious)
            scores = [self.electra.score(block) for block in blocks]
            return min(scores)

        # Score raw text
        return self.electra.score(text)

    def filter_sample(self, sample: dict) -> tuple[bool, float]:
        """Check if a sample should be accepted.

        Returns:
            (accepted, score)
        """
        score = self.score_sample(sample)
        accepted = self.config.min_score <= score <= self.config.max_score
        return accepted, score

    def filter_jsonl(
        self,
        input_path: Path,
        output_path: Path,
        rejected_path: Path | None = None,
    ) -> FilterResult:
        """Filter a JSONL file.

        Args:
            input_path: Input JSONL file
            output_path: Output JSONL for accepted samples
            rejected_path: Optional output for rejected samples

        Returns:
            FilterResult with statistics
        """
        accepted_count = 0
        rejected_count = 0
        total_score = 0.0
        score_hist = {
            "0.0-0.2": 0,
            "0.2-0.4": 0,
            "0.4-0.6": 0,
            "0.6-0.8": 0,
            "0.8-1.0": 0,
        }

        accepted_file = open(output_path, "w")
        rejected_file = open(rejected_path, "w") if rejected_path else None

        try:
            with open(input_path) as f:
                # Collect samples for batch processing
                samples = []
                for line in f:
                    sample = json.loads(line)
                    samples.append(sample)

            # Process in batches
            texts = [s.get(self.config.score_field, "") for s in samples]

            # Extract code and score
            for i, (sample, text) in enumerate(zip(samples, texts, strict=False)):
                blocks = extract_code_blocks(text)
                if blocks:
                    scores = self.electra.score_batch(blocks)
                    score = min(scores)
                else:
                    score = self.electra.score(text) if text else 0.0

                total_score += score

                # Update histogram
                if score < 0.2:
                    score_hist["0.0-0.2"] += 1
                elif score < 0.4:
                    score_hist["0.2-0.4"] += 1
                elif score < 0.6:
                    score_hist["0.4-0.6"] += 1
                elif score < 0.8:
                    score_hist["0.6-0.8"] += 1
                else:
                    score_hist["0.8-1.0"] += 1

                # Check acceptance
                accepted = self.config.min_score <= score <= self.config.max_score

                if self.config.include_scores:
                    sample["_electra_score"] = score

                if accepted:
                    accepted_count += 1
                    accepted_file.write(json.dumps(sample) + "\n")
                else:
                    rejected_count += 1
                    if rejected_file:
                        rejected_file.write(json.dumps(sample) + "\n")

                # Progress
                if (i + 1) % 100 == 0:
                    print(f"  Processed {i + 1}/{len(samples)} samples...")

        finally:
            accepted_file.close()
            if rejected_file:
                rejected_file.close()

        total = accepted_count + rejected_count
        return FilterResult(
            accepted=accepted_count,
            rejected=rejected_count,
            total=total,
            mean_score=total_score / total if total > 0 else 0.0,
            score_distribution=score_hist,
        )

    def filter_stream(
        self,
        samples: Iterator[dict],
    ) -> Iterator[tuple[dict, float, bool]]:
        """Filter samples from an iterator.

        Yields:
            (sample, score, accepted) tuples
        """
        for sample in samples:
            accepted, score = self.filter_sample(sample)
            if self.config.include_scores:
                sample["_electra_score"] = score
            yield sample, score, accepted


def filter_training_data(
    model_path: Path | str,
    input_path: Path,
    output_path: Path,
    min_score: float = 0.7,
    rejected_path: Path | None = None,
) -> FilterResult:
    """Convenience function to filter training data.

    Args:
        model_path: Path to trained ASM-ELECTRA model
        input_path: Input training JSONL
        output_path: Output filtered JSONL
        min_score: Minimum score to accept (0-1)
        rejected_path: Optional path for rejected samples

    Returns:
        FilterResult with statistics
    """
    config = FilterConfig(min_score=min_score)
    filter = SampleFilter(model_path=model_path, config=config)

    print(f"Filtering {input_path} with min_score={min_score}")
    result = filter.filter_jsonl(input_path, output_path, rejected_path)
    print(result)

    return result
