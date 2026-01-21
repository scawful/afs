"""Dataset splitting with stratification support."""

from __future__ import annotations

import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..generators.base import TrainingSample


@dataclass
class SplitResult:
    """Result of dataset splitting."""

    train: list[TrainingSample]
    val: list[TrainingSample]
    test: list[TrainingSample]
    stats: dict[str, dict[str, int]]  # Split -> domain -> count

    @property
    def total(self) -> int:
        """Total samples across all splits."""
        return len(self.train) + len(self.val) + len(self.test)

    def summary(self) -> str:
        """Generate a summary string."""
        lines = [
            "Dataset Split Summary:",
            f"  Train: {len(self.train)} samples",
            f"  Val:   {len(self.val)} samples",
            f"  Test:  {len(self.test)} samples",
            f"  Total: {self.total} samples",
        ]

        if self.stats:
            lines.append("\nDomain distribution:")
            for split_name, domains in self.stats.items():
                lines.append(f"  {split_name}:")
                for domain, count in sorted(domains.items()):
                    lines.append(f"    {domain}: {count}")

        return "\n".join(lines)


class DatasetSplitter:
    """Split datasets with stratification support.

    Example:
        ```python
        splitter = DatasetSplitter(
            train_ratio=0.8,
            val_ratio=0.1,
            test_ratio=0.1,
            stratify_by="domain",
        )
        result = splitter.split(samples)
        ```
    """

    def __init__(
        self,
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        stratify_by: str | None = "domain",
        shuffle: bool = True,
        seed: int | None = 42,
    ):
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.stratify_by = stratify_by
        self.shuffle = shuffle
        self.seed = seed

        # Validate ratios
        total = train_ratio + val_ratio + test_ratio
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Ratios must sum to 1.0, got {total}")

    def split(self, samples: list[TrainingSample]) -> SplitResult:
        """Split samples into train/val/test sets.

        Args:
            samples: List of training samples

        Returns:
            SplitResult with train, val, test lists
        """
        if self.seed is not None:
            random.seed(self.seed)

        if self.stratify_by:
            return self._stratified_split(samples)
        else:
            return self._random_split(samples)

    def _random_split(self, samples: list[TrainingSample]) -> SplitResult:
        """Simple random split without stratification."""
        if self.shuffle:
            samples = samples.copy()
            random.shuffle(samples)

        n = len(samples)
        train_end = int(n * self.train_ratio)
        val_end = train_end + int(n * self.val_ratio)

        train = samples[:train_end]
        val = samples[train_end:val_end]
        test = samples[val_end:]

        stats = self._compute_stats(train, val, test)
        return SplitResult(train=train, val=val, test=test, stats=stats)

    def _stratified_split(self, samples: list[TrainingSample]) -> SplitResult:
        """Split with stratification by a field (e.g., domain)."""
        # Group by stratification field
        groups: dict[str, list[TrainingSample]] = defaultdict(list)

        for sample in samples:
            key = getattr(sample, self.stratify_by, "unknown")
            groups[key].append(sample)

        train: list[TrainingSample] = []
        val: list[TrainingSample] = []
        test: list[TrainingSample] = []

        # Split each group proportionally
        for group_name, group_samples in groups.items():
            if self.shuffle:
                random.shuffle(group_samples)

            n = len(group_samples)
            train_end = int(n * self.train_ratio)
            val_end = train_end + int(n * self.val_ratio)

            train.extend(group_samples[:train_end])
            val.extend(group_samples[train_end:val_end])
            test.extend(group_samples[val_end:])

        # Final shuffle of combined splits
        if self.shuffle:
            random.shuffle(train)
            random.shuffle(val)
            random.shuffle(test)

        stats = self._compute_stats(train, val, test)
        return SplitResult(train=train, val=val, test=test, stats=stats)

    def _compute_stats(
        self,
        train: list[TrainingSample],
        val: list[TrainingSample],
        test: list[TrainingSample],
    ) -> dict[str, dict[str, int]]:
        """Compute domain distribution stats for each split."""
        stats = {}

        for name, split in [("train", train), ("val", val), ("test", test)]:
            domain_counts: dict[str, int] = defaultdict(int)
            for sample in split:
                domain_counts[sample.domain] += 1
            stats[name] = dict(domain_counts)

        return stats


def split_dataset(
    input_path: Path,
    output_dir: Path,
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    stratify_by: str | None = "domain",
    shuffle: bool = True,
    seed: int | None = 42,
) -> SplitResult:
    """Convenience function to split a JSONL file.

    Args:
        input_path: Path to input JSONL file
        output_dir: Directory for output files
        train_ratio: Fraction for training set
        val_ratio: Fraction for validation set
        test_ratio: Fraction for test set
        stratify_by: Field to stratify by (None for random)
        shuffle: Whether to shuffle samples
        seed: Random seed for reproducibility

    Returns:
        SplitResult with statistics
    """
    from ..generators.base import read_jsonl, write_jsonl

    # Load samples
    samples = read_jsonl(input_path)

    # Split
    splitter = DatasetSplitter(
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        stratify_by=stratify_by,
        shuffle=shuffle,
        seed=seed,
    )
    result = splitter.split(samples)

    # Write outputs
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    write_jsonl(result.train, output_dir / "train.jsonl")
    write_jsonl(result.val, output_dir / "val.jsonl")
    write_jsonl(result.test, output_dir / "test.jsonl")

    return result
