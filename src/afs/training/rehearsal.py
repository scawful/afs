"""Rehearsal buffer system to prevent catastrophic forgetting.

The rehearsal buffer maintains high-quality samples from previous training versions,
allowing new training runs to preserve knowledge from earlier iterations.

This prevents catastrophic forgetting when training new versions of models like Veran,
where each version (v1→v2→v3→v4) was losing knowledge from previous versions.

Usage:
    # Build buffer from previous training data
    buffer = RehearsalBuffer()
    buffer.load_from_jsonl(Path("models/veran_v4_training.jsonl"), version="v4")

    # Select top samples by quality
    buffer.select_top_samples(ratio=0.3)  # Keep top 30%

    # Mix with new training data
    new_samples = load_samples("models/veran_v5_new.jsonl")
    mixed = buffer.merge_with_new_data(new_samples, rehearsal_ratio=0.3)

    # Save buffer for future use
    buffer.save(Path("~/.context/training/rehearsal/veran_v4.jsonl"))
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..generators.base import TrainingSample


@dataclass
class RehearsalBufferConfig:
    """Configuration for rehearsal buffer."""

    # Quality-based selection
    quality_threshold: float = 0.5  # Minimum quality score to include
    top_ratio: float = 0.3  # Keep top 30% of samples

    # Diversity sampling
    enable_diversity: bool = True  # Use diversity sampling
    max_per_domain: int | None = None  # Max samples per domain

    # Version tracking
    track_provenance: bool = True  # Track which version samples came from

    # Storage
    compress: bool = False  # Gzip compression for storage


@dataclass
class RehearsalBuffer:
    """Maintains samples from previous training runs to prevent forgetting.

    Attributes:
        samples: List of training samples in the buffer
        config: Buffer configuration
        version_counts: Count of samples per version
    """

    samples: list[TrainingSample] = field(default_factory=list)
    config: RehearsalBufferConfig = field(default_factory=RehearsalBufferConfig)
    version_counts: dict[str, int] = field(default_factory=dict)

    def load_from_jsonl(
        self,
        path: Path,
        version: str | None = None,
        max_samples: int | None = None
    ) -> int:
        """Load samples from JSONL file.

        Args:
            path: Path to JSONL file
            version: Version tag for provenance tracking
            max_samples: Maximum samples to load (None = all)

        Returns:
            Number of samples loaded
        """
        path = path.expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Training data not found: {path}")

        loaded = 0
        with open(path) as f:
            for line in f:
                if max_samples and loaded >= max_samples:
                    break

                line = line.strip()
                if not line:
                    continue

                data = json.loads(line)
                sample = TrainingSample(**data)

                # Add provenance if tracking
                if self.config.track_provenance and version:
                    if "_metadata" not in data:
                        sample._metadata = {}
                    sample._metadata["rehearsal_version"] = version

                self.samples.append(sample)
                loaded += 1

        # Update version counts
        if version:
            self.version_counts[version] = self.version_counts.get(version, 0) + loaded

        return loaded

    def select_top_samples(
        self,
        ratio: float | None = None,
        threshold: float | None = None
    ) -> int:
        """Select top samples by quality score.

        Args:
            ratio: Keep top X ratio of samples (e.g., 0.3 = top 30%)
            threshold: Minimum quality score (overrides ratio if provided)

        Returns:
            Number of samples retained
        """
        if not self.samples:
            return 0

        # Use config defaults if not provided
        ratio = self.config.top_ratio if ratio is None else ratio
        threshold = self.config.quality_threshold if threshold is None else threshold

        # Filter by threshold first
        filtered = [s for s in self.samples if s.quality_score >= threshold]

        # Then select top ratio
        if ratio < 1.0:
            filtered.sort(key=lambda s: s.quality_score, reverse=True)
            keep_count = max(1, int(len(filtered) * ratio))
            filtered = filtered[:keep_count]

        original_count = len(self.samples)
        self.samples = filtered

        return len(self.samples)

    def diversity_sample(
        self,
        target_count: int,
        domain_key: str = "domain"
    ) -> int:
        """Sample for diversity across domains.

        Args:
            target_count: Target number of samples
            domain_key: Metadata key for domain (default: "domain")

        Returns:
            Number of samples after sampling
        """
        if len(self.samples) <= target_count:
            return len(self.samples)

        # Group by domain
        by_domain: dict[str, list[TrainingSample]] = {}
        for sample in self.samples:
            domain = getattr(sample, domain_key, "unknown")
            if domain not in by_domain:
                by_domain[domain] = []
            by_domain[domain].append(sample)

        # Calculate samples per domain
        domains = list(by_domain.keys())
        per_domain = target_count // len(domains)
        remainder = target_count % len(domains)

        # Sample from each domain
        selected = []
        for i, domain in enumerate(domains):
            domain_samples = by_domain[domain]
            # First domains get remainder samples
            count = per_domain + (1 if i < remainder else 0)
            count = min(count, len(domain_samples))

            # Select top quality from this domain
            domain_samples.sort(key=lambda s: s.quality_score, reverse=True)
            selected.extend(domain_samples[:count])

        self.samples = selected
        return len(self.samples)

    def merge_with_new_data(
        self,
        new_samples: list[TrainingSample],
        rehearsal_ratio: float = 0.3,
        shuffle: bool = True,
        seed: int | None = None
    ) -> list[TrainingSample]:
        """Merge rehearsal buffer with new training data.

        Args:
            new_samples: New training samples
            rehearsal_ratio: Proportion of rehearsal samples (0.0-1.0)
            shuffle: Shuffle combined samples
            seed: Random seed for reproducibility

        Returns:
            Combined and shuffled samples
        """
        if not 0.0 <= rehearsal_ratio <= 1.0:
            raise ValueError(f"rehearsal_ratio must be in [0.0, 1.0], got {rehearsal_ratio}")

        # Calculate counts
        total_new = len(new_samples)
        n_rehearsal = int(total_new * rehearsal_ratio / (1 - rehearsal_ratio))
        n_rehearsal = min(n_rehearsal, len(self.samples))

        # Select rehearsal samples (top quality)
        rehearsal_samples = sorted(
            self.samples,
            key=lambda s: s.quality_score,
            reverse=True
        )[:n_rehearsal]

        # Combine
        combined = new_samples + rehearsal_samples

        # Shuffle if requested
        if shuffle:
            if seed is not None:
                random.seed(seed)
            random.shuffle(combined)

        return combined

    def version_balance(
        self,
        target_count: int,
        versions: list[str] | None = None
    ) -> int:
        """Balance samples across versions.

        Args:
            target_count: Target total sample count
            versions: Specific versions to include (None = all)

        Returns:
            Number of samples after balancing
        """
        if not self.config.track_provenance:
            raise ValueError("Provenance tracking not enabled")

        # Group by version
        by_version: dict[str, list[TrainingSample]] = {}
        for sample in self.samples:
            version = sample._metadata.get("rehearsal_version", "unknown")
            if versions and version not in versions:
                continue
            if version not in by_version:
                by_version[version] = []
            by_version[version].append(sample)

        if not by_version:
            return 0

        # Calculate samples per version
        version_list = list(by_version.keys())
        per_version = target_count // len(version_list)

        # Sample from each version
        selected = []
        for version in version_list:
            version_samples = by_version[version]
            count = min(per_version, len(version_samples))

            # Select top quality
            version_samples.sort(key=lambda s: s.quality_score, reverse=True)
            selected.extend(version_samples[:count])

        self.samples = selected
        return len(self.samples)

    def save(self, output_path: Path) -> int:
        """Save buffer to JSONL file.

        Args:
            output_path: Path for output file

        Returns:
            Number of samples saved
        """
        output_path = output_path.expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            for sample in self.samples:
                f.write(json.dumps(sample.to_dict()) + '\n')

        return len(self.samples)

    def stats(self) -> dict[str, Any]:
        """Get buffer statistics.

        Returns:
            Dictionary of statistics
        """
        if not self.samples:
            return {
                "total": 0,
                "versions": {},
                "quality": {},
            }

        quality_scores = [s.quality_score for s in self.samples]

        return {
            "total": len(self.samples),
            "versions": dict(self.version_counts),
            "quality": {
                "min": min(quality_scores) if quality_scores else 0.0,
                "max": max(quality_scores) if quality_scores else 0.0,
                "avg": sum(quality_scores) / len(quality_scores) if quality_scores else 0.0,
            },
            "domains": self._count_by_domain(),
        }

    def _count_by_domain(self) -> dict[str, int]:
        """Count samples by domain."""
        counts: dict[str, int] = {}
        for sample in self.samples:
            domain = sample.domain
            counts[domain] = counts.get(domain, 0) + 1
        return counts


def load_rehearsal_buffer(
    buffer_path: Path,
    config: RehearsalBufferConfig | None = None
) -> RehearsalBuffer:
    """Load an existing rehearsal buffer from file.

    Args:
        buffer_path: Path to saved buffer JSONL
        config: Buffer configuration (uses defaults if None)

    Returns:
        RehearsalBuffer instance
    """
    buffer = RehearsalBuffer(config=config or RehearsalBufferConfig())
    buffer.load_from_jsonl(buffer_path)
    return buffer
