#!/usr/bin/env python3
"""Apply improvements to training datasets based on quality analysis.

Usage:
    python scripts/improve_dataset.py <dataset.jsonl> --action remove_duplicates
    python scripts/improve_dataset.py data.jsonl --min-quality 0.6 --output improved_data.jsonl
    python scripts/improve_dataset.py data.jsonl --remove-duplicates --remove-anomalies --min-quality 0.5
"""

import argparse
import json
import logging
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from afs.quality import DatasetAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class ImprovementResult:
    """Result of dataset improvements."""

    original_count: int
    final_count: int
    duplicates_removed: int
    anomalies_removed: int
    low_quality_removed: int
    other_removed: int

    def summary(self) -> str:
        """Get summary text."""
        return (
            f"Original: {self.original_count} samples\n"
            f"Final: {self.final_count} samples ({self.final_count/self.original_count*100:.1f}%)\n"
            f"Removed: {self.original_count - self.final_count} samples\n"
            f"  - Duplicates: {self.duplicates_removed}\n"
            f"  - Anomalies: {self.anomalies_removed}\n"
            f"  - Low quality: {self.low_quality_removed}\n"
            f"  - Other: {self.other_removed}"
        )


def improve_dataset(
    input_path: Path,
    output_path: Path,
    remove_duplicates: bool = False,
    remove_anomalies: bool = False,
    min_quality: float = 0.0,
    dedup_semantic: bool = False,
    domain: str = "general",
    report_path: Optional[Path] = None,
) -> ImprovementResult:
    """Improve dataset by removing low-quality samples.

    Args:
        input_path: Path to input dataset
        output_path: Path to save improved dataset
        remove_duplicates: Remove exact duplicate samples
        remove_anomalies: Remove anomalous samples
        min_quality: Minimum quality score (0.0-1.0)
        dedup_semantic: Also remove semantic duplicates
        domain: Domain type for analysis
        report_path: Path to save improvement report

    Returns:
        ImprovementResult with statistics
    """
    logger.info(f"Loading dataset: {input_path}")
    samples = []
    with open(input_path) as f:
        if input_path.suffix == ".jsonl":
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        else:
            data = json.load(f)
            samples = data if isinstance(data, list) else [data]

    logger.info(f"Loaded {len(samples)} samples")

    # Analyze
    logger.info("Analyzing dataset for improvements...")
    analyzer = DatasetAnalyzer(domain=domain)
    report = analyzer.analyze(samples)

    # Track removals
    result = ImprovementResult(
        original_count=len(samples),
        final_count=len(samples),
        duplicates_removed=0,
        anomalies_removed=0,
        low_quality_removed=0,
        other_removed=0,
    )

    # Build set of indices to remove
    remove_indices = set()

    # Remove duplicates
    if remove_duplicates:
        duplicate_indices = defaultdict(list)
        for quality in report.sample_qualities:
            if quality.is_duplicate:
                # Keep first occurrence, remove others
                if quality.duplicate_info.exact_duplicates:
                    for dup_idx in quality.duplicate_info.exact_duplicates:
                        duplicate_indices[min(quality.index, dup_idx)].append(
                            max(quality.index, dup_idx)
                        )

        for keep_idx, remove_idxs in duplicate_indices.items():
            for idx in remove_idxs:
                if idx not in remove_indices and idx != keep_idx:
                    remove_indices.add(idx)
                    result.duplicates_removed += 1

        logger.info(f"Marked {result.duplicates_removed} duplicates for removal")

    # Remove anomalies
    if remove_anomalies:
        for quality in report.sample_qualities:
            if quality.is_anomaly and quality.index not in remove_indices:
                remove_indices.add(quality.index)
                result.anomalies_removed += 1

        logger.info(f"Marked {result.anomalies_removed} anomalies for removal")

    # Remove low quality
    if min_quality > 0:
        for quality in report.sample_qualities:
            if quality.overall_quality_score < min_quality and quality.index not in remove_indices:
                remove_indices.add(quality.index)
                result.low_quality_removed += 1

        logger.info(f"Marked {result.low_quality_removed} low-quality samples for removal")

    # Keep samples that passed filters
    improved_samples = [s for i, s in enumerate(samples) if i not in remove_indices]
    result.final_count = len(improved_samples)

    # Save improved dataset
    logger.info(f"Saving improved dataset to {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix == ".jsonl":
        with open(output_path, "w") as f:
            for sample in improved_samples:
                f.write(json.dumps(sample) + "\n")
    else:
        with open(output_path, "w") as f:
            json.dump(improved_samples, f, indent=2)

    # Save report if requested
    if report_path:
        logger.info(f"Saving improvement report to {report_path}")
        report_path.parent.mkdir(parents=True, exist_ok=True)

        report_data = {
            "input": str(input_path),
            "output": str(output_path),
            "original_count": result.original_count,
            "final_count": result.final_count,
            "retained": result.final_count / result.original_count,
            "removals": {
                "duplicates": result.duplicates_removed,
                "anomalies": result.anomalies_removed,
                "low_quality": result.low_quality_removed,
                "total": result.original_count - result.final_count,
            },
            "filters": {
                "remove_duplicates": remove_duplicates,
                "remove_anomalies": remove_anomalies,
                "min_quality": min_quality,
                "domain": domain,
            },
            "original_stats": report.statistics.to_dict(),
        }

        with open(report_path, "w") as f:
            json.dump(report_data, f, indent=2)

    return result


def augment_dataset(
    input_path: Path,
    output_path: Path,
    high_quality_only: bool = False,
    quality_threshold: float = 0.8,
    augmentation_factor: int = 2,
) -> None:
    """Augment dataset by expanding high-quality samples.

    Args:
        input_path: Path to input dataset
        output_path: Path to save augmented dataset
        high_quality_only: Only augment high-quality samples
        quality_threshold: Threshold for high quality
        augmentation_factor: How many variations to generate per sample
    """
    logger.info(f"Loading dataset: {input_path}")
    samples = []
    with open(input_path) as f:
        if input_path.suffix == ".jsonl":
            for line in f:
                if line.strip():
                    samples.append(json.loads(line))
        else:
            data = json.load(f)
            samples = data if isinstance(data, list) else [data]

    # Analyze
    logger.info("Analyzing for augmentation candidates...")
    analyzer = DatasetAnalyzer()
    report = analyzer.analyze(samples)

    # Find candidates
    augmentation_candidates = []
    for quality in report.sample_qualities:
        if quality.overall_quality_score >= quality_threshold:
            augmentation_candidates.append(quality.index)

    logger.info(f"Found {len(augmentation_candidates)} high-quality samples for augmentation")

    # Augment (simple approach: duplicate with variation)
    augmented_samples = list(samples)

    for idx in augmentation_candidates:
        sample = samples[idx]
        for _ in range(augmentation_factor - 1):
            # Simple augmentation: add slight variation
            aug_sample = dict(sample)
            # In a real implementation, would do semantic variations
            augmented_samples.append(aug_sample)

    logger.info(f"Generated {len(augmented_samples) - len(samples)} augmented samples")

    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.suffix == ".jsonl":
        with open(output_path, "w") as f:
            for sample in augmented_samples:
                f.write(json.dumps(sample) + "\n")
    else:
        with open(output_path, "w") as f:
            json.dump(augmented_samples, f, indent=2)

    logger.info(f"Saved augmented dataset to {output_path}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Improve training datasets by removing low-quality samples",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Remove duplicates and anomalies
  python scripts/improve_dataset.py data.jsonl --remove-duplicates --remove-anomalies

  # Keep only high-quality samples
  python scripts/improve_dataset.py data.jsonl --min-quality 0.7 --output clean_data.jsonl

  # Augment high-quality samples
  python scripts/improve_dataset.py data.jsonl --augment --output augmented_data.jsonl

  # Generate improvement report
  python scripts/improve_dataset.py data.jsonl --remove-duplicates --report improvements.json
""",
    )

    parser.add_argument(
        "dataset",
        help="Path to input dataset (JSONL or JSON)",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output path for improved dataset",
    )
    parser.add_argument(
        "--remove-duplicates",
        action="store_true",
        help="Remove duplicate samples",
    )
    parser.add_argument(
        "--remove-anomalies",
        action="store_true",
        help="Remove anomalous samples",
    )
    parser.add_argument(
        "--min-quality",
        type=float,
        default=0.0,
        help="Minimum quality score (0.0-1.0) to keep sample (default: 0.0)",
    )
    parser.add_argument(
        "-d", "--domain",
        choices=["general", "assembly", "code"],
        default="general",
        help="Domain type for analysis (default: general)",
    )
    parser.add_argument(
        "-r", "--report",
        help="Save improvement report to file",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Augment high-quality samples",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    input_path = Path(args.dataset)

    if args.augment:
        output_path = args.output or f"{input_path.stem}_augmented{input_path.suffix}"
        augment_dataset(input_path, Path(output_path))
    else:
        output_path = args.output or f"{input_path.stem}_improved{input_path.suffix}"

        result = improve_dataset(
            input_path,
            Path(output_path),
            remove_duplicates=args.remove_duplicates,
            remove_anomalies=args.remove_anomalies,
            min_quality=args.min_quality,
            domain=args.domain,
            report_path=Path(args.report) if args.report else None,
        )

        print("\n" + "=" * 80)
        print("DATASET IMPROVEMENT SUMMARY")
        print("=" * 80)
        print(result.summary())
        print(f"\nImproved dataset saved to: {output_path}")
        if args.report:
            print(f"Improvement report saved to: {args.report}")
        print("=" * 80)


if __name__ == "__main__":
    main()
