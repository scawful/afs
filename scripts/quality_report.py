#!/usr/bin/env python3
"""Generate comprehensive quality report for training datasets.

Usage:
    python scripts/quality_report.py <dataset.jsonl> [--output report.json] [--domain general|assembly|code]
    python scripts/quality_report.py training_data/agahnim/*.jsonl --domain assembly
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Optional

from afs.quality import DatasetAnalyzer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def generate_report(
    dataset_paths: list[Path],
    output_path: Optional[Path] = None,
    domain: str = "general",
    save_samples: bool = True,
) -> None:
    """Generate quality report for dataset(s).

    Args:
        dataset_paths: Paths to dataset files (JSONL or JSON)
        output_path: Where to save the report
        domain: Domain type for analysis
        save_samples: Whether to save per-sample quality data
    """
    analyzer = DatasetAnalyzer(domain=domain)

    # Load all samples
    all_samples = []
    for path in dataset_paths:
        logger.info(f"Loading dataset: {path}")
        with open(path) as f:
            if path.suffix == ".jsonl":
                for line in f:
                    if line.strip():
                        all_samples.append(json.loads(line))
            elif path.suffix == ".json":
                data = json.load(f)
                if isinstance(data, list):
                    all_samples.extend(data)
                else:
                    all_samples.append(data)

    logger.info(f"Loaded {len(all_samples)} total samples")

    # Analyze
    logger.info("Analyzing dataset...")
    report = analyzer.analyze(
        all_samples,
        dataset_name=dataset_paths[0].stem if len(dataset_paths) == 1 else "combined",
        dataset_path=str(dataset_paths[0]),
    )

    # Determine output paths
    if output_path is None:
        output_path = Path("quality_report.json")
    else:
        output_path = Path(output_path)

    samples_path = output_path.parent / f"{output_path.stem}_samples.jsonl"

    # Save reports
    logger.info(f"Saving report to {output_path}")
    report.save_json(output_path)

    if save_samples:
        logger.info(f"Saving sample details to {samples_path}")
        report.save_samples_jsonl(samples_path)

    # Print summary
    print("\n" + "=" * 80)
    print(f"QUALITY REPORT: {report.dataset_name}")
    print("=" * 80)
    print(f"Total Samples: {report.statistics.total_samples}")
    print(f"Unique Samples: {report.statistics.unique_samples}")
    print(f"Average Quality Score: {report.average_quality_score:.1%}")
    print(f"Dataset Size: {report.statistics.total_size_bytes / 1024 / 1024:.2f} MB")
    print()

    print("QUALITY BREAKDOWN:")
    print(f"  High Quality (0.8-1.0): {sum(1 for s in report.sample_qualities if s.overall_quality_score >= 0.8)}")
    print(f"  Medium Quality (0.5-0.8): {sum(1 for s in report.sample_qualities if 0.5 <= s.overall_quality_score < 0.8)}")
    print(f"  Low Quality (0.0-0.5): {sum(1 for s in report.sample_qualities if s.overall_quality_score < 0.5)}")
    print()

    print("ISSUES DETECTED:")
    print(f"  Duplicates: {sum(1 for s in report.sample_qualities if s.is_duplicate)}")
    print(f"  Anomalies: {sum(1 for s in report.sample_qualities if s.is_anomaly)}")
    print()

    print("BIAS ANALYSIS:")
    print(f"  Overall Bias Score: {report.bias_report.overall_bias_score:.2f}/1.0")
    print(f"  Gender Bias: {report.bias_report.gender_bias.bias_score:.2f}/1.0")
    print(f"  Cultural Bias: {report.bias_report.cultural_bias.bias_score:.2f}/1.0")
    print(f"  Technical Bias: {report.bias_report.technical_bias.bias_score:.2f}/1.0")
    print()

    print("IMPROVEMENTS RECOMMENDED:")
    for i, rec in enumerate(report.improvement_opportunities, 1):
        print(f"  {i}. {rec}")
    print()

    print("BIAS RECOMMENDATIONS:")
    for i, rec in enumerate(report.bias_report.recommendations, 1):
        print(f"  {i}. {rec}")
    print()

    print(f"Full report saved to: {output_path}")
    if save_samples:
        print(f"Sample details saved to: {samples_path}")
    print("=" * 80)


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate quality report for training datasets",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/quality_report.py training_data/dataset.jsonl
  python scripts/quality_report.py training_data/*.jsonl --output results/report.json
  python scripts/quality_report.py data.json --domain assembly --save-samples
""",
    )

    parser.add_argument(
        "datasets",
        nargs="+",
        help="Path(s) to dataset file(s) (JSONL or JSON)",
    )
    parser.add_argument(
        "-o", "--output",
        help="Output path for report (default: quality_report.json)",
    )
    parser.add_argument(
        "-d", "--domain",
        choices=["general", "assembly", "code"],
        default="general",
        help="Domain type for analysis (default: general)",
    )
    parser.add_argument(
        "--no-samples",
        action="store_true",
        help="Don't save per-sample quality data",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose output",
    )

    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Expand paths
    dataset_paths = []
    for pattern in args.datasets:
        path = Path(pattern)
        if "*" in pattern:
            dataset_paths.extend(sorted(Path(pattern).parent.glob(Path(pattern).name)))
        else:
            dataset_paths.append(path)

    if not dataset_paths:
        logger.error("No datasets found")
        return

    generate_report(
        dataset_paths,
        output_path=Path(args.output) if args.output else None,
        domain=args.domain,
        save_samples=not args.no_samples,
    )


if __name__ == "__main__":
    main()
