#!/usr/bin/env python3
"""Build rehearsal buffer from previous training data versions.

This script creates a rehearsal buffer by selecting top-quality samples
from previous training runs to prevent catastrophic forgetting.

Usage:
    python3 scripts/build_rehearsal_buffer.py --help
    python3 scripts/build_rehearsal_buffer.py --input models/veran_v4.jsonl --output ~/.context/training/rehearsal/veran_v4.jsonl
    python3 scripts/build_rehearsal_buffer.py --input models/veran_v2.jsonl --input models/veran_v3.jsonl --input models/veran_v4.jsonl --output ~/.context/training/rehearsal/veran_combined.jsonl
"""

from __future__ import annotations

import argparse
from pathlib import Path

import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from afs.training.rehearsal import RehearsalBuffer, RehearsalBufferConfig


def main():
    parser = argparse.ArgumentParser(
        description="Build rehearsal buffer from previous training data.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--input",
        action="append",
        required=True,
        help="Input training JSONL file (repeatable for multiple versions)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output rehearsal buffer JSONL path",
    )
    parser.add_argument(
        "--version",
        action="append",
        help="Version tag for each input (repeatable, must match --input count)",
    )
    parser.add_argument(
        "--quality-threshold",
        type=float,
        default=0.5,
        help="Minimum quality score to include (default: 0.5)",
    )
    parser.add_argument(
        "--top-ratio",
        type=float,
        default=0.3,
        help="Keep top X ratio of samples (default: 0.3 = 30%%)",
    )
    parser.add_argument(
        "--max-per-domain",
        type=int,
        help="Maximum samples per domain (optional)",
    )
    parser.add_argument(
        "--enable-diversity",
        action="store_true",
        default=True,
        help="Use diversity sampling (default: enabled)",
    )
    parser.add_argument(
        "--no-diversity",
        action="store_false",
        dest="enable_diversity",
        help="Disable diversity sampling",
    )
    parser.add_argument(
        "--version-balance",
        type=int,
        help="Balance samples across versions to this count",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        help="Maximum samples to load from each input file",
    )

    args = parser.parse_args()

    # Validate ratios
    if not 0.0 <= args.quality_threshold <= 1.0:
        print(f"Error: --quality-threshold must be in [0.0, 1.0], got {args.quality_threshold}")
        return 1

    if not 0.0 < args.top_ratio <= 1.0:
        print(f"Error: --top-ratio must be in (0.0, 1.0], got {args.top_ratio}")
        return 1

    # Validate version tags
    if args.version and len(args.version) != len(args.input):
        print(f"Error: Number of --version tags ({len(args.version)}) must match --input files ({len(args.input)})")
        return 1

    # Create config
    config = RehearsalBufferConfig(
        quality_threshold=args.quality_threshold,
        top_ratio=args.top_ratio,
        enable_diversity=args.enable_diversity,
        max_per_domain=args.max_per_domain,
        track_provenance=bool(args.version),
    )

    # Create buffer
    buffer = RehearsalBuffer(config=config)

    print("=" * 60)
    print("Building Rehearsal Buffer")
    print("=" * 60)

    # Load input files
    for i, input_file in enumerate(args.input):
        input_path = Path(input_file).expanduser().resolve()
        version = args.version[i] if args.version else f"input_{i+1}"

        print(f"\n[{i+1}/{len(args.input)}] Loading {input_path.name}...")
        print(f"  Version: {version}")

        if not input_path.exists():
            print(f"  Error: File not found, skipping")
            continue

        try:
            loaded = buffer.load_from_jsonl(
                input_path,
                version=version,
                max_samples=args.max_samples
            )
            print(f"  Loaded: {loaded} samples")
        except Exception as e:
            print(f"  Error: {e}")
            continue

    if not buffer.samples:
        print("\nError: No samples loaded. Check input paths.")
        return 1

    print(f"\nTotal samples loaded: {len(buffer.samples)}")

    # Apply filtering
    print("\n" + "=" * 60)
    print("Filtering Samples")
    print("=" * 60)

    print(f"\nQuality threshold: {args.quality_threshold}")
    print(f"Top ratio: {args.top_ratio}")

    before = len(buffer.samples)
    retained = buffer.select_top_samples(
        ratio=args.top_ratio,
        threshold=args.quality_threshold
    )
    removed = before - retained

    print(f"\nFiltered: {removed} samples removed")
    print(f"Retained: {retained} samples ({100*retained/before:.1f}%)")

    # Version balancing
    if args.version_balance:
        print(f"\n" + "=" * 60)
        print("Balancing Across Versions")
        print("=" * 60)

        before = len(buffer.samples)
        balanced = buffer.version_balance(
            target_count=args.version_balance,
            versions=args.version if args.version else None
        )
        print(f"\nBalanced to {balanced} samples across {len(buffer.version_counts)} versions")

    # Show statistics
    print("\n" + "=" * 60)
    print("Buffer Statistics")
    print("=" * 60)

    stats = buffer.stats()
    print(f"\nTotal samples: {stats['total']}")
    print(f"\nQuality scores:")
    print(f"  Min: {stats['quality']['min']:.3f}")
    print(f"  Max: {stats['quality']['max']:.3f}")
    print(f"  Avg: {stats['quality']['avg']:.3f}")

    if stats['versions']:
        print(f"\nVersions:")
        for version, count in stats['versions'].items():
            print(f"  {version}: {count} samples")

    if stats['domains']:
        print(f"\nDomains:")
        for domain, count in stats['domains'].items():
            print(f"  {domain}: {count} samples")

    # Save buffer
    print("\n" + "=" * 60)
    print("Saving Buffer")
    print("=" * 60)

    output_path = args.output.expanduser().resolve()
    saved = buffer.save(output_path)

    print(f"\nSaved {saved} samples to:")
    print(f"  {output_path}")

    print("\n" + "=" * 60)
    print("Complete")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
