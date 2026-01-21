#!/usr/bin/env python3
"""Create mixed training datasets with ToolBench integration.

This script combines existing agent-specific training data with ToolBench samples
to improve tool use capabilities across all agents.

Usage:
    python3 scripts/create_tooluse_dataset.py --help
    python3 scripts/create_tooluse_dataset.py --agents nayru,din --toolbench-ratio 0.2
    python3 scripts/create_tooluse_dataset.py --all-agents --toolbench-ratio 0.3 --output models/
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any


def load_samples(filepath: Path) -> list[dict[str, Any]]:
    """Load samples from JSONL file."""
    samples = []
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                samples.append(json.loads(line))
    return samples


def save_samples(samples: list[dict[str, Any]], output_path: Path) -> None:
    """Save samples to JSONL file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for sample in samples:
            f.write(json.dumps(sample) + '\n')


def create_mixed_dataset(
    agent_name: str,
    existing_data_path: Path,
    toolbench_data_path: Path,
    output_dir: Path,
    toolbench_ratio: float = 0.2,
    seed: int = 42,
) -> tuple[Path, int, int]:
    """Mix agent-specific data with ToolBench samples.

    Args:
        agent_name: Agent name (nayru, din, farore, veran, majora, etc.)
        existing_data_path: Path to agent's existing training.jsonl
        toolbench_data_path: Path to processed ToolBench samples
        output_dir: Directory for output files
        toolbench_ratio: Proportion of ToolBench samples (0.0-1.0)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (output_path, existing_count, toolbench_count)
    """
    random.seed(seed)

    # Load existing agent data
    if not existing_data_path.exists():
        print(f"Warning: No existing data found for {agent_name} at {existing_data_path}")
        existing_samples = []
    else:
        existing_samples = load_samples(existing_data_path)
        print(f"  Loaded {len(existing_samples)} existing samples for {agent_name}")

    # Load ToolBench samples
    if not toolbench_data_path.exists():
        raise FileNotFoundError(f"ToolBench data not found: {toolbench_data_path}")

    toolbench_samples = load_samples(toolbench_data_path)
    print(f"  Loaded {len(toolbench_samples)} ToolBench samples")

    # Calculate sample counts
    if len(existing_samples) == 0:
        # No existing data, use ToolBench only
        n_existing = 0
        n_toolbench = len(toolbench_samples)
    else:
        # Mix based on ratio
        total_existing = len(existing_samples)
        n_existing = int(total_existing * (1 - toolbench_ratio))
        n_toolbench = int(total_existing * toolbench_ratio)

        # Ensure we don't exceed available samples
        n_existing = min(n_existing, len(existing_samples))
        n_toolbench = min(n_toolbench, len(toolbench_samples))

    # Sample from each
    selected_existing = random.sample(existing_samples, n_existing) if n_existing > 0 else []
    selected_toolbench = random.sample(toolbench_samples, n_toolbench) if n_toolbench > 0 else []

    # Combine and shuffle
    mixed = selected_existing + selected_toolbench
    random.shuffle(mixed)

    # Save
    output_path = output_dir / f"{agent_name}_with_tooluse_v1.jsonl"
    save_samples(mixed, output_path)

    return output_path, n_existing, n_toolbench


def main():
    parser = argparse.ArgumentParser(
        description="Create mixed training datasets with ToolBench integration.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--agents",
        help="Comma-separated list of agent names (e.g., nayru,din,farore)",
    )
    parser.add_argument(
        "--all-agents",
        action="store_true",
        help="Process all known agents",
    )
    parser.add_argument(
        "--toolbench-data",
        type=Path,
        default=Path("~/.context/training/toolbench/processed/train.jsonl"),
        help="Path to processed ToolBench training data (default: ~/.context/training/toolbench/processed/train.jsonl)",
    )
    parser.add_argument(
        "--toolbench-ratio",
        type=float,
        default=0.2,
        help="Proportion of ToolBench samples (0.0-1.0, default: 0.2 = 20%%)",
    )
    parser.add_argument(
        "--existing-data-dir",
        type=Path,
        default=Path("models"),
        help="Directory containing existing agent training data (default: models/)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("models"),
        help="Output directory for mixed datasets (default: models/)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )

    args = parser.parse_args()

    # Validate ratio
    if not 0.0 <= args.toolbench_ratio <= 1.0:
        print("Error: --toolbench-ratio must be between 0.0 and 1.0")
        return 1

    # Resolve paths
    toolbench_data = args.toolbench_data.expanduser().resolve()
    existing_data_dir = args.existing_data_dir.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()

    if not toolbench_data.exists():
        print(f"Error: ToolBench data not found: {toolbench_data}")
        print("\nFirst run: afs training toolbench-export --dataset-dir ~/.context/training/datasets/ToolBench --output ~/.context/training/toolbench/processed/train.jsonl")
        return 1

    # Determine agents to process
    if args.all_agents:
        agents = ["nayru", "din", "farore", "veran", "majora", "hylia", "agahnim"]
    elif args.agents:
        agents = [a.strip() for a in args.agents.split(",")]
    else:
        print("Error: Must specify --agents or --all-agents")
        parser.print_help()
        return 1

    print(f"Creating mixed datasets with {args.toolbench_ratio:.0%} ToolBench samples\n")

    # Process each agent
    results = []
    for agent in agents:
        print(f"Processing {agent}:")

        # Try common naming patterns for existing data
        existing_paths = [
            existing_data_dir / f"{agent}_current_training.jsonl",
            existing_data_dir / f"{agent}_training.jsonl",
            existing_data_dir / f"{agent}.jsonl",
            existing_data_dir / f"{agent}_v*_training.jsonl",  # Versioned
        ]

        existing_path = None
        for path in existing_paths:
            # Handle glob pattern
            if "*" in str(path):
                matches = list(existing_data_dir.glob(path.name))
                if matches:
                    # Use most recent version
                    existing_path = max(matches, key=lambda p: p.stat().st_mtime)
                    break
            elif path.exists():
                existing_path = path
                break

        if existing_path is None:
            print(f"  Warning: No existing training data found for {agent}, skipping...")
            continue

        try:
            output_path, n_existing, n_toolbench = create_mixed_dataset(
                agent_name=agent,
                existing_data_path=existing_path,
                toolbench_data_path=toolbench_data,
                output_dir=output_dir,
                toolbench_ratio=args.toolbench_ratio,
                seed=args.seed,
            )

            print(f"  Created: {output_path}")
            print(f"    Existing: {n_existing}")
            print(f"    ToolBench: {n_toolbench}")
            print(f"    Total: {n_existing + n_toolbench}\n")

            results.append({
                "agent": agent,
                "output": str(output_path),
                "existing": n_existing,
                "toolbench": n_toolbench,
                "total": n_existing + n_toolbench,
            })

        except Exception as e:
            print(f"  Error: {e}\n")
            continue

    # Summary
    if results:
        print(f"\nSuccessfully created {len(results)} mixed datasets:")
        for r in results:
            print(f"  {r['agent']}: {r['total']} samples ({r['existing']} existing + {r['toolbench']} ToolBench)")
    else:
        print("\nNo datasets created. Check agent names and data paths.")
        return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
