#!/usr/bin/env python3
"""
Merge training datasets for 5 models with deduplication and quality filtering.

Models to merge:
1. majora_v1 - Oracle + CodeSearchNet + synthetic
2. veran_v5 - SNES hardware + synthetic
3. din_v2 - optimization + synthetic
4. nayru_v6 - generation + CodeSearchNet + synthetic
5. farore_v6 - debugging + synthetic

Mix ratios: 60% expert data, 25% CodeSearchNet/ToolBench, 15% synthetic
"""

import json
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple
from datetime import datetime

# Configuration
MODELS_DIR = Path("/Users/scawful/src/lab/afs/models")
TRAINING_DIR = Path("/Users/scawful/.context/training")

# Quality threshold (set to 0.0 to include all samples, including those with no score)
MIN_QUALITY_SCORE = 0.0

# Model configurations: (expert_files, description)
MODEL_CONFIGS = {
    "majora_v1": {
        "expert_files": [
            MODELS_DIR / "majora_v1_training.jsonl",
        ],
        "description": "Oracle + CodeSearchNet + synthetic",
    },
    "veran_v5": {
        "expert_files": [
            MODELS_DIR / "veran_snes_hardware_v2.jsonl",
            MODELS_DIR / "veran_combined_v2.jsonl",
        ],
        "description": "SNES hardware + synthetic",
    },
    "din_v2": {
        "expert_files": [
            MODELS_DIR / "din_optimization_training_v2.jsonl",
            MODELS_DIR / "din_combined_training.jsonl",
        ],
        "description": "optimization + synthetic",
    },
    "nayru_v6": {
        "expert_files": [
            MODELS_DIR / "train_validated_cleaned.jsonl",
        ],
        "description": "generation + CodeSearchNet + synthetic",
    },
    "farore_v6": {
        "expert_files": [
            MODELS_DIR / "farore_debugging_training.jsonl",
        ],
        "description": "debugging + synthetic",
    },
}


def load_jsonl(filepath: Path) -> List[Dict]:
    """Load JSONL file, skipping invalid lines."""
    samples = []
    if not filepath.exists():
        print(f"  WARNING: File not found: {filepath}")
        return samples

    with open(filepath, "r") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                sample = json.loads(line)
                samples.append(sample)
            except json.JSONDecodeError as e:
                print(f"  WARNING: Invalid JSON at {filepath}:{line_no}: {e}")
                continue

    return samples


def get_quality_score(sample: Dict) -> float:
    """Extract quality score from sample."""
    if isinstance(sample.get("quality_score"), (int, float)):
        return sample["quality_score"]

    # Try metadata
    if isinstance(sample.get("_metadata"), dict):
        if isinstance(sample["_metadata"].get("quality_score"), (int, float)):
            return sample["_metadata"]["quality_score"]

    # Try metadata in metadata field
    if isinstance(sample.get("metadata"), dict):
        if isinstance(sample["metadata"].get("quality_score"), (int, float)):
            return sample["metadata"]["quality_score"]

    return 0.5  # Default score


def get_instruction_text(sample: Dict) -> str:
    """Extract instruction text for deduplication."""
    # Try standard instruction field
    instr = sample.get("instruction", "").strip()
    if instr:
        return instr

    # Try messages format (for chat-based samples)
    messages = sample.get("messages", [])
    if messages and isinstance(messages, list):
        for msg in messages:
            if isinstance(msg, dict) and msg.get("role") == "user":
                return msg.get("content", "").strip()

    # Try input field
    return sample.get("input", "").strip()


def collect_synthetic_samples() -> List[Dict]:
    """Collect all synthetic samples from subdirectories."""
    synthetic_dir = TRAINING_DIR / "synthetic"
    samples = []

    if not synthetic_dir.exists():
        print(f"  WARNING: Synthetic dir not found: {synthetic_dir}")
        return samples

    # Look for all JSON/JSONL files in subdirectories
    for subdir in synthetic_dir.iterdir():
        if not subdir.is_dir():
            continue

        # Check for data files
        for json_file in subdir.glob("*.json"):
            try:
                with open(json_file, "r") as f:
                    data = json.load(f)
                    if isinstance(data, list):
                        samples.extend(data)
                    elif isinstance(data, dict):
                        samples.append(data)
            except Exception as e:
                print(f"  WARNING: Could not load {json_file}: {e}")

        # Check for JSONL files
        for jsonl_file in subdir.glob("*.jsonl"):
            samples.extend(load_jsonl(jsonl_file))

    print(f"  Collected {len(samples)} synthetic samples")
    return samples


def merge_datasets(
    expert_files: List[Path],
    codesearchnet_samples: List[Dict],
    toolbench_samples: List[Dict],
    synthetic_samples: List[Dict],
    model_name: str,
) -> Tuple[List[Dict], Dict]:
    """
    Merge datasets with specified ratios.

    Returns:
        - Merged and deduplicated samples
        - Statistics dict
    """
    stats = {
        "expert": 0,
        "codesearchnet": 0,
        "toolbench": 0,
        "synthetic": 0,
        "deduplicated": 0,
        "quality_filtered": 0,
    }

    # Load expert data
    expert_samples = []
    for filepath in expert_files:
        loaded = load_jsonl(filepath)
        expert_samples.extend(loaded)
        print(f"    Loaded {len(loaded)} from {filepath.name}")

    stats["expert"] = len(expert_samples)

    # Filter by quality score
    expert_samples = [
        s for s in expert_samples
        if get_quality_score(s) >= MIN_QUALITY_SCORE
    ]
    stats["quality_filtered"] = stats["expert"] - len(expert_samples)

    # Deduplication tracker (by instruction text)
    seen_instructions = set()
    merged_samples = []

    def add_sample(sample: Dict, source: str):
        """Add sample with deduplication."""
        instruction = get_instruction_text(sample)
        if not instruction:
            return False

        if instruction in seen_instructions:
            stats["deduplicated"] += 1
            return False

        seen_instructions.add(instruction)
        merged_samples.append(sample)
        stats[source] += 1
        return True

    # Add expert samples first (60% target)
    print(f"    Adding expert samples ({len(expert_samples)} available)...")
    for sample in expert_samples:
        add_sample(sample, "expert")

    expert_count = len(merged_samples)
    target_codesearchnet = max(1, int(expert_count * 0.25 / 0.60))
    target_synthetic = max(1, int(expert_count * 0.15 / 0.60))

    print(f"    Targets: CodeSearchNet={target_codesearchnet}, Synthetic={target_synthetic}")

    # Add CodeSearchNet samples (25% target)
    print(f"    Adding CodeSearchNet samples (target={target_codesearchnet}, available={len(codesearchnet_samples)})...")
    for sample in codesearchnet_samples[:target_codesearchnet]:
        add_sample(sample, "codesearchnet")

    # Add ToolBench samples as part of the 25% allocation
    if model_name == "nayru_v6":  # nayru might benefit from tool use examples
        toolbench_target = min(len(toolbench_samples), max(1, int(target_codesearchnet * 0.2)))
        print(f"    Adding ToolBench samples (target={toolbench_target}, available={len(toolbench_samples)})...")
        for sample in toolbench_samples[:toolbench_target]:
            add_sample(sample, "toolbench")

    # Add synthetic samples (15% target)
    print(f"    Adding synthetic samples (target={target_synthetic}, available={len(synthetic_samples)})...")
    for sample in synthetic_samples[:target_synthetic]:
        add_sample(sample, "synthetic")

    print(f"    Final merged count: {len(merged_samples)}")
    print(f"    Deduplication removed: {stats['deduplicated']} duplicates")

    return merged_samples, stats


def save_jsonl(samples: List[Dict], filepath: Path) -> None:
    """Save samples to JSONL file."""
    with open(filepath, "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")


def main():
    """Main merge pipeline."""
    print("=" * 80)
    print("Training Dataset Merger")
    print("=" * 80)
    print()

    # Load shared data sources
    print("Loading shared data sources...")
    print()

    print("  CodeSearchNet...")
    codesearchnet_file = TRAINING_DIR / "codesearchnet" / "processed" / "train.jsonl"
    codesearchnet_samples = load_jsonl(codesearchnet_file)
    print(f"    Loaded {len(codesearchnet_samples)} samples")

    print("  ToolBench...")
    toolbench_file = TRAINING_DIR / "toolbench" / "processed" / "train_sample.jsonl"
    toolbench_samples = load_jsonl(toolbench_file)
    print(f"    Loaded {len(toolbench_samples)} samples")

    print("  Synthetic samples...")
    synthetic_samples = collect_synthetic_samples()

    print()
    print("=" * 80)
    print("Merging datasets for each model...")
    print("=" * 80)
    print()

    all_stats = {}

    for model_name, config in MODEL_CONFIGS.items():
        print(f"Processing {model_name}...")
        print(f"  Description: {config['description']}")
        print()

        expert_files = config["expert_files"]

        # Merge datasets
        merged_samples, stats = merge_datasets(
            expert_files,
            codesearchnet_samples,
            toolbench_samples,
            synthetic_samples,
            model_name,
        )

        # Save to output file
        output_file = MODELS_DIR / f"{model_name}_merged.jsonl"
        save_jsonl(merged_samples, output_file)

        print(f"    Saved to {output_file.name}")
        print()

        all_stats[model_name] = {
            "output_file": str(output_file),
            "total_samples": len(merged_samples),
            **stats,
        }

    # Print summary report
    print("=" * 80)
    print("SUMMARY REPORT")
    print("=" * 80)
    print()

    for model_name, stats in all_stats.items():
        print(f"{model_name}_merged.jsonl")
        print(f"  Total samples: {stats['total_samples']}")
        print(f"  Expert: {stats['expert']}")
        print(f"  CodeSearchNet: {stats['codesearchnet']}")
        print(f"  ToolBench: {stats.get('toolbench', 0)}")
        print(f"  Synthetic: {stats['synthetic']}")
        print(f"  Quality filtered: {stats['quality_filtered']}")
        print(f"  Deduplicated: {stats['deduplicated']}")

        # Calculate ratios
        if stats['total_samples'] > 0:
            expert_ratio = stats['expert'] / stats['total_samples']
            code_ratio = stats['codesearchnet'] / stats['total_samples']
            syn_ratio = stats['synthetic'] / stats['total_samples']
            print(f"  Ratios: {expert_ratio:.1%} expert, {code_ratio:.1%} code, {syn_ratio:.1%} synthetic")
        print()

    # Save summary
    summary_file = MODELS_DIR / "merge_summary.json"
    with open(summary_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "min_quality_score": MIN_QUALITY_SCORE,
            "models": all_stats,
        }, f, indent=2)

    print(f"Summary saved to {summary_file.name}")
    print()
    print("=" * 80)
    print("Merge complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
