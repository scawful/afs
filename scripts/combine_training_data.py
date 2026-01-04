#!/usr/bin/env python3
"""Combine training datasets with register-emphasis examples first."""

import json
from pathlib import Path

def load_jsonl(path):
    """Load JSONL file."""
    examples = []
    with open(path) as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples


def main():
    models_dir = Path(__file__).parent.parent / "models"

    # Load datasets
    register_data = load_jsonl(models_dir / "veran_register_emphasis.jsonl")
    original_data = load_jsonl(models_dir / "veran_snes_hardware.jsonl")

    print(f"Register-emphasis examples: {len(register_data)}")
    print(f"Original examples: {len(original_data)}")

    # Track codes we've already added
    seen_codes = set()
    for ex in register_data:
        seen_codes.add(ex["instruction"])

    # Add original examples that aren't duplicates
    unique_original = []
    for ex in original_data:
        if ex["instruction"] not in seen_codes:
            unique_original.append(ex)
            seen_codes.add(ex["instruction"])

    print(f"Unique original examples to add: {len(unique_original)}")

    # Combine: register-emphasis FIRST, then original
    combined = register_data + unique_original

    print(f"Total combined: {len(combined)}")

    # Save combined dataset
    output_file = models_dir / "veran_snes_hardware_v2.jsonl"
    with open(output_file, "w") as f:
        for ex in combined:
            f.write(json.dumps(ex) + "\n")

    print(f"\nSaved to: {output_file}")

    # Show category breakdown
    categories = {}
    for ex in combined:
        cat = ex.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1

    print("\nCategory breakdown:")
    for cat, count in sorted(categories.items()):
        print(f"  {cat}: {count}")


if __name__ == "__main__":
    main()
