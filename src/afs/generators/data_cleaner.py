"""Training data cleaner for the nayru model.

Cleans malformed training samples by:
1. Stripping file markers from instructions (e.g., "; *$EE1ED-$EE213 LOCAL", "====")
2. Filtering out samples with short/empty outputs (< 100 chars)
3. Marking samples that need regeneration (bad summaries)

Usage:
    python -m afs generators clean --input X --output Y

NOTE: The cleaning logic is now shared via base.py. This module uses those
shared utilities and provides a CLI interface for batch cleaning.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .base import (
    TrainingSample,
    clean_instruction,
    is_malformed_output,
    read_jsonl,
    write_jsonl,
)


@dataclass
class CleaningStats:
    """Statistics from a cleaning run."""

    total_input: int = 0
    cleaned: int = 0
    filtered_short_output: int = 0
    marked_for_regen: int = 0
    instruction_cleaned: int = 0
    output_retained: int = 0
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        """Return a formatted summary of the cleaning run."""
        lines = [
            f"Input samples: {self.total_input}",
            f"Output samples: {self.output_retained}",
            f"Filtered (short output): {self.filtered_short_output}",
            f"Marked for regeneration: {self.marked_for_regen}",
            f"Instructions cleaned: {self.instruction_cleaned}",
            f"Samples modified: {self.cleaned}",
        ]
        if self.errors:
            lines.append(f"Errors: {len(self.errors)}")
        return "\n".join(lines)


def clean_sample(sample: TrainingSample) -> tuple[TrainingSample | None, dict[str, Any]]:
    """
    Clean a single training sample.

    Returns:
        Tuple of (cleaned_sample_or_None, stats_dict)
        Returns None if the sample should be filtered out
    """
    stats: dict[str, Any] = {
        "instruction_cleaned": False,
        "filtered": False,
        "needs_regen": False,
        "filter_reason": None,
    }

    # Check output length first
    if len(sample.output) < 100:
        stats["filtered"] = True
        stats["filter_reason"] = "short_output"
        return None, stats

    # Check for bad outputs (file markers as summaries)
    if is_malformed_output(sample.output, sample.instruction):
        stats["needs_regen"] = True
        # Mark in metadata for later regeneration
        sample._metadata["needs_regeneration"] = True
        sample._metadata["regen_reason"] = "output_is_file_marker"

    # Clean the instruction
    cleaned_instruction, was_modified = clean_instruction(sample.instruction)
    if was_modified:
        sample.instruction = cleaned_instruction
        stats["instruction_cleaned"] = True

    # Don't output samples with empty instructions after cleaning
    if not sample.instruction.strip():
        stats["filtered"] = True
        stats["filter_reason"] = "empty_instruction"
        return None, stats

    return sample, stats


def clean_dataset(
    input_path: Path,
    output_path: Path | None = None,
    regen_output_path: Path | None = None,
    min_output_length: int = 100,
) -> CleaningStats:
    """
    Clean a JSONL training dataset.

    Args:
        input_path: Path to input JSONL file
        output_path: Path to write cleaned samples (default: input_cleaned.jsonl)
        regen_output_path: Path to write samples needing regeneration (optional)
        min_output_length: Minimum output length to retain sample

    Returns:
        CleaningStats with details of what was cleaned/filtered
    """
    stats = CleaningStats()

    samples = read_jsonl(input_path)
    stats.total_input = len(samples)

    cleaned_samples: list[TrainingSample] = []
    regen_samples: list[TrainingSample] = []

    for sample in samples:
        try:
            cleaned, sample_stats = clean_sample(sample)

            if sample_stats["filtered"]:
                if sample_stats["filter_reason"] == "short_output":
                    stats.filtered_short_output += 1
                continue

            if sample_stats["instruction_cleaned"]:
                stats.instruction_cleaned += 1
                stats.cleaned += 1

            if sample_stats["needs_regen"]:
                stats.marked_for_regen += 1
                if regen_output_path:
                    regen_samples.append(sample)

            if cleaned:
                cleaned_samples.append(cleaned)

        except Exception as e:
            stats.errors.append(f"Error processing sample {sample.sample_id}: {e}")

    stats.output_retained = len(cleaned_samples)

    # Write outputs
    if output_path is None:
        output_path = input_path.parent / f"{input_path.stem}_cleaned.jsonl"

    write_jsonl(cleaned_samples, output_path)

    if regen_output_path and regen_samples:
        write_jsonl(regen_samples, regen_output_path)

    return stats


def main(args: list[str] | None = None) -> int:
    """CLI entry point for the data cleaner."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Clean training data for the nayru model"
    )
    parser.add_argument(
        "--input", "-i", required=True, help="Input JSONL file"
    )
    parser.add_argument(
        "--output", "-o", help="Output JSONL file (default: input_cleaned.jsonl)"
    )
    parser.add_argument(
        "--regen-output",
        help="Output file for samples needing regeneration (optional)",
    )
    parser.add_argument(
        "--min-output-length",
        type=int,
        default=100,
        help="Minimum output length to retain (default: 100)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose output"
    )

    parsed = parser.parse_args(args)

    input_path = Path(parsed.input).expanduser().resolve()
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1

    output_path = None
    if parsed.output:
        output_path = Path(parsed.output).expanduser().resolve()

    regen_output_path = None
    if parsed.regen_output:
        regen_output_path = Path(parsed.regen_output).expanduser().resolve()

    if parsed.verbose:
        print(f"Reading from: {input_path}")

    stats = clean_dataset(
        input_path=input_path,
        output_path=output_path,
        regen_output_path=regen_output_path,
        min_output_length=parsed.min_output_length,
    )

    print("\nCleaning Results:")
    print("-" * 40)
    print(stats.summary())

    if output_path:
        print(f"\nOutput written to: {output_path}")
    else:
        print(f"\nOutput written to: {input_path.parent / f'{input_path.stem}_cleaned.jsonl'}")

    if regen_output_path and stats.marked_for_regen > 0:
        print(f"Samples for regeneration: {regen_output_path}")

    if stats.errors:
        print("\nErrors encountered:")
        for error in stats.errors[:10]:
            print(f"  - {error}")
        if len(stats.errors) > 10:
            print(f"  ... and {len(stats.errors) - 10} more")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
