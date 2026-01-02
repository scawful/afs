"""Generators CLI commands: augmentation, CoT, cleaning, validation."""

from __future__ import annotations

import argparse
from pathlib import Path


def generators_asm_augment_command(args: argparse.Namespace) -> int:
    """Augment ASM training samples via paraphrasing."""
    from ..generators import AsmAugmentConfig, AsmAugmentGenerator, write_jsonl

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        return 1

    config = AsmAugmentConfig(
        paraphrase_count=args.paraphrase_count,
        include_original=args.include_original,
        shuffle_output=not args.no_shuffle,
        min_instruction_len=args.min_len,
        random_seed=args.seed,
    )

    generator = AsmAugmentGenerator(input_path=input_path, config=config)
    result = generator.generate()

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    else:
        output_path = input_path.parent / f"{input_path.stem}_augmented.jsonl"

    count = write_jsonl(result.samples, output_path)

    print(f"Source samples: {result.source_count}")
    print(f"Generated samples: {result.total}")
    print(f"Skipped: {result.skipped}")
    if result.errors:
        print(f"Errors: {len(result.errors)}")
        for error in result.errors[:5]:
            print(f"  - {error}")
    print(f"Output: {output_path}")
    print(f"Wrote {count} samples")
    return 0


def generators_cot_command(args: argparse.Namespace) -> int:
    """Generate Chain of Thought reasoning for samples."""
    from ..generators.cot import CotConfig, CotGenerator, CotFormat
    from ..generators import write_jsonl

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        return 1

    try:
        cot_format = CotFormat(args.format)
    except ValueError:
        print(f"Invalid format: {args.format}")
        print(f"Valid formats: {', '.join(f.value for f in CotFormat)}")
        return 1

    config = CotConfig(
        api_provider=args.provider,
        model_name=args.model,
        cot_format=cot_format,
        requests_per_minute=args.rpm,
        batch_size=args.batch_size,
        temperature=args.temperature,
    )

    generator = CotGenerator(input_path=input_path, config=config)

    if args.limit:
        print(f"Limiting to {args.limit} samples")

    result = generator.generate()

    if args.limit and len(result.samples) > args.limit:
        result.samples = result.samples[: args.limit]

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    else:
        output_path = input_path.parent / f"{input_path.stem}_cot.jsonl"

    count = write_jsonl(result.samples, output_path)

    print("\nResults:")
    print(f"  Source samples: {result.source_count}")
    print(f"  Generated CoT: {result.total}")
    print(f"  Skipped: {result.skipped}")
    if result.errors:
        print(f"  Errors: {len(result.errors)}")
        for error in result.errors[:5]:
            print(f"    - {error}")
    print(f"  Output: {output_path}")
    print(f"  Wrote {count} samples")
    return 0


def generators_clean_command(args: argparse.Namespace) -> int:
    """Clean training data by fixing malformed samples."""
    from ..generators.data_cleaner import clean_dataset

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        return 1

    output_path = None
    if args.output:
        output_path = Path(args.output).expanduser().resolve()

    regen_output_path = None
    if args.regen_output:
        regen_output_path = Path(args.regen_output).expanduser().resolve()

    stats = clean_dataset(
        input_path=input_path,
        output_path=output_path,
        regen_output_path=regen_output_path,
        min_output_length=args.min_output_length,
    )

    print("\nCleaning Results:")
    print("-" * 40)
    print(stats.summary())

    actual_output = output_path or input_path.parent / f"{input_path.stem}_cleaned.jsonl"
    print(f"\nOutput: {actual_output}")

    if regen_output_path and stats.marked_for_regen > 0:
        print(f"Samples for regeneration: {regen_output_path}")

    if stats.errors:
        print("\nErrors:")
        for error in stats.errors[:5]:
            print(f"  - {error}")
        if len(stats.errors) > 5:
            print(f"  ... and {len(stats.errors) - 5} more")

    return 0


def generators_validate_command(args: argparse.Namespace) -> int:
    """Validate assembly code in training samples using asar."""
    from ..generators.asar_validator import (
        AsarValidatorConfig,
        check_asar_available,
        validate_training_data,
    )

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        return 1

    if not check_asar_available():
        print("Error: asar not found. Install asar SNES assembler and ensure it's in PATH.")
        print("  - macOS: brew install asar (if available) or build from source")
        print("  - Linux: build from source at https://github.com/RPGHacker/asar")
        print("  - You can also specify --asar-path to point to the executable")
        return 1

    # Determine output paths
    if args.output:
        output_pass_path = Path(args.output).expanduser().resolve()
    else:
        output_pass_path = input_path.parent / f"{input_path.stem}_valid.jsonl"

    if args.invalid_output:
        output_fail_path = Path(args.invalid_output).expanduser().resolve()
    else:
        output_fail_path = input_path.parent / f"{input_path.stem}_invalid.jsonl"

    # Build config
    config = AsarValidatorConfig(
        asar_path=args.asar_path if hasattr(args, 'asar_path') else None,
        include_alttp_context=not getattr(args, 'no_alttp_context', False),
        min_output_length=getattr(args, 'min_length', 10),
        keep_temp_files=getattr(args, 'keep_temp', False),
        timeout=args.timeout,
    )

    if hasattr(args, 'include_path') and args.include_path:
        config.include_paths = [Path(p).expanduser().resolve() for p in args.include_path]

    if hasattr(args, 'skip_domain') and args.skip_domain:
        config.skip_domains = list(args.skip_domain)

    print(f"Input: {input_path}")
    print(f"Pass output: {output_pass_path}")
    print(f"Fail output: {output_fail_path}")
    print()

    stats = validate_training_data(
        input_path=input_path,
        output_pass_path=output_pass_path,
        output_fail_path=output_fail_path,
        config=config,
        verbose=True,
    )

    print()
    print(f"Wrote {stats.passed} samples to: {output_pass_path}")
    print(f"Wrote {stats.failed + stats.skipped} samples to: {output_fail_path}")

    if stats.errors:
        print("\nErrors encountered:")
        for error in stats.errors[:10]:
            print(f"  - {error}")
        if len(stats.errors) > 10:
            print(f"  ... and {len(stats.errors) - 10} more")

    return 0


def register_parsers(subparsers: argparse._SubParsersAction) -> None:
    """Register generators command parsers."""
    generators_parser = subparsers.add_parser(
        "generators", help="Training data generators."
    )
    generators_sub = generators_parser.add_subparsers(dest="generators_command")

    # asm-augment
    gen_asm_augment = generators_sub.add_parser(
        "asm-augment", help="Augment ASM training samples via paraphrasing."
    )
    gen_asm_augment.add_argument(
        "--input", required=True, help="Source JSONL file with training samples."
    )
    gen_asm_augment.add_argument(
        "--output", help="Output JSONL path (default: input_augmented.jsonl)."
    )
    gen_asm_augment.add_argument(
        "--paraphrase-count",
        type=int,
        default=5,
        help="Number of paraphrases per sample (default: 5).",
    )
    gen_asm_augment.add_argument(
        "--no-original",
        action="store_false",
        dest="include_original",
        help="Exclude original samples from output.",
    )
    gen_asm_augment.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Don't shuffle output samples.",
    )
    gen_asm_augment.add_argument(
        "--min-len",
        type=int,
        default=10,
        help="Minimum instruction length to augment (default: 10).",
    )
    gen_asm_augment.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility.",
    )
    gen_asm_augment.set_defaults(func=generators_asm_augment_command)

    # cot
    gen_cot = generators_sub.add_parser(
        "cot", help="Generate Chain of Thought reasoning for samples."
    )
    gen_cot.add_argument(
        "--input", required=True, help="Source JSONL file with training samples."
    )
    gen_cot.add_argument(
        "--output", help="Output JSONL path (default: input_cot.jsonl)."
    )
    gen_cot.add_argument(
        "--provider",
        default="gemini",
        choices=["gemini", "claude", "openai"],
        help="LLM provider for CoT generation (default: gemini).",
    )
    gen_cot.add_argument(
        "--model",
        default="gemini-2.0-flash-exp",
        help="Model name (default: gemini-2.0-flash-exp).",
    )
    gen_cot.add_argument(
        "--format",
        default="separate",
        choices=["separate", "embedded", "special_tokens"],
        help="CoT output format (default: separate).",
    )
    gen_cot.add_argument(
        "--rpm",
        type=int,
        default=60,
        help="Requests per minute rate limit (default: 60).",
    )
    gen_cot.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for processing (default: 10).",
    )
    gen_cot.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="LLM temperature (default: 0.7).",
    )
    gen_cot.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples to process (for testing).",
    )
    gen_cot.set_defaults(func=generators_cot_command)

    # clean
    gen_clean = generators_sub.add_parser(
        "clean", help="Clean training data by fixing malformed samples."
    )
    gen_clean.add_argument(
        "--input", required=True, help="Source JSONL file with training samples."
    )
    gen_clean.add_argument(
        "--output", help="Output JSONL path (default: input_cleaned.jsonl)."
    )
    gen_clean.add_argument(
        "--regen-output",
        help="Output file for samples needing regeneration (optional).",
    )
    gen_clean.add_argument(
        "--min-output-length",
        type=int,
        default=100,
        help="Minimum output length to retain sample (default: 100).",
    )
    gen_clean.set_defaults(func=generators_clean_command)

    # validate
    gen_validate = generators_sub.add_parser(
        "validate", help="Validate assembly code samples using asar SNES assembler."
    )
    gen_validate.add_argument(
        "--input", required=True, help="Source JSONL file with training samples."
    )
    gen_validate.add_argument(
        "--output", help="Output JSONL path for valid samples."
    )
    gen_validate.add_argument(
        "--invalid-output", help="Output JSONL path for invalid samples."
    )
    gen_validate.add_argument(
        "--stats-output", help="Output JSON path for validation statistics."
    )
    gen_validate.add_argument(
        "--base-rom", help="Base ROM path for asar validation."
    )
    gen_validate.add_argument(
        "--timeout",
        type=float,
        default=5.0,
        help="Timeout per sample in seconds (default: 5.0).",
    )
    gen_validate.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers (default: CPU count).",
    )
    gen_validate.set_defaults(func=generators_validate_command)
