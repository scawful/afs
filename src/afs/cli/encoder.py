"""Encoder CLI commands: analyze, filter, dedupe, sample, pipeline, train."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def encoder_analyze_command(args: argparse.Namespace) -> int:
    """Analyze training data quality using encoder."""
    from ..tokenizer import ASMTokenizer
    from ..training import EncoderDataProcessor, EncoderConfig

    # Load tokenizer
    if args.tokenizer:
        tokenizer = ASMTokenizer.load(args.tokenizer)
    else:
        tokenizer = ASMTokenizer()

    config = EncoderConfig(
        min_instruction_tokens=getattr(args, "min_instruction_tokens", 5),
        min_output_tokens=getattr(args, "min_output_tokens", 10),
        max_unk_ratio=getattr(args, "max_unk_ratio", 0.1),
    )
    processor = EncoderDataProcessor(config=config, tokenizer=tokenizer)

    # Load samples
    input_path = Path(args.input)
    samples = []
    with open(input_path) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    print(f"Analyzing {len(samples)} samples...")

    # Analyze
    issues_count: dict[str, int] = {}
    valid_count = 0
    unk_ratios = []

    for sample in samples:
        analysis = processor.analyze_sample(sample)
        if analysis["is_valid"]:
            valid_count += 1
        unk_ratios.append(analysis["output_unk_ratio"])
        for issue in analysis["issues"]:
            issues_count[issue] = issues_count.get(issue, 0) + 1

    # Report
    print(f"\nResults:")
    print(f"  Valid samples: {valid_count}/{len(samples)} ({100*valid_count/len(samples):.1f}%)")
    print(f"  Mean UNK ratio: {sum(unk_ratios)/len(unk_ratios):.3f}")
    print(f"\nIssues:")
    for issue, count in sorted(issues_count.items(), key=lambda x: -x[1]):
        print(f"  {issue}: {count} ({100*count/len(samples):.1f}%)")

    return 0


def encoder_filter_command(args: argparse.Namespace) -> int:
    """Filter training data by quality."""
    from ..tokenizer import ASMTokenizer
    from ..training import EncoderDataProcessor, EncoderConfig

    # Load tokenizer
    if args.tokenizer:
        tokenizer = ASMTokenizer.load(args.tokenizer)
    else:
        tokenizer = ASMTokenizer()

    config = EncoderConfig(
        min_instruction_tokens=getattr(args, "min_instruction_tokens", 5),
        min_output_tokens=getattr(args, "min_output_tokens", 10),
        max_unk_ratio=args.max_unk_ratio,
    )
    processor = EncoderDataProcessor(config=config, tokenizer=tokenizer)

    # Load samples
    input_path = Path(args.input)
    samples = []
    with open(input_path) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    print(f"Filtering {len(samples)} samples...")

    # Filter
    passed, failed = processor.filter_by_quality(samples, verbose=True)

    # Save passed
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        for sample in passed:
            f.write(json.dumps(sample) + "\n")
    print(f"Passed: {len(passed)} -> {output_path}")

    # Save failed if requested
    if args.rejected:
        rejected_path = Path(args.rejected)
        with open(rejected_path, "w") as f:
            for sample in failed:
                f.write(json.dumps(sample) + "\n")
        print(f"Rejected: {len(failed)} -> {rejected_path}")

    return 0


def encoder_dedupe_command(args: argparse.Namespace) -> int:
    """Deduplicate training data using semantic similarity."""
    from ..tokenizer import ASMTokenizer
    from ..training import EncoderDataProcessor, EncoderConfig

    # Load tokenizer
    if args.tokenizer:
        tokenizer = ASMTokenizer.load(args.tokenizer)
    else:
        tokenizer = ASMTokenizer()

    config = EncoderConfig(similarity_threshold=args.threshold)
    processor = EncoderDataProcessor(config=config, tokenizer=tokenizer)

    # Load samples
    input_path = Path(args.input)
    samples = []
    with open(input_path) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    print(f"Deduplicating {len(samples)} samples (threshold={args.threshold})...")

    # Deduplicate
    deduped = processor.deduplicate(samples, field=args.field, keep=args.keep)

    # Save
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        for sample in deduped:
            f.write(json.dumps(sample) + "\n")

    removed = len(samples) - len(deduped)
    print(f"Kept: {len(deduped)}, Removed: {removed} ({100*removed/len(samples):.1f}% duplicates)")
    print(f"Saved to {output_path}")

    return 0


def encoder_sample_command(args: argparse.Namespace) -> int:
    """Sample diverse subset from training data."""
    from ..tokenizer import ASMTokenizer
    from ..training import EncoderDataProcessor, EncoderConfig

    # Load tokenizer
    if args.tokenizer:
        tokenizer = ASMTokenizer.load(args.tokenizer)
    else:
        tokenizer = ASMTokenizer()

    config = EncoderConfig(num_clusters=args.clusters)
    processor = EncoderDataProcessor(config=config, tokenizer=tokenizer)

    # Load samples
    input_path = Path(args.input)
    samples = []
    with open(input_path) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))

    print(f"Sampling {args.n} diverse samples from {len(samples)}...")

    # Sample
    diverse = processor.sample_diverse(samples, n_samples=args.n, field=args.field)

    # Save
    output_path = Path(args.output)
    with open(output_path, "w") as f:
        for sample in diverse:
            f.write(json.dumps(sample) + "\n")

    print(f"Sampled {len(diverse)} samples -> {output_path}")

    return 0


def encoder_pipeline_command(args: argparse.Namespace) -> int:
    """Run full preprocessing pipeline: expand vocab, filter, dedupe."""
    from ..tokenizer import ASMTokenizer
    from ..training import EncoderDataProcessor, EncoderConfig

    print("=" * 60)
    print("AFS Pretraining Data Pipeline")
    print("=" * 60)

    # Step 1: Load or create tokenizer
    print("\n[1/4] Loading tokenizer...")
    if args.tokenizer and Path(args.tokenizer).exists():
        tokenizer = ASMTokenizer.load(args.tokenizer)
        print(f"  Loaded tokenizer with {len(tokenizer)} tokens")
    else:
        tokenizer = ASMTokenizer()
        print(f"  Created new tokenizer with {len(tokenizer)} base tokens")

    # Load samples
    input_path = Path(args.input)
    samples = []
    with open(input_path) as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    print(f"  Loaded {len(samples)} samples from {input_path}")

    # Step 2: Expand vocabulary
    if not getattr(args, "skip_vocab_expansion", False):
        print("\n[2/4] Expanding vocabulary...")
        outputs = [s.get("output", "") for s in samples]
        added = tokenizer.train_on_corpus(outputs, min_frequency=args.min_frequency)
        print(f"  Added {added} tokens. New vocab: {len(tokenizer)}")

        # Save expanded tokenizer
        tokenizer_out = Path(args.output_dir) / "tokenizer"
        tokenizer.save(tokenizer_out)
        print(f"  Saved tokenizer to {tokenizer_out}")
    else:
        print("\n[2/4] Skipping vocabulary expansion")

    # Step 3: Filter by quality
    print("\n[3/4] Filtering by quality...")
    config = EncoderConfig(
        max_unk_ratio=args.max_unk_ratio,
    )
    processor = EncoderDataProcessor(config=config, tokenizer=tokenizer)

    passed, failed = processor.filter_by_quality(samples)
    print(f"  Passed: {len(passed)}, Failed: {len(failed)}")

    # Step 4: Deduplicate
    if not getattr(args, "skip_dedupe", False):
        print("\n[4/4] Deduplicating...")
        config = EncoderConfig(similarity_threshold=args.dedupe_threshold)
        processor = EncoderDataProcessor(config=config, tokenizer=tokenizer)
        final = processor.deduplicate(passed, keep="longest")
        removed = len(passed) - len(final)
        print(f"  Removed {removed} duplicates. Final: {len(final)}")
    else:
        print("\n[4/4] Skipping deduplication")
        final = passed

    # Save outputs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save cleaned data
    cleaned_path = output_dir / "train_cleaned.jsonl"
    with open(cleaned_path, "w") as f:
        for sample in final:
            f.write(json.dumps(sample) + "\n")

    # Save rejected data
    rejected_path = output_dir / "train_rejected.jsonl"
    with open(rejected_path, "w") as f:
        for sample in failed:
            f.write(json.dumps(sample) + "\n")

    print("\n" + "=" * 60)
    print("Pipeline Complete")
    print("=" * 60)
    print(f"  Input:    {len(samples)} samples")
    print(f"  Output:   {len(final)} samples ({100*len(final)/len(samples):.1f}% retained)")
    print(f"  Cleaned:  {cleaned_path}")
    print(f"  Rejected: {rejected_path}")
    print(f"  Tokenizer: {output_dir / 'tokenizer'}")

    return 0


def encoder_train_command(args: argparse.Namespace) -> int:
    """Train ASM encoder model."""
    from ..training import ASMTrainer, ASMTrainerConfig
    from ..tokenizer import ASMTokenizer

    tokenizer_path = Path(args.tokenizer)
    output_path = Path(args.output)
    train_path = Path(args.train)

    # Load tokenizer
    print(f"Loading tokenizer from {tokenizer_path}")
    tokenizer = ASMTokenizer.load(tokenizer_path)

    # Load training texts
    print(f"Loading training data from {train_path}")
    train_texts = []
    with open(train_path) as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                if "output" in data:
                    train_texts.append(data["output"])
                elif "text" in data:
                    train_texts.append(data["text"])

    print(f"  Loaded {len(train_texts)} samples")

    # Load validation texts if provided
    val_texts = None
    if args.val:
        val_path = Path(args.val)
        print(f"Loading validation data from {val_path}")
        val_texts = []
        with open(val_path) as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    if "output" in data:
                        val_texts.append(data["output"])
                    elif "text" in data:
                        val_texts.append(data["text"])
        print(f"  Loaded {len(val_texts)} validation samples")

    # Configure training
    config = ASMTrainerConfig(
        output_dir=output_path,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
    )

    # Train
    print(f"\nTraining encoder with config:")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Layers: {config.num_layers}")
    print(f"  Heads: {config.num_heads}")

    trainer = ASMTrainer(tokenizer=tokenizer, config=config)
    metrics = trainer.train(train_texts, val_texts)

    print(f"\nTraining complete!")
    print(f"  Final loss: {metrics.get('train_loss', 'N/A')}")
    print(f"  Model saved to: {output_path}")

    return 0


def register_parsers(subparsers: argparse._SubParsersAction) -> None:
    """Register encoder command parsers."""
    enc_parser = subparsers.add_parser(
        "encoder", help="Encoder-based data preprocessing for pretraining."
    )
    enc_sub = enc_parser.add_subparsers(dest="encoder_command")

    # encoder analyze
    enc_analyze = enc_sub.add_parser(
        "analyze", help="Analyze training data quality."
    )
    enc_analyze.add_argument("--input", required=True, help="Input JSONL file.")
    enc_analyze.add_argument(
        "--tokenizer", help="Path to tokenizer (uses default if not specified)."
    )
    enc_analyze.add_argument(
        "--field", default="output", help="Field to analyze (default: output)."
    )
    enc_analyze.add_argument(
        "--min-instruction-tokens",
        type=int,
        default=5,
        help="Minimum instruction tokens (default: 5).",
    )
    enc_analyze.add_argument(
        "--min-output-tokens",
        type=int,
        default=10,
        help="Minimum output tokens (default: 10).",
    )
    enc_analyze.add_argument(
        "--max-unk-ratio",
        type=float,
        default=0.1,
        help="Maximum unknown token ratio (default: 0.1).",
    )
    enc_analyze.add_argument(
        "--detailed", action="store_true", help="Show detailed statistics."
    )
    enc_analyze.set_defaults(func=encoder_analyze_command)

    # encoder filter
    enc_filter = enc_sub.add_parser("filter", help="Filter training data by quality.")
    enc_filter.add_argument("--input", required=True, help="Input JSONL file.")
    enc_filter.add_argument("--output", required=True, help="Output JSONL file.")
    enc_filter.add_argument("--rejected", help="Output file for rejected samples.")
    enc_filter.add_argument(
        "--tokenizer", help="Path to tokenizer."
    )
    enc_filter.add_argument(
        "--min-tokens", type=int, help="Minimum tokens to keep sample."
    )
    enc_filter.add_argument(
        "--max-tokens", type=int, help="Maximum tokens to keep sample."
    )
    enc_filter.add_argument(
        "--max-unk-ratio",
        type=float,
        default=0.1,
        help="Maximum unknown token ratio (default: 0.1).",
    )
    enc_filter.set_defaults(func=encoder_filter_command)

    # encoder dedupe
    enc_dedupe = enc_sub.add_parser(
        "dedupe", help="Deduplicate training data using semantic similarity."
    )
    enc_dedupe.add_argument("--input", required=True, help="Input JSONL file.")
    enc_dedupe.add_argument("--output", required=True, help="Output JSONL file.")
    enc_dedupe.add_argument(
        "--field", default="output", help="Field to use for similarity (default: output)."
    )
    enc_dedupe.add_argument(
        "--threshold",
        type=float,
        default=0.95,
        help="Similarity threshold for deduplication (default: 0.95).",
    )
    enc_dedupe.add_argument(
        "--keep",
        choices=["first", "longest", "shortest"],
        default="longest",
        help="Which duplicate to keep (default: longest).",
    )
    enc_dedupe.add_argument(
        "--tokenizer", help="Path to tokenizer."
    )
    enc_dedupe.set_defaults(func=encoder_dedupe_command)

    # encoder sample
    enc_sample = enc_sub.add_parser(
        "sample", help="Sample diverse subset from training data."
    )
    enc_sample.add_argument("--input", required=True, help="Input JSONL file.")
    enc_sample.add_argument("--output", required=True, help="Output JSONL file.")
    enc_sample.add_argument(
        "--n", type=int, required=True, help="Number of samples to select."
    )
    enc_sample.add_argument(
        "--field", default="output", help="Field to use for diversity (default: output)."
    )
    enc_sample.add_argument(
        "--clusters",
        type=int,
        default=50,
        help="Number of clusters for diversity sampling (default: 50).",
    )
    enc_sample.add_argument(
        "--tokenizer", help="Path to tokenizer."
    )
    enc_sample.add_argument(
        "--seed", type=int, default=42, help="Random seed (default: 42)."
    )
    enc_sample.set_defaults(func=encoder_sample_command)

    # encoder pipeline
    enc_pipeline = enc_sub.add_parser(
        "pipeline", help="Run full preprocessing pipeline: expand vocab, filter, dedupe."
    )
    enc_pipeline.add_argument("--input", required=True, help="Input JSONL file.")
    enc_pipeline.add_argument(
        "--output-dir", required=True, help="Output directory."
    )
    enc_pipeline.add_argument(
        "--tokenizer", help="Existing tokenizer path (creates new if not specified)."
    )
    enc_pipeline.add_argument(
        "--skip-vocab-expansion",
        action="store_true",
        help="Skip vocabulary expansion step.",
    )
    enc_pipeline.add_argument(
        "--min-frequency",
        type=int,
        default=2,
        help="Minimum frequency for vocab expansion (default: 2).",
    )
    enc_pipeline.add_argument(
        "--max-unk-ratio",
        type=float,
        default=0.1,
        help="Maximum unknown token ratio (default: 0.1).",
    )
    enc_pipeline.add_argument(
        "--skip-dedupe",
        action="store_true",
        help="Skip deduplication step.",
    )
    enc_pipeline.add_argument(
        "--dedupe-threshold",
        type=float,
        default=0.95,
        help="Similarity threshold for deduplication (default: 0.95).",
    )
    enc_pipeline.set_defaults(func=encoder_pipeline_command)

    # encoder train
    enc_train = enc_sub.add_parser("train", help="Train ASM encoder model.")
    enc_train.add_argument(
        "--tokenizer", required=True, help="Path to tokenizer."
    )
    enc_train.add_argument(
        "--train", required=True, help="Training data JSONL."
    )
    enc_train.add_argument(
        "--output", required=True, help="Output directory for model."
    )
    enc_train.add_argument("--val", help="Validation data JSONL.")
    enc_train.add_argument(
        "--epochs", type=int, default=10, help="Training epochs (default: 10)."
    )
    enc_train.add_argument(
        "--batch-size", type=int, default=32, help="Batch size (default: 32)."
    )
    enc_train.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate (default: 1e-4).",
    )
    enc_train.add_argument(
        "--hidden-size", type=int, default=256, help="Hidden size (default: 256)."
    )
    enc_train.add_argument(
        "--num-layers", type=int, default=4, help="Number of layers (default: 4)."
    )
    enc_train.add_argument(
        "--num-heads", type=int, default=4, help="Number of attention heads (default: 4)."
    )
    enc_train.set_defaults(func=encoder_train_command)
