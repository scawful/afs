"""Training CLI commands: training, discriminator, tokenizer, encoder."""

from __future__ import annotations

import argparse
from pathlib import Path


# =============================================================================
# Training Commands
# =============================================================================


def training_prepare_command(args: argparse.Namespace) -> int:
    """Split dataset into train/val/test sets."""
    from ..training import split_dataset

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        return 1

    output_dir = Path(args.output).expanduser().resolve()

    result = split_dataset(
        input_path=input_path,
        output_dir=output_dir,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=1.0 - args.train_ratio - args.val_ratio,
        stratify_by=args.stratify_by if not args.no_stratify else None,
        shuffle=not args.no_shuffle,
        seed=args.seed,
    )

    print(result.summary())
    print(f"\nOutput directory: {output_dir}")
    return 0


def training_convert_command(args: argparse.Namespace) -> int:
    """Convert training data to framework format."""
    from ..training import get_converter

    input_path = Path(args.input).expanduser().resolve()
    if not input_path.exists():
        print(f"Input file not found: {input_path}")
        return 1

    if args.output:
        output_path = Path(args.output).expanduser().resolve()
    else:
        output_path = input_path.parent / f"{input_path.stem}_{args.format}.jsonl"

    converter = get_converter(
        format_name=args.format,
        include_cot=not args.no_cot,
        cot_mode=args.cot_mode,
    )

    count = converter.convert_file(input_path, output_path)
    print(f"Converted {count} samples to {args.format} format")
    print(f"Output: {output_path}")
    return 0


def training_registry_list_command(args: argparse.Namespace) -> int:
    """List experiments in registry."""
    from ..training import ModelRegistry

    registry = ModelRegistry()
    experiments = registry.list(
        status=args.status,
        ab_group=args.ab_group,
        framework=args.framework,
    )

    if not experiments:
        print("No experiments found.")
        return 0

    print(f"Found {len(experiments)} experiments:\n")
    for exp in experiments:
        loss_str = f"loss={exp.metrics.final_loss:.4f}" if exp.metrics.final_loss else ""
        print(f"  {exp.experiment_id}: {exp.run_name}")
        print(f"    Model: {exp.base_model}")
        print(f"    Status: {exp.status} {loss_str}")
        if exp.ab_group:
            print(f"    A/B Group: {exp.ab_group} ({exp.ab_variant or 'unassigned'})")
        print()

    return 0


def training_registry_create_command(args: argparse.Namespace) -> int:
    """Create a new experiment."""
    from ..training import ModelRegistry

    registry = ModelRegistry()
    exp = registry.create_experiment(
        run_name=args.name,
        base_model=args.model,
        framework=args.framework,
        dataset_path=args.dataset,
        ab_group=args.ab_group,
        ab_variant=args.ab_variant,
        tags=args.tag or [],
        notes=args.notes or "",
    )

    print(f"Created experiment: {exp.experiment_id}")
    print(f"  Run name: {exp.run_name}")
    print(f"  Base model: {exp.base_model}")
    print(f"  Framework: {exp.framework}")
    return 0


def training_registry_compare_command(args: argparse.Namespace) -> int:
    """Compare experiments."""
    from ..training import ModelRegistry

    registry = ModelRegistry()
    results = registry.compare(args.experiments)

    if not results:
        print("No experiments found for comparison.")
        return 1

    # Print comparison table
    headers = list(results[0].keys())
    print(" | ".join(f"{h:15}" for h in headers))
    print("-" * (17 * len(headers)))

    for row in results:
        values = []
        for h in headers:
            v = row.get(h)
            if isinstance(v, float):
                values.append(f"{v:.4f}")
            elif v is None:
                values.append("-")
            else:
                values.append(str(v)[:15])
        print(" | ".join(f"{v:15}" for v in values))

    return 0


# =============================================================================
# Discriminator Commands
# =============================================================================


def discriminator_data_command(args: argparse.Namespace) -> int:
    """Create ELECTRA training data from assembly sources."""
    from ..discriminator import create_training_data

    sources = [Path(s).expanduser() for s in args.sources]
    output = Path(args.output).expanduser()

    print("Creating ELECTRA training data...")
    print(f"  Sources: {len(sources)} paths")
    print(f"  Fake ratio: {args.fake_ratio}")

    dataset = create_training_data(
        real_sources=sources,
        fake_ratio=args.fake_ratio,
        min_lines=args.min_lines,
        max_lines=args.max_lines,
    )

    dataset.to_jsonl(output)
    stats = dataset.stats()

    print("\nResults:")
    print(f"  Total: {stats['total']}")
    print(f"  Real: {stats['real']}")
    print(f"  Fake: {stats['fake']}")
    print(f"  Output: {output}")

    return 0


def discriminator_train_command(args: argparse.Namespace) -> int:
    """Train ASM-ELECTRA discriminator."""
    from ..discriminator import ASMElectra, ElectraConfig, ElectraDataset

    input_path = Path(args.input).expanduser()
    output_dir = Path(args.output).expanduser()
    val_path = Path(args.val).expanduser() if args.val else None

    config = ElectraConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        output_dir=output_dir,
    )

    print("Training ASM-ELECTRA...")
    print(f"  Input: {input_path}")
    print(f"  Epochs: {config.epochs}")
    print(f"  Batch size: {config.batch_size}")

    # Load data
    train_dataset = ElectraDataset.from_jsonl(input_path)
    train_data = train_dataset.to_hf_format()

    val_data = None
    if val_path:
        val_dataset = ElectraDataset.from_jsonl(val_path)
        val_data = val_dataset.to_hf_format()

    # Train
    electra = ASMElectra(config=config)
    metrics = electra.train(train_data, val_data)

    print("\nTraining complete:")
    print(f"  Loss: {metrics['train_loss']:.4f}")
    print(f"  Steps: {metrics['steps']}")
    print(f"  Model saved: {output_dir / 'final'}")

    return 0


def discriminator_filter_command(args: argparse.Namespace) -> int:
    """Filter training data using trained discriminator."""
    from ..discriminator import SampleFilter, FilterConfig

    model_path = Path(args.model).expanduser()
    input_path = Path(args.input).expanduser()
    output_path = Path(args.output).expanduser()
    rejected_path = Path(args.rejected).expanduser() if args.rejected else None

    config = FilterConfig(min_score=args.min_score)

    print("Filtering training data...")
    print(f"  Model: {model_path}")
    print(f"  Min score: {config.min_score}")

    sample_filter = SampleFilter(model_path=model_path, config=config)
    result = sample_filter.filter_jsonl(input_path, output_path, rejected_path)

    print(f"\n{result}")
    print("\nScore distribution:")
    for bucket, count in result.score_distribution.items():
        print(f"  {bucket}: {count}")

    return 0


def discriminator_score_command(args: argparse.Namespace) -> int:
    """Score assembly code quality."""
    from ..discriminator import ASMElectra

    model_path = Path(args.model).expanduser()

    if args.file:
        text = Path(args.file).expanduser().read_text()
    elif args.text:
        text = args.text
    else:
        print("Error: must provide --text or --file")
        return 1

    electra = ASMElectra(model_path=model_path)
    score = electra.score(text)
    prediction, confidence = electra.predict(text)

    label = "REAL" if prediction == 0 else "FAKE"
    print(f"Score: {score:.4f}")
    print(f"Prediction: {label} (confidence: {confidence:.2%})")

    return 0


# =============================================================================
# Parser Registration
# =============================================================================


def register_parsers(subparsers: argparse._SubParsersAction) -> None:
    """Register training and discriminator command parsers."""

    # =========================================================================
    # Training
    # =========================================================================
    training_parser = subparsers.add_parser("training", help="Training data utilities.")
    training_sub = training_parser.add_subparsers(dest="training_command")

    # training prepare
    train_prepare = training_sub.add_parser("prepare", help="Prepare train/val/test split.")
    train_prepare.add_argument("--input", required=True, help="Input JSONL file.")
    train_prepare.add_argument("--output", required=True, help="Output directory.")
    train_prepare.add_argument("--train-ratio", type=float, default=0.8, help="Train ratio.")
    train_prepare.add_argument("--val-ratio", type=float, default=0.1, help="Validation ratio.")
    train_prepare.add_argument("--no-stratify", action="store_true", help="Disable stratification.")
    train_prepare.add_argument("--stratify-by", default="domain", help="Field to stratify by.")
    train_prepare.add_argument("--no-shuffle", action="store_true", help="Don't shuffle data.")
    train_prepare.add_argument("--seed", type=int, default=42, help="Random seed.")
    train_prepare.set_defaults(func=training_prepare_command)

    # training convert
    train_convert = training_sub.add_parser("convert", help="Convert to training format.")
    train_convert.add_argument("--input", required=True, help="Input JSONL file.")
    train_convert.add_argument("--output", help="Output path.")
    train_convert.add_argument(
        "--format", default="alpaca",
        choices=["alpaca", "sharegpt", "openai"],
        help="Format.",
    )
    train_convert.add_argument("--no-cot", action="store_true", help="Exclude CoT.")
    train_convert.add_argument(
        "--cot-mode", default="separate",
        choices=["separate", "embedded"],
        help="CoT mode.",
    )
    train_convert.set_defaults(func=training_convert_command)

    # training registry-list
    train_reg_list = training_sub.add_parser("registry-list", help="List experiments.")
    train_reg_list.add_argument("--status", help="Filter by status.")
    train_reg_list.add_argument("--ab-group", help="Filter by A/B group.")
    train_reg_list.add_argument("--framework", help="Filter by framework.")
    train_reg_list.set_defaults(func=training_registry_list_command)

    # training registry-create
    train_reg_create = training_sub.add_parser("registry-create", help="Create experiment.")
    train_reg_create.add_argument("--name", required=True, help="Run name.")
    train_reg_create.add_argument("--model", required=True, help="Base model.")
    train_reg_create.add_argument("--framework", default="mlx", help="Training framework.")
    train_reg_create.add_argument("--dataset", help="Dataset path.")
    train_reg_create.add_argument("--ab-group", help="A/B test group.")
    train_reg_create.add_argument("--ab-variant", help="A/B variant (a or b).")
    train_reg_create.add_argument("--tag", action="append", help="Experiment tags.")
    train_reg_create.add_argument("--notes", help="Experiment notes.")
    train_reg_create.set_defaults(func=training_registry_create_command)

    # training registry-compare
    train_reg_compare = training_sub.add_parser("registry-compare", help="Compare experiments.")
    train_reg_compare.add_argument("experiments", nargs="+", help="Experiment IDs to compare.")
    train_reg_compare.set_defaults(func=training_registry_compare_command)

    # =========================================================================
    # Discriminator
    # =========================================================================
    disc_parser = subparsers.add_parser("discriminator", help="ELECTRA discriminator tools.")
    disc_sub = disc_parser.add_subparsers(dest="discriminator_command")

    # discriminator data
    disc_data = disc_sub.add_parser("data", help="Create ELECTRA training data.")
    disc_data.add_argument("--sources", nargs="+", required=True, help="Source paths.")
    disc_data.add_argument("--output", required=True, help="Output JSONL path.")
    disc_data.add_argument("--fake-ratio", type=float, default=0.5, help="Fake sample ratio.")
    disc_data.add_argument("--min-lines", type=int, default=3, help="Min lines per sample.")
    disc_data.add_argument("--max-lines", type=int, default=50, help="Max lines per sample.")
    disc_data.set_defaults(func=discriminator_data_command)

    # discriminator train
    disc_train = disc_sub.add_parser("train", help="Train ELECTRA discriminator.")
    disc_train.add_argument("--input", required=True, help="Training data JSONL.")
    disc_train.add_argument("--output", required=True, help="Output directory.")
    disc_train.add_argument("--val", help="Validation data JSONL.")
    disc_train.add_argument("--epochs", type=int, default=3, help="Training epochs.")
    disc_train.add_argument("--batch-size", type=int, default=32, help="Batch size.")
    disc_train.add_argument("--learning-rate", type=float, default=5e-5, help="Learning rate.")
    disc_train.set_defaults(func=discriminator_train_command)

    # discriminator filter
    disc_filter = disc_sub.add_parser("filter", help="Filter data using discriminator.")
    disc_filter.add_argument("--model", required=True, help="Discriminator model path.")
    disc_filter.add_argument("--input", required=True, help="Input JSONL.")
    disc_filter.add_argument("--output", required=True, help="Output JSONL (passed samples).")
    disc_filter.add_argument("--rejected", help="Output JSONL for rejected samples.")
    disc_filter.add_argument("--min-score", type=float, default=0.5, help="Minimum score.")
    disc_filter.set_defaults(func=discriminator_filter_command)

    # discriminator score
    disc_score = disc_sub.add_parser("score", help="Score assembly code.")
    disc_score.add_argument("--model", required=True, help="Discriminator model path.")
    disc_score.add_argument("--text", help="Text to score.")
    disc_score.add_argument("--file", help="File to score.")
    disc_score.set_defaults(func=discriminator_score_command)
