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

    if args.train_ratio <= 0 or args.val_ratio < 0:
        print("Train ratio must be > 0 and validation ratio must be >= 0.")
        return 1
    if args.train_ratio + args.val_ratio >= 1.0:
        print("Train ratio + validation ratio must be < 1.0.")
        return 1

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


def training_memory_export_command(args: argparse.Namespace) -> int:
    """Export memory entries to TrainingSample JSONL."""
    from ..config import load_config_model
    from ..training import export_memory_to_dataset

    config = load_config_model(config_path=Path(args.config) if args.config else None)
    memory_cfg = config.memory_export
    context_root = (
        Path(args.context_root).expanduser().resolve()
        if args.context_root
        else config.general.context_root
    )
    memory_root = (
        Path(args.memory_root).expanduser().resolve()
        if args.memory_root
        else (Path(context_root) / "memory")
    )
    output_path = Path(args.output).expanduser().resolve()

    require_quality = memory_cfg.require_quality
    if args.require_quality is True:
        require_quality = True
    elif args.no_require_quality is True:
        require_quality = False

    result = export_memory_to_dataset(
        memory_root,
        output_path,
        default_domain=args.domain,
        allow_raw=args.allow_raw or memory_cfg.allow_raw,
        allow_raw_tags=args.allow_raw_tag or memory_cfg.allow_raw_tags,
        default_instruction=args.default_instruction or memory_cfg.default_instruction,
        include_tags=args.include_tag,
        exclude_tags=args.exclude_tag,
        limit=args.limit if args.limit is not None else (memory_cfg.limit or None),
        require_quality=require_quality,
        min_quality_score=(
            args.min_quality_score
            if args.min_quality_score is not None
            else memory_cfg.min_quality_score
        ),
        score_profile=args.score_profile or memory_cfg.score_profile,
        enable_asar=args.enable_asar or memory_cfg.enable_asar,
        redact=not args.no_redact,
    )

    print(result.summary())
    print(f"output: {output_path}")
    return 0


def training_history_export_command(args: argparse.Namespace) -> int:
    """Export history events to TrainingSample JSONL."""
    from ..config import load_config_model
    from ..training import export_history_to_dataset

    config = load_config_model(config_path=Path(args.config) if args.config else None)
    history_root = (
        Path(args.history_root).expanduser().resolve()
        if args.history_root
        else (config.general.context_root / "history")
    )
    output_path = Path(args.output).expanduser().resolve()

    require_quality = config.memory_export.require_quality
    if args.require_quality is True:
        require_quality = True
    elif args.no_require_quality is True:
        require_quality = False

    if args.event_type:
        event_types = args.event_type
    else:
        event_types = ["model"]
        if args.include_tools:
            event_types.append("tool")
        if args.include_fs:
            event_types.append("fs")
        if args.include_cli:
            event_types.append("cli")

    result = export_history_to_dataset(
        history_root,
        output_path,
        event_types=event_types,
        include_tools=args.include_tools,
        include_fs=args.include_fs,
        include_cli=args.include_cli,
        default_domain=args.domain,
        tool_domain=args.tool_domain,
        limit=args.limit,
        require_quality=require_quality,
        min_quality_score=(
            args.min_quality_score
            if args.min_quality_score is not None
            else config.memory_export.min_quality_score
        ),
        score_profile=args.score_profile or config.memory_export.score_profile,
        enable_asar=args.enable_asar or config.memory_export.enable_asar,
        redact=not args.no_redact,
    )

    print(result.summary())
    print(f"output: {output_path}")
    return 0


# =============================================================================
# Antigravity Export
# =============================================================================


def training_antigravity_export_command(args: argparse.Namespace) -> int:
    """Export Antigravity logs to TrainingSample JSONL."""
    from ..config import load_config_model
    from ..training import export_antigravity_to_dataset
    from ..training.antigravity_export import DEFAULT_ANTIGRAVITY_DB

    config = load_config_model(config_path=Path(args.config) if args.config else None)
    output_path = Path(args.output).expanduser().resolve()
    db_path = (
        Path(args.db_path).expanduser().resolve()
        if args.db_path
        else None
    )

    require_quality = config.memory_export.require_quality
    if args.require_quality is True:
        require_quality = True
    elif args.no_require_quality is True:
        require_quality = False

    result = export_antigravity_to_dataset(
        db_path=db_path or DEFAULT_ANTIGRAVITY_DB,
        output_path=output_path,
        state_keys=args.state_key,
        default_domain=args.domain,
        limit=args.limit,
        include_paths_content=args.include_paths_content,
        max_path_chars=args.max_path_chars,
        require_quality=require_quality,
        min_quality_score=(
            args.min_quality_score
            if args.min_quality_score is not None
            else config.memory_export.min_quality_score
        ),
        score_profile=args.score_profile or config.memory_export.score_profile,
        enable_asar=args.enable_asar or config.memory_export.enable_asar,
        redact=not args.no_redact,
    )

    print(result.summary())
    print(f"output: {output_path}")
    return 0


# =============================================================================
# Gemini Export
# =============================================================================


def training_gemini_export_command(args: argparse.Namespace) -> int:
    """Export Gemini CLI logs to TrainingSample JSONL."""
    from ..config import load_config_model
    from ..training import export_gemini_logs_to_dataset

    config = load_config_model(config_path=Path(args.config) if args.config else None)
    output_path = Path(args.output).expanduser().resolve()
    roots = [Path(root).expanduser().resolve() for root in (args.root or [])]
    scan_roots = [Path(root).expanduser().resolve() for root in (args.scan_root or [])]

    require_quality = config.memory_export.require_quality
    if args.require_quality is True:
        require_quality = True
    elif args.no_require_quality is True:
        require_quality = False

    result = export_gemini_logs_to_dataset(
        roots if roots else None,
        output_path=output_path,
        scan_roots=scan_roots if scan_roots else None,
        max_scan_depth=args.max_scan_depth,
        default_domain=args.domain,
        include_tools=not args.no_tools,
        include_thoughts=args.include_thoughts,
        max_tool_output_chars=args.max_tool_output_chars,
        limit=args.limit,
        require_quality=require_quality,
        min_quality_score=(
            args.min_quality_score
            if args.min_quality_score is not None
            else config.memory_export.min_quality_score
        ),
        score_profile=args.score_profile or config.memory_export.score_profile,
        enable_asar=args.enable_asar or config.memory_export.enable_asar,
        redact=not args.no_redact,
    )

    print(result.summary())
    print(f"output: {output_path}")
    return 0


# =============================================================================
# Claude Export
# =============================================================================


def training_claude_export_command(args: argparse.Namespace) -> int:
    """Export Claude Code logs to TrainingSample JSONL."""
    from ..config import load_config_model
    from ..training import export_claude_logs_to_dataset

    config = load_config_model(config_path=Path(args.config) if args.config else None)
    output_path = Path(args.output).expanduser().resolve()
    roots = [Path(root).expanduser().resolve() for root in (args.root or [])]
    scan_roots = [Path(root).expanduser().resolve() for root in (args.scan_root or [])]

    require_quality = config.memory_export.require_quality
    if args.require_quality is True:
        require_quality = True
    elif args.no_require_quality is True:
        require_quality = False

    result = export_claude_logs_to_dataset(
        roots if roots else None,
        output_path=output_path,
        scan_roots=scan_roots if scan_roots else None,
        max_scan_depth=args.max_scan_depth,
        default_domain=args.domain,
        include_tools=not args.no_tools,
        max_tool_output_chars=args.max_tool_output_chars,
        limit=args.limit,
        require_quality=require_quality,
        min_quality_score=(
            args.min_quality_score
            if args.min_quality_score is not None
            else config.memory_export.min_quality_score
        ),
        score_profile=args.score_profile or config.memory_export.score_profile,
        enable_asar=args.enable_asar or config.memory_export.enable_asar,
        redact=not args.no_redact,
    )

    print(result.summary())
    print(f"output: {output_path}")
    return 0


# =============================================================================
# Codex Export
# =============================================================================


def training_codex_export_command(args: argparse.Namespace) -> int:
    """Export Codex CLI logs to TrainingSample JSONL."""
    from ..config import load_config_model
    from ..training import export_codex_logs_to_dataset

    config = load_config_model(config_path=Path(args.config) if args.config else None)
    output_path = Path(args.output).expanduser().resolve()
    roots = [Path(root).expanduser().resolve() for root in (args.root or [])]
    scan_roots = [Path(root).expanduser().resolve() for root in (args.scan_root or [])]

    require_quality = config.memory_export.require_quality
    if args.require_quality is True:
        require_quality = True
    elif args.no_require_quality is True:
        require_quality = False

    result = export_codex_logs_to_dataset(
        roots if roots else None,
        output_path=output_path,
        scan_roots=scan_roots if scan_roots else None,
        max_scan_depth=args.max_scan_depth,
        default_domain=args.domain,
        include_tools=not args.no_tools,
        include_system=args.include_system,
        max_tool_output_chars=args.max_tool_output_chars,
        limit=args.limit,
        require_quality=require_quality,
        min_quality_score=(
            args.min_quality_score
            if args.min_quality_score is not None
            else config.memory_export.min_quality_score
        ),
        score_profile=args.score_profile or config.memory_export.score_profile,
        enable_asar=args.enable_asar or config.memory_export.enable_asar,
        redact=not args.no_redact,
    )

    print(result.summary())
    print(f"output: {output_path}")
    return 0


# =============================================================================
# Codex History Import
# =============================================================================


def training_codex_history_import_command(args: argparse.Namespace) -> int:
    """Import Codex CLI logs into AFS history."""
    from ..config import load_config_model
    from ..training import import_codex_logs_to_history

    config = load_config_model(config_path=Path(args.config) if args.config else None)
    history_root = (
        Path(args.history_root).expanduser().resolve()
        if args.history_root
        else (config.general.context_root / "history")
    )
    roots = [Path(root).expanduser().resolve() for root in (args.root or [])]
    scan_roots = [Path(root).expanduser().resolve() for root in (args.scan_root or [])]

    result = import_codex_logs_to_history(
        roots if roots else None,
        history_root=history_root,
        scan_roots=scan_roots if scan_roots else None,
        max_scan_depth=args.max_scan_depth,
        include_tools=not args.no_tools,
        include_system=args.include_system,
        max_tool_output_chars=args.max_tool_output_chars,
        limit=args.limit,
        redact=not args.no_redact,
    )

    print(result.summary())
    print(f"history_root: {history_root}")
    return 0


# =============================================================================
# ToolBench Export
# =============================================================================


def training_toolbench_export_command(args: argparse.Namespace) -> int:
    """Export ToolBench dataset to TrainingSample JSONL."""
    from ..training.converters import export_toolbench_to_jsonl

    dataset_dir = Path(args.dataset_dir).expanduser().resolve()
    if not dataset_dir.exists():
        print(f"ToolBench dataset directory not found: {dataset_dir}")
        return 1

    output_path = Path(args.output).expanduser().resolve()

    count = export_toolbench_to_jsonl(
        dataset_dir=dataset_dir,
        output_path=output_path,
        split=args.split,
        max_samples=args.limit,
    )

    print(f"Exported {count} samples to {output_path}")
    return 0


# =============================================================================
# Rebalance
# =============================================================================


def training_rebalance_command(args: argparse.Namespace) -> int:
    """Rebalance training datasets by source or domain."""
    from ..training import rebalance_dataset

    raw_inputs = args.input or []
    flattened: list[str] = []
    for item in raw_inputs:
        if isinstance(item, list):
            flattened.extend(item)
        else:
            flattened.append(item)
    input_paths = [Path(path).expanduser().resolve() for path in flattened]
    output_path = Path(args.output).expanduser().resolve()
    append_paths = [Path(path).expanduser().resolve() for path in (args.append or [])]

    weights: dict[str, float] = {}
    for item in args.weight or []:
        if "=" not in item:
            print(f"Invalid weight: {item} (expected name=value)")
            return 1
        name, value = item.split("=", 1)
        try:
            weights[name.strip()] = float(value)
        except ValueError:
            print(f"Invalid weight value: {item}")
            return 1

    result = rebalance_dataset(
        input_paths=input_paths,
        output_path=output_path,
        group_by=args.group_by,
        weights=weights or None,
        max_total=args.max_total,
        allow_oversample=args.allow_oversample,
        include_unweighted=args.include_unweighted,
        seed=args.seed,
        shuffle=not args.no_shuffle,
        append_paths=append_paths or None,
    )

    print(result.summary())
    print(f"output: {output_path}")
    if result.errors:
        print("errors:")
        for err in result.errors:
            print(f"- {err}")
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

    # training memory-export
    train_memory = training_sub.add_parser(
        "memory-export", help="Export memory entries to TrainingSample JSONL."
    )
    train_memory.add_argument("--config", help="Config path.")
    train_memory.add_argument("--context-root", help="Context root override.")
    train_memory.add_argument("--memory-root", help="Memory directory override.")
    train_memory.add_argument("--output", required=True, help="Output JSONL path.")
    train_memory.add_argument("--domain", default="memory", help="Default domain.")
    train_memory.add_argument(
        "--allow-raw",
        action="store_true",
        help="Allow raw content entries without explicit instruction/output.",
    )
    train_memory.add_argument(
        "--allow-raw-tag",
        action="append",
        help="Tag required to include raw content entries (repeatable).",
    )
    train_memory.add_argument(
        "--include-tag",
        action="append",
        help="Only export entries matching this tag (repeatable).",
    )
    train_memory.add_argument(
        "--exclude-tag",
        action="append",
        help="Skip entries matching this tag (repeatable).",
    )
    train_memory.add_argument(
        "--default-instruction",
        help="Instruction to use when allow-raw is set.",
    )
    quality_group = train_memory.add_mutually_exclusive_group()
    quality_group.add_argument(
        "--require-quality",
        dest="require_quality",
        action="store_true",
        default=None,
        help="Require quality scoring (overrides config).",
    )
    quality_group.add_argument(
        "--no-require-quality",
        dest="no_require_quality",
        action="store_true",
        default=None,
        help="Disable quality scoring (overrides config).",
    )
    train_memory.add_argument(
        "--min-quality-score",
        type=float,
        help="Minimum quality score to export.",
    )
    train_memory.add_argument(
        "--score-profile",
        help="Scoring profile (generic, dialogue, or asm).",
    )
    train_memory.add_argument(
        "--enable-asar",
        action="store_true",
        help="Enable asar validation in scoring.",
    )
    train_memory.add_argument(
        "--no-redact",
        action="store_true",
        help="Disable secret redaction.",
    )
    train_memory.add_argument(
        "--limit",
        type=int,
        help="Maximum number of exported samples.",
    )
    train_memory.set_defaults(func=training_memory_export_command)

    # training history-export
    train_history = training_sub.add_parser(
        "history-export", help="Export history events to TrainingSample JSONL."
    )
    train_history.add_argument("--config", help="Config path.")
    train_history.add_argument("--history-root", help="History directory override.")
    train_history.add_argument("--output", required=True, help="Output JSONL path.")
    train_history.add_argument("--domain", default="history", help="Default domain.")
    train_history.add_argument("--tool-domain", default="history_tools", help="Domain for tool/FS/CLI events.")
    train_history.add_argument(
        "--event-type",
        action="append",
        help="Event type to include (repeatable).",
    )
    train_history.add_argument(
        "--include-tools",
        action="store_true",
        help="Include tool execution events.",
    )
    train_history.add_argument(
        "--include-fs",
        action="store_true",
        help="Include filesystem read/write events.",
    )
    train_history.add_argument(
        "--include-cli",
        action="store_true",
        help="Include CLI invocation events.",
    )
    train_history.add_argument(
        "--limit",
        type=int,
        help="Maximum number of exported samples.",
    )
    history_quality = train_history.add_mutually_exclusive_group()
    history_quality.add_argument(
        "--require-quality",
        dest="require_quality",
        action="store_true",
        default=None,
        help="Require quality scoring (overrides config).",
    )
    history_quality.add_argument(
        "--no-require-quality",
        dest="no_require_quality",
        action="store_true",
        default=None,
        help="Disable quality scoring (overrides config).",
    )
    train_history.add_argument(
        "--min-quality-score",
        type=float,
        help="Minimum quality score to export.",
    )
    train_history.add_argument(
        "--score-profile",
        help="Scoring profile (generic, dialogue, or asm).",
    )
    train_history.add_argument(
        "--enable-asar",
        action="store_true",
        help="Enable asar validation in scoring.",
    )
    train_history.add_argument(
        "--no-redact",
        action="store_true",
        help="Disable secret redaction.",
    )
    train_history.set_defaults(func=training_history_export_command)

    # training antigravity-export
    train_antigravity = training_sub.add_parser(
        "antigravity-export",
        help="Export Antigravity trajectory summaries to TrainingSample JSONL.",
    )
    train_antigravity.add_argument("--config", help="Config path.")
    train_antigravity.add_argument("--db-path", help="Antigravity state.vscdb path.")
    train_antigravity.add_argument("--output", required=True, help="Output JSONL path.")
    train_antigravity.add_argument("--domain", default="general", help="Default domain.")
    train_antigravity.add_argument(
        "--state-key",
        action="append",
        help="State key to read from the DB (repeatable).",
    )
    train_antigravity.add_argument(
        "--include-paths-content",
        action="store_true",
        help="Include file contents for PathsToReview.",
    )
    train_antigravity.add_argument(
        "--max-path-chars",
        type=int,
        default=2000,
        help="Maximum characters to include per path.",
    )
    train_antigravity.add_argument(
        "--limit",
        type=int,
        help="Maximum number of exported samples.",
    )
    antigravity_quality = train_antigravity.add_mutually_exclusive_group()
    antigravity_quality.add_argument(
        "--require-quality",
        dest="require_quality",
        action="store_true",
        default=None,
        help="Require quality scoring (overrides config).",
    )
    antigravity_quality.add_argument(
        "--no-require-quality",
        dest="no_require_quality",
        action="store_true",
        default=None,
        help="Disable quality scoring (overrides config).",
    )
    train_antigravity.add_argument(
        "--min-quality-score",
        type=float,
        help="Minimum quality score to export.",
    )
    train_antigravity.add_argument(
        "--score-profile",
        help="Scoring profile (generic, dialogue, or asm).",
    )
    train_antigravity.add_argument(
        "--enable-asar",
        action="store_true",
        help="Enable asar validation in scoring.",
    )
    train_antigravity.add_argument(
        "--no-redact",
        action="store_true",
        help="Disable secret redaction.",
    )
    train_antigravity.set_defaults(func=training_antigravity_export_command)

    # training gemini-export
    train_gemini = training_sub.add_parser(
        "gemini-export", help="Export Gemini CLI logs to TrainingSample JSONL."
    )
    train_gemini.add_argument("--config", help="Config path.")
    train_gemini.add_argument(
        "--root",
        action="append",
        help="Gemini root directory (repeatable).",
    )
    train_gemini.add_argument(
        "--scan-root",
        action="append",
        help="Scan for .gemini directories under this root (repeatable).",
    )
    train_gemini.add_argument(
        "--max-scan-depth",
        type=int,
        default=4,
        help="Maximum .gemini scan depth.",
    )
    train_gemini.add_argument("--output", required=True, help="Output JSONL path.")
    train_gemini.add_argument("--domain", default="gemini", help="Default domain.")
    train_gemini.add_argument(
        "--no-tools",
        action="store_true",
        help="Exclude tool outputs from the input field.",
    )
    train_gemini.add_argument(
        "--include-thoughts",
        action="store_true",
        help="Include Gemini thoughts in the thinking field.",
    )
    train_gemini.add_argument(
        "--max-tool-output-chars",
        type=int,
        default=2000,
        help="Maximum characters per tool output.",
    )
    train_gemini.add_argument(
        "--limit",
        type=int,
        help="Maximum number of exported samples.",
    )
    gemini_quality = train_gemini.add_mutually_exclusive_group()
    gemini_quality.add_argument(
        "--require-quality",
        dest="require_quality",
        action="store_true",
        default=None,
        help="Require quality scoring (overrides config).",
    )
    gemini_quality.add_argument(
        "--no-require-quality",
        dest="no_require_quality",
        action="store_true",
        default=None,
        help="Disable quality scoring (overrides config).",
    )
    train_gemini.add_argument(
        "--min-quality-score",
        type=float,
        help="Minimum quality score to export.",
    )
    train_gemini.add_argument(
        "--score-profile",
        help="Scoring profile (generic, dialogue, or asm).",
    )
    train_gemini.add_argument(
        "--enable-asar",
        action="store_true",
        help="Enable asar validation in scoring.",
    )
    train_gemini.add_argument(
        "--no-redact",
        action="store_true",
        help="Disable secret redaction.",
    )
    train_gemini.set_defaults(func=training_gemini_export_command)

    # training claude-export
    train_claude = training_sub.add_parser(
        "claude-export", help="Export Claude Code logs to TrainingSample JSONL."
    )
    train_claude.add_argument("--config", help="Config path.")
    train_claude.add_argument(
        "--root",
        action="append",
        help="Claude root directory (repeatable).",
    )
    train_claude.add_argument(
        "--scan-root",
        action="append",
        help="Scan for .claude directories under this root (repeatable).",
    )
    train_claude.add_argument(
        "--max-scan-depth",
        type=int,
        default=4,
        help="Maximum .claude scan depth.",
    )
    train_claude.add_argument("--output", required=True, help="Output JSONL path.")
    train_claude.add_argument("--domain", default="claude", help="Default domain.")
    train_claude.add_argument(
        "--no-tools",
        action="store_true",
        help="Exclude tool outputs from the input field.",
    )
    train_claude.add_argument(
        "--max-tool-output-chars",
        type=int,
        default=2000,
        help="Maximum characters per tool output.",
    )
    train_claude.add_argument(
        "--limit",
        type=int,
        help="Maximum number of exported samples.",
    )
    claude_quality = train_claude.add_mutually_exclusive_group()
    claude_quality.add_argument(
        "--require-quality",
        dest="require_quality",
        action="store_true",
        default=None,
        help="Require quality scoring (overrides config).",
    )
    claude_quality.add_argument(
        "--no-require-quality",
        dest="no_require_quality",
        action="store_true",
        default=None,
        help="Disable quality scoring (overrides config).",
    )
    train_claude.add_argument(
        "--min-quality-score",
        type=float,
        help="Minimum quality score to export.",
    )
    train_claude.add_argument(
        "--score-profile",
        help="Scoring profile (generic, dialogue, or asm).",
    )
    train_claude.add_argument(
        "--enable-asar",
        action="store_true",
        help="Enable asar validation in scoring.",
    )
    train_claude.add_argument(
        "--no-redact",
        action="store_true",
        help="Disable secret redaction.",
    )
    train_claude.set_defaults(func=training_claude_export_command)

    # training codex-export
    train_codex = training_sub.add_parser(
        "codex-export", help="Export Codex CLI logs to TrainingSample JSONL."
    )
    train_codex.add_argument("--config", help="Config path.")
    train_codex.add_argument(
        "--root",
        action="append",
        help="Codex root directory (repeatable).",
    )
    train_codex.add_argument(
        "--scan-root",
        action="append",
        help="Scan for .codex directories under this root (repeatable).",
    )
    train_codex.add_argument(
        "--max-scan-depth",
        type=int,
        default=4,
        help="Maximum .codex scan depth.",
    )
    train_codex.add_argument("--output", required=True, help="Output JSONL path.")
    train_codex.add_argument("--domain", default="codex", help="Default domain.")
    train_codex.add_argument(
        "--no-tools",
        action="store_true",
        help="Exclude tool outputs from the input field.",
    )
    train_codex.add_argument(
        "--include-system",
        action="store_true",
        help="Include system instructions in the input field.",
    )
    train_codex.add_argument(
        "--max-tool-output-chars",
        type=int,
        default=2000,
        help="Maximum characters per tool output.",
    )
    train_codex.add_argument(
        "--limit",
        type=int,
        help="Maximum number of exported samples.",
    )
    codex_quality = train_codex.add_mutually_exclusive_group()
    codex_quality.add_argument(
        "--require-quality",
        dest="require_quality",
        action="store_true",
        default=None,
        help="Require quality scoring (overrides config).",
    )
    codex_quality.add_argument(
        "--no-require-quality",
        dest="no_require_quality",
        action="store_true",
        default=None,
        help="Disable quality scoring (overrides config).",
    )
    train_codex.add_argument(
        "--min-quality-score",
        type=float,
        help="Minimum quality score to export.",
    )
    train_codex.add_argument(
        "--score-profile",
        help="Scoring profile (generic, dialogue, or asm).",
    )
    train_codex.add_argument(
        "--enable-asar",
        action="store_true",
        help="Enable asar validation in scoring.",
    )
    train_codex.add_argument(
        "--no-redact",
        action="store_true",
        help="Disable secret redaction.",
    )
    train_codex.set_defaults(func=training_codex_export_command)

    # training codex-history-import
    train_codex_history = training_sub.add_parser(
        "codex-history-import",
        help="Import Codex CLI logs into AFS history.",
    )
    train_codex_history.add_argument("--config", help="Config path.")
    train_codex_history.add_argument("--history-root", help="History directory override.")
    train_codex_history.add_argument(
        "--root",
        action="append",
        help="Codex root directory (repeatable).",
    )
    train_codex_history.add_argument(
        "--scan-root",
        action="append",
        help="Scan for .codex directories under this root (repeatable).",
    )
    train_codex_history.add_argument(
        "--max-scan-depth",
        type=int,
        default=4,
        help="Maximum .codex scan depth.",
    )
    train_codex_history.add_argument(
        "--no-tools",
        action="store_true",
        help="Skip tool execution events.",
    )
    train_codex_history.add_argument(
        "--include-system",
        action="store_true",
        help="Include system instructions in model payloads.",
    )
    train_codex_history.add_argument(
        "--max-tool-output-chars",
        type=int,
        default=4000,
        help="Maximum characters per tool output.",
    )
    train_codex_history.add_argument(
        "--limit",
        type=int,
        help="Maximum number of model events to import.",
    )
    train_codex_history.add_argument(
        "--no-redact",
        action="store_true",
        help="Disable secret redaction.",
    )
    train_codex_history.set_defaults(func=training_codex_history_import_command)

    # training toolbench-export
    train_toolbench = training_sub.add_parser(
        "toolbench-export",
        help="Export ToolBench dataset to TrainingSample JSONL.",
    )
    train_toolbench.add_argument(
        "--dataset-dir",
        required=True,
        help="ToolBench dataset directory (containing data/ subdirectory).",
    )
    train_toolbench.add_argument(
        "--output",
        required=True,
        help="Output JSONL path.",
    )
    train_toolbench.add_argument(
        "--split",
        choices=["train", "validation"],
        default="train",
        help="Dataset split to export (default: train).",
    )
    train_toolbench.add_argument(
        "--limit",
        type=int,
        help="Maximum number of samples to export (default: all).",
    )
    train_toolbench.set_defaults(func=training_toolbench_export_command)

    # training rebalance
    train_rebalance = training_sub.add_parser(
        "rebalance",
        help="Rebalance datasets by source or domain.",
    )
    train_rebalance.add_argument(
        "--input",
        action="append",
        nargs="+",
        required=True,
        help="Input JSONL files.",
    )
    train_rebalance.add_argument(
        "--output",
        required=True,
        help="Output JSONL path.",
    )
    train_rebalance.add_argument(
        "--group-by",
        choices=["source", "domain"],
        default="source",
        help="Field to rebalance by.",
    )
    train_rebalance.add_argument(
        "--weight",
        action="append",
        help="Weight as name=value (repeatable).",
    )
    train_rebalance.add_argument(
        "--max-total",
        type=int,
        help="Maximum total samples (optional).",
    )
    train_rebalance.add_argument(
        "--allow-oversample",
        action="store_true",
        help="Allow oversampling smaller groups.",
    )
    train_rebalance.add_argument(
        "--include-unweighted",
        action="store_true",
        help="Include groups without weights.",
    )
    train_rebalance.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    train_rebalance.add_argument(
        "--no-shuffle",
        action="store_true",
        help="Preserve original ordering.",
    )
    train_rebalance.add_argument(
        "--append",
        action="append",
        help="Append JSONL files without rebalancing.",
    )
    train_rebalance.set_defaults(func=training_rebalance_command)

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
