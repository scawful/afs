"""Pipeline CLI commands: scoring, pipeline, and evaluation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


# =============================================================================
# Scoring Commands
# =============================================================================


def scoring_score_command(args: argparse.Namespace) -> int:
    """Score training samples."""
    from ..training.scoring import score_jsonl, ScoringWeights

    input_path = Path(args.input)
    output_path = Path(args.output)
    electra_path = Path(args.electra) if args.electra else None

    weights = ScoringWeights(
        electra=getattr(args, "weight_electra", 0.4),
        asar=getattr(args, "weight_asar", 0.3),
        entity=getattr(args, "weight_entity", 0.2),
        length=getattr(args, "weight_length", 0.1),
    )

    print(f"Scoring samples from {input_path}")
    if electra_path:
        print(f"  Using ELECTRA model: {electra_path}")
    print(f"  Weights: electra={weights.electra}, asar={weights.asar}, entity={weights.entity}, length={weights.length}")

    stats = score_jsonl(
        input_path=input_path,
        output_path=output_path,
        electra_path=electra_path,
        weights=weights,
        min_score=getattr(args, "min_score", None),
    )

    print(f"\nScoring Complete")
    print(f"  Input: {stats['input_count']} samples")
    print(f"  Output: {stats['output_count']} samples")
    if getattr(args, "min_score", None):
        print(f"  Filtered: {stats['filtered_count']} samples (below {args.min_score})")
    print(f"  Mean score: {stats['mean_score']:.3f}")
    print(f"  Score range: {stats['min_score']:.3f} - {stats['max_score']:.3f}")
    print(f"  Output: {output_path}")

    return 0


def scoring_analyze_command(args: argparse.Namespace) -> int:
    """Analyze score distribution."""
    from ..generators.base import TrainingSample
    from ..training.scoring import QualityScorer, ScoringConfig, analyze_scores

    input_path = Path(args.input)
    electra_path = Path(args.electra) if args.electra else None

    # Load samples
    samples = []
    with open(input_path) as f:
        for line in f:
            if line.strip():
                samples.append(TrainingSample.from_dict(json.loads(line)))

    print(f"Analyzing {len(samples)} samples...")

    config = ScoringConfig(electra_model_path=electra_path)
    scorer = QualityScorer(config=config)
    scores = scorer.score_batch(samples, update_samples=False)

    analysis = analyze_scores(scores)

    print(f"\nScore Analysis")
    print("=" * 60)
    print(f"  Total samples: {analysis['count']}")
    print(f"\nOverall Score:")
    print(f"  Mean: {analysis['overall']['mean']:.3f}")
    print(f"  Min:  {analysis['overall']['min']:.3f}")
    print(f"  Max:  {analysis['overall']['max']:.3f}")

    if args.histogram:
        print(f"\n  Distribution:")
        for bucket, count in sorted(analysis['overall']['histogram'].items()):
            bar = "#" * (count * 40 // len(samples)) if samples else ""
            print(f"    {bucket}: {count:4} {bar}")

    print(f"\nComponent Scores:")
    print(f"  ELECTRA mean: {analysis['electra']['mean']:.3f}")
    print(f"  Entity coverage mean: {analysis['entity_coverage']['mean']:.3f}")
    print(f"  Asar pass rate: {100 * analysis['asar_pass_rate']:.1f}%")

    print(f"\nEntity Stats:")
    print(f"  Total entities: {analysis['entity_stats']['total_entities']}")
    print(f"  Known entities: {analysis['entity_stats']['known_entities']}")

    return 0


# =============================================================================
# Pipeline Commands
# =============================================================================


def pipeline_run_command(args: argparse.Namespace) -> int:
    """Run the full data pipeline."""
    from ..training.pipeline import DataPipeline, PipelineConfig
    from ..training.scoring import ScoringWeights

    # Parse input paths
    input_paths = [Path(p) for p in args.input]

    # Build config
    config = PipelineConfig(
        input_paths=input_paths,
        output_dir=Path(args.output),
        expand_vocab=not getattr(args, "skip_vocab", False),
        extract_entities=not getattr(args, "skip_entities", False),
        score_quality=not getattr(args, "skip_scoring", False),
        min_quality_score=getattr(args, "min_score", 0.6),
        apply_phase1_augment=not getattr(args, "skip_augment", False),
        phase1_paraphrase_count=getattr(args, "paraphrase_count", 3),
        apply_phase2_augment=not getattr(args, "skip_augment", False) and not getattr(args, "skip_phase2", False),
        deduplicate=not getattr(args, "skip_dedupe", False),
        dedupe_threshold=getattr(args, "dedupe_threshold", 0.95),
        split_data=not getattr(args, "skip_split", False),
        train_ratio=getattr(args, "train_ratio", 0.8),
        val_ratio=getattr(args, "val_ratio", 0.1),
        test_ratio=1.0 - getattr(args, "train_ratio", 0.8) - getattr(args, "val_ratio", 0.1),
        verbose=not getattr(args, "quiet", False),
    )

    if args.tokenizer:
        config.tokenizer_path = Path(args.tokenizer)

    if args.electra:
        config.electra_model_path = Path(args.electra)

    # Run pipeline
    pipeline = DataPipeline(config)
    result = pipeline.run()

    if result.errors:
        print("\nErrors:")
        for error in result.errors:
            print(f"  - {error}")
        return 1

    return 0


def pipeline_status_command(args: argparse.Namespace) -> int:
    """Show status of a pipeline run."""
    output_dir = Path(args.dir)
    result_path = output_dir / "pipeline_result.json"

    if not result_path.exists():
        print(f"No pipeline result found at {result_path}")
        return 1

    with open(result_path) as f:
        result = json.load(f)

    print("\nPipeline Result")
    print("=" * 60)
    print(f"  Input samples: {result['input_count']}")
    print(f"  Output samples: {result['output_count']}")
    print(f"  Filtered: {result['filtered_count']}")
    print(f"  Augmented: {result['augmented_count']}")
    print(f"  Deduped: {result['dedupe_removed']}")
    print(f"\nQuality:")
    print(f"  Mean score: {result['mean_quality_score']:.3f}")
    print(f"  Range: {result['min_quality_score']:.3f} - {result['max_quality_score']:.3f}")
    print(f"\nEntities:")
    print(f"  Total: {result['total_entities']}")
    print(f"  Known: {result['known_entities']}")
    print(f"  Coverage: {100 * result['entity_coverage']:.1f}%")
    print(f"\nDuration: {result['duration_seconds']:.1f} seconds")
    print(f"\nOutput files:")
    for name, path in result['output_paths'].items():
        print(f"  {name}: {path}")

    return 0


# =============================================================================
# Evaluation Commands
# =============================================================================


def evaluation_run_command(args: argparse.Namespace) -> int:
    """Run evaluation on training samples."""
    from ..generators.base import TrainingSample
    from ..evaluation import EvaluationHarness
    from ..training.scoring import QualityScorer, ScoringConfig

    # Load samples
    samples = []
    with open(args.input) as f:
        for line in f:
            if line.strip():
                samples.append(TrainingSample.from_dict(json.loads(line)))

    print(f"Loaded {len(samples)} samples from {args.input}")

    # Create scorer
    config = ScoringConfig()
    if args.electra:
        config.electra_model_path = Path(args.electra)
    scorer = QualityScorer(config=config)

    # Run evaluation
    harness = EvaluationHarness(scorer=scorer)
    result = harness.evaluate(samples)

    # Output
    print()
    print(result.summary())

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"\nResults saved to {args.output}")

    return 0


def evaluation_compare_command(args: argparse.Namespace) -> int:
    """Compare two datasets."""
    from ..generators.base import TrainingSample
    from ..evaluation import EvaluationHarness
    from ..training.scoring import QualityScorer, ScoringConfig

    def load_samples(path: str) -> list[TrainingSample]:
        samples = []
        with open(path) as f:
            for line in f:
                if line.strip():
                    samples.append(TrainingSample.from_dict(json.loads(line)))
        return samples

    baseline = load_samples(args.baseline)
    candidate = load_samples(args.candidate)

    print(f"Baseline: {len(baseline)} samples from {args.baseline}")
    print(f"Candidate: {len(candidate)} samples from {args.candidate}")

    # Create scorer
    config = ScoringConfig()
    if args.electra:
        config.electra_model_path = Path(args.electra)
    scorer = QualityScorer(config=config)

    # Compare
    harness = EvaluationHarness(scorer=scorer)
    result = harness.compare(baseline, candidate)

    print()
    print(result.summary())

    if args.output:
        with open(args.output, "w") as f:
            json.dump(result.to_dict(), f, indent=2)
        print(f"\nResults saved to {args.output}")

    return 0


def evaluation_human_create_command(args: argparse.Namespace) -> int:
    """Create human evaluation batch."""
    from ..generators.base import TrainingSample
    from ..evaluation.human import HumanEvaluationManager, SamplingStrategy
    from ..training.scoring import QualityScorer, ScoringConfig

    # Load samples
    samples = []
    with open(args.input) as f:
        for line in f:
            if line.strip():
                samples.append(TrainingSample.from_dict(json.loads(line)))

    print(f"Loaded {len(samples)} samples")

    # Score samples if needed
    config = ScoringConfig()
    if args.electra:
        config.electra_model_path = Path(args.electra)
    scorer = QualityScorer(config=config)
    scorer.score_batch(samples, update_samples=True)

    # Create batch
    manager = HumanEvaluationManager()
    batch = manager.create_batch(
        samples,
        name=getattr(args, "name", "") or "",
        n=args.n,
        strategy=SamplingStrategy(args.strategy),
    )

    # Save
    output_path = Path(args.output)
    manager.save_batch(batch, output_path)
    print(f"Created batch with {len(batch.tasks)} tasks")
    print(f"Saved to {output_path}")

    # Export CSV if requested
    if getattr(args, "csv", False):
        csv_path = output_path.with_suffix(".csv")
        manager.export_csv(batch, csv_path)
        print(f"Exported CSV to {csv_path}")

    return 0


def evaluation_human_import_command(args: argparse.Namespace) -> int:
    """Import human evaluation results."""
    from ..evaluation.human import HumanEvaluationManager

    manager = HumanEvaluationManager()

    # Load batch
    batch = manager.load_batch(Path(args.batch))
    print(f"Loaded batch {batch.batch_id} with {len(batch.tasks)} tasks")

    # Import results
    results_path = Path(args.results)
    if results_path.suffix == ".csv":
        updated = manager.import_csv(batch, results_path)
    else:
        updated = manager.import_results(batch, results_path)

    print(f"Updated {updated} tasks")

    # Save updated batch
    output_path = Path(args.output) if args.output else Path(args.batch)
    manager.save_batch(batch, output_path)
    print(f"Saved updated batch to {output_path}")

    # Show summary
    summary = manager.get_batch_summary(batch)
    print(f"\nBatch Summary:")
    print(f"  Completed: {summary['completed']}/{summary['total_tasks']}")
    if summary['ratings']['count'] > 0:
        print(f"  Mean rating: {summary['ratings']['mean']:.2f}")

    return 0


# =============================================================================
# Parser Registration
# =============================================================================


def register_parsers(subparsers: argparse._SubParsersAction) -> None:
    """Register scoring, pipeline, and evaluation command parsers."""

    # =========================================================================
    # Scoring
    # =========================================================================
    scoring_parser = subparsers.add_parser(
        "scoring", help="Quality scoring for training samples."
    )
    scoring_sub = scoring_parser.add_subparsers(dest="scoring_command")

    # scoring score
    scr_score = scoring_sub.add_parser("score", help="Score training samples.")
    scr_score.add_argument("--input", required=True, help="Input JSONL file.")
    scr_score.add_argument("--output", required=True, help="Output JSONL file.")
    scr_score.add_argument("--electra", help="Path to ELECTRA model.")
    scr_score.add_argument(
        "--min-score", type=float, help="Minimum score to include in output."
    )
    scr_score.add_argument(
        "--weight-electra",
        type=float,
        default=0.4,
        help="ELECTRA score weight (default: 0.4).",
    )
    scr_score.add_argument(
        "--weight-asar",
        type=float,
        default=0.3,
        help="ASAR validation weight (default: 0.3).",
    )
    scr_score.add_argument(
        "--weight-entity",
        type=float,
        default=0.2,
        help="Entity coverage weight (default: 0.2).",
    )
    scr_score.add_argument(
        "--weight-length",
        type=float,
        default=0.1,
        help="Length score weight (default: 0.1).",
    )
    scr_score.set_defaults(func=scoring_score_command)

    # scoring analyze
    scr_analyze = scoring_sub.add_parser(
        "analyze", help="Analyze score distribution."
    )
    scr_analyze.add_argument("--input", required=True, help="Input JSONL file.")
    scr_analyze.add_argument("--electra", help="Path to ELECTRA model.")
    scr_analyze.add_argument(
        "--histogram", action="store_true", help="Show score histogram."
    )
    scr_analyze.set_defaults(func=scoring_analyze_command)

    # =========================================================================
    # Pipeline
    # =========================================================================
    pipeline_parser = subparsers.add_parser(
        "pipeline", help="Full data processing pipeline."
    )
    pipeline_sub = pipeline_parser.add_subparsers(dest="pipeline_command")

    # pipeline run
    pipe_run = pipeline_sub.add_parser("run", help="Run the full data pipeline.")
    pipe_run.add_argument(
        "--input", nargs="+", required=True, help="Input JSONL file(s)."
    )
    pipe_run.add_argument("--output", required=True, help="Output directory.")
    pipe_run.add_argument("--tokenizer", help="Path to existing tokenizer.")
    pipe_run.add_argument("--electra", help="Path to ELECTRA model for scoring.")
    pipe_run.add_argument(
        "--min-score",
        type=float,
        default=0.6,
        help="Minimum quality score to keep (default: 0.6).",
    )
    pipe_run.add_argument(
        "--train-ratio",
        type=float,
        default=0.8,
        help="Train split ratio (default: 0.8).",
    )
    pipe_run.add_argument(
        "--val-ratio",
        type=float,
        default=0.1,
        help="Validation split ratio (default: 0.1).",
    )
    pipe_run.add_argument(
        "--dedupe-threshold",
        type=float,
        default=0.95,
        help="Deduplication similarity threshold (default: 0.95).",
    )
    pipe_run.add_argument(
        "--paraphrase-count",
        type=int,
        default=3,
        help="Number of paraphrases per sample (default: 3).",
    )
    pipe_run.add_argument(
        "--skip-vocab", action="store_true", help="Skip vocabulary expansion."
    )
    pipe_run.add_argument(
        "--skip-entities", action="store_true", help="Skip entity extraction."
    )
    pipe_run.add_argument(
        "--skip-scoring", action="store_true", help="Skip quality scoring."
    )
    pipe_run.add_argument(
        "--skip-augment", action="store_true", help="Skip augmentation."
    )
    pipe_run.add_argument(
        "--skip-phase2", action="store_true", help="Skip phase 2 augmentation."
    )
    pipe_run.add_argument(
        "--skip-dedupe", action="store_true", help="Skip deduplication."
    )
    pipe_run.add_argument(
        "--skip-split", action="store_true", help="Skip train/val/test split."
    )
    pipe_run.add_argument(
        "--quiet", action="store_true", help="Minimal output."
    )
    pipe_run.set_defaults(func=pipeline_run_command)

    # pipeline status
    pipe_status = pipeline_sub.add_parser(
        "status", help="Show status of a pipeline run."
    )
    pipe_status.add_argument(
        "--dir", required=True, help="Pipeline output directory."
    )
    pipe_status.set_defaults(func=pipeline_status_command)

    # =========================================================================
    # Evaluation
    # =========================================================================
    eval_parser = subparsers.add_parser(
        "evaluation", help="Evaluation harness and human evaluation."
    )
    eval_sub = eval_parser.add_subparsers(dest="evaluation_command")

    # evaluation run
    eval_run = eval_sub.add_parser("run", help="Run evaluation on training samples.")
    eval_run.add_argument("--input", required=True, help="Input JSONL file.")
    eval_run.add_argument("--output", help="Output JSON file for results.")
    eval_run.add_argument("--electra", help="Path to ELECTRA model.")
    eval_run.set_defaults(func=evaluation_run_command)

    # evaluation compare
    eval_compare = eval_sub.add_parser("compare", help="Compare two datasets.")
    eval_compare.add_argument(
        "--baseline", required=True, help="Baseline JSONL file."
    )
    eval_compare.add_argument(
        "--candidate", required=True, help="Candidate JSONL file."
    )
    eval_compare.add_argument("--output", help="Output JSON file for results.")
    eval_compare.add_argument("--electra", help="Path to ELECTRA model.")
    eval_compare.set_defaults(func=evaluation_compare_command)

    # evaluation human-create
    eval_human_create = eval_sub.add_parser(
        "human-create", help="Create human evaluation batch."
    )
    eval_human_create.add_argument(
        "--input", required=True, help="Input JSONL file."
    )
    eval_human_create.add_argument(
        "--output", required=True, help="Output batch file."
    )
    eval_human_create.add_argument(
        "--n", type=int, default=50, help="Number of samples to include (default: 50)."
    )
    eval_human_create.add_argument(
        "--strategy",
        choices=["uncertainty", "random", "stratified"],
        default="uncertainty",
        help="Sampling strategy (default: uncertainty).",
    )
    eval_human_create.add_argument("--name", help="Batch name.")
    eval_human_create.add_argument("--electra", help="Path to ELECTRA model.")
    eval_human_create.add_argument(
        "--csv", action="store_true", help="Also export as CSV."
    )
    eval_human_create.set_defaults(func=evaluation_human_create_command)

    # evaluation human-import
    eval_human_import = eval_sub.add_parser(
        "human-import", help="Import human evaluation results."
    )
    eval_human_import.add_argument(
        "--batch", required=True, help="Batch file to update."
    )
    eval_human_import.add_argument(
        "--results", required=True, help="Results file (CSV or JSON)."
    )
    eval_human_import.add_argument(
        "--output", help="Output file (default: update batch in place)."
    )
    eval_human_import.set_defaults(func=evaluation_human_import_command)
