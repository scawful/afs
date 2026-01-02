"""Active Learning CLI commands: sample, curriculum, queue operations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def active_learning_sample_command(args: argparse.Namespace) -> int:
    """Sample using uncertainty strategy."""
    from ..generators.base import TrainingSample
    from ..active_learning import UncertaintySampler
    from ..training.scoring import QualityScorer, ScoringConfig

    # Load samples
    samples = []
    with open(args.input) as f:
        for line in f:
            if line.strip():
                samples.append(TrainingSample.from_dict(json.loads(line)))

    print(f"Loaded {len(samples)} samples")

    # Score if needed
    config = ScoringConfig()
    if args.electra:
        config.electra_model_path = Path(args.electra)
    scorer = QualityScorer(config=config)
    scorer.score_batch(samples, update_samples=True)

    # Sample
    sampler = UncertaintySampler()
    selected = sampler.sample(samples, args.n, scorer=None)  # Already scored

    print(f"Selected {len(selected)} samples by uncertainty")

    # Show distribution
    dist = sampler.get_uncertainty_distribution(samples)
    print(f"\nUncertainty distribution (all samples):")
    for level, count in dist.items():
        print(f"  {level}: {count}")

    # Save if output specified
    if args.output:
        with open(args.output, "w") as f:
            for sample in selected:
                f.write(json.dumps(sample.to_dict()) + "\n")
        print(f"\nSaved to {args.output}")

    return 0


def active_learning_curriculum_command(args: argparse.Namespace) -> int:
    """Get samples for curriculum stage."""
    from ..generators.base import TrainingSample
    from ..active_learning import CurriculumManager, CurriculumStage
    from ..knowledge import EntityExtractor

    # Load samples
    samples = []
    with open(args.input) as f:
        for line in f:
            if line.strip():
                samples.append(TrainingSample.from_dict(json.loads(line)))

    print(f"Loaded {len(samples)} samples")

    # Populate kg_entities if needed
    extractor = EntityExtractor()
    for sample in samples:
        if not sample.kg_entities:
            sample.populate_kg_entities(extractor)

    # Get curriculum distribution
    manager = CurriculumManager()

    if args.plan:
        # Show curriculum plan
        plan = manager.get_curriculum_plan(samples)
        print(f"\nCurriculum Plan ({plan['total_samples']} total samples):")
        print(f"{'Stage':<12} {'Count':>8} {'%':>8} {'Cumulative':>12}")
        print("-" * 45)
        for stage in plan['stages']:
            print(f"{stage['stage']:<12} {stage['sample_count']:>8} {stage['percentage']:>7.1f}% {stage['cumulative_count']:>12}")
        return 0

    # Get samples for specific stage
    stage = CurriculumStage(args.stage)
    stage_samples = manager.get_samples_for_stage(samples, stage)

    print(f"\nStage '{stage.value}': {len(stage_samples)} samples")

    if args.output:
        with open(args.output, "w") as f:
            for sample in stage_samples:
                f.write(json.dumps(sample.to_dict()) + "\n")
        print(f"Saved to {args.output}")

    return 0


def active_learning_queue_add_command(args: argparse.Namespace) -> int:
    """Add samples to priority queue."""
    from ..generators.base import TrainingSample
    from ..active_learning import PriorityQueue
    from ..training.scoring import QualityScorer, ScoringConfig

    # Load samples
    samples = []
    with open(args.input) as f:
        for line in f:
            if line.strip():
                samples.append(TrainingSample.from_dict(json.loads(line)))

    print(f"Loaded {len(samples)} samples")

    # Score if needed
    config = ScoringConfig()
    if args.electra:
        config.electra_model_path = Path(args.electra)
    scorer = QualityScorer(config=config)
    scorer.score_batch(samples, update_samples=True)

    # Add to queue
    queue_path = Path(args.queue)
    queue = PriorityQueue(storage_path=queue_path)
    added = queue.add(samples, scorer=None)  # Already scored

    print(f"Added {added} samples to queue")

    stats = queue.get_stats()
    print(f"Queue now has {stats['total_items']} items")
    print(f"  Pending: {stats['by_status']['pending']}")
    print(f"  In progress: {stats['by_status']['in_progress']}")
    print(f"  Reviewed: {stats['by_status']['reviewed']}")

    return 0


def active_learning_queue_get_command(args: argparse.Namespace) -> int:
    """Get next batch from priority queue."""
    from ..active_learning import PriorityQueue

    queue_path = Path(args.queue)
    if not queue_path.exists():
        print(f"Queue not found: {queue_path}")
        return 1

    queue = PriorityQueue(storage_path=queue_path)
    items = queue.get_batch(args.n)

    print(f"Retrieved {len(items)} items from queue")

    if args.output:
        output_data = [item.to_dict() for item in items]
        with open(args.output, "w") as f:
            json.dump(output_data, f, indent=2)
        print(f"Saved to {args.output}")
    else:
        # Print items
        for i, item in enumerate(items):
            print(f"\n[{i+1}] {item.item_id}")
            print(f"    Domain: {item.domain}")
            print(f"    Priority: {item.priority:.3f}")
            print(f"    Uncertainty: {item.uncertainty:.3f}")
            print(f"    Instruction: {item.instruction[:60]}...")

    return 0


def active_learning_queue_status_command(args: argparse.Namespace) -> int:
    """Show priority queue status."""
    from ..active_learning import PriorityQueue

    queue_path = Path(args.queue)
    if not queue_path.exists():
        print(f"Queue not found: {queue_path}")
        return 1

    queue = PriorityQueue(storage_path=queue_path)
    stats = queue.get_stats()

    print(f"Priority Queue Status")
    print("=" * 40)
    print(f"Total items: {stats['total_items']}")
    print(f"\nBy status:")
    for status, count in stats['by_status'].items():
        print(f"  {status}: {count}")

    print(f"\nBy domain:")
    for domain, count in stats['by_domain'].items():
        print(f"  {domain}: {count}")

    if stats['ratings']['count'] > 0:
        print(f"\nRatings:")
        print(f"  Count: {stats['ratings']['count']}")
        print(f"  Mean: {stats['ratings']['mean']:.2f}")
        print(f"  Range: {stats['ratings']['min']:.2f} - {stats['ratings']['max']:.2f}")

    return 0


def register_parsers(subparsers: argparse._SubParsersAction) -> None:
    """Register active learning command parsers."""
    al_parser = subparsers.add_parser(
        "active-learning", help="Active learning utilities."
    )
    al_sub = al_parser.add_subparsers(dest="active_learning_command")

    # active-learning sample
    al_sample = al_sub.add_parser(
        "sample", help="Sample using uncertainty or diversity strategy."
    )
    al_sample.add_argument("--input", required=True, help="Input JSONL file.")
    al_sample.add_argument("--output", help="Output JSONL file.")
    al_sample.add_argument(
        "--n", type=int, default=100, help="Number of samples to select (default: 100)."
    )
    al_sample.add_argument(
        "--strategy",
        choices=["uncertainty", "random", "diverse"],
        default="uncertainty",
        help="Sampling strategy (default: uncertainty).",
    )
    al_sample.add_argument("--electra", help="Path to ELECTRA model.")
    al_sample.set_defaults(func=active_learning_sample_command)

    # active-learning curriculum
    al_curriculum = al_sub.add_parser(
        "curriculum", help="Get samples for curriculum learning stage."
    )
    al_curriculum.add_argument("--input", required=True, help="Input JSONL file.")
    al_curriculum.add_argument("--output", help="Output JSONL file.")
    al_curriculum.add_argument(
        "--stage",
        choices=["simple", "moderate", "complex", "advanced"],
        help="Curriculum stage to filter to.",
    )
    al_curriculum.add_argument(
        "--plan",
        action="store_true",
        help="Show curriculum plan instead of filtering.",
    )
    al_curriculum.set_defaults(func=active_learning_curriculum_command)

    # active-learning queue-add
    al_queue_add = al_sub.add_parser(
        "queue-add", help="Add samples to annotation priority queue."
    )
    al_queue_add.add_argument("--input", required=True, help="Input JSONL file.")
    al_queue_add.add_argument("--queue", required=True, help="Queue file path.")
    al_queue_add.add_argument("--electra", help="Path to ELECTRA model.")
    al_queue_add.set_defaults(func=active_learning_queue_add_command)

    # active-learning queue-get
    al_queue_get = al_sub.add_parser(
        "queue-get", help="Get next batch from annotation queue."
    )
    al_queue_get.add_argument("--queue", required=True, help="Queue file path.")
    al_queue_get.add_argument("--output", help="Output file for batch.")
    al_queue_get.add_argument(
        "--n", type=int, default=10, help="Batch size (default: 10)."
    )
    al_queue_get.set_defaults(func=active_learning_queue_get_command)

    # active-learning queue-status
    al_queue_status = al_sub.add_parser(
        "queue-status", help="Show annotation queue status."
    )
    al_queue_status.add_argument("--queue", required=True, help="Queue file path.")
    al_queue_status.set_defaults(func=active_learning_queue_status_command)
