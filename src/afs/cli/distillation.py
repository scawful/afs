"""CLI commands for distillation pipeline.

Commands:
- distill generate: Generate training data from teacher ensemble
- distill status: Show generation progress
- distill export: Export to training format
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def register_parsers(subparsers: argparse._SubParsersAction) -> None:
    """Register distillation command parsers."""
    distill_parser = subparsers.add_parser(
        "distill",
        help="Distillation pipeline commands"
    )
    distill_subs = distill_parser.add_subparsers(dest="distill_command")

    # Generate command
    gen_parser = distill_subs.add_parser(
        "generate",
        help="Generate training data from teacher ensemble"
    )
    gen_parser.add_argument(
        "--count",
        type=int,
        default=1000,
        help="Number of samples to generate (default: 1000)"
    )
    gen_parser.add_argument(
        "--output",
        type=str,
        default="distillation_data",
        help="Output directory (default: distillation_data)"
    )
    gen_parser.add_argument(
        "--domains",
        type=str,
        nargs="+",
        default=["din", "nayru", "farore", "veran"],
        help="Domains to generate for (default: all)"
    )
    gen_parser.add_argument(
        "--providers",
        type=str,
        nargs="+",
        default=["openai", "google", "anthropic"],
        help="Providers to use (default: all)"
    )
    gen_parser.add_argument(
        "--min-quality",
        type=float,
        default=0.5,
        help="Minimum quality score (default: 0.5)"
    )
    gen_parser.add_argument(
        "--format",
        type=str,
        choices=["chatml", "alpaca", "sharegpt"],
        default="chatml",
        help="Training data format (default: chatml)"
    )
    gen_parser.set_defaults(func=cmd_generate)

    # Status command
    status_parser = distill_subs.add_parser(
        "status",
        help="Show generation progress from checkpoint"
    )
    status_parser.add_argument(
        "--checkpoint",
        type=str,
        default="distillation_data/checkpoint.jsonl",
        help="Checkpoint file path"
    )
    status_parser.set_defaults(func=cmd_status)

    # Export command
    export_parser = distill_subs.add_parser(
        "export",
        help="Export checkpoint to training format"
    )
    export_parser.add_argument(
        "--checkpoint",
        type=str,
        default="distillation_data/checkpoint.jsonl",
        help="Checkpoint file path"
    )
    export_parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output file path (.jsonl)"
    )
    export_parser.add_argument(
        "--format",
        type=str,
        choices=["chatml", "alpaca", "sharegpt"],
        default="chatml",
        help="Training data format (default: chatml)"
    )
    export_parser.set_defaults(func=cmd_export)

    # Teachers command
    teachers_parser = distill_subs.add_parser(
        "teachers",
        help="List available teacher models and their status"
    )
    teachers_parser.set_defaults(func=cmd_teachers)


def cmd_generate(args: argparse.Namespace) -> int:
    """Generate training data from teacher ensemble."""
    from ..distillation import (
        DistillationConfig,
        DistillationDataGenerator,
        Provider,
        TeacherEnsemble,
    )
    from ..distillation.teacher import (
        AnthropicTeacher,
        GoogleTeacher,
        OpenAITeacher,
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    # Build ensemble based on providers
    teachers = []
    provider_map = {
        "openai": (Provider.OPENAI, OpenAITeacher),
        "google": (Provider.GOOGLE, GoogleTeacher),
        "anthropic": (Provider.ANTHROPIC, AnthropicTeacher),
    }

    for provider_name in args.providers:
        if provider_name in provider_map:
            _, teacher_cls = provider_map[provider_name]
            teachers.append(teacher_cls())
            logger.info(f"Added {provider_name} teacher")

    if not teachers:
        logger.error("No valid providers specified")
        return 1

    ensemble = TeacherEnsemble(teachers=teachers)

    config = DistillationConfig(
        target_count=args.count,
        min_quality_score=args.min_quality,
        output_dir=Path(args.output),
        domains=args.domains,
        format_type=args.format,
    )

    generator = DistillationDataGenerator(
        ensemble=ensemble,
        config=config,
    )

    def progress_callback(progress):
        logger.info(
            f"Generated: {progress.generated}/{progress.total_requested} "
            f"({progress.samples_per_minute:.1f}/min) | "
            f"Failed: {progress.failed} | Quality filtered: {progress.quality_filtered}"
        )

    logger.info(f"Starting generation: {args.count} samples across {args.domains}")
    logger.info(f"Providers: {args.providers}")
    logger.info(f"Output: {args.output}")

    try:
        samples = asyncio.run(
            generator.generate_batch(
                count=args.count,
                progress_callback=progress_callback,
            )
        )

        # Export final training data
        output_path = Path(args.output) / f"train_{args.format}.jsonl"
        generator.export_training_data(output_path)

        stats = generator.get_statistics()
        logger.info("\nGeneration complete!")
        logger.info(f"Total samples: {stats['total_samples']}")
        logger.info(f"Provider distribution: {stats['providers']}")
        logger.info(f"Domain distribution: {stats['domains']}")
        logger.info(f"Average quality: {stats['avg_quality']:.2f}")

        return 0
    except KeyboardInterrupt:
        logger.info("Generation interrupted. Progress saved to checkpoint.")
        return 0
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        return 1


def cmd_status(args: argparse.Namespace) -> int:
    """Show generation progress from checkpoint."""
    import json
    from collections import Counter

    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.exists():
        print(f"No checkpoint found at {checkpoint_path}")
        return 1

    samples = []
    with open(checkpoint_path) as f:
        for line in f:
            samples.append(json.loads(line.strip()))

    if not samples:
        print("Checkpoint is empty")
        return 0

    # Compute statistics
    providers = Counter(s.get("provider", "unknown") for s in samples)
    domains = Counter(s.get("domain", "unknown") for s in samples)
    difficulties = Counter(s.get("difficulty", "unknown") for s in samples)
    qualities = [s.get("quality_score", 0) for s in samples]

    print("\nDistillation Progress")
    print("=" * 50)
    print(f"Total samples: {len(samples)}")
    print("\nBy Provider:")
    for provider, count in providers.most_common():
        print(f"  {provider}: {count}")
    print("\nBy Domain:")
    for domain, count in domains.most_common():
        print(f"  {domain}: {count}")
    print("\nBy Difficulty:")
    for diff, count in difficulties.most_common():
        print(f"  {diff}: {count}")
    print("\nQuality Scores:")
    print(f"  Average: {sum(qualities) / len(qualities):.2f}")
    print(f"  Min: {min(qualities):.2f}")
    print(f"  Max: {max(qualities):.2f}")

    return 0


def cmd_export(args: argparse.Namespace) -> int:
    """Export checkpoint to training format."""
    import json

    checkpoint_path = Path(args.checkpoint)
    output_path = Path(args.output)

    if not checkpoint_path.exists():
        print(f"No checkpoint found at {checkpoint_path}")
        return 1

    samples = []
    with open(checkpoint_path) as f:
        for line in f:
            samples.append(json.loads(line.strip()))

    if not samples:
        print("Checkpoint is empty")
        return 1

    # Convert to training format
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        for sample in samples:
            if args.format == "alpaca":
                formatted = {
                    "instruction": sample.get("prompt", ""),
                    "input": "",
                    "output": sample.get("response", ""),
                }
            elif args.format == "chatml":
                messages = []
                if sample.get("system_prompt"):
                    messages.append({"role": "system", "content": sample["system_prompt"]})
                messages.append({"role": "user", "content": sample.get("prompt", "")})
                messages.append({"role": "assistant", "content": sample.get("response", "")})
                formatted = {"messages": messages}
            elif args.format == "sharegpt":
                conversations = []
                if sample.get("system_prompt"):
                    conversations.append({"from": "system", "value": sample["system_prompt"]})
                conversations.append({"from": "human", "value": sample.get("prompt", "")})
                conversations.append({"from": "gpt", "value": sample.get("response", "")})
                formatted = {"conversations": conversations}
            else:
                formatted = sample

            f.write(json.dumps(formatted) + "\n")

    print(f"Exported {len(samples)} samples to {output_path} in {args.format} format")
    return 0


def cmd_teachers(args: argparse.Namespace) -> int:
    """List available teacher models and their status."""
    import os

    teachers = [
        ("OpenAI", "OPENAI_API_KEY", "gpt-5.2"),
        ("Google", "GEMINI_API_KEY", "gemini-3-flash-preview"),
        ("Anthropic", "CLAUDE_API_KEY", "claude-opus-4.5"),
    ]

    print("\nAvailable Teacher Models")
    print("=" * 60)

    for name, env_var, models in teachers:
        has_key = bool(os.getenv(env_var))
        status = "Configured" if has_key else "Not configured"
        status_icon = "+" if has_key else "-"

        print(f"\n[{status_icon}] {name}")
        print(f"    Status: {status}")
        print(f"    Env var: {env_var}")
        print(f"    Models: {models}")

    print("\n" + "=" * 60)
    configured = sum(1 for _, env_var, _ in teachers if os.getenv(env_var))
    print(f"Total: {configured}/{len(teachers)} providers configured")

    return 0
