"""Generator CLI commands: model, knowledge, batch generation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def generator_model_command(args: argparse.Namespace) -> int:
    """Generate assembly code using a trained model."""
    from ..generators.model_generator import (
        ModelType,
        create_generator,
    )

    # Determine model type
    model_type = args.type or "api"

    # Create generator
    kwargs = {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }

    if model_type == "api":
        kwargs["api_provider"] = args.provider or "gemini"
        if args.api_key:
            kwargs["api_key"] = args.api_key
        if args.model_name:
            kwargs["model_name"] = args.model_name
    else:
        if not args.model:
            print(f"--model required for type '{model_type}'")
            return 1
        kwargs["model_path"] = Path(args.model)

    generator = create_generator(model_type=model_type, **kwargs)

    # Generate
    instruction = args.instruction
    context = args.context or ""

    sample = generator.generate_one(instruction, context=context)

    if sample is None:
        print("Generation failed (quality check did not pass)")
        return 1

    print(f"Instruction: {instruction}")
    print("=" * 60)
    print(sample.output)
    print("=" * 60)
    print(f"Quality: {sample.quality_score:.2f}")
    if "generation_attempt" in sample._metadata:
        print(f"Attempts: {sample._metadata['generation_attempt']}")

    # Save if output specified
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(sample.output)
        print(f"\nSaved to: {output_path}")

    return 0


def generator_knowledge_command(args: argparse.Namespace) -> int:
    """Generate assembly code with ALTTP knowledge context."""
    from ..generators.knowledge_generator import create_knowledge_generator

    # Parse entities
    required_entities = []
    if args.entities:
        required_entities = [e.strip() for e in args.entities.split(",")]

    # Create generator
    kwargs = {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
        "include_entity_context": True,
        "extract_entities_from_instruction": True,
    }

    if args.provider:
        kwargs["api_provider"] = args.provider
    if args.api_key:
        kwargs["api_key"] = args.api_key

    generator = create_knowledge_generator(**kwargs)

    # Suggest entities first if requested
    if args.suggest:
        suggestions = generator.suggest_entities(args.instruction)
        print(f"Suggested entities for: {args.instruction}")
        for entity in suggestions:
            print(f"  - {entity}")
        return 0

    # Generate
    sample = generator.generate_with_context(
        args.instruction,
        required_entities=required_entities,
    )

    if sample is None:
        print("Generation failed (quality check did not pass)")
        return 1

    print(f"Instruction: {args.instruction}")
    if required_entities:
        print(f"Required entities: {', '.join(required_entities)}")
    print("=" * 60)
    print(sample.output)
    print("=" * 60)
    print(f"Quality: {sample.quality_score:.2f}")

    # Show knowledge context info
    if "knowledge_context" in sample._metadata:
        kc = sample._metadata["knowledge_context"]
        if kc.get("entity_hints"):
            print(f"Extracted hints: {', '.join(kc['entity_hints'])}")
        if kc.get("context_entities"):
            print(f"Context entities: {', '.join(kc['context_entities'])}")

    # Save if output specified
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(sample.output)
        print(f"\nSaved to: {output_path}")

    return 0


def generator_batch_command(args: argparse.Namespace) -> int:
    """Generate assembly code for multiple instructions."""
    from ..generators.knowledge_generator import create_knowledge_generator

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Input not found: {input_path}")
        return 1

    # Load instructions
    instructions = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                # Support JSON lines or plain text
                if line.startswith("{"):
                    data = json.loads(line)
                    instructions.append(data.get("instruction", data.get("text", line)))
                else:
                    instructions.append(line)

    if not instructions:
        print("No instructions found in input file")
        return 1

    print(f"Loaded {len(instructions)} instructions")

    # Create generator
    kwargs = {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }
    if args.provider:
        kwargs["api_provider"] = args.provider
    if args.api_key:
        kwargs["api_key"] = args.api_key

    generator = create_knowledge_generator(**kwargs)

    # Generate batch
    samples = generator.generate_batch(instructions)

    print(f"Generated {len(samples)}/{len(instructions)} samples")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for sample in samples:
            record = {
                "sample_id": sample.sample_id,
                "instruction": sample.instruction,
                "output": sample.output,
                "quality_score": sample.quality_score,
                "domain": sample.domain,
            }
            f.write(json.dumps(record) + "\n")

    print(f"Saved to: {output_path}")
    return 0


def generator_scale_command(args: argparse.Namespace) -> int:
    """Generate large-scale training data using curriculum learning."""
    import asyncio
    from ..generators.curriculum_generator import (
        create_curriculum_generator,
        Difficulty,
        ExpertDomain,
    )
    from ..generators import write_jsonl

    # Validate domain
    try:
        domain = ExpertDomain(args.domain.lower())
    except ValueError:
        print(f"Invalid domain: {args.domain}")
        print(f"Valid domains: {', '.join(d.value for d in ExpertDomain)}")
        return 1

    # Parse providers
    providers = args.providers.split(",") if args.providers else ["gemini"]

    # Create generator
    generator = create_curriculum_generator(
        domain=domain,
        target_per_difficulty=args.target_per_difficulty,
        providers=providers,
    )

    # Load checkpoint if resuming
    if args.resume:
        checkpoint_path = Path(args.resume).expanduser()
        if checkpoint_path.exists():
            print(f"Resuming from checkpoint: {checkpoint_path}")
            generator.load_checkpoint(checkpoint_path)
        else:
            print(f"Checkpoint not found: {checkpoint_path}")
            return 1

    # Parse difficulty filter
    difficulty = None
    if args.difficulty:
        try:
            difficulty = Difficulty(args.difficulty.lower())
        except ValueError:
            print(f"Invalid difficulty: {args.difficulty}")
            print(f"Valid difficulties: {', '.join(d.value for d in Difficulty)}")
            return 1

    # Progress callback
    def on_progress(progress):
        print(f"\r[{progress.total_generated} samples | "
              f"{progress.samples_per_minute():.1f}/min | "
              f"Failed: {progress.total_failed}]", end="", flush=True)

    print(f"Generating {args.count} samples for {domain.value}")
    print(f"Providers: {', '.join(providers)}")
    if difficulty:
        print(f"Difficulty: {difficulty.value}")
    print()

    # Generate
    try:
        samples = generator.generate_batch_sync(
            count=args.count,
            difficulty=difficulty,
            progress_callback=on_progress,
        )
    except KeyboardInterrupt:
        print("\n\nInterrupted! Saving checkpoint...")
        checkpoint_path = generator.save_checkpoint()
        print(f"Checkpoint saved: {checkpoint_path}")
        return 1

    print()  # Newline after progress

    # Save results
    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    count = write_jsonl(samples, output_path)
    print(f"\nGenerated {count} samples")
    print(f"Output: {output_path}")

    # Show summary
    print(f"\n{generator._progress.summary()}")

    # Save checkpoint if requested
    if args.checkpoint:
        checkpoint_path = generator.save_checkpoint()
        print(f"\nCheckpoint: {checkpoint_path}")

    return 0


def register_parsers(subparsers: argparse._SubParsersAction) -> None:
    """Register generator command parsers."""
    gen_parser = subparsers.add_parser(
        "generator", help="Generate assembly code using models."
    )
    gen_sub = gen_parser.add_subparsers(dest="generator_command")

    # generator model
    gen_model = gen_sub.add_parser(
        "model", help="Generate assembly code using a model."
    )
    gen_model.add_argument(
        "--instruction", "-i", required=True, help="Generation instruction."
    )
    gen_model.add_argument(
        "--model", "-m", help="Path to local model."
    )
    gen_model.add_argument(
        "--type", "-t",
        choices=["mlx", "huggingface", "llama_cpp", "api"],
        default="api",
        help="Model type (default: api).",
    )
    gen_model.add_argument(
        "--provider", "-p",
        choices=["gemini", "claude", "openai"],
        default="gemini",
        help="API provider (default: gemini).",
    )
    gen_model.add_argument(
        "--model-name", help="Model name for API provider."
    )
    gen_model.add_argument(
        "--api-key", help="API key (or set via environment)."
    )
    gen_model.add_argument(
        "--context", "-c", help="Additional context for generation."
    )
    gen_model.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7).",
    )
    gen_model.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum tokens to generate (default: 1024).",
    )
    gen_model.add_argument(
        "--output", "-o", help="Output file path."
    )
    gen_model.set_defaults(func=generator_model_command)

    # generator knowledge
    gen_knowledge = gen_sub.add_parser(
        "knowledge", help="Generate with ALTTP knowledge context."
    )
    gen_knowledge.add_argument(
        "--instruction", "-i", required=True, help="Generation instruction."
    )
    gen_knowledge.add_argument(
        "--entities", "-e",
        help="Required entities (comma-separated, e.g., 'link_health,sprite_x').",
    )
    gen_knowledge.add_argument(
        "--suggest",
        action="store_true",
        help="Only suggest relevant entities, don't generate.",
    )
    gen_knowledge.add_argument(
        "--provider", "-p",
        choices=["gemini", "claude", "openai"],
        default="gemini",
        help="API provider (default: gemini).",
    )
    gen_knowledge.add_argument(
        "--api-key", help="API key (or set via environment)."
    )
    gen_knowledge.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7).",
    )
    gen_knowledge.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum tokens to generate (default: 1024).",
    )
    gen_knowledge.add_argument(
        "--output", "-o", help="Output file path."
    )
    gen_knowledge.set_defaults(func=generator_knowledge_command)

    # generator batch
    gen_batch = gen_sub.add_parser(
        "batch", help="Batch generation from instruction file."
    )
    gen_batch.add_argument(
        "--input", required=True, help="Input file with instructions (text or JSONL)."
    )
    gen_batch.add_argument(
        "--output", "-o", required=True, help="Output JSONL file."
    )
    gen_batch.add_argument(
        "--provider", "-p",
        choices=["gemini", "claude", "openai"],
        default="gemini",
        help="API provider (default: gemini).",
    )
    gen_batch.add_argument(
        "--api-key", help="API key (or set via environment)."
    )
    gen_batch.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature (default: 0.7).",
    )
    gen_batch.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum tokens to generate (default: 1024).",
    )
    gen_batch.set_defaults(func=generator_batch_command)

    # generator scale
    gen_scale = gen_sub.add_parser(
        "scale", help="Large-scale curriculum-based data generation."
    )
    gen_scale.add_argument(
        "--domain", "-d", required=True,
        choices=["din", "nayru", "farore", "veran"],
        help="Expert domain to generate for.",
    )
    gen_scale.add_argument(
        "--count", "-n", type=int, default=1000,
        help="Number of samples to generate (default: 1000).",
    )
    gen_scale.add_argument(
        "--output", "-o", required=True,
        help="Output JSONL file path.",
    )
    gen_scale.add_argument(
        "--providers", "-p",
        default="gemini",
        help="Comma-separated list of providers (gemini,claude,openai).",
    )
    gen_scale.add_argument(
        "--difficulty",
        choices=["basic", "intermediate", "advanced", "expert"],
        help="Generate only this difficulty level (default: all).",
    )
    gen_scale.add_argument(
        "--target-per-difficulty",
        type=int,
        default=2500,
        help="Target samples per difficulty level (default: 2500).",
    )
    gen_scale.add_argument(
        "--resume",
        help="Path to checkpoint file to resume from.",
    )
    gen_scale.add_argument(
        "--checkpoint",
        action="store_true",
        help="Save checkpoint after completion.",
    )
    gen_scale.set_defaults(func=generator_scale_command)
