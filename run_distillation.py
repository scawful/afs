#!/usr/bin/env python3
"""Run large-scale distillation data generation.

Usage:
    .venv/bin/python run_distillation.py --count 1000 --output distillation_data/

Generates training data from multi-provider teacher ensemble with:
- OpenAI (gpt-5.2)
- Google (gemini-2.5-flash)
- Anthropic (claude-opus-4.5)
"""

import asyncio
import argparse
import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(f"distillation_{datetime.now():%Y%m%d_%H%M%S}.log")
    ]
)
logger = logging.getLogger(__name__)


async def main():
    parser = argparse.ArgumentParser(description="Run distillation data generation")
    parser.add_argument("--count", type=int, default=1000, help="Number of samples")
    parser.add_argument("--output", type=str, default="distillation_data", help="Output dir")
    parser.add_argument("--domains", nargs="+", default=["din", "nayru", "farore", "veran"])
    parser.add_argument("--min-quality", type=float, default=0.5)
    args = parser.parse_args()

    from afs.distillation import (
        TeacherEnsemble,
        DistillationDataGenerator,
        DistillationConfig,
    )
    from afs.distillation.teacher import (
        OpenAITeacher,
        GoogleTeacher,
        AnthropicTeacher,
    )

    logger.info("=" * 60)
    logger.info("DISTILLATION DATA GENERATION")
    logger.info("=" * 60)
    logger.info(f"Target samples: {args.count}")
    logger.info(f"Output: {args.output}")
    logger.info(f"Domains: {args.domains}")
    logger.info(f"Min quality: {args.min_quality}")
    logger.info("=" * 60)

    # Build ensemble with all available providers
    teachers = []
    if os.getenv("OPENAI_API_KEY"):
        teachers.append(OpenAITeacher())
        logger.info("Added OpenAI (gpt-5.2)")
    if os.getenv("GEMINI_API_KEY"):
        teachers.append(GoogleTeacher())
        logger.info("Added Google (gemini-3-flash-preview)")
    if os.getenv("CLAUDE_API_KEY"):
        teachers.append(AnthropicTeacher())
        logger.info("Added Anthropic (claude-opus-4.5)")

    if not teachers:
        logger.error("No API keys found!")
        return 1

    ensemble = TeacherEnsemble(teachers=teachers)

    config = DistillationConfig(
        target_count=args.count,
        min_quality_score=args.min_quality,
        checkpoint_interval=50,
        output_dir=Path(args.output),
        domains=args.domains,
    )

    generator = DistillationDataGenerator(
        ensemble=ensemble,
        config=config,
    )

    start_time = datetime.now()

    def progress_callback(progress):
        elapsed = (datetime.now() - start_time).total_seconds()
        eta_minutes = (progress.total_requested - progress.generated) / progress.samples_per_minute if progress.samples_per_minute > 0 else 0
        logger.info(
            f"Progress: {progress.generated}/{progress.total_requested} "
            f"({progress.samples_per_minute:.1f}/min) | "
            f"Failed: {progress.failed} | "
            f"Quality filtered: {progress.quality_filtered} | "
            f"ETA: {eta_minutes:.0f} min"
        )

    try:
        samples = await generator.generate_batch(
            count=args.count,
            progress_callback=progress_callback,
        )

        # Export to training formats
        output_path = Path(args.output)
        for fmt in ["chatml", "alpaca"]:
            export_path = output_path / f"train_{fmt}.jsonl"
            generator.export_training_data(export_path, format_type=fmt)
            logger.info(f"Exported {fmt} format to {export_path}")

        stats = generator.get_statistics()
        elapsed = (datetime.now() - start_time).total_seconds()

        logger.info("=" * 60)
        logger.info("GENERATION COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Total samples: {stats['total_samples']}")
        logger.info(f"By provider: {stats['providers']}")
        logger.info(f"By domain: {stats['domains']}")
        logger.info(f"Average quality: {stats['avg_quality']:.2f}")
        logger.info(f"Total time: {elapsed/60:.1f} minutes")
        logger.info(f"Rate: {stats['total_samples']/elapsed*60:.1f} samples/min")
        logger.info("=" * 60)

        return 0

    except KeyboardInterrupt:
        logger.info("Generation interrupted. Progress saved to checkpoint.")
        stats = generator.get_statistics()
        logger.info(f"Samples generated before interrupt: {stats['total_samples']}")
        return 0
    except Exception as e:
        logger.error(f"Generation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
