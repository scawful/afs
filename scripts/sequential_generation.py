#!/usr/bin/env python3
"""Sequential data generation for overnight runs.

Generates training data for all 4 domains sequentially to avoid rate limit issues.
"""

import asyncio
import logging
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from afs.generators.curriculum_generator import (
    CurriculumGenerator,
    ExpertDomain,
    ScaleConfig,
    ProviderConfig,
    GenerationProgress,
)
from afs.generators.base import write_jsonl

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def generate_domain(domain: ExpertDomain, count: int, output_dir: Path) -> int:
    """Generate samples for a single domain."""
    logger.info(f"Starting generation for {domain.value}: {count} samples")

    config = ScaleConfig(
        target_samples_per_difficulty=count // 4,
        providers=[
            ProviderConfig(
                name="gemini",
                model="gemini-2.0-flash-exp",
                requests_per_minute=8,  # Conservative for free tier
            ),
        ],
    )

    generator = CurriculumGenerator(domain=domain, config=config)

    def progress_callback(progress: GenerationProgress):
        logger.info(
            f"[{domain.value}] {progress.total_generated}/{count} "
            f"({progress.samples_per_minute():.1f}/min)"
        )

    samples = await generator.generate_batch_async(
        count=count,
        progress_callback=progress_callback,
    )

    output_path = output_dir / f"{domain.value}_train.jsonl"
    saved = generator.export_samples(output_path)
    logger.info(f"Saved {saved} samples to {output_path}")

    return saved


async def main():
    """Run sequential generation for all domains."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--count", type=int, default=250,
                        help="Samples per domain (default: 250)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output directory")
    args = parser.parse_args()

    # Setup output directory
    if args.output:
        output_dir = Path(args.output)
    else:
        output_dir = Path("generated_data") / f"overnight_{datetime.now():%Y%m%d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Generating {args.count} samples per domain")

    # Generate for each domain sequentially
    domains = [ExpertDomain.DIN, ExpertDomain.NAYRU, ExpertDomain.FARORE, ExpertDomain.VERAN]
    total = 0

    for domain in domains:
        try:
            count = await generate_domain(domain, args.count, output_dir)
            total += count
        except Exception as e:
            logger.error(f"Failed to generate {domain.value}: {e}")

    logger.info(f"Generation complete! Total samples: {total}")
    logger.info(f"Output: {output_dir}")


if __name__ == "__main__":
    asyncio.run(main())
