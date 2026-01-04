#!/usr/bin/env python3
"""Pilot test for distillation pipeline.

Tests each provider individually, then runs a small batch.
"""

import asyncio
import os
import sys
import logging
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


async def test_single_provider(teacher_cls, name: str) -> bool:
    """Test a single provider."""
    logger.info(f"\n{'='*50}")
    logger.info(f"Testing {name}...")
    logger.info(f"{'='*50}")

    try:
        teacher = teacher_cls()

        response = await teacher.generate(
            prompt="Write 65816 assembly to store $42 at address $7E0100",
            system_prompt="You are an expert 65816 assembly programmer for SNES."
        )

        if response.success:
            logger.info(f"[OK] {name} succeeded!")
            logger.info(f"  Model: {response.model}")
            logger.info(f"  Latency: {response.latency_ms:.0f}ms")
            logger.info(f"  Response length: {len(response.content)} chars")
            logger.info(f"  First 200 chars: {response.content[:200]}...")
            return True
        else:
            logger.error(f"[FAIL] {name} failed: {response.error}")
            return False

    except Exception as e:
        logger.error(f"[FAIL] {name} exception: {e}")
        return False


async def test_ensemble_batch(count: int = 5) -> bool:
    """Test ensemble with a small batch."""
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

    logger.info(f"\n{'='*50}")
    logger.info(f"Testing ensemble batch ({count} samples)...")
    logger.info(f"{'='*50}")

    # Only use providers with valid keys
    teachers = []
    if os.getenv("OPENAI_API_KEY"):
        teachers.append(OpenAITeacher())
        logger.info("  Added OpenAI teacher")
    if os.getenv("GEMINI_API_KEY"):
        teachers.append(GoogleTeacher())
        logger.info("  Added Google teacher")
    if os.getenv("CLAUDE_API_KEY"):
        teachers.append(AnthropicTeacher())
        logger.info("  Added Anthropic teacher")

    if not teachers:
        logger.error("No API keys found!")
        return False

    ensemble = TeacherEnsemble(teachers=teachers)

    config = DistillationConfig(
        target_count=count,
        min_quality_score=0.3,  # Lower for pilot
        checkpoint_interval=5,
        output_dir=Path("pilot_output"),
        domains=["din", "nayru"],  # Subset for pilot
    )

    generator = DistillationDataGenerator(
        ensemble=ensemble,
        config=config,
    )

    def progress_callback(progress):
        logger.info(
            f"  Progress: {progress.generated}/{progress.total_requested} "
            f"({progress.samples_per_minute:.1f}/min) | "
            f"Failed: {progress.failed}"
        )

    try:
        samples = await generator.generate_batch(
            count=count,
            progress_callback=progress_callback,
        )

        stats = generator.get_statistics()
        logger.info(f"\n[OK] Batch complete!")
        logger.info(f"  Total samples: {stats['total_samples']}")
        logger.info(f"  Providers: {stats['providers']}")
        logger.info(f"  Domains: {stats['domains']}")
        logger.info(f"  Avg quality: {stats['avg_quality']:.2f}")

        return True

    except Exception as e:
        logger.error(f"[FAIL] Batch failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def main():
    """Run all pilot tests."""
    from afs.distillation.teacher import (
        OpenAITeacher,
        GoogleTeacher,
        AnthropicTeacher,
    )

    logger.info("=" * 60)
    logger.info("DISTILLATION PILOT TEST")
    logger.info("=" * 60)

    # Check API keys
    logger.info("\nAPI Key Status:")
    keys = {
        "OPENAI_API_KEY": bool(os.getenv("OPENAI_API_KEY")),
        "GEMINI_API_KEY": bool(os.getenv("GEMINI_API_KEY")),
        "CLAUDE_API_KEY": bool(os.getenv("CLAUDE_API_KEY")),
    }
    for key, present in keys.items():
        status = "[+]" if present else "[-]"
        logger.info(f"  {status} {key}")

    results = {}

    # Test individual providers (only if key exists)
    if os.getenv("OPENAI_API_KEY"):
        results["OpenAI"] = await test_single_provider(OpenAITeacher, "OpenAI")
    else:
        logger.warning("Skipping OpenAI (no API key)")
        results["OpenAI"] = None

    if os.getenv("GEMINI_API_KEY"):
        results["Google"] = await test_single_provider(GoogleTeacher, "Google")
    else:
        logger.warning("Skipping Google (no API key)")
        results["Google"] = None

    if os.getenv("CLAUDE_API_KEY"):
        results["Anthropic"] = await test_single_provider(AnthropicTeacher, "Anthropic")
    else:
        logger.warning("Skipping Anthropic (no API key)")
        results["Anthropic"] = None

    # Test ensemble batch
    working_providers = sum(1 for v in results.values() if v is True)
    if working_providers > 0:
        results["Ensemble"] = await test_ensemble_batch(count=5)
    else:
        logger.error("No working providers - skipping ensemble test")
        results["Ensemble"] = False

    # Summary
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY")
    logger.info("=" * 60)
    for name, success in results.items():
        if success is None:
            status = "[SKIP]"
        elif success:
            status = "[PASS]"
        else:
            status = "[FAIL]"
        logger.info(f"  {status} {name}")

    all_passed = all(v in (True, None) for v in results.values()) and results.get("Ensemble", False)

    if all_passed:
        logger.info("\n[OK] All tests passed! Ready for large generation.")
        return 0
    else:
        logger.error("\n[FAIL] Some tests failed. Fix issues before large generation.")
        return 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
