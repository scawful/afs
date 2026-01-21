#!/usr/bin/env python3
"""Main script for running the continuous learning loop.

Usage:
    # Run indefinitely
    python scripts/continuous_loop.py

    # Run with custom config
    python scripts/continuous_loop.py --config config.json

    # Run for testing (limited iterations)
    python scripts/continuous_loop.py --max-iterations 5

    # Check status
    python scripts/continuous_loop.py --status

    # Manual trigger
    python scripts/continuous_loop.py --trigger
"""

import argparse
import json
import logging
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from afs.continuous import (
    ContinuousLearningLoop,
    LoopConfig,
    TriggerConfig,
    DataGeneratorConfig,
    ABTestConfig,
)

logger = logging.getLogger(__name__)


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_config(config_path: Path) -> LoopConfig:
    """Load loop configuration from JSON file."""
    with open(config_path) as f:
        data = json.load(f)

    return LoopConfig(
        db_path=Path(data.get("db_path", "~/.context/training/continuous/usage.db")).expanduser(),
        output_dir=Path(data.get("output_dir", "~/.context/training/continuous")).expanduser(),
        models_dir=Path(data.get("models_dir", "~/.context/training/continuous/models")).expanduser(),
        trigger_config=TriggerConfig(**data.get("trigger_config", {})),
        generator_config=DataGeneratorConfig(**data.get("generator_config", {})),
        ab_test_config=ABTestConfig(**data.get("ab_test_config", {})),
        check_interval_seconds=data.get("check_interval_seconds", 3600),
        enable_ab_testing=data.get("enable_ab_testing", True),
        enable_auto_promotion=data.get("enable_auto_promotion", True),
    )


def example_training_function(data_path: Path) -> dict:
    """Example training function.

    Replace this with your actual training logic.

    Args:
        data_path: Path to training data (JSONL)

    Returns:
        Dict with training metrics and model_path
    """
    logger.info(f"Training on data: {data_path}")

    # Count samples
    with open(data_path) as f:
        samples = sum(1 for _ in f)

    logger.info(f"Dataset contains {samples} samples")

    # Simulate training
    # In real implementation:
    # 1. Load data
    # 2. Initialize/load model
    # 3. Train with LoRA or full fine-tuning
    # 4. Save model
    # 5. Evaluate on validation set
    # 6. Return metrics

    model_path = data_path.with_suffix(".model")
    logger.info(f"Model saved to {model_path}")

    return {
        "samples": samples,
        "epochs": 3,
        "loss": 0.25,
        "val_accuracy": 0.85,
        "model_path": str(model_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Continuous learning loop")
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to config JSON file",
    )
    parser.add_argument(
        "--max-iterations",
        type=int,
        help="Maximum iterations (for testing)",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show current status and exit",
    )
    parser.add_argument(
        "--trigger",
        action="store_true",
        help="Manually trigger retraining",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging",
    )
    parser.add_argument(
        "--no-train",
        action="store_true",
        help="Skip actual training (only prepare data)",
    )

    args = parser.parse_args()

    setup_logging(args.verbose)

    # Load config
    if args.config:
        config = load_config(args.config)
    else:
        config = LoopConfig()

    # Initialize loop
    loop = ContinuousLearningLoop(config)

    # Handle commands
    if args.status:
        # Show status
        summary = loop.get_status_summary()
        print(json.dumps(summary, indent=2, default=str))
        return

    if args.trigger:
        # Manual trigger
        logger.info("Manual retrain trigger")
        train_fn = None if args.no_train else example_training_function
        result = loop.run_iteration(train_fn)
        print(json.dumps(result, indent=2, default=str))
        return

    # Run loop
    logger.info("Starting continuous learning loop")
    logger.info(f"Database: {config.db_path}")
    logger.info(f"Output directory: {config.output_dir}")
    logger.info(f"Check interval: {config.check_interval_seconds}s")

    train_fn = None if args.no_train else example_training_function

    try:
        loop.run_loop(
            train_fn=train_fn,
            max_iterations=args.max_iterations,
        )
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        loop.stop()


if __name__ == "__main__":
    main()
