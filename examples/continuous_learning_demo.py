#!/usr/bin/env python3
"""Demo of the continuous learning system.

Demonstrates:
1. Logging model usage
2. Recording user feedback
3. Automatic retraining when thresholds are met
4. A/B testing new models
5. Auto-promotion of better models
"""

import logging
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from afs.continuous import (
    ContinuousLearningLoop,
    LoopConfig,
    TriggerConfig,
    DataGeneratorConfig,
    ABTestConfig,
    UsageLogger,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def demo_basic_logging():
    """Demo 1: Basic usage logging."""
    print("\n" + "=" * 60)
    print("DEMO 1: Basic Usage Logging")
    print("=" * 60)

    db_path = Path("/tmp/continuous_demo.db")
    db_path.unlink(missing_ok=True)  # Clean slate

    usage_logger = UsageLogger(db_path)

    # Log some usage
    queries = [
        ("What is 65816 assembly?", "65816 is the CPU used in the SNES...", 0.85),
        ("How do I write LDA instruction?", "LDA loads a value into the accumulator...", 0.90),
        ("Debug my code", "Here are some tips for debugging...", 0.65),
    ]

    for query, response, quality in queries:
        record_id = usage_logger.log(
            query=query,
            response=response,
            model="nayru-v1",
            expert="asm",
            latency_ms=150,
            quality_score=quality,
        )
        print(f"Logged: {record_id} (quality: {quality:.2f})")

    # Record feedback
    print("\nRecording user feedback...")
    records = list(usage_logger.get_records(limit=3))
    usage_logger.record_feedback(records[0].id, feedback=1, feedback_text="Great!")
    usage_logger.record_feedback(records[1].id, feedback=1, feedback_text="Helpful")
    usage_logger.record_feedback(records[2].id, feedback=-1, feedback_text="Not clear")

    # Show stats
    stats = usage_logger.get_statistics()
    print("\nUsage Statistics:")
    print(f"  Total records: {stats['total']}")
    print(f"  With feedback: {stats['with_feedback']}")
    print(f"  Positive: {stats['positive_feedback']}")
    print(f"  Negative: {stats['negative_feedback']}")
    print(f"  Avg quality: {stats['avg_quality_score']:.3f}")

    db_path.unlink()  # Cleanup


def demo_data_generation():
    """Demo 2: Training data generation."""
    print("\n" + "=" * 60)
    print("DEMO 2: Training Data Generation")
    print("=" * 60)

    db_path = Path("/tmp/continuous_demo.db")
    output_dir = Path("/tmp/continuous_demo_output")
    db_path.unlink(missing_ok=True)
    output_dir.mkdir(exist_ok=True)

    usage_logger = UsageLogger(db_path)

    # Simulate production usage
    print("Simulating production usage...")
    for i in range(50):
        quality = 0.7 + (i % 3) * 0.1  # Mix of quality scores
        record_id = usage_logger.log(
            query=f"Question {i}",
            response=f"Answer {i}",
            model="nayru-v1",
            quality_score=quality,
        )

        # Add feedback to some
        if i % 5 == 0:
            feedback = 1 if quality > 0.75 else -1
            usage_logger.record_feedback(record_id, feedback)

    # Generate training data
    from afs.continuous import TrainingDataGenerator

    config = DataGeneratorConfig(
        min_quality_score=0.75,
        min_user_feedback=1,
        deduplicate=True,
    )

    generator = TrainingDataGenerator(usage_logger, config)
    result = generator.generate(output_dir / "training.jsonl")

    print(f"\nGeneration Result:")
    print(f"  Total candidates: {result.total_candidates}")
    print(f"  Filtered by quality: {result.filtered_by_quality}")
    print(f"  Filtered by feedback: {result.filtered_by_feedback}")
    print(f"  Duplicates removed: {result.duplicates_removed}")
    print(f"  Final count: {result.final_count}")
    print(f"  Output: {result.output_path}")

    # Show sample
    if result.final_count > 0:
        import json

        with open(result.output_path) as f:
            sample = json.loads(f.readline())
            print(f"\nSample training data:")
            print(json.dumps(sample, indent=2))

    # Cleanup
    db_path.unlink()
    import shutil

    shutil.rmtree(output_dir)


def demo_triggers():
    """Demo 3: Automatic retraining triggers."""
    print("\n" + "=" * 60)
    print("DEMO 3: Automatic Retraining Triggers")
    print("=" * 60)

    db_path = Path("/tmp/continuous_demo.db")
    output_dir = Path("/tmp/continuous_demo_output")
    db_path.unlink(missing_ok=True)
    output_dir.mkdir(exist_ok=True)

    usage_logger = UsageLogger(db_path)

    # Configure triggers
    trigger_config = TriggerConfig(
        min_new_samples=10,  # Low threshold for demo
        schedule_interval_hours=1,
        cooldown_hours=0,  # No cooldown for demo
    )

    from afs.continuous import AutoRetrainer

    retrainer = AutoRetrainer(
        usage_logger,
        trigger_config=trigger_config,
        output_dir=output_dir,
    )

    # Simulate usage below threshold
    print("Simulating 5 samples (below threshold)...")
    for i in range(5):
        usage_logger.log(
            query=f"Query {i}",
            response=f"Response {i}",
            model="nayru-v1",
            quality_score=0.8,
        )

    result = retrainer.check_and_retrain()
    if result:
        print("Retrain triggered!")
    else:
        print("No retrain triggered (as expected)")

    # Add more samples to reach threshold
    print("\nAdding 10 more samples (reach threshold)...")
    for i in range(5, 15):
        usage_logger.log(
            query=f"Query {i}",
            response=f"Response {i}",
            model="nayru-v1",
            quality_score=0.8,
        )

    result = retrainer.check_and_retrain()
    if result:
        print("Retrain triggered!")
        print(f"  Status: {result['status']}")
        print(f"  Samples: {result['generation']['final_count']}")
    else:
        print("No retrain triggered")

    # Cleanup
    db_path.unlink()
    import shutil

    shutil.rmtree(output_dir)


def demo_ab_testing():
    """Demo 4: A/B testing."""
    print("\n" + "=" * 60)
    print("DEMO 4: A/B Testing")
    print("=" * 60)

    db_path = Path("/tmp/continuous_demo.db")
    db_path.unlink(missing_ok=True)

    usage_logger = UsageLogger(db_path)

    from afs.continuous import ABTestManager, ABTestConfig

    config = ABTestConfig(
        initial_challenger_traffic=0.1,  # 10% to challenger
        min_samples_per_version=5,  # Low for demo
        min_duration_hours=0,  # No minimum for demo
    )

    ab_test = ABTestManager(usage_logger, config)

    # Deploy champion
    champion_path = Path("/tmp/champion.model")
    champion_path.touch()

    from afs.continuous import ModelVersion, ModelStatus

    ab_test.champion = ModelVersion(
        id="champion_v1",
        name="nayru-v1",
        path=champion_path,
        status=ModelStatus.CHAMPION,
        traffic_weight=0.9,
    )

    # Deploy challenger
    challenger_path = Path("/tmp/challenger.model")
    challenger_path.touch()

    challenger = ab_test.deploy_challenger(
        model_name="nayru-v2",
        model_path=challenger_path,
        traffic_weight=0.1,
    )

    print(f"Champion: {ab_test.champion.name} ({ab_test.champion.traffic_weight*100:.0f}%)")
    print(f"Challenger: {challenger.name} ({challenger.traffic_weight*100:.0f}%)")

    # Simulate traffic routing
    print("\nSimulating 100 requests:")
    champion_count = 0
    challenger_count = 0

    for _ in range(100):
        routed = ab_test.route_request()
        if routed == ab_test.champion:
            champion_count += 1
        else:
            challenger_count += 1

    print(f"  Champion: {champion_count} requests")
    print(f"  Challenger: {challenger_count} requests")

    # Cleanup
    db_path.unlink()
    champion_path.unlink()
    challenger_path.unlink()


def demo_full_loop():
    """Demo 5: Full continuous learning loop (one iteration)."""
    print("\n" + "=" * 60)
    print("DEMO 5: Full Continuous Learning Loop")
    print("=" * 60)

    # Setup
    db_path = Path("/tmp/continuous_demo.db")
    output_dir = Path("/tmp/continuous_demo_output")
    db_path.unlink(missing_ok=True)
    output_dir.mkdir(exist_ok=True)

    # Configure with low thresholds for demo
    config = LoopConfig(
        db_path=db_path,
        output_dir=output_dir,
        trigger_config=TriggerConfig(
            min_new_samples=10,
            cooldown_hours=0,
        ),
        generator_config=DataGeneratorConfig(
            min_quality_score=0.7,
        ),
        enable_ab_testing=False,  # Disable for simpler demo
    )

    loop = ContinuousLearningLoop(config)

    # Simulate production usage
    print("Simulating production usage...")
    for i in range(20):
        record_id = loop.log_usage(
            query=f"Query {i}",
            response=f"Response {i}",
            model="nayru-v1",
            quality_score=0.75 + (i % 3) * 0.05,
        )

        # Add feedback to some
        if i % 3 == 0:
            loop.record_feedback(record_id, feedback=1)

    # Run one iteration
    print("\nRunning continuous learning iteration...")

    def mock_train_fn(data_path: Path) -> dict:
        """Mock training function."""
        print(f"  Training on: {data_path}")
        return {"loss": 0.25, "accuracy": 0.85}

    result = loop.run_iteration(train_fn=mock_train_fn)

    print("\nIteration Result:")
    import json

    print(json.dumps(result, indent=2, default=str))

    # Show status
    print("\nLoop Status:")
    summary = loop.get_status_summary()
    print(f"  Total retrains: {summary['status']['total_retrains']}")
    print(f"  Total usage records: {summary['usage_stats']['total']}")

    # Cleanup
    db_path.unlink()
    import shutil

    shutil.rmtree(output_dir)


def main():
    """Run all demos."""
    print("\n" + "=" * 60)
    print("CONTINUOUS LEARNING SYSTEM DEMO")
    print("=" * 60)

    try:
        demo_basic_logging()
        time.sleep(1)

        demo_data_generation()
        time.sleep(1)

        demo_triggers()
        time.sleep(1)

        demo_ab_testing()
        time.sleep(1)

        demo_full_loop()

        print("\n" + "=" * 60)
        print("ALL DEMOS COMPLETED SUCCESSFULLY!")
        print("=" * 60)

    except Exception as e:
        logger.error(f"Demo failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
