#!/usr/bin/env python3
"""Comprehensive demo of the model registry and version management system."""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from afs.registry import (
    EvaluationScores,
    LineageTracker,
    ModelRegistry,
    ModelVersion,
    TrainingMetadata,
)


def print_section(title: str) -> None:
    """Print a formatted section header."""
    print(f"\n{'=' * 70}")
    print(f"  {title}")
    print(f"{'=' * 70}\n")


def demo_basic_operations() -> None:
    """Demonstrate basic registry operations."""
    print_section("Basic Registry Operations")

    registry = ModelRegistry()

    # List all models
    print("1. Available Models:")
    models = registry.list_models()
    for model_name in models:
        print(f"   - {model_name}")

    # Get model info
    print("\n2. Model Info (majora):")
    majora = registry.get_model("majora")
    if majora:
        print(f"   Name: {majora.model_name}")
        print(f"   Created: {majora.created_at}")
        print(f"   Tags: {', '.join(majora.tags)}")

    # List versions
    print("\n3. Versions of majora:")
    versions = registry.list_versions("majora")
    for v in versions:
        status = "●" if v.deployed else "○"
        accuracy = (
            f"accuracy={v.evaluation_scores.accuracy:.3f}"
            if v.evaluation_scores.accuracy
            else ""
        )
        print(f"   {status} {v.version:8} ({v.status.value:10}) {accuracy}")

    # Get latest version
    print("\n4. Latest Version:")
    latest = registry.get_latest("majora")
    if latest:
        print(f"   Version: {latest.version}")
        print(f"   Status: {latest.status.value}")
        print(f"   Deployed: {latest.deployed}")


def demo_version_details() -> None:
    """Demonstrate detailed version information."""
    print_section("Detailed Version Information")

    registry = ModelRegistry()
    version = registry.get_version("majora", "v1")

    if not version:
        print("Version v1 not found")
        return

    print(f"Model: {version.model_name}:{version.version}")
    print(f"Status: {version.status.value}")
    print(f"Created: {version.created_at}")

    print("\nTraining Details:")
    if version.training:
        t = version.training
        print(f"  Framework: {t.framework}")
        print(f"  Base Model: {t.base_model}")
        print(f"  Samples: {t.samples}")
        print(f"  Epochs: {t.epochs}")
        print(f"  Batch Size: {t.batch_size}")
        print(f"  Learning Rate: {t.learning_rate}")
        if t.duration_hours:
            print(f"  Duration: {t.duration_hours:.2f} hours")
        if t.cost_usd:
            print(f"  Cost: ${t.cost_usd:.2f}")

    print("\nEvaluation Scores:")
    scores = version.evaluation_scores
    print(f"  Accuracy: {scores.accuracy}")
    print(f"  F1 Score: {scores.f1_score}")
    print(f"  Perplexity: {scores.perplexity}")
    print(f"  BLEU Score: {scores.bleu_score}")
    if scores.inference_speed_tokens_per_sec:
        print(f"  Speed: {scores.inference_speed_tokens_per_sec:.0f} tokens/sec")

    print("\nFile Locations:")
    if version.lora_path:
        print(f"  LoRA: {version.lora_path}")
    if version.gguf_path:
        print(f"  GGUF: {version.gguf_path}")
    if version.checkpoint_path:
        print(f"  Checkpoint: {version.checkpoint_path}")

    print("\nMetadata:")
    print(f"  Training Data: {', '.join(version.training_data_sources)}")
    if version.git_commit:
        print(f"  Git Commit: {version.git_commit}")
    if version.tags:
        print(f"  Tags: {', '.join(version.tags)}")
    if version.notes:
        print(f"  Notes: {version.notes}")


def demo_version_comparison() -> None:
    """Demonstrate comparing versions."""
    print_section("Version Comparison")

    registry = ModelRegistry()

    try:
        diff = registry.compare_versions("majora", "v1", "v2")

        print("Differences between v1 and v2:\n")
        print(f"{'Metric':<30} {'v1':<15} {'v2':<15}")
        print("-" * 60)

        for key, values in sorted(diff.items()):
            v1_val = values["v1"]
            v2_val = values["v2"]
            v1_str = str(v1_val)[:14] if v1_val is not None else "None"
            v2_str = str(v2_val)[:14] if v2_val is not None else "None"
            print(f"{key:<30} {v1_str:<15} {v2_str:<15}")

        # Highlight improvements
        print("\nImprovement Analysis:")
        for key, values in diff.items():
            if key.startswith(("accuracy", "f1", "bleu")):
                v1 = values["v1"]
                v2 = values["v2"]
                if v1 is not None and v2 is not None:
                    improvement = ((v2 - v1) / v1) * 100
                    direction = "↑" if improvement > 0 else "↓"
                    print(f"  {key}: {direction} {abs(improvement):.1f}%")

    except ValueError as e:
        print(f"Error: {e}")


def demo_lineage() -> None:
    """Demonstrate lineage tracking."""
    print_section("Model Lineage Tracking")

    tracker = LineageTracker()

    # Show version tree
    print("Version Tree:")
    print(tracker.build_tree("majora"))

    # Get training history
    print("\nTraining History:")
    history = tracker.get_training_history("majora")
    for record in history:
        print(f"\n{record['version']}:")
        if record.get("parent_version"):
            print(f"  Parent: {record['parent_version']}")
        if record.get("training_data"):
            print(f"  Data: {', '.join(record['training_data'])}")
        if record.get("git_commit"):
            print(f"  Commit: {record['git_commit'][:8]}")

    # Get ancestors
    print("\nAncestors of v2:")
    ancestors = tracker.get_ancestors("majora", "v2")
    if ancestors:
        print(f"  {' <- '.join(ancestors)}")
    else:
        print("  None (root version)")

    # Get data lineage
    print("\nData Lineage for v2:")
    data_lineage = tracker.get_data_lineage("majora", "v2")
    for version, sources in data_lineage.items():
        print(f"  {version}: {sources}")


def demo_deployment() -> None:
    """Demonstrate deployment and rollback."""
    print_section("Deployment Management")

    registry = ModelRegistry()

    print("Current Deployment Status:")
    for version in registry.list_versions("majora"):
        status = "●" if version.deployed else "○"
        print(f"  {status} {version.version} (deployed: {version.deployed})")

    if registry.get_version("majora", "v1"):
        print("\nDeployment Timeline:")
        v1 = registry.get_version("majora", "v1")
        v2 = registry.get_version("majora", "v2")

        if v1:
            print(f"  v1: deployed={v1.deployed}, date={v1.deployed_at}")
        if v2:
            print(f"  v2: deployed={v2.deployed}, date={v2.deployed_at}")


def demo_summary() -> None:
    """Demonstrate summary reports."""
    print_section("Registry Summary Reports")

    registry = ModelRegistry()
    tracker = LineageTracker()

    print("Registry Summary (All Models):")
    print(registry.summary())

    print("\n\nRegistry Summary (majora only):")
    print(registry.summary("majora"))

    print("\n\nLineage Summary:")
    print(tracker.summary())


def demo_programmatic_workflow() -> None:
    """Demonstrate a complete programmatic workflow."""
    print_section("Complete Programmatic Workflow")

    registry = ModelRegistry()
    tracker = LineageTracker()

    print("Workflow Steps:\n")

    # Step 1: Get current best model
    print("1. Finding current best model...")
    best = max(
        registry.list_versions("majora"),
        key=lambda v: v.evaluation_scores.accuracy or 0,
    )
    print(f"   Best: {best.version} (accuracy={best.evaluation_scores.accuracy})")

    # Step 2: Get its lineage
    print("\n2. Getting lineage...")
    ancestors = tracker.get_ancestors("majora", best.version)
    print(f"   Lineage: {[best.version] + list(reversed(ancestors))}")

    # Step 3: Check deployment status
    print("\n3. Checking deployment...")
    deployed = [v for v in registry.list_versions("majora") if v.deployed]
    print(f"   Deployed versions: {[v.version for v in deployed]}")

    # Step 4: Show training data across versions
    print("\n4. Training data evolution...")
    for v in registry.list_versions("majora"):
        data = ", ".join(v.training_data_sources) or "none"
        print(f"   {v.version}: {data}")

    # Step 5: Cost analysis
    print("\n5. Training cost analysis...")
    total_cost = 0
    for v in registry.list_versions("majora"):
        if v.training and v.training.cost_usd:
            cost = v.training.cost_usd
            total_cost += cost
            print(f"   {v.version}: ${cost:.2f}")
    print(f"   Total: ${total_cost:.2f}")


def main() -> None:
    """Run all demonstrations."""
    print("\n" + "=" * 70)
    print(" " * 15 + "MODEL REGISTRY DEMONSTRATION")
    print("=" * 70)

    demo_basic_operations()
    demo_version_details()
    demo_version_comparison()
    demo_lineage()
    demo_deployment()
    demo_summary()
    demo_programmatic_workflow()

    print("\n" + "=" * 70)
    print(" " * 20 + "DEMO COMPLETE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
