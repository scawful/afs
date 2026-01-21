"""Example usage of the model registry system."""

from __future__ import annotations

from .database import ModelRegistry
from .lineage import LineageTracker


def example_register_model() -> None:
    """Example: Register a new model version."""
    registry = ModelRegistry()

    # Register first version of Majora model
    v1 = registry.register_model(
        model_name="majora",
        version="v1",
        base_model="Qwen2.5-Coder-7B",
        framework="unsloth",
        samples=223,
        epochs=3,
        batch_size=4,
        learning_rate=2e-4,
        max_seq_length=2048,
        training_data=["oracle", "toolbench"],
        lora_path="/path/to/majora-v1-lora",
        gguf_path="/path/to/majora-v1-Q8_0.gguf",
        evaluation_scores={
            "accuracy": 0.85,
            "f1_score": 0.82,
        },
        duration_hours=3.0,
        cost_usd=0.80,
        hardware="M2 Max",
        notes="Initial training with oracle and toolbench datasets",
    )

    print(f"✓ Registered {v1.model_name}:{v1.version}")
    print(registry.summary("majora"))


def example_list_versions() -> None:
    """Example: List all versions of a model."""
    registry = ModelRegistry()

    versions = registry.list_versions("majora")
    print("\nVersions of majora:")
    for v in versions:
        print(f"  {v.version}: {v.status.value}")
        if v.evaluation_scores.accuracy:
            print(f"    Accuracy: {v.evaluation_scores.accuracy:.3f}")


def example_deploy_version() -> None:
    """Example: Deploy a version."""
    registry = ModelRegistry()

    # Set v1 as deployed
    registry.set_deployed("majora", "v1", deployed=True)
    print("✓ Deployed majora:v1")

    # Later, deploy v2 (which automatically marks v1 as deprecated)
    registry.register_model(
        model_name="majora",
        version="v2",
        base_model="Qwen2.5-Coder-7B",
        samples=223,
        epochs=3,
        evaluation_scores={"accuracy": 0.87},
    )

    registry.rollback("majora", "v2")
    print("✓ Deployed majora:v2 (marked v1 as deprecated)")


def example_compare_versions() -> None:
    """Example: Compare two versions."""
    registry = ModelRegistry()

    diff = registry.compare_versions("majora", "v1", "v2")

    print("\nDifferences between v1 and v2:")
    for key, values in diff.items():
        print(f"  {key}:")
        print(f"    v1: {values['v1']}")
        print(f"    v2: {values['v2']}")


def example_update_scores() -> None:
    """Example: Update evaluation scores."""
    registry = ModelRegistry()

    # After running evaluation
    registry.update_evaluation_scores(
        "majora",
        "v1",
        accuracy=0.86,
        f1_score=0.83,
        perplexity=2.15,
    )

    print("✓ Updated evaluation scores for majora:v1")


def example_track_lineage() -> None:
    """Example: Track model lineage."""
    tracker = LineageTracker()
    registry = ModelRegistry()

    # Register base model
    registry.register_model(
        model_name="majora",
        version="v1",
        base_model="Qwen2.5-Coder-7B",
        samples=223,
        epochs=3,
    )

    # Track lineage
    tracker.add_version(
        model_name="majora",
        version="v1",
        base_model="Qwen2.5-Coder-7B",
        training_data=["oracle", "toolbench"],
        git_commit="abc123def456",
    )

    # Fine-tune from v1
    registry.register_model(
        model_name="majora",
        version="v2",
        base_model="majora:v1",
        samples=100,
        epochs=2,
        parent_version="v1",
    )

    tracker.add_version(
        model_name="majora",
        version="v2",
        parent_version="v1",
        training_data=["oracle-additional"],
        git_commit="def456ghi789",
    )

    print("✓ Tracked lineage for majora")
    print(tracker.build_tree("majora"))


def example_get_ancestors() -> None:
    """Example: Get ancestor versions."""
    tracker = LineageTracker()

    ancestors = tracker.get_ancestors("majora", "v3")
    print(f"\nAncestors of majora:v3: {ancestors}")

    # Get complete data lineage
    data_lineage = tracker.get_data_lineage("majora", "v3")
    print("\nData lineage:")
    for version, data_sources in data_lineage.items():
        print(f"  {version}: {data_sources}")


def example_full_workflow() -> None:
    """Example: Complete workflow from training to deployment."""
    registry = ModelRegistry()
    tracker = LineageTracker()

    # Step 1: Register initial training
    print("Step 1: Registering initial model...")
    v1 = registry.register_model(
        model_name="oracle",
        version="v1",
        base_model="Qwen2.5-Coder-7B",
        samples=1000,
        epochs=3,
        learning_rate=2e-4,
        lora_path="/models/oracle-v1-lora",
        gguf_path="/models/oracle-v1-Q8_0.gguf",
        evaluation_scores={"accuracy": 0.84},
        git_commit="abc123",
        notes="Initial training",
    )

    # Step 2: Track lineage
    print("Step 2: Tracking lineage...")
    tracker.add_version(
        model_name="oracle",
        version="v1",
        base_model="Qwen2.5-Coder-7B",
        training_data=["oracle-dataset"],
        git_commit="abc123",
    )

    # Step 3: Deploy
    print("Step 3: Deploying...")
    registry.set_deployed("oracle", "v1", deployed=True)

    # Step 4: Fine-tune for new task
    print("Step 4: Fine-tuning for new task...")
    v2 = registry.register_model(
        model_name="oracle",
        version="v2",
        base_model="oracle:v1",
        samples=500,
        epochs=2,
        lora_path="/models/oracle-v2-lora",
        evaluation_scores={"accuracy": 0.86},
        parent_version="v1",
        git_commit="def456",
        notes="Fine-tuned on additional data",
    )

    tracker.add_version(
        model_name="oracle",
        version="v2",
        parent_version="v1",
        training_data=["oracle-dataset", "additional-dataset"],
        git_commit="def456",
    )

    # Step 5: Compare and deploy if better
    print("Step 5: Comparing versions...")
    diff = registry.compare_versions("oracle", "v1", "v2")
    if "accuracy" in diff and diff["accuracy"]["v2"] > diff["accuracy"]["v1"]:
        print("  v2 is better, deploying...")
        registry.rollback("oracle", "v2")

    # Step 6: View summary
    print("\nStep 6: Summary")
    print(registry.summary("oracle"))
    print("\n" + tracker.build_tree("oracle"))


if __name__ == "__main__":
    example_full_workflow()
