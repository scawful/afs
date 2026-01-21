"""CLI commands for model registry management.

Provides command-line interface for:
- Listing and inspecting models
- Registering new versions
- Comparing versions
- Managing deployments
- Viewing lineage
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

from .database import ModelRegistry
from .lineage import LineageTracker


@click.group()
def registry() -> None:
    """Model registry management commands."""
    pass


@registry.command()
@click.option(
    "--registry",
    type=click.Path(),
    help="Path to registry JSON file",
)
def list(registry: str | None) -> None:
    """List all registered models."""
    reg = ModelRegistry(Path(registry) if registry else None)
    models = reg.list_models()

    if not models:
        click.echo("No models registered")
        return

    click.echo("Registered Models:")
    click.echo("=" * 60)

    for model_name in models:
        versions = reg.list_versions(model_name)
        deployed = sum(1 for v in versions if v.deployed)
        click.echo(f"\n{model_name}")
        click.echo(f"  Versions: {len(versions)} ({deployed} deployed)")

        for v in sorted(versions, key=lambda x: x.version)[-5:]:  # Show last 5
            status = "●" if v.deployed else "○"
            accuracy = (
                f"acc={v.evaluation_scores.accuracy:.3f}"
                if v.evaluation_scores.accuracy
                else ""
            )
            click.echo(f"  {status} {v.version:8} ({v.status.value:10}) {accuracy}")


@registry.command()
@click.argument("model_name")
@click.option("--registry", type=click.Path(), help="Path to registry JSON file")
def info(model_name: str, registry: str | None) -> None:
    """Show detailed information about a model."""
    reg = ModelRegistry(Path(registry) if registry else None)
    latest = reg.get_latest(model_name)

    if not latest:
        click.echo(f"Model not found: {model_name}")
        sys.exit(1)

    click.echo(f"Model: {model_name}")
    click.echo("=" * 60)
    click.echo(f"Latest Version: {latest.version}")
    click.echo(f"Status: {latest.status.value}")
    click.echo(f"Created: {latest.created_at}")
    click.echo(f"Deployed: {latest.deployed}")

    if latest.training:
        click.echo("\nTraining:")
        click.echo(f"  Base Model: {latest.training.base_model}")
        click.echo(f"  Framework: {latest.training.framework}")
        click.echo(f"  Samples: {latest.training.samples}")
        click.echo(f"  Epochs: {latest.training.epochs}")
        click.echo(f"  Batch Size: {latest.training.batch_size}")
        click.echo(f"  Learning Rate: {latest.training.learning_rate}")
        if latest.training.duration_hours:
            click.echo(f"  Duration: {latest.training.duration_hours:.2f} hours")
        if latest.training.cost_usd:
            click.echo(f"  Cost: ${latest.training.cost_usd:.2f}")

    click.echo("\nEvaluation Scores:")
    scores = latest.evaluation_scores
    if scores.accuracy:
        click.echo(f"  Accuracy: {scores.accuracy:.4f}")
    if scores.f1_score:
        click.echo(f"  F1 Score: {scores.f1_score:.4f}")
    if scores.perplexity:
        click.echo(f"  Perplexity: {scores.perplexity:.4f}")
    if scores.bleu_score:
        click.echo(f"  BLEU Score: {scores.bleu_score:.4f}")
    if scores.inference_speed_tokens_per_sec:
        click.echo(f"  Speed: {scores.inference_speed_tokens_per_sec:.0f} tokens/sec")

    if latest.lora_path:
        click.echo(f"\nLoRA Path: {latest.lora_path}")
    if latest.gguf_path:
        click.echo(f"GGUF Path: {latest.gguf_path}")
    if latest.git_commit:
        click.echo(f"Git Commit: {latest.git_commit}")

    if latest.notes:
        click.echo(f"\nNotes: {latest.notes}")


@registry.command()
@click.argument("model_name")
@click.option("--registry", type=click.Path(), help="Path to registry JSON file")
def versions(model_name: str, registry: str | None) -> None:
    """List all versions of a model."""
    reg = ModelRegistry(Path(registry) if registry else None)
    model_versions = reg.list_versions(model_name)

    if not model_versions:
        click.echo(f"No versions found for model: {model_name}")
        sys.exit(1)

    click.echo(f"Versions of {model_name}:")
    click.echo("=" * 80)

    for v in sorted(model_versions, key=lambda x: x.version):
        status = "●" if v.deployed else "○"
        accuracy = (
            f"acc={v.evaluation_scores.accuracy:.3f}"
            if v.evaluation_scores.accuracy
            else ""
        )
        samples = f"n={v.training.samples}" if v.training else ""
        click.echo(f"{status} {v.version:8} ({v.status.value:10}) {accuracy:15} {samples}")

        if v.parent_version:
            click.echo(f"    └─ Parent: {v.parent_version}")
        if v.training_data_sources:
            click.echo(f"    └─ Data: {', '.join(v.training_data_sources)}")


@registry.command()
@click.argument("model_name")
@click.argument("version1")
@click.argument("version2")
@click.option("--registry", type=click.Path(), help="Path to registry JSON file")
def compare(model_name: str, version1: str, version2: str, registry: str | None) -> None:
    """Compare two versions of a model."""
    reg = ModelRegistry(Path(registry) if registry else None)

    try:
        diff = reg.compare_versions(model_name, version1, version2)
    except ValueError as e:
        click.echo(f"Error: {e}")
        sys.exit(1)

    click.echo(f"Comparing {model_name} versions: {version1} vs {version2}")
    click.echo("=" * 80)

    if not diff:
        click.echo("No differences found")
        return

    for key, values in sorted(diff.items()):
        v1 = values["v1"]
        v2 = values["v2"]
        click.echo(f"\n{key}:")
        click.echo(f"  {version1}: {v1}")
        click.echo(f"  {version2}: {v2}")


@registry.command()
@click.argument("model_name")
@click.argument("version")
@click.option("--base-model", required=True, help="Base model used")
@click.option("--samples", type=int, default=0, help="Number of training samples")
@click.option("--epochs", type=int, default=1, help="Number of epochs")
@click.option("--batch-size", type=int, default=32, help="Batch size")
@click.option("--learning-rate", type=float, default=1e-4, help="Learning rate")
@click.option("--framework", default="unsloth", help="Training framework")
@click.option("--lora-path", help="Path to LoRA weights")
@click.option("--gguf-path", help="Path to GGUF model")
@click.option("--accuracy", type=float, help="Accuracy score")
@click.option("--f1-score", type=float, help="F1 score")
@click.option("--registry", type=click.Path(), help="Path to registry JSON file")
def add(
    model_name: str,
    version: str,
    base_model: str,
    samples: int,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    framework: str,
    lora_path: str | None,
    gguf_path: str | None,
    accuracy: float | None,
    f1_score: float | None,
    registry: str | None,
) -> None:
    """Register a new model version."""
    reg = ModelRegistry(Path(registry) if registry else None)

    scores = {}
    if accuracy is not None:
        scores["accuracy"] = accuracy
    if f1_score is not None:
        scores["f1_score"] = f1_score

    try:
        model_version = reg.register_model(
            model_name=model_name,
            version=version,
            base_model=base_model,
            samples=samples,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            framework=framework,
            lora_path=lora_path,
            gguf_path=gguf_path,
            evaluation_scores=scores,
        )
        click.echo(f"✓ Registered {model_name}:{model_version.version}")
    except Exception as e:
        click.echo(f"✗ Error: {e}")
        sys.exit(1)


@registry.command()
@click.argument("model_name")
@click.argument("version")
@click.option("--registry", type=click.Path(), help="Path to registry JSON file")
def deploy(model_name: str, version: str, registry: str | None) -> None:
    """Mark a version as deployed."""
    reg = ModelRegistry(Path(registry) if registry else None)

    try:
        reg.set_deployed(model_name, version, deployed=True)
        click.echo(f"✓ Deployed {model_name}:{version}")
    except ValueError as e:
        click.echo(f"✗ Error: {e}")
        sys.exit(1)


@registry.command()
@click.argument("model_name")
@click.argument("version")
@click.option("--registry", type=click.Path(), help="Path to registry JSON file")
def rollback(model_name: str, version: str, registry: str | None) -> None:
    """Rollback to a previous version."""
    reg = ModelRegistry(Path(registry) if registry else None)

    try:
        reg.rollback(model_name, version)
        click.echo(f"✓ Rolled back to {model_name}:{version}")
    except ValueError as e:
        click.echo(f"✗ Error: {e}")
        sys.exit(1)


@registry.command()
@click.argument("model_name")
@click.argument("version")
@click.option("--accuracy", type=float, help="Accuracy score")
@click.option("--f1-score", type=float, help="F1 score")
@click.option("--perplexity", type=float, help="Perplexity score")
@click.option("--bleu-score", type=float, help="BLEU score")
@click.option("--speed", type=float, help="Inference speed (tokens/sec)")
@click.option("--registry", type=click.Path(), help="Path to registry JSON file")
def score(
    model_name: str,
    version: str,
    accuracy: float | None,
    f1_score: float | None,
    perplexity: float | None,
    bleu_score: float | None,
    speed: float | None,
    registry: str | None,
) -> None:
    """Update evaluation scores for a version."""
    reg = ModelRegistry(Path(registry) if registry else None)

    scores = {}
    if accuracy is not None:
        scores["accuracy"] = accuracy
    if f1_score is not None:
        scores["f1_score"] = f1_score
    if perplexity is not None:
        scores["perplexity"] = perplexity
    if bleu_score is not None:
        scores["bleu_score"] = bleu_score
    if speed is not None:
        scores["inference_speed_tokens_per_sec"] = speed

    try:
        reg.update_evaluation_scores(model_name, version, **scores)
        click.echo(f"✓ Updated scores for {model_name}:{version}")
    except ValueError as e:
        click.echo(f"✗ Error: {e}")
        sys.exit(1)


@registry.command()
@click.argument("model_name")
@click.option("--registry", type=click.Path(), help="Path to registry JSON file")
def tree(model_name: str, registry: str | None) -> None:
    """Show version lineage tree."""
    tracker = LineageTracker()
    tree_output = tracker.build_tree(model_name)
    click.echo(tree_output)


@registry.command()
@click.argument("model_name")
@click.option("--registry", type=click.Path(), help="Path to registry JSON file")
def lineage(model_name: str, registry: str | None) -> None:
    """Show model lineage and training history."""
    tracker = LineageTracker()
    history = tracker.get_training_history(model_name)

    if not history:
        click.echo(f"No lineage found for {model_name}")
        return

    click.echo(f"Training History for {model_name}")
    click.echo("=" * 80)

    for record in history:
        click.echo(f"\n{record['version']}")
        if record.get("parent_version"):
            click.echo(f"  Parent: {record['parent_version']}")
        if record.get("training_data"):
            click.echo(f"  Data: {', '.join(record['training_data'])}")
        if record.get("git_commit"):
            click.echo(f"  Commit: {record['git_commit']}")
        if record.get("notes"):
            click.echo(f"  Notes: {record['notes']}")


def main() -> None:
    """Main entry point for registry CLI."""
    registry()


if __name__ == "__main__":
    main()
