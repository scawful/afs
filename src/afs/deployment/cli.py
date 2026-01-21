"""CLI commands for deployment validation and management."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import click

from afs.deployment.validator import PreDeploymentValidator
from afs.logging_config import get_logger
from afs.notifications.base import NotificationManager
from afs.notifications.desktop import DesktopNotificationHandler
from afs.notifications.email import EmailNotificationHandler
from afs.registry.database import ModelRegistry

logger = get_logger(__name__)


@click.group("deploy", help="Model deployment and validation commands")
def deploy_group():
    """Deployment management."""
    pass


@deploy_group.command("validate", help="Validate model before deployment")
@click.argument("model_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--model-name",
    default="unknown",
    help="Name of the model",
)
@click.option(
    "--version",
    default="v1",
    help="Version string",
)
@click.option(
    "--baseline",
    default=None,
    help="Baseline version for regression testing",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Directory to save validation reports",
)
@click.option(
    "--json",
    "output_json",
    is_flag=True,
    help="Output JSON report",
)
@click.option(
    "--markdown",
    "output_markdown",
    is_flag=True,
    help="Output markdown report",
)
@click.option(
    "--notify",
    is_flag=True,
    help="Send notifications on failures",
)
@click.option(
    "--strict",
    is_flag=True,
    help="Exit with error if any warnings occur",
)
def validate_command(
    model_path: Path,
    model_name: str,
    version: str,
    baseline: Optional[str],
    output_dir: Optional[Path],
    output_json: bool,
    output_markdown: bool,
    notify: bool,
    strict: bool,
) -> None:
    """Validate a model before deployment.

    Example:
        python -m afs.deployment.cli validate model.gguf --model-name majora --version v5 --baseline v4
    """
    click.echo(f"Validating {model_name} v{version}...")

    # Setup notification manager if requested
    notification_manager = None
    if notify:
        notification_manager = NotificationManager()
        notification_manager.register_handler("desktop", DesktopNotificationHandler())
        try:
            notification_manager.register_handler("email", EmailNotificationHandler())
        except Exception:
            pass  # Email not configured

    # Create validator
    validator = PreDeploymentValidator(
        model_path=model_path,
        model_name=model_name,
        version=version,
        baseline_version=baseline,
        notification_manager=notification_manager,
    )

    # Run validation
    report = validator.validate_all()

    # Display summary
    summary = report.summary()
    click.echo("\n" + "=" * 60)
    click.echo(f"Validation Report: {model_name} v{version}")
    click.echo("=" * 60)
    click.echo(f"Status: {summary['overall_status']}")
    click.echo(f"Passed: {summary['passed']}/{summary['total_checks']}")
    if summary["warnings"] > 0:
        click.echo(f"Warnings: {summary['warnings']}")
    if summary["failed"] > 0:
        click.echo(f"Failed: {summary['failed']}")

    # Display failed checks
    if report.failed_checks:
        click.echo("\nFailed Checks:")
        for result in report.failed_checks:
            click.echo(f"  ✗ {result.check_name}: {result.message}")

    # Display warnings
    if report.warning_checks:
        click.echo("\nWarnings:")
        for result in report.warning_checks:
            click.echo(f"  ⚠ {result.check_name}: {result.message}")

    # Rollback recommendation
    rollback = validator.get_rollback_recommendation()
    if rollback:
        click.echo(f"\n{rollback}")

    # Save reports
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = report.timestamp.replace(":", "-").replace(".", "-")

        if output_json:
            json_path = output_dir / f"{model_name}_{version}_{timestamp}.json"
            report.save_json(json_path)
            click.echo(f"\nJSON report: {json_path}")

        if output_markdown:
            md_path = output_dir / f"{model_name}_{version}_{timestamp}.md"
            report.save_markdown(md_path)
            click.echo(f"Markdown report: {md_path}")

    # Exit with appropriate code
    if not report.passed:
        if strict or report.failed_checks:
            sys.exit(1)
        elif report.warning_checks:
            sys.exit(0) if not strict else sys.exit(1)


@deploy_group.command("registry-check", help="Check model in registry")
@click.option(
    "--model-name",
    required=True,
    help="Model name to check",
)
@click.option(
    "--version",
    default=None,
    help="Specific version to check",
)
@click.option(
    "--registry-path",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to registry database",
)
def registry_check_command(
    model_name: str,
    version: Optional[str],
    registry_path: Optional[Path],
) -> None:
    """Check model status in registry."""
    try:
        registry = ModelRegistry(registry_path)

        if version:
            model_version = registry.get_version(model_name, version)
            if not model_version:
                click.echo(f"Version {version} not found for {model_name}")
                sys.exit(1)

            click.echo(f"Model: {model_name} {version}")
            click.echo(f"Status: {model_version.status.value}")
            click.echo(f"Created: {model_version.created_at}")
            click.echo(f"Deployed: {model_version.deployed}")

            if model_version.gguf_path:
                gguf_path = Path(model_version.gguf_path)
                if gguf_path.exists():
                    size_gb = gguf_path.stat().st_size / (1024**3)
                    click.echo(f"GGUF: {model_version.gguf_path} ({size_gb:.2f} GB)")
                else:
                    click.echo(f"GGUF: {model_version.gguf_path} (NOT FOUND)")

            if model_version.evaluation_scores:
                click.echo("Evaluation Scores:")
                scores = model_version.evaluation_scores.to_dict()
                for key, value in scores.items():
                    if value is not None:
                        click.echo(f"  {key}: {value}")

        else:
            versions = registry.list_versions(model_name)
            if not versions:
                click.echo(f"No versions found for {model_name}")
                sys.exit(1)

            click.echo(f"Model: {model_name}")
            click.echo(f"Total versions: {len(versions)}")
            click.echo("\nVersions:")
            for v in versions:
                status_emoji = "✓" if v.deployed else " "
                click.echo(f"  [{status_emoji}] {v.version} ({v.status})")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@deploy_group.command("rollback", help="Rollback to previous model version")
@click.option(
    "--model-name",
    required=True,
    help="Model name",
)
@click.option(
    "--target-version",
    required=True,
    help="Version to rollback to",
)
@click.option(
    "--registry-path",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to registry database",
)
@click.option(
    "--force",
    is_flag=True,
    help="Force rollback without confirmation",
)
def rollback_command(
    model_name: str,
    target_version: str,
    registry_path: Optional[Path],
    force: bool,
) -> None:
    """Rollback to a previous model version."""
    try:
        registry = ModelRegistry(registry_path)

        # Get target version
        version = registry.get_version(model_name, target_version)
        if not version:
            click.echo(f"Version {target_version} not found for {model_name}")
            sys.exit(1)

        # Confirm rollback
        if not force:
            click.echo(f"Rolling back {model_name} to {target_version}")
            if not click.confirm("Continue?"):
                click.echo("Cancelled.")
                sys.exit(0)

        # Mark previous version as active
        registry.set_deployed(model_name, target_version, deployed=True)

        click.echo(f"✓ Rolled back {model_name} to {target_version}")
        click.echo(f"  Created: {version.created_at}")
        click.echo(f"  Status: {version.status.value}")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@deploy_group.command("compare", help="Compare two model versions")
@click.option(
    "--model-name",
    required=True,
    help="Model name",
)
@click.option(
    "--version1",
    required=True,
    help="First version to compare",
)
@click.option(
    "--version2",
    required=True,
    help="Second version to compare",
)
@click.option(
    "--registry-path",
    type=click.Path(path_type=Path),
    default=None,
    help="Path to registry database",
)
def compare_command(
    model_name: str,
    version1: str,
    version2: str,
    registry_path: Optional[Path],
) -> None:
    """Compare two model versions."""
    try:
        registry = ModelRegistry(registry_path)

        v1 = registry.get_version(model_name, version1)
        v2 = registry.get_version(model_name, version2)

        if not v1 or not v2:
            click.echo("One or both versions not found")
            sys.exit(1)

        # Compare evaluation scores
        click.echo(f"Comparing {model_name}:")
        click.echo(f"  {version1} vs {version2}")
        click.echo()

        scores1 = v1.evaluation_scores.to_dict()
        scores2 = v2.evaluation_scores.to_dict()

        metrics = set(scores1.keys()) | set(scores2.keys())
        for metric in sorted(metrics):
            val1 = scores1.get(metric)
            val2 = scores2.get(metric)

            if val1 is None:
                click.echo(f"  {metric}: (N/A) → {val2}")
            elif val2 is None:
                click.echo(f"  {metric}: {val1} → (N/A)")
            else:
                diff = val2 - val1
                pct = (diff / val1 * 100) if val1 != 0 else 0
                symbol = "↑" if diff > 0 else ("↓" if diff < 0 else "→")
                click.echo(f"  {metric}: {val1} → {val2} ({symbol} {pct:+.1f}%)")

    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


if __name__ == "__main__":
    deploy_group()
