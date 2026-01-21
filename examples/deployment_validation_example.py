#!/usr/bin/env python3
"""Example: Pre-deployment validation workflow.

Demonstrates the complete validation process:
1. Load model from registry
2. Run pre-deployment validation
3. Generate reports
4. Make deployment decision
5. Handle failures with rollback
"""

from pathlib import Path

from afs.deployment import PreDeploymentValidator
from afs.notifications.base import NotificationManager
from afs.notifications.desktop import DesktopNotificationHandler
from afs.registry.database import ModelRegistry


def example_basic_validation():
    """Basic validation of a model file."""
    print("=" * 60)
    print("Example 1: Basic Validation")
    print("=" * 60)

    # Create a test model file (mock)
    model_path = Path("/tmp/test_model.gguf")
    model_path.write_bytes(b"GGUF" + b"\x00" * (500 * 1024 * 1024))

    # Create validator
    validator = PreDeploymentValidator(
        model_path=model_path,
        model_name="test-model",
        version="v1",
    )

    # Run validation
    report = validator.validate_all()

    # Display results
    summary = report.summary()
    print(f"\nModel: {summary['model_name']} {summary['version']}")
    print(f"Status: {summary['overall_status']}")
    print(f"Passed: {summary['passed']}/{summary['total_checks']}")
    if summary["warnings"] > 0:
        print(f"Warnings: {summary['warnings']}")
    if summary["failed"] > 0:
        print(f"Failed: {summary['failed']}")

    # Show results
    print("\nValidation Results:")
    for result in report.results:
        emoji = {
            "passed": "✓",
            "warning": "⚠",
            "failed": "✗",
            "skipped": "⊘",
        }
        print(f"  {emoji.get(result.status.value, '?')} {result.check_name}: {result.message}")

    return report


def example_validation_with_baseline():
    """Validation with regression testing against baseline."""
    print("\n" + "=" * 60)
    print("Example 2: Validation with Regression Testing")
    print("=" * 60)

    model_path = Path("/tmp/majora_v5.gguf")
    model_path.write_bytes(b"GGUF" + b"\x00" * (7 * 1024 * 1024 * 1024))

    # Create validator with baseline
    validator = PreDeploymentValidator(
        model_path=model_path,
        model_name="majora",
        version="v5",
        baseline_version="v4",  # Compare against v4
    )

    # Run validation
    report = validator.validate_all()

    summary = report.summary()
    print(f"\nModel: {summary['model_name']} {summary['version']}")
    print(f"Baseline: {summary['baseline_version']}")
    print(f"Status: {summary['overall_status']}")
    print(f"Passed: {summary['passed']}/{summary['total_checks']}")

    return report


def example_with_notifications():
    """Validation with notification alerts."""
    print("\n" + "=" * 60)
    print("Example 3: Validation with Notifications")
    print("=" * 60)

    # Setup notifications
    notification_manager = NotificationManager()
    notification_manager.register_handler("desktop", DesktopNotificationHandler())

    model_path = Path("/tmp/test_notified.gguf")
    model_path.write_bytes(b"GGUF" + b"\x00" * (500 * 1024 * 1024))

    # Create validator with notifications
    validator = PreDeploymentValidator(
        model_path=model_path,
        model_name="notified-model",
        version="v1",
        notification_manager=notification_manager,
    )

    print("Running validation with notifications enabled...")
    report = validator.validate_all()

    summary = report.summary()
    print(f"Status: {summary['overall_status']}")

    if report.failed_checks:
        print(f"Failed checks (notifications sent):")
        for result in report.failed_checks:
            print(f"  - {result.check_name}: {result.message}")

    return report


def example_generate_reports(report):
    """Generate JSON and markdown reports."""
    print("\n" + "=" * 60)
    print("Example 4: Generate Reports")
    print("=" * 60)

    report_dir = Path("/tmp/validation_reports")
    report_dir.mkdir(exist_ok=True)

    # Save JSON report
    json_path = report_dir / "validation_report.json"
    report.save_json(json_path)
    print(f"JSON Report: {json_path}")
    print(f"  Size: {json_path.stat().st_size} bytes")

    # Save Markdown report
    md_path = report_dir / "validation_report.md"
    report.save_markdown(md_path)
    print(f"Markdown Report: {md_path}")
    print(f"  Size: {md_path.stat().st_size} bytes")

    return report_dir


def example_registry_integration():
    """Integration with model registry."""
    print("\n" + "=" * 60)
    print("Example 5: Registry Integration")
    print("=" * 60)

    try:
        registry = ModelRegistry()

        # Get latest version of a model
        models = list(registry.models.keys())
        if models:
            model_name = models[0]
            print(f"Found models: {models}")

            # Get latest version
            latest = registry.get_latest(model_name)
            if latest and latest.gguf_path:
                print(f"\nValidating {model_name} {latest.version}")

                # Run validator
                validator = PreDeploymentValidator(
                    model_path=Path(latest.gguf_path),
                    model_name=model_name,
                    version=latest.version,
                    baseline_version=latest.parent_version,
                )

                report = validator.validate_all()
                summary = report.summary()
                print(f"Status: {summary['overall_status']}")

                return report
    except Exception as e:
        print(f"Registry integration example (no registry found): {e}")

    return None


def example_deployment_decision(report):
    """Make deployment decision based on validation."""
    print("\n" + "=" * 60)
    print("Example 6: Deployment Decision")
    print("=" * 60)

    summary = report.summary()
    rollback = report.get_rollback_recommendation()

    print(f"\nValidation Status: {summary['overall_status']}")

    if report.passed:
        print("\n✓ DEPLOYMENT APPROVED")
        print("\nReasons:")
        print("- All critical checks passed")
        print("- No failures detected")

        if report.warning_checks:
            print(f"\nNote: {len(report.warning_checks)} warning(s):")
            for w in report.warning_checks:
                print(f"  - {w.check_name}: {w.message}")
            print("\nThese do not block deployment but should be reviewed.")

        print("\nNext steps:")
        print("1. Review all warnings")
        print("2. Notify deployment team")
        print("3. Schedule deployment")
        print("4. Monitor model performance")

    else:
        print("\n✗ DEPLOYMENT BLOCKED")
        print(f"\nFailed checks ({len(report.failed_checks)}):")
        for f in report.failed_checks:
            print(f"  - {f.check_name}: {f.message}")

        if rollback:
            print(f"\n{rollback}")

        print("\nRequired actions:")
        print("1. Fix all failures")
        print("2. Re-run validation")
        print("3. Do not deploy until all checks pass")


def example_comparison():
    """Compare two model versions."""
    print("\n" + "=" * 60)
    print("Example 7: Version Comparison")
    print("=" * 60)

    try:
        registry = ModelRegistry()
        model_name = "majora"

        versions = registry.list_versions(model_name)
        if len(versions) >= 2:
            v1 = versions[-2]
            v2 = versions[-1]

            print(f"\nComparing {model_name}:")
            print(f"  Version 1: {v1.version} (Created: {v1.created_at})")
            print(f"  Version 2: {v2.version} (Created: {v2.created_at})")

            diff = registry.compare_versions(model_name, v1.version, v2.version)
            print(f"\nDifferences:")
            for key, (val1, val2) in diff.items():
                if val1 != val2:
                    change = "↑" if val2 > val1 else "↓"
                    print(f"  {key}: {val1} → {val2} ({change})")
    except Exception as e:
        print(f"Comparison example (data not available): {e}")


def main():
    """Run all examples."""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 58 + "║")
    print("║" + "  Pre-Deployment Validation System Examples".center(58) + "║")
    print("║" + " " * 58 + "║")
    print("╚" + "═" * 58 + "╝")

    # Example 1: Basic validation
    report1 = example_basic_validation()

    # Example 2: Validation with baseline
    report2 = example_validation_with_baseline()

    # Example 3: Validation with notifications
    report3 = example_with_notifications()

    # Example 4: Generate reports
    example_generate_reports(report1)

    # Example 5: Registry integration
    example_registry_integration()

    # Example 6: Deployment decision
    example_deployment_decision(report1)

    # Example 7: Version comparison
    example_comparison()

    print("\n" + "=" * 60)
    print("Examples completed!")
    print("=" * 60)
    print("\nTo use in your code:")
    print("  from afs.deployment import PreDeploymentValidator")
    print("  validator = PreDeploymentValidator(...)")
    print("  report = validator.validate_all()")
    print("\nFor CLI:")
    print("  python -m afs.deployment.cli validate model.gguf \\")
    print("    --model-name majora --version v5 --baseline v4")


if __name__ == "__main__":
    main()
