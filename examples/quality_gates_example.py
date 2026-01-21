#!/usr/bin/env python3
"""Example: Quality gate enforcement for model deployment.

Demonstrates:
1. Creating context-aware gates
2. Running comprehensive checks
3. Integrating with CI/CD
4. Managing version approvals
5. Controlling deployments
"""

import json
import logging
from pathlib import Path

from afs.gates import (
    DeploymentContext,
    ModelMetrics,
    QualityGate,
    SecurityScanResults,
    TestMetrics,
)
from afs.gates.ci_integration import LocalFileIntegration
from afs.gates.registry_integration import DeploymentController, RegistryIntegration

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_example_metrics():
    """Load example test and model metrics."""
    test_metrics = TestMetrics(
        total_tests=1250,
        passed_tests=1240,
        failed_tests=10,
        skipped_tests=0,
        duration_seconds=145.2,
        coverage_percent=88.5,
    )

    # Current model metrics
    current_metrics = ModelMetrics(
        quality_score=0.874,
        accuracy=0.921,
        f1_score=0.897,
        perplexity=24.8,
        latency_ms=127.3,
        throughput_tokens_per_sec=48.7,
        memory_mb=2185.0,
    )

    # Baseline for comparison
    baseline_metrics = ModelMetrics(
        quality_score=0.851,
        latency_ms=125.0,
        memory_mb=2100.0,
    )

    security_results = SecurityScanResults(
        critical_vulnerabilities=0,
        high_vulnerabilities=0,
        medium_vulnerabilities=4,
        low_vulnerabilities=12,
        scan_tool="trivy",
    )

    return test_metrics, current_metrics, baseline_metrics, security_results


def example_development_gate():
    """Example: Development gate (relaxed for fast iteration)."""
    logger.info("=== EXAMPLE 1: Development Gate ===\n")

    gate = QualityGate.development()
    logger.info(f"Context: {gate.context.value}")
    logger.info(f"Min test pass rate: {gate.thresholds.min_test_pass_rate:.0%}")
    logger.info(f"Min code coverage: {gate.thresholds.min_code_coverage:.0%}")
    logger.info(f"Min quality score: {gate.thresholds.min_quality_score:.2f}\n")

    test_metrics, current_metrics, baseline_metrics, security_results = (
        load_example_metrics()
    )

    report = gate.check_all(
        model_name="veran",
        model_version="v6.2-dev",
        test_metrics=test_metrics,
        model_metrics=current_metrics,
        security_results=security_results,
        baseline_model_metrics=baseline_metrics,
    )

    print(gate.summary_string(report))
    print(f"\nResult: {'PASSED' if report.all_passed() else 'FAILED'}\n")


def example_staging_gate():
    """Example: Staging gate (standard validation)."""
    logger.info("=== EXAMPLE 2: Staging Gate ===\n")

    gate = QualityGate.staging()
    logger.info(f"Context: {gate.context.value}")
    logger.info(f"Min test pass rate: {gate.thresholds.min_test_pass_rate:.0%}")
    logger.info(f"Min code coverage: {gate.thresholds.min_code_coverage:.0%}")
    logger.info(f"Min quality score: {gate.thresholds.min_quality_score:.2f}\n")

    test_metrics, current_metrics, baseline_metrics, security_results = (
        load_example_metrics()
    )

    report = gate.check_all(
        model_name="veran",
        model_version="v6.2",
        test_metrics=test_metrics,
        model_metrics=current_metrics,
        security_results=security_results,
        baseline_model_metrics=baseline_metrics,
    )

    print(gate.summary_string(report))
    print(f"\nResult: {'PASSED' if report.all_passed() else 'FAILED'}\n")


def example_production_gate():
    """Example: Production gate (strict validation)."""
    logger.info("=== EXAMPLE 3: Production Gate (Strict) ===\n")

    gate = QualityGate.production()
    logger.info(f"Context: {gate.context.value}")
    logger.info(f"Min test pass rate: {gate.thresholds.min_test_pass_rate:.0%}")
    logger.info(f"Min code coverage: {gate.thresholds.min_code_coverage:.0%}")
    logger.info(f"Min quality score: {gate.thresholds.min_quality_score:.2f}")
    logger.info(f"Max critical vulns: {gate.thresholds.max_critical_vulnerabilities}\n")

    test_metrics, current_metrics, baseline_metrics, security_results = (
        load_example_metrics()
    )

    # This will fail some production thresholds
    report = gate.check_all(
        model_name="veran",
        model_version="v6.2",
        test_metrics=test_metrics,
        model_metrics=current_metrics,
        security_results=security_results,
        baseline_model_metrics=baseline_metrics,
    )

    print(gate.summary_string(report))

    if not report.all_passed():
        print("\nFailed checks details:")
        for check in report.failed_checks():
            print(f"  {check.gate_name}: {check.message}")

    print(f"\nResult: {'PASSED' if report.all_passed() else 'FAILED'}\n")


def example_ci_integration():
    """Example: CI/CD integration."""
    logger.info("=== EXAMPLE 4: CI/CD Integration ===\n")

    gate = QualityGate.staging()
    test_metrics, current_metrics, baseline_metrics, security_results = (
        load_example_metrics()
    )

    report = gate.check_all(
        model_name="veran",
        model_version="v6.2",
        test_metrics=test_metrics,
        model_metrics=current_metrics,
        security_results=security_results,
        baseline_model_metrics=baseline_metrics,
    )

    # Integrate with local CI system
    ci_integration = LocalFileIntegration(".quality-gates")
    success = ci_integration.report(report)
    logger.info(f"Report written: {success}")

    if report.all_passed():
        # Approve for deployment
        success = ci_integration.approve_deployment("veran", "v6.2")
        logger.info(f"Approval recorded: {success}")
    else:
        # Block merge
        ci_integration.block_merge(report, "Quality gates failed")
        logger.info("Merge blocked due to failed gates")

    print(f"\nCI integration complete\n")


def example_registry_approval():
    """Example: Model registry approval."""
    logger.info("=== EXAMPLE 5: Registry Approval ===\n")

    gate = QualityGate.staging()
    test_metrics, current_metrics, baseline_metrics, security_results = (
        load_example_metrics()
    )

    report = gate.check_all(
        model_name="veran",
        model_version="v6.2",
        test_metrics=test_metrics,
        model_metrics=current_metrics,
        security_results=security_results,
        baseline_model_metrics=baseline_metrics,
    )

    # Register model version
    registry = RegistryIntegration()

    if report.all_passed():
        # Approve version
        success = registry.approve_version(
            report,
            approved_by="ci-system@afs.local",
            notes="Passed all staging quality gates",
        )
        logger.info(f"Version approved: {success}")

        # Check approval status
        status = registry.get_approval_status("veran", "v6.2", "staging")
        if status:
            logger.info(f"Approval timestamp: {status.timestamp}")
            logger.info(f"Approved by: {status.approved_by}")

        # List approved versions
        approved = registry.list_approved_versions("veran", "staging")
        logger.info(f"Approved versions for staging: {approved}")
    else:
        # Reject version
        success = registry.reject_version(
            "veran",
            "v6.2",
            "staging",
            reason="Failed security checks",
            rejected_by="ci-system@afs.local",
        )
        logger.info(f"Version rejected: {success}")

    print()


def example_deployment_control():
    """Example: Deployment control."""
    logger.info("=== EXAMPLE 6: Deployment Control ===\n")

    # Setup registry with approved version
    registry = RegistryIntegration()
    gate = QualityGate.staging()
    test_metrics, current_metrics, baseline_metrics, security_results = (
        load_example_metrics()
    )

    report = gate.check_all(
        model_name="veran",
        model_version="v6.2",
        test_metrics=test_metrics,
        model_metrics=current_metrics,
        security_results=security_results,
        baseline_model_metrics=baseline_metrics,
    )

    # Approve version
    registry.approve_version(report)

    # Setup deployment controller
    controller = DeploymentController(registry)

    # Check if version can be deployed
    can_deploy, reason = controller.can_deploy("veran", "v6.2", "staging")
    logger.info(f"Can deploy: {can_deploy}")
    logger.info(f"Reason: {reason}")

    if can_deploy:
        # Execute deployment
        success = controller.execute_deployment(
            "veran",
            "v6.2",
            "staging",
            deployment_target="staging-cluster",
        )
        logger.info(f"Deployment executed: {success}")

        # In case of issues, rollback
        # controller.rollback("veran", "v6.2", "staging", reason="Anomaly detected")

    print()


def example_event_callbacks():
    """Example: Event callbacks."""
    logger.info("=== EXAMPLE 7: Event Callbacks ===\n")

    gate = QualityGate.staging()

    # Register callbacks
    def on_gate_passed(result):
        logger.info(f"✓ Gate passed: {result.gate_name}")

    def on_gate_failed(result):
        logger.error(f"✗ Gate failed: {result.gate_name}")
        logger.error(f"  Message: {result.message}")

    def on_gates_blocked(report):
        logger.critical(f"🚫 Deployment blocked for {report.model_name}")
        for check in report.failed_checks():
            logger.critical(f"  - {check.gate_name}: {check.message}")

    gate.register_callback("gate_passed", on_gate_passed)
    gate.register_callback("gate_failed", on_gate_failed)
    gate.register_callback("gates_blocked", on_gates_blocked)

    test_metrics, current_metrics, baseline_metrics, security_results = (
        load_example_metrics()
    )

    report = gate.check_all(
        model_name="veran",
        model_version="v6.2",
        test_metrics=test_metrics,
        model_metrics=current_metrics,
        security_results=security_results,
        baseline_model_metrics=baseline_metrics,
    )

    print(f"All callbacks fired during check_all()\n")


def example_metric_calculations():
    """Example: Metric calculations."""
    logger.info("=== EXAMPLE 8: Metric Calculations ===\n")

    current = ModelMetrics(
        quality_score=0.87,
        baseline_quality_score=0.85,
        latency_ms=127.0,
        baseline_latency_ms=125.0,
        memory_mb=2150.0,
        baseline_memory_mb=2100.0,
    )

    regression = current.regression_percent()
    latency_increase = current.latency_increase_percent()
    memory_increase = current.memory_increase_percent()

    logger.info(f"Quality regression: {regression:.2%}")
    logger.info(f"Latency increase: {latency_increase:.2%}")
    logger.info(f"Memory increase: {memory_increase:.2%}\n")


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("Quality Gates System Examples")
    print("=" * 70 + "\n")

    example_development_gate()
    example_staging_gate()
    example_production_gate()
    example_ci_integration()
    example_registry_approval()
    example_deployment_control()
    example_event_callbacks()
    example_metric_calculations()

    print("=" * 70)
    print("Examples complete!")
    print("=" * 70)
    print("\nCheck .quality-gates/ directory for generated reports and approvals")


if __name__ == "__main__":
    main()
