"""Quality gate system for deployment validation.

This package provides automated quality gate enforcement to prevent bad code
and models from reaching production. Features include:

- Configurable rules per deployment context
- Pre-merge, pre-deployment, and post-deployment validation
- Integration with CI/CD pipelines (GitHub Actions)
- Model registry version approval
- Automatic rollback on failures
- Detailed reporting and notifications

Example usage:

    from afs.gates import QualityGate, TestMetrics, ModelMetrics
    from afs.gates.ci_integration import GitHubActionsIntegration

    # Create production-grade gate
    gate = QualityGate.production()

    # Check tests
    test_metrics = TestMetrics(
        total_tests=100,
        passed_tests=99,
        failed_tests=1,
        skipped_tests=0,
        duration_seconds=45.0,
        coverage_percent=88.5,
    )
    test_result = gate.check_tests(test_metrics)

    # Check model quality
    model_metrics = ModelMetrics(
        quality_score=0.87,
        baseline_quality_score=0.85,
        latency_ms=125.0,
        baseline_latency_ms=120.0,
    )
    model_result = gate.check_model_quality(model_metrics)

    # Get full report
    report = gate.check_all(
        model_name="veran",
        model_version="v6.2",
        test_metrics=test_metrics,
        model_metrics=model_metrics,
    )

    # Integrate with CI/CD
    gh_integration = GitHubActionsIntegration()
    gh_integration.report(report)
"""

from .quality_gates import (
    DeploymentContext,
    GateCheckResult,
    GateStatus,
    GateThresholds,
    ModelMetrics,
    QualityGate,
    QualityGateReport,
    SecurityScanResults,
    TestMetrics,
)

__all__ = [
    "QualityGate",
    "DeploymentContext",
    "GateStatus",
    "GateThresholds",
    "GateCheckResult",
    "QualityGateReport",
    "TestMetrics",
    "ModelMetrics",
    "SecurityScanResults",
]
