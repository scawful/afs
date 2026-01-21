"""Tests for quality gate system."""

import tempfile
from pathlib import Path
from unittest.mock import Mock

import pytest

from afs.gates import (
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
from afs.gates.ci_integration import (
    LocalFileIntegration,
)
from afs.gates.registry_integration import (
    DeploymentController,
    RegistryIntegration,
)


class TestQualityGate:
    """Tests for QualityGate class."""

    def test_development_gate_relaxed_thresholds(self):
        """Development gate should have relaxed thresholds."""
        gate = QualityGate.development()
        assert gate.context == DeploymentContext.DEVELOPMENT
        assert gate.thresholds.min_test_pass_rate == 0.80
        assert gate.thresholds.min_code_coverage == 0.60
        assert gate.thresholds.min_quality_score == 0.50

    def test_production_gate_strict_thresholds(self):
        """Production gate should have strict thresholds."""
        gate = QualityGate.production()
        assert gate.context == DeploymentContext.PRODUCTION
        assert gate.thresholds.min_test_pass_rate == 0.98
        assert gate.thresholds.min_code_coverage == 0.90
        assert gate.thresholds.min_quality_score == 0.85

    def test_staging_gate_standard_thresholds(self):
        """Staging gate should have standard thresholds."""
        gate = QualityGate.staging()
        assert gate.context == DeploymentContext.STAGING
        assert gate.thresholds.min_test_pass_rate == 0.95

    def test_custom_thresholds(self):
        """Should support custom thresholds."""
        custom = GateThresholds(
            min_test_pass_rate=0.90,
            min_code_coverage=0.75,
        )
        gate = QualityGate(DeploymentContext.STAGING, custom)
        assert gate.thresholds.min_test_pass_rate == 0.90
        assert gate.thresholds.min_code_coverage == 0.75


class TestTestMetricsGate:
    """Tests for test metrics gate checks."""

    def test_passing_test_metrics(self):
        """Should pass with good test metrics."""
        gate = QualityGate.staging()
        metrics = TestMetrics(
            total_tests=100,
            passed_tests=96,
            failed_tests=4,
            skipped_tests=0,
            duration_seconds=60.0,
            coverage_percent=85.0,
        )
        result = gate.check_tests(metrics)
        assert result.passed
        assert result.status == GateStatus.PASSED
        assert result.gate_name == "tests"

    def test_failing_test_pass_rate(self):
        """Should fail with low test pass rate."""
        gate = QualityGate.staging()
        metrics = TestMetrics(
            total_tests=100,
            passed_tests=90,  # 90% < 95% threshold
            failed_tests=10,
            skipped_tests=0,
            duration_seconds=60.0,
            coverage_percent=85.0,
        )
        result = gate.check_tests(metrics)
        assert not result.passed
        assert result.status == GateStatus.BLOCKED

    def test_failing_code_coverage(self):
        """Should fail with low code coverage."""
        gate = QualityGate.staging()
        metrics = TestMetrics(
            total_tests=100,
            passed_tests=96,
            failed_tests=4,
            skipped_tests=0,
            duration_seconds=60.0,
            coverage_percent=75.0,  # 75% < 80% threshold
        )
        result = gate.check_tests(metrics)
        assert not result.passed
        assert "coverage" in result.message.lower()

    def test_test_pass_rate_calculation(self):
        """Should correctly calculate pass rate."""
        metrics = TestMetrics(
            total_tests=50,
            passed_tests=45,
            failed_tests=5,
            skipped_tests=0,
            duration_seconds=30.0,
            coverage_percent=80.0,
        )
        assert metrics.pass_rate() == 0.9

    def test_zero_tests(self):
        """Should handle zero tests gracefully."""
        metrics = TestMetrics(
            total_tests=0,
            passed_tests=0,
            failed_tests=0,
            skipped_tests=0,
            duration_seconds=0.0,
            coverage_percent=0.0,
        )
        assert metrics.pass_rate() == 1.0


class TestModelQualityGate:
    """Tests for model quality gate checks."""

    def test_passing_model_metrics(self):
        """Should pass with good model metrics."""
        gate = QualityGate.staging()
        metrics = ModelMetrics(
            quality_score=0.82,
            baseline_quality_score=0.80,
            latency_ms=120.0,
            baseline_latency_ms=115.0,
        )
        result = gate.check_model_quality(metrics)
        assert result.passed
        assert result.status == GateStatus.PASSED

    def test_failing_quality_score(self):
        """Should fail with low quality score."""
        gate = QualityGate.staging()
        metrics = ModelMetrics(
            quality_score=0.60,  # 0.60 < 0.70 threshold
            baseline_quality_score=0.80,
        )
        result = gate.check_model_quality(metrics)
        assert not result.passed
        assert "quality score" in result.message.lower()

    def test_regression_exceeds_threshold(self):
        """Should fail with excessive regression."""
        gate = QualityGate.staging()
        metrics = ModelMetrics(
            quality_score=0.75,
            baseline_quality_score=0.80,
            # Regression: 5/80 = 6.25% > 5% threshold
        )
        result = gate.check_model_quality(metrics)
        assert not result.passed
        assert "regression" in result.message.lower()

    def test_latency_increase_exceeds_threshold(self):
        """Should fail with excessive latency increase."""
        gate = QualityGate.staging()
        metrics = ModelMetrics(
            quality_score=0.82,
            latency_ms=145.0,
            baseline_latency_ms=100.0,
            # Increase: 45/100 = 45% > 20% threshold
        )
        result = gate.check_model_quality(metrics)
        assert not result.passed
        assert "latency" in result.message.lower()

    def test_regression_calculation(self):
        """Should correctly calculate regression."""
        metrics = ModelMetrics(
            quality_score=0.75,
            baseline_quality_score=0.80,
        )
        assert metrics.regression_percent() == pytest.approx(0.0625)

    def test_latency_increase_calculation(self):
        """Should correctly calculate latency increase."""
        metrics = ModelMetrics(
            quality_score=0.85,
            latency_ms=125.0,
            baseline_latency_ms=100.0,
        )
        assert metrics.latency_increase_percent() == pytest.approx(0.25)

    def test_memory_increase_calculation(self):
        """Should correctly calculate memory increase."""
        metrics = ModelMetrics(
            quality_score=0.85,
            memory_mb=2400.0,
            baseline_memory_mb=2000.0,
        )
        assert metrics.memory_increase_percent() == pytest.approx(0.20)


class TestSecurityGate:
    """Tests for security scan gate checks."""

    def test_passing_security_scan(self):
        """Should pass with clean security scan."""
        gate = QualityGate.staging()
        scan = SecurityScanResults(
            critical_vulnerabilities=0,
            high_vulnerabilities=0,
            medium_vulnerabilities=5,
            low_vulnerabilities=10,
        )
        result = gate.check_security(scan)
        assert result.passed
        assert result.status == GateStatus.PASSED

    def test_critical_vulnerability_blocks(self):
        """Should block on critical vulnerabilities."""
        gate = QualityGate.staging()
        scan = SecurityScanResults(
            critical_vulnerabilities=1,
            high_vulnerabilities=0,
            medium_vulnerabilities=0,
            low_vulnerabilities=0,
        )
        result = gate.check_security(scan)
        assert not result.passed
        assert result.status == GateStatus.BLOCKED

    def test_high_vulnerabilities_warning(self):
        """High vulnerabilities should warn in staging."""
        gate = QualityGate.staging()
        scan = SecurityScanResults(
            critical_vulnerabilities=0,
            high_vulnerabilities=5,  # > 2 threshold
            medium_vulnerabilities=0,
            low_vulnerabilities=0,
        )
        result = gate.check_security(scan)
        assert not result.passed

    def test_production_strict_security(self):
        """Production should have strict security standards."""
        gate = QualityGate.production()
        scan = SecurityScanResults(
            critical_vulnerabilities=0,
            high_vulnerabilities=1,  # > 0 threshold for production
            medium_vulnerabilities=0,
            low_vulnerabilities=0,
        )
        result = gate.check_security(scan)
        assert not result.passed


class TestQualityGateReport:
    """Tests for quality gate reports."""

    def test_all_passed_report(self):
        """Report should mark all passed."""
        check1 = GateCheckResult("test1", GateStatus.PASSED, True, "Passed")
        check2 = GateCheckResult("test2", GateStatus.PASSED, True, "Passed")
        report = QualityGateReport(
            context=DeploymentContext.STAGING,
            model_name="test_model",
            model_version="v1",
            checks=[check1, check2],
        )
        assert report.all_passed()
        assert not report.any_blocked()
        assert len(report.failed_checks()) == 0

    def test_failed_report(self):
        """Report should track failures."""
        check1 = GateCheckResult("test1", GateStatus.PASSED, True, "Passed")
        check2 = GateCheckResult(
            "test2", GateStatus.FAILED, False, "Failed"
        )
        report = QualityGateReport(
            context=DeploymentContext.STAGING,
            model_name="test_model",
            model_version="v1",
            checks=[check1, check2],
        )
        assert not report.all_passed()
        assert len(report.failed_checks()) == 1
        assert report.failed_checks()[0].gate_name == "test2"

    def test_blocked_report(self):
        """Report should track blocks."""
        check = GateCheckResult(
            "test1", GateStatus.BLOCKED, False, "Blocked"
        )
        report = QualityGateReport(
            context=DeploymentContext.STAGING,
            model_name="test_model",
            model_version="v1",
            checks=[check],
        )
        assert report.any_blocked()

    def test_report_summary(self):
        """Report summary should be accurate."""
        check1 = GateCheckResult("test1", GateStatus.PASSED, True, "Passed")
        check2 = GateCheckResult(
            "test2", GateStatus.FAILED, False, "Failed"
        )
        report = QualityGateReport(
            context=DeploymentContext.STAGING,
            model_name="test_model",
            model_version="v1",
            checks=[check1, check2],
        )
        summary = report.summary()
        assert summary["total_checks"] == 2
        assert summary["passed_checks"] == 1
        assert summary["failed_checks"] == 1
        assert not summary["all_passed"]

    def test_report_serialization(self):
        """Report should serialize to dict."""
        check = GateCheckResult("test1", GateStatus.PASSED, True, "Passed")
        report = QualityGateReport(
            context=DeploymentContext.STAGING,
            model_name="test_model",
            model_version="v1",
            checks=[check],
        )
        data = report.to_dict()
        assert data["model_name"] == "test_model"
        assert data["context"] == "staging"
        assert len(data["checks"]) == 1


class TestCheckAll:
    """Tests for comprehensive check_all method."""

    def test_check_all_passes(self):
        """Should pass all gates."""
        gate = QualityGate.staging()

        test_metrics = TestMetrics(
            total_tests=100,
            passed_tests=96,
            failed_tests=4,
            skipped_tests=0,
            duration_seconds=60.0,
            coverage_percent=85.0,
        )

        model_metrics = ModelMetrics(
            quality_score=0.82,
            baseline_quality_score=0.80,
        )

        scan = SecurityScanResults(
            critical_vulnerabilities=0,
            high_vulnerabilities=0,
            medium_vulnerabilities=5,
        )

        report = gate.check_all(
            model_name="test_model",
            model_version="v1",
            test_metrics=test_metrics,
            model_metrics=model_metrics,
            security_results=scan,
        )

        assert report.all_passed()
        assert len(report.checks) == 3  # tests, model_quality, security

    def test_check_all_with_failures(self):
        """Should report failures."""
        gate = QualityGate.staging()

        test_metrics = TestMetrics(
            total_tests=100,
            passed_tests=90,  # Below threshold
            failed_tests=10,
            skipped_tests=0,
            duration_seconds=60.0,
            coverage_percent=85.0,
        )

        report = gate.check_all(
            model_name="test_model",
            model_version="v1",
            test_metrics=test_metrics,
        )

        assert not report.all_passed()
        assert len(report.failed_checks()) >= 1


class TestCallbacks:
    """Tests for event callbacks."""

    def test_gate_passed_callback(self):
        """Should trigger passed callback."""
        gate = QualityGate.staging()
        callback = Mock()
        gate.register_callback("gate_passed", callback)

        metrics = TestMetrics(
            total_tests=100,
            passed_tests=96,
            failed_tests=4,
            skipped_tests=0,
            duration_seconds=60.0,
            coverage_percent=85.0,
        )

        gate.check_tests(metrics)
        assert callback.called

    def test_gate_failed_callback(self):
        """Should trigger failed callback."""
        gate = QualityGate.staging()
        callback = Mock()
        gate.register_callback("gate_failed", callback)

        metrics = TestMetrics(
            total_tests=100,
            passed_tests=90,
            failed_tests=10,
            skipped_tests=0,
            duration_seconds=60.0,
            coverage_percent=85.0,
        )

        gate.check_tests(metrics)
        assert callback.called


class TestCIIntegration:
    """Tests for CI integration."""

    def test_local_file_integration(self):
        """Should write reports to local files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            integration = LocalFileIntegration(tmpdir)
            report = QualityGateReport(
                context=DeploymentContext.STAGING,
                model_name="test_model",
                model_version="v1",
            )

            success = integration.report(report)
            assert success
            assert list(Path(tmpdir).glob("report-*.json"))

    def test_local_file_approval(self):
        """Should log approvals to file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            integration = LocalFileIntegration(tmpdir)
            success = integration.approve_deployment(
                "test_model", "v1"
            )
            assert success
            approvals_file = Path(tmpdir) / "approvals.jsonl"
            assert approvals_file.exists()


class TestRegistryIntegration:
    """Tests for registry integration."""

    def test_approve_version(self):
        """Should approve a version."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = RegistryIntegration(tmpdir)
            report = QualityGateReport(
                context=DeploymentContext.STAGING,
                model_name="test_model",
                model_version="v1",
                checks=[
                    GateCheckResult(
                        "test", GateStatus.PASSED, True, "Passed"
                    )
                ],
            )

            success = registry.approve_version(report)
            assert success

    def test_get_approval_status(self):
        """Should retrieve approval status."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = RegistryIntegration(tmpdir)
            report = QualityGateReport(
                context=DeploymentContext.STAGING,
                model_name="test_model",
                model_version="v1",
                checks=[
                    GateCheckResult(
                        "test", GateStatus.PASSED, True, "Passed"
                    )
                ],
            )

            registry.approve_version(report)
            status = registry.get_approval_status(
                "test_model", "v1", "staging"
            )
            assert status is not None
            assert status.approved

    def test_reject_version(self):
        """Should reject a version."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = RegistryIntegration(tmpdir)
            success = registry.reject_version(
                "test_model", "v1", "staging", reason="Test failure"
            )
            assert success

    def test_is_deployable(self):
        """Should check if version is deployable."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = RegistryIntegration(tmpdir)
            report = QualityGateReport(
                context=DeploymentContext.STAGING,
                model_name="test_model",
                model_version="v1",
                checks=[
                    GateCheckResult(
                        "test", GateStatus.PASSED, True, "Passed"
                    )
                ],
            )

            registry.approve_version(report)
            assert registry.is_deployable("test_model", "v1", "staging")
            assert not registry.is_deployable(
                "test_model", "v2", "staging"
            )


class TestDeploymentController:
    """Tests for deployment controller."""

    def test_can_deploy_approved_version(self):
        """Should allow deployment of approved version."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = RegistryIntegration(tmpdir)
            report = QualityGateReport(
                context=DeploymentContext.STAGING,
                model_name="test_model",
                model_version="v1",
                checks=[
                    GateCheckResult(
                        "test", GateStatus.PASSED, True, "Passed"
                    )
                ],
            )
            registry.approve_version(report)

            controller = DeploymentController(registry)
            can_deploy, reason = controller.can_deploy(
                "test_model", "v1", "staging"
            )
            assert can_deploy

    def test_cannot_deploy_unapproved_version(self):
        """Should block deployment of unapproved version."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = RegistryIntegration(tmpdir)
            controller = DeploymentController(registry)

            can_deploy, reason = controller.can_deploy(
                "test_model", "v1", "staging"
            )
            assert not can_deploy

    def test_execute_deployment(self):
        """Should log deployment execution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            registry = RegistryIntegration(tmpdir)
            report = QualityGateReport(
                context=DeploymentContext.STAGING,
                model_name="test_model",
                model_version="v1",
                checks=[
                    GateCheckResult(
                        "test", GateStatus.PASSED, True, "Passed"
                    )
                ],
            )
            registry.approve_version(report)

            controller = DeploymentController(registry)
            success = controller.execute_deployment(
                "test_model", "v1", "staging"
            )
            assert success

    def test_rollback_deployment(self):
        """Should log rollback."""
        with tempfile.TemporaryDirectory() as tmpdir:
            controller = DeploymentController()
            success = controller.rollback(
                "test_model", "v1", "staging", reason="Test failure"
            )
            assert success
