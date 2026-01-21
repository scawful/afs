"""Automated quality gate system for deployment validation.

Enforces quality standards before allowing deployments through configurable rules.
Supports multiple contexts (development, staging, production) with escalating
strictness. Integrates with CI/CD, model registry, and deployment pipeline.

Key features:
- Configurable rules per context
- Pre-merge, pre-deployment, and post-deployment validation
- Automatic rollback on production failures
- Integration with GitHub Actions and model registry
- Detailed reporting and alert notifications
"""

from __future__ import annotations

import logging
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Optional

logger = logging.getLogger(__name__)


class DeploymentContext(str, Enum):
    """Deployment context determining gate strictness."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"


class GateStatus(str, Enum):
    """Status of a quality gate check."""

    PASSED = "passed"
    FAILED = "failed"
    BLOCKED = "blocked"
    WARNING = "warning"
    NOT_RUN = "not_run"


@dataclass
class GateThresholds:
    """Configurable thresholds for quality gates."""

    # Testing thresholds
    min_test_pass_rate: float = 0.95  # Minimum 95% tests passing
    min_code_coverage: float = 0.80  # Minimum 80% code coverage

    # Model quality thresholds
    min_quality_score: float = 0.70  # Minimum 0.7 overall quality
    max_regression: float = 0.05  # Maximum 5% regression vs baseline
    max_latency_increase: float = 0.20  # Maximum 20% latency increase

    # Security thresholds
    max_critical_vulnerabilities: int = 0  # Zero tolerance for critical
    max_high_vulnerabilities: int = 2  # Allow max 2 high-severity
    max_medium_vulnerabilities: int = 10  # Allow max 10 medium-severity

    # Performance thresholds
    max_memory_increase_percent: float = 15.0  # Max 15% memory increase
    min_throughput_tokens_per_sec: float = 10.0  # Min throughput requirement

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GateThresholds:
        """Create from dictionary, using defaults for missing keys."""
        known_fields = set(cls.__dataclass_fields__.keys())
        kwargs = {k: v for k, v in data.items() if k in known_fields}
        return cls(**kwargs)


@dataclass
class TestMetrics:
    """Test execution metrics."""

    total_tests: int
    passed_tests: int
    failed_tests: int
    skipped_tests: int
    duration_seconds: float
    coverage_percent: float

    def pass_rate(self) -> float:
        """Calculate test pass rate."""
        if self.total_tests == 0:
            return 1.0
        return self.passed_tests / self.total_tests

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class ModelMetrics:
    """Model quality and performance metrics."""

    quality_score: float  # 0.0-1.0
    accuracy: Optional[float] = None
    f1_score: Optional[float] = None
    perplexity: Optional[float] = None
    latency_ms: Optional[float] = None
    throughput_tokens_per_sec: Optional[float] = None
    memory_mb: Optional[float] = None

    # Baseline comparison
    baseline_quality_score: Optional[float] = None
    baseline_latency_ms: Optional[float] = None
    baseline_memory_mb: Optional[float] = None

    def regression_percent(self) -> float:
        """Calculate regression vs baseline as percentage."""
        if not self.baseline_quality_score:
            return 0.0
        diff = self.baseline_quality_score - self.quality_score
        if self.baseline_quality_score == 0:
            return 0.0
        return abs(diff) / self.baseline_quality_score

    def latency_increase_percent(self) -> float:
        """Calculate latency increase vs baseline."""
        if not self.baseline_latency_ms or self.baseline_latency_ms == 0:
            return 0.0
        diff = self.latency_ms - self.baseline_latency_ms if self.latency_ms else 0
        return diff / self.baseline_latency_ms

    def memory_increase_percent(self) -> float:
        """Calculate memory increase vs baseline."""
        if not self.baseline_memory_mb or self.baseline_memory_mb == 0:
            return 0.0
        diff = self.memory_mb - self.baseline_memory_mb if self.memory_mb else 0
        return diff / self.baseline_memory_mb

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class SecurityScanResults:
    """Results from security scanning."""

    critical_vulnerabilities: int = 0
    high_vulnerabilities: int = 0
    medium_vulnerabilities: int = 0
    low_vulnerabilities: int = 0
    scan_timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    scan_tool: str = "unknown"

    def total_vulnerabilities(self) -> int:
        """Get total vulnerability count."""
        return (
            self.critical_vulnerabilities
            + self.high_vulnerabilities
            + self.medium_vulnerabilities
            + self.low_vulnerabilities
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return asdict(self)


@dataclass
class GateCheckResult:
    """Result of a single quality gate check."""

    gate_name: str
    status: GateStatus
    passed: bool
    message: str
    details: dict[str, Any] = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "gate_name": self.gate_name,
            "status": self.status.value,
            "passed": self.passed,
            "message": self.message,
            "details": self.details,
            "timestamp": self.timestamp,
        }


@dataclass
class QualityGateReport:
    """Complete quality gate check report."""

    context: DeploymentContext
    model_name: str
    model_version: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    checks: list[GateCheckResult] = field(default_factory=list)

    def all_passed(self) -> bool:
        """Check if all gates passed."""
        return all(check.passed for check in self.checks)

    def any_blocked(self) -> bool:
        """Check if any gates resulted in blocked status."""
        return any(check.status == GateStatus.BLOCKED for check in self.checks)

    def failed_checks(self) -> list[GateCheckResult]:
        """Get all failed checks."""
        return [check for check in self.checks if not check.passed]

    def summary(self) -> dict[str, Any]:
        """Get summary of report."""
        return {
            "context": self.context.value,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "timestamp": self.timestamp,
            "total_checks": len(self.checks),
            "passed_checks": sum(1 for c in self.checks if c.passed),
            "failed_checks": len(self.failed_checks()),
            "all_passed": self.all_passed(),
            "any_blocked": self.any_blocked(),
        }

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "context": self.context.value,
            "model_name": self.model_name,
            "model_version": self.model_version,
            "timestamp": self.timestamp,
            "checks": [check.to_dict() for check in self.checks],
            "summary": self.summary(),
        }


class QualityGate:
    """Main quality gate enforcer with configurable rules."""

    def __init__(
        self,
        context: DeploymentContext = DeploymentContext.STAGING,
        thresholds: Optional[GateThresholds] = None,
    ):
        """Initialize quality gate.

        Args:
            context: Deployment context (development, staging, production)
            thresholds: Custom thresholds (uses defaults if None)
        """
        self.context = context
        self.thresholds = thresholds or self._default_thresholds_for_context(context)
        self._callbacks: dict[str, list[Callable]] = {}

    @classmethod
    def development(cls, thresholds: Optional[GateThresholds] = None) -> QualityGate:
        """Create development context gate (relaxed)."""
        default_thresholds = GateThresholds(
            min_test_pass_rate=0.80,
            min_code_coverage=0.60,
            min_quality_score=0.50,
            max_regression=0.20,
            max_latency_increase=0.50,
            max_critical_vulnerabilities=1,
            max_high_vulnerabilities=5,
            max_medium_vulnerabilities=20,
        )
        merged = cls._merge_thresholds(default_thresholds, thresholds)
        return cls(DeploymentContext.DEVELOPMENT, merged)

    @classmethod
    def staging(cls, thresholds: Optional[GateThresholds] = None) -> QualityGate:
        """Create staging context gate (standard)."""
        default_thresholds = GateThresholds()  # Uses all defaults
        merged = cls._merge_thresholds(default_thresholds, thresholds)
        return cls(DeploymentContext.STAGING, merged)

    @classmethod
    def production(cls, thresholds: Optional[GateThresholds] = None) -> QualityGate:
        """Create production context gate (strict)."""
        default_thresholds = GateThresholds(
            min_test_pass_rate=0.98,
            min_code_coverage=0.90,
            min_quality_score=0.85,
            max_regression=0.02,
            max_latency_increase=0.10,
            max_critical_vulnerabilities=0,
            max_high_vulnerabilities=0,
            max_medium_vulnerabilities=2,
            max_memory_increase_percent=5.0,
        )
        merged = cls._merge_thresholds(default_thresholds, thresholds)
        return cls(DeploymentContext.PRODUCTION, merged)

    @staticmethod
    def _merge_thresholds(
        defaults: GateThresholds, custom: Optional[GateThresholds]
    ) -> GateThresholds:
        """Merge custom thresholds with defaults."""
        if not custom:
            return defaults
        defaults_dict = defaults.to_dict()
        custom_dict = custom.to_dict()
        defaults_dict.update(custom_dict)
        return GateThresholds.from_dict(defaults_dict)

    def _default_thresholds_for_context(self, context: DeploymentContext) -> GateThresholds:
        """Get default thresholds for context."""
        if context == DeploymentContext.DEVELOPMENT:
            return GateThresholds(
                min_test_pass_rate=0.80,
                min_code_coverage=0.60,
                min_quality_score=0.50,
                max_regression=0.20,
                max_latency_increase=0.50,
            )
        elif context == DeploymentContext.PRODUCTION:
            return GateThresholds(
                min_test_pass_rate=0.98,
                min_code_coverage=0.90,
                min_quality_score=0.85,
                max_regression=0.02,
                max_latency_increase=0.10,
                max_critical_vulnerabilities=0,
            )
        else:  # STAGING
            return GateThresholds()

    def register_callback(self, event: str, callback: Callable) -> None:
        """Register callback for gate events.

        Args:
            event: Event name (gate_passed, gate_failed, gate_blocked)
            callback: Callable to invoke on event
        """
        if event not in self._callbacks:
            self._callbacks[event] = []
        self._callbacks[event].append(callback)

    def _trigger_callbacks(self, event: str, *args: Any, **kwargs: Any) -> None:
        """Trigger registered callbacks."""
        for callback in self._callbacks.get(event, []):
            try:
                callback(*args, **kwargs)
            except Exception as e:
                logger.warning(f"Callback error for event {event}: {e}")

    def check_tests(self, metrics: TestMetrics) -> GateCheckResult:
        """Check test metrics against thresholds.

        Args:
            metrics: Test execution metrics

        Returns:
            Check result with pass/fail status
        """
        pass_rate = metrics.pass_rate()
        coverage = metrics.coverage_percent / 100.0

        passed = True
        messages = []

        if pass_rate < self.thresholds.min_test_pass_rate:
            passed = False
            messages.append(
                f"Test pass rate {pass_rate:.1%} below threshold "
                f"{self.thresholds.min_test_pass_rate:.1%}"
            )

        if coverage < self.thresholds.min_code_coverage:
            passed = False
            messages.append(
                f"Code coverage {coverage:.1%} below threshold "
                f"{self.thresholds.min_code_coverage:.1%}"
            )

        status = GateStatus.BLOCKED if not passed else GateStatus.PASSED

        result = GateCheckResult(
            gate_name="tests",
            status=status,
            passed=passed,
            message=" | ".join(messages) or "All test metrics passed",
            details={
                "pass_rate": pass_rate,
                "coverage_percent": coverage,
                "total_tests": metrics.total_tests,
                "passed_tests": metrics.passed_tests,
                "failed_tests": metrics.failed_tests,
                "duration_seconds": metrics.duration_seconds,
            },
        )

        if not passed:
            self._trigger_callbacks("gate_failed", result)
        else:
            self._trigger_callbacks("gate_passed", result)

        return result

    def check_model_quality(
        self, metrics: ModelMetrics, baseline: Optional[ModelMetrics] = None
    ) -> GateCheckResult:
        """Check model quality metrics against thresholds.

        Args:
            metrics: Current model metrics
            baseline: Baseline metrics for comparison (optional)

        Returns:
            Check result with pass/fail status
        """
        # Set baseline if not provided
        if baseline and not metrics.baseline_quality_score:
            metrics.baseline_quality_score = baseline.quality_score
            metrics.baseline_latency_ms = baseline.latency_ms
            metrics.baseline_memory_mb = baseline.memory_mb

        passed = True
        messages = []

        if metrics.quality_score < self.thresholds.min_quality_score:
            passed = False
            messages.append(
                f"Quality score {metrics.quality_score:.2f} below threshold "
                f"{self.thresholds.min_quality_score:.2f}"
            )

        regression = metrics.regression_percent()
        if regression > self.thresholds.max_regression:
            passed = False
            messages.append(
                f"Quality regression {regression:.1%} exceeds threshold "
                f"{self.thresholds.max_regression:.1%}"
            )

        latency_increase = metrics.latency_increase_percent()
        if latency_increase > self.thresholds.max_latency_increase:
            passed = False
            messages.append(
                f"Latency increase {latency_increase:.1%} exceeds threshold "
                f"{self.thresholds.max_latency_increase:.1%}"
            )

        if (
            metrics.throughput_tokens_per_sec
            and metrics.throughput_tokens_per_sec < self.thresholds.min_throughput_tokens_per_sec
        ):
            passed = False
            messages.append(
                f"Throughput {metrics.throughput_tokens_per_sec:.1f} tokens/sec "
                f"below threshold {self.thresholds.min_throughput_tokens_per_sec:.1f}"
            )

        status = GateStatus.BLOCKED if not passed else GateStatus.PASSED

        result = GateCheckResult(
            gate_name="model_quality",
            status=status,
            passed=passed,
            message=" | ".join(messages) or "Model quality acceptable",
            details={
                "quality_score": metrics.quality_score,
                "regression_percent": regression,
                "latency_increase_percent": latency_increase,
                "memory_increase_percent": metrics.memory_increase_percent(),
                "baseline_quality": metrics.baseline_quality_score,
            },
        )

        if not passed:
            self._trigger_callbacks("gate_failed", result)
        else:
            self._trigger_callbacks("gate_passed", result)

        return result

    def check_security(self, scan_results: SecurityScanResults) -> GateCheckResult:
        """Check security scan results against thresholds.

        Args:
            scan_results: Security scan results

        Returns:
            Check result with pass/fail status
        """
        passed = True
        messages = []
        status = GateStatus.PASSED

        if scan_results.critical_vulnerabilities > self.thresholds.max_critical_vulnerabilities:
            passed = False
            status = GateStatus.BLOCKED
            messages.append(
                f"Critical vulnerabilities {scan_results.critical_vulnerabilities} "
                f"exceed threshold {self.thresholds.max_critical_vulnerabilities}"
            )

        if scan_results.high_vulnerabilities > self.thresholds.max_high_vulnerabilities:
            passed = False
            messages.append(
                f"High vulnerabilities {scan_results.high_vulnerabilities} "
                f"exceed threshold {self.thresholds.max_high_vulnerabilities}"
            )

        if scan_results.medium_vulnerabilities > self.thresholds.max_medium_vulnerabilities:
            if self.context == DeploymentContext.PRODUCTION:
                passed = False
            messages.append(
                f"Medium vulnerabilities {scan_results.medium_vulnerabilities} "
                f"exceed threshold {self.thresholds.max_medium_vulnerabilities}"
            )

        if not passed and status != GateStatus.BLOCKED:
            status = GateStatus.WARNING

        result = GateCheckResult(
            gate_name="security",
            status=status,
            passed=passed,
            message=" | ".join(messages) or "Security scan passed",
            details={
                "critical": scan_results.critical_vulnerabilities,
                "high": scan_results.high_vulnerabilities,
                "medium": scan_results.medium_vulnerabilities,
                "low": scan_results.low_vulnerabilities,
                "total": scan_results.total_vulnerabilities(),
                "scan_tool": scan_results.scan_tool,
                "scan_timestamp": scan_results.scan_timestamp,
            },
        )

        if not passed:
            if status == GateStatus.BLOCKED:
                self._trigger_callbacks("gate_blocked", result)
            else:
                self._trigger_callbacks("gate_failed", result)
        else:
            self._trigger_callbacks("gate_passed", result)

        return result

    def check_all(
        self,
        model_name: str,
        model_version: str,
        test_metrics: Optional[TestMetrics] = None,
        model_metrics: Optional[ModelMetrics] = None,
        security_results: Optional[SecurityScanResults] = None,
        baseline_model_metrics: Optional[ModelMetrics] = None,
    ) -> QualityGateReport:
        """Run all quality gate checks.

        Args:
            model_name: Name of the model
            model_version: Version of the model
            test_metrics: Test execution metrics
            model_metrics: Model quality metrics
            security_results: Security scan results
            baseline_model_metrics: Baseline for comparison

        Returns:
            Complete quality gate report
        """
        report = QualityGateReport(
            context=self.context,
            model_name=model_name,
            model_version=model_version,
        )

        # Run all enabled checks
        if test_metrics:
            report.checks.append(self.check_tests(test_metrics))

        if model_metrics:
            report.checks.append(self.check_model_quality(model_metrics, baseline_model_metrics))

        if security_results:
            report.checks.append(self.check_security(security_results))

        # Trigger final callbacks
        if report.all_passed():
            self._trigger_callbacks("all_gates_passed", report)
        else:
            if report.any_blocked():
                self._trigger_callbacks("gates_blocked", report)
            else:
                self._trigger_callbacks("gates_failed", report)

        return report

    def summary_string(self, report: QualityGateReport) -> str:
        """Get human-readable summary of gate report."""
        summary = report.summary()
        lines = [
            f"Quality Gate Report - {self.context.value.upper()}",
            f"Model: {report.model_name} v{report.model_version}",
            f"Timestamp: {report.timestamp}",
            "",
            f"Summary: {summary['passed_checks']}/{summary['total_checks']} checks passed",
        ]

        if not summary["all_passed"]:
            lines.append("\nFailed checks:")
            for check in report.failed_checks():
                lines.append(f"  - {check.gate_name}: {check.message}")

        return "\n".join(lines)
