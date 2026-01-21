"""Tests for health check system."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from afs.health.enhanced_checks import (
    EnhancedHealthChecker,
    HealthCheckLevel,
    HealthCheckResult,
    HealthScore,
    HealthStatus,
)
from afs.health.daemon import HealthMonitoringDaemon


class TestHealthScore:
    """Test HealthScore data class."""

    def test_score_creation(self):
        """Test creating a health score."""
        score = HealthScore(
            category="system",
            metric="cpu",
            score=0.85,
            status=HealthStatus.GOOD,
            message="CPU at 85%",
        )
        assert score.category == "system"
        assert score.metric == "cpu"
        assert score.score == 0.85
        assert score.status == HealthStatus.GOOD

    def test_score_to_status(self):
        """Test score-to-status mapping."""
        assert (
            EnhancedHealthChecker()._score_to_status(0.95) == HealthStatus.EXCELLENT
        )
        assert EnhancedHealthChecker()._score_to_status(0.8) == HealthStatus.GOOD
        assert (
            EnhancedHealthChecker()._score_to_status(0.6) == HealthStatus.DEGRADED
        )
        assert (
            EnhancedHealthChecker()._score_to_status(0.3) == HealthStatus.CRITICAL
        )

    def test_score_color_coding(self):
        """Test ANSI color codes for terminal output."""
        excellent_score = HealthScore(
            category="test",
            metric="test",
            score=0.95,
            status=HealthStatus.EXCELLENT,
            message="Test",
        )
        assert "\033[92m" in excellent_score.color  # Green


class TestEnhancedHealthChecker:
    """Test EnhancedHealthChecker class."""

    @pytest.fixture
    def checker(self):
        """Create a health checker with temp directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield EnhancedHealthChecker(context_root=Path(tmpdir))

    def test_checker_initialization(self, checker):
        """Test checker initialization."""
        assert checker.healing_enabled is False
        assert isinstance(checker.config, dict)
        assert checker.config["cpu_threshold"] == 85.0
        assert checker.config["test_queries"] == 5

    def test_checker_custom_config(self):
        """Test checker with custom config."""
        config = {"cpu_threshold": 75.0, "memory_threshold": 70.0}
        with tempfile.TemporaryDirectory() as tmpdir:
            checker = EnhancedHealthChecker(context_root=Path(tmpdir), config=config)
            assert checker.config["cpu_threshold"] == 75.0
            assert checker.config["memory_threshold"] == 70.0

    def test_basic_health_check(self, checker):
        """Test basic health check level."""
        result = checker.check(level=HealthCheckLevel.BASIC, save_report=False)

        assert isinstance(result, HealthCheckResult)
        assert result.check_level == HealthCheckLevel.BASIC
        assert result.overall_score >= 0.0
        assert result.overall_score <= 1.0
        assert result.overall_status in HealthStatus

    def test_standard_health_check(self, checker):
        """Test standard health check level."""
        result = checker.check(level=HealthCheckLevel.STANDARD, save_report=False)

        assert result.check_level == HealthCheckLevel.STANDARD
        assert len(result.scores) > 0
        assert len(result.checks) > 0

    def test_comprehensive_health_check(self, checker):
        """Test comprehensive health check level."""
        result = checker.check(level=HealthCheckLevel.COMPREHENSIVE, save_report=False)

        assert result.check_level == HealthCheckLevel.COMPREHENSIVE
        assert len(result.scores) >= len(result.checks)

    def test_health_check_with_auto_heal(self, checker):
        """Test health check with auto-healing enabled."""
        result = checker.check(
            level=HealthCheckLevel.BASIC,
            auto_heal=True,
            save_report=False,
        )

        assert result is not None
        # Auto-heal flag should have been enabled during check
        assert isinstance(result, HealthCheckResult)

    def test_health_score_calculation(self, checker):
        """Test overall score calculation."""
        # Add some test scores
        checker.scores = [
            HealthScore(
                category="system",
                metric="cpu",
                score=0.9,
                status=HealthStatus.EXCELLENT,
                message="Test",
            ),
            HealthScore(
                category="system",
                metric="memory",
                score=0.8,
                status=HealthStatus.GOOD,
                message="Test",
            ),
            HealthScore(
                category="service",
                metric="api",
                score=1.0,
                status=HealthStatus.EXCELLENT,
                message="Test",
            ),
        ]

        overall = checker._calculate_overall_score()
        assert 0.0 <= overall <= 1.0

    def test_health_check_report_summary(self, checker):
        """Test health check report summary generation."""
        result = checker.check(level=HealthCheckLevel.BASIC, save_report=False)
        summary = result.summary()

        assert "Health Check Report" in summary
        assert "BASIC" in summary  # Uppercase level in report
        assert "Overall Score" in summary

    def test_health_check_json_serialization(self, checker):
        """Test converting health result to JSON."""
        result = checker.check(level=HealthCheckLevel.BASIC, save_report=False)
        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert "check_level" in result_dict
        assert "overall_score" in result_dict
        assert "timestamp" in result_dict
        assert "scores" in result_dict
        assert "checks" in result_dict

    @patch("afs.health.enhanced_checks.psutil")
    def test_system_health_check(self, mock_psutil, checker):
        """Test system health check."""
        mock_psutil.cpu_percent.return_value = 50.0
        mock_psutil.virtual_memory.return_value = Mock(
            percent=40.0, available=1024 * 1024 * 1024
        )
        mock_psutil.disk_usage.return_value = Mock(percent=30.0, free=100 * 1024**3)

        checker._check_system_health()

        assert len(checker.scores) > 0
        assert any(s.metric == "cpu_usage" for s in checker.scores)

    def test_lmstudio_api_check(self, checker):
        """Test LMStudio API check."""
        score = checker._check_lmstudio_api()
        assert 0.0 <= score <= 1.0

    def test_dependencies_check(self, checker):
        """Test Python dependencies check."""
        score = checker._check_dependencies()
        assert 0.0 <= score <= 1.0

    def test_data_integrity_check(self, checker):
        """Test training data integrity check."""
        score = checker._check_data_integrity()
        assert 0.0 <= score <= 1.0

    def test_cache_validity_check(self, checker):
        """Test cache validity check."""
        score = checker._check_cache_validity()
        assert 0.0 <= score <= 1.0

    def test_retry_with_backoff(self, checker):
        """Test retry with exponential backoff."""
        call_count = 0

        def failing_func():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise ValueError("First attempt fails")
            return "success"

        result = checker._retry_with_backoff(failing_func, max_attempts=3, backoff_s=0.01)
        assert result == "success"
        assert call_count == 2

    def test_report_persistence(self):
        """Test saving and loading health reports."""
        with tempfile.TemporaryDirectory() as tmpdir:
            context_root = Path(tmpdir)
            checker = EnhancedHealthChecker(context_root=context_root)

            result = checker.check(level=HealthCheckLevel.BASIC, save_report=True)

            # Check that report file was created
            health_dir = context_root / "health"
            assert health_dir.exists()
            reports = list(health_dir.glob("report-*.json"))
            assert len(reports) > 0

            # Load and verify report
            with open(reports[0]) as f:
                saved_data = json.load(f)
                assert saved_data["overall_score"] == result.overall_score


class TestHealthMonitoringDaemon:
    """Test HealthMonitoringDaemon class."""

    def test_daemon_initialization(self):
        """Test daemon initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            daemon = HealthMonitoringDaemon(context_root=Path(tmpdir))

            assert daemon.running is False
            assert daemon.check_interval_s == 60
            assert daemon.alert_threshold == 0.1

    def test_daemon_custom_config(self):
        """Test daemon with custom configuration."""
        with tempfile.TemporaryDirectory() as tmpdir:
            daemon = HealthMonitoringDaemon(
                context_root=Path(tmpdir),
                check_interval_s=30,
                alert_threshold=0.2,
                auto_heal=True,
            )

            assert daemon.check_interval_s == 30
            assert daemon.alert_threshold == 0.2
            assert daemon.auto_heal is True

    def test_trend_calculation(self):
        """Test trend calculation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            daemon = HealthMonitoringDaemon(context_root=Path(tmpdir))

            # Simulate trend data
            from datetime import datetime, timedelta

            now = datetime.now()
            daemon.check_history = [
                (now - timedelta(hours=1), 0.8),
                (now - timedelta(minutes=30), 0.7),
                (now, 0.75),
            ]

            trend = daemon.get_trend(hours=2)
            assert trend["current"] == 0.75
            assert trend["average"] > 0
            assert "trend" in trend


class TestHealthCheckLevels:
    """Test different health check levels."""

    def test_check_level_enum(self):
        """Test HealthCheckLevel enum."""
        levels = list(HealthCheckLevel)
        assert HealthCheckLevel.BASIC in levels
        assert HealthCheckLevel.STANDARD in levels
        assert HealthCheckLevel.COMPREHENSIVE in levels
        assert HealthCheckLevel.STRESS in levels

    def test_check_level_string_conversion(self):
        """Test converting string to HealthCheckLevel."""
        level = HealthCheckLevel("basic")
        assert level == HealthCheckLevel.BASIC

        with pytest.raises(ValueError):
            HealthCheckLevel("invalid")


class TestHealthAutoHealing:
    """Test auto-healing functionality."""

    def test_healing_disabled_by_default(self):
        """Test that healing is disabled by default."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checker = EnhancedHealthChecker(context_root=Path(tmpdir))
            assert checker.healing_enabled is False

    def test_healing_actions_logged(self):
        """Test that healing actions are tracked."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checker = EnhancedHealthChecker(context_root=Path(tmpdir))
            checker.healing_enabled = True

            checker._heal_high_memory()
            # Healing actions should be recorded
            assert isinstance(checker.healing_actions, list)


class TestHealthMetrics:
    """Test health metric calculations."""

    def test_score_to_status_boundaries(self):
        """Test status boundary calculations."""
        checker = EnhancedHealthChecker()

        # Test boundaries
        assert checker._score_to_status(1.0) == HealthStatus.EXCELLENT
        assert checker._score_to_status(0.9) == HealthStatus.EXCELLENT
        assert checker._score_to_status(0.89) == HealthStatus.GOOD
        assert checker._score_to_status(0.7) == HealthStatus.GOOD
        assert checker._score_to_status(0.69) == HealthStatus.DEGRADED
        assert checker._score_to_status(0.5) == HealthStatus.DEGRADED
        assert checker._score_to_status(0.49) == HealthStatus.CRITICAL
        assert checker._score_to_status(0.0) == HealthStatus.CRITICAL


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
