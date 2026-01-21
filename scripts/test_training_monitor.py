#!/usr/bin/env python3
"""
Unit and integration tests for training_monitor_tui.py

Run with: pytest scripts/test_training_monitor.py -v
Or: python3 -m pytest scripts/test_training_monitor.py -v
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.training_monitor_tui import TrainingMetrics, VastAIMonitor, TrainingMonitorUI


class TestTrainingMetrics:
    """Test TrainingMetrics data model."""

    def test_initialization(self):
        """Test creating a new TrainingMetrics instance."""
        m = TrainingMetrics(
            instance_id=30007012,
            gpu_type="RTX 4090",
            status="running",
        )
        assert m.instance_id == 30007012
        assert m.gpu_type == "RTX 4090"
        assert m.status == "running"
        assert m.epoch == 0
        assert m.step == 0
        assert m.loss == 0.0

    def test_progress_percentage_zero_steps(self):
        """Test progress calculation with zero steps."""
        m = TrainingMetrics(instance_id=1, gpu_type="A100", status="running")
        m.total_steps = 1000
        m.step = 0
        assert m.progress_percentage() == 0.0

    def test_progress_percentage_halfway(self):
        """Test progress calculation at halfway point."""
        m = TrainingMetrics(instance_id=1, gpu_type="A100", status="running")
        m.total_steps = 1000
        m.step = 500
        assert m.progress_percentage() == 50.0

    def test_progress_percentage_complete(self):
        """Test progress calculation at completion."""
        m = TrainingMetrics(instance_id=1, gpu_type="A100", status="running")
        m.total_steps = 1000
        m.step = 1000
        assert m.progress_percentage() == 100.0

    def test_progress_percentage_exceeds_total(self):
        """Test progress capping at 100%."""
        m = TrainingMetrics(instance_id=1, gpu_type="A100", status="running")
        m.total_steps = 1000
        m.step = 1500  # Over total
        assert m.progress_percentage() == 100.0

    def test_health_status_healthy(self):
        """Test health status for healthy instance."""
        m = TrainingMetrics(instance_id=1, gpu_type="A100", status="running")
        m.gpu_util = 85.0
        m.memory_util = 45.0
        m.loss = 2.5
        assert m.health_status() == "healthy"

    def test_health_status_error(self):
        """Test health status for exited instance."""
        m = TrainingMetrics(instance_id=1, gpu_type="A100", status="exited")
        assert m.health_status() == "error"

    def test_health_status_warning_low_gpu(self):
        """Test health status warning for low GPU utilization."""
        m = TrainingMetrics(instance_id=1, gpu_type="A100", status="running")
        m.gpu_util = 5.0  # Too low
        m.memory_util = 40.0
        m.loss = 2.5
        assert m.health_status() == "warning"

    def test_health_status_warning_high_loss(self):
        """Test health status warning for diverging loss."""
        m = TrainingMetrics(instance_id=1, gpu_type="A100", status="running")
        m.gpu_util = 85.0
        m.memory_util = 40.0
        m.loss = 15.0  # Too high
        assert m.health_status() == "warning"

    def test_health_status_warning_high_gpu(self):
        """Test health status warning for over-utilized GPU."""
        m = TrainingMetrics(instance_id=1, gpu_type="A100", status="running")
        m.gpu_util = 96.0  # Too high
        m.memory_util = 40.0
        m.loss = 2.5
        assert m.health_status() == "warning"

    def test_estimated_completion_time_no_steps(self):
        """Test ETA with no steps completed."""
        m = TrainingMetrics(instance_id=1, gpu_type="A100", status="running")
        m.step = 0
        m.total_steps = 1000
        assert m.estimated_completion_time() is None

    def test_estimated_completion_time_halfway(self):
        """Test ETA at halfway point."""
        m = TrainingMetrics(instance_id=1, gpu_type="A100", status="running")
        m.step = 500
        m.total_steps = 1000
        m.runtime_seconds = 3600  # 1 hour

        eta = m.estimated_completion_time()
        assert eta is not None
        # At 500 steps in 3600s, 1 step per 7.2s
        # 500 remaining steps = ~3600 more seconds
        assert 3500 < eta.total_seconds() < 3700

    def test_estimated_completion_time_almost_done(self):
        """Test ETA near completion."""
        m = TrainingMetrics(instance_id=1, gpu_type="A100", status="running")
        m.step = 990
        m.total_steps = 1000
        m.runtime_seconds = 3600

        eta = m.estimated_completion_time()
        assert eta is not None
        # 10 remaining steps at 3.6s/step = ~36 seconds
        # (3600 seconds / 990 steps = 3.636 s/step)
        assert 30 < eta.total_seconds() < 45


class TestVastAIMonitor:
    """Test VastAIMonitor data collection."""

    def test_initialization(self):
        """Test monitor initialization."""
        monitor = VastAIMonitor([1, 2, 3], use_mock=True)
        assert len(monitor.metrics) == 3
        assert 1 in monitor.metrics
        assert 2 in monitor.metrics
        assert 3 in monitor.metrics

    def test_mock_status_generation(self):
        """Test mock data generation."""
        monitor = VastAIMonitor([1], use_mock=True)
        status = monitor._generate_mock_status(1)

        assert "id" in status
        assert "gpu_name" in status
        assert "actual_status" in status
        assert "gpu_util" in status
        assert "disk_util" in status
        assert "dph_total" in status

    def test_mock_status_values_in_range(self):
        """Test mock data values are in valid ranges."""
        monitor = VastAIMonitor([1], use_mock=True)
        status = monitor._generate_mock_status(1)

        assert 0 <= status["gpu_util"] <= 100
        assert 0 <= status["disk_util"] <= 100
        assert status["dph_total"] > 0
        assert status["actual_status"] in ["running", "loading"]

    def test_update_all_metrics_mock(self):
        """Test updating metrics with mock data."""
        monitor = VastAIMonitor([1, 2, 3], use_mock=True)
        monitor.update_all_metrics()

        # All instances should have updated metrics
        for metrics in monitor.metrics.values():
            assert metrics.status in ["running", "loading", "exited", "created", "unknown"]
            assert metrics.gpu_util >= 0
            assert metrics.memory_util >= 0
            assert metrics.cost_per_hour > 0

    def test_mock_logs_generation(self):
        """Test mock log generation."""
        monitor = VastAIMonitor([1], use_mock=True)
        logs = monitor._generate_mock_logs(1)

        assert isinstance(logs, list)
        assert len(logs) > 0
        # Should contain training-related text
        assert any("Step" in log or "Epoch" in log or "Loss" in log for log in logs)

    def test_parse_training_metrics_epoch(self):
        """Test parsing epoch from logs."""
        monitor = VastAIMonitor([1], use_mock=True)
        instance_data = {
            "id": 1,
            "gpu_name": "RTX 4090",
            "actual_status": "running",
            "gpu_util": 85.0,
            "memory": 42.0,
            "disk_util": 50.0,
            "dph_total": 0.5,
            "total_cost": 10.0,
            "duration": 3600.0,
        }
        # Note: regex searches in reverse, stops after first match
        # Put epoch log at the end so it's found first in reverse search
        logs = ["Step 500: Loss = 2.345", "Epoch 2/3 completed at 2025-01-14"]

        monitor._parse_training_metrics(1, instance_data, logs)
        metrics = monitor.metrics[1]

        assert metrics.epoch == 2
        assert metrics.total_epochs == 3

    def test_parse_training_metrics_step(self):
        """Test parsing step from logs."""
        monitor = VastAIMonitor([1], use_mock=True)
        instance_data = {
            "id": 1,
            "gpu_name": "RTX 4090",
            "actual_status": "running",
            "gpu_util": 85.0,
            "memory": 42.0,
            "disk_util": 50.0,
            "dph_total": 0.5,
            "total_cost": 10.0,
            "duration": 3600.0,
        }
        logs = ["Step 1000"]

        monitor._parse_training_metrics(1, instance_data, logs)
        metrics = monitor.metrics[1]

        assert metrics.step == 1000

    def test_parse_training_metrics_loss(self):
        """Test parsing loss from logs."""
        monitor = VastAIMonitor([1], use_mock=True)
        instance_data = {
            "id": 1,
            "gpu_name": "RTX 4090",
            "actual_status": "running",
            "gpu_util": 85.0,
            "memory": 42.0,
            "disk_util": 50.0,
            "dph_total": 0.5,
            "total_cost": 10.0,
            "duration": 3600.0,
        }
        logs = ["Loss = 2.345"]

        monitor._parse_training_metrics(1, instance_data, logs)
        metrics = monitor.metrics[1]

        assert abs(metrics.loss - 2.345) < 0.001

    def test_circular_log_buffer(self):
        """Test that log buffer maintains fixed size."""
        monitor = VastAIMonitor([1], use_mock=True)
        metrics = monitor.metrics[1]

        # Add more than maxlen logs
        for i in range(15):
            metrics.recent_logs.append(f"Log line {i}")

        # Should only keep last 10
        assert len(metrics.recent_logs) == 10
        # Should have newest logs
        assert "Log line 14" in metrics.recent_logs
        assert "Log line 4" not in metrics.recent_logs


class TestTrainingMonitorUI:
    """Test TrainingMonitorUI rendering."""

    def test_ui_initialization(self):
        """Test UI initialization."""
        monitor = VastAIMonitor([1, 2, 3], use_mock=True)
        ui = TrainingMonitorUI(monitor)

        assert ui.monitor == monitor
        assert ui.refresh_interval == 10
        assert ui.should_exit is False
        assert ui.paused is False

    def test_format_time_seconds(self):
        """Test time formatting for seconds."""
        ui = TrainingMonitorUI(VastAIMonitor([1], use_mock=True))
        assert ui._format_time(45) == "45s"
        assert ui._format_time(5) == "5s"

    def test_format_time_minutes(self):
        """Test time formatting for minutes."""
        ui = TrainingMonitorUI(VastAIMonitor([1], use_mock=True))
        assert ui._format_time(300) == "5m"
        assert ui._format_time(3540) == "59m"

    def test_format_time_hours(self):
        """Test time formatting for hours."""
        ui = TrainingMonitorUI(VastAIMonitor([1], use_mock=True))
        assert "2.0h" in ui._format_time(7200)
        assert "1.5h" in ui._format_time(5400)

    def test_format_cost(self):
        """Test cost formatting."""
        ui = TrainingMonitorUI(VastAIMonitor([1], use_mock=True))
        assert ui._format_cost(0.5) == "$0.500"
        assert ui._format_cost(1.234) == "$1.234"
        assert ui._format_cost(10.0) == "$10.000"

    def test_get_color_for_health(self):
        """Test health status color mapping."""
        ui = TrainingMonitorUI(VastAIMonitor([1], use_mock=True))

        assert ui._get_color_for_health("healthy") == "green"
        assert ui._get_color_for_health("warning") == "yellow"
        assert ui._get_color_for_health("error") == "red"
        assert ui._get_color_for_health("unknown") == "white"

    def test_build_metrics_table(self):
        """Test metrics table building."""
        monitor = VastAIMonitor([1, 2], use_mock=True)
        monitor.update_all_metrics()
        ui = TrainingMonitorUI(monitor)

        table = ui._build_metrics_table()
        assert table is not None
        assert hasattr(table, "title")

    def test_build_logs_panel(self):
        """Test logs panel building."""
        monitor = VastAIMonitor([1, 2], use_mock=True)
        monitor.update_all_metrics()
        ui = TrainingMonitorUI(monitor)

        panel = ui._build_logs_panel()
        assert panel is not None
        assert hasattr(panel, "title")

    def test_build_stats_panel(self):
        """Test stats panel building."""
        monitor = VastAIMonitor([1, 2], use_mock=True)
        monitor.update_all_metrics()
        ui = TrainingMonitorUI(monitor)

        panel = ui._build_stats_panel()
        assert panel is not None
        assert hasattr(panel, "title")

    def test_build_controls_panel(self):
        """Test controls panel building."""
        monitor = VastAIMonitor([1, 2], use_mock=True)
        ui = TrainingMonitorUI(monitor)

        panel = ui._build_controls_panel()
        assert panel is not None
        assert hasattr(panel, "title")

    def test_build_layout(self):
        """Test complete layout building."""
        monitor = VastAIMonitor([1, 2], use_mock=True)
        monitor.update_all_metrics()
        ui = TrainingMonitorUI(monitor)

        layout = ui.build_layout()
        assert layout is not None
        assert hasattr(layout, "split_column")


class TestIntegration:
    """Integration tests for complete workflow."""

    def test_complete_monitoring_cycle(self):
        """Test complete monitoring cycle."""
        # Initialize monitor
        monitor = VastAIMonitor([1, 2, 3, 4, 5], use_mock=True)

        # First update
        monitor.update_all_metrics()
        assert len(monitor.metrics) == 5
        assert all(m.status in ["running", "loading"] for m in monitor.metrics.values())

        # Second update (should have new values)
        monitor.update_all_metrics()
        assert all(m.status in ["running", "loading"] for m in monitor.metrics.values())

    def test_monitor_with_ui(self):
        """Test monitor and UI together."""
        monitor = VastAIMonitor([1, 2, 3], use_mock=True)
        ui = TrainingMonitorUI(monitor)

        # Update metrics
        monitor.update_all_metrics()

        # Build complete UI
        layout = ui.build_layout()
        assert layout is not None

        # All metrics should be populated
        for metrics in monitor.metrics.values():
            assert metrics.gpu_type != "Unknown" or metrics.use_mock is True

    def test_high_instance_count(self):
        """Test with many instances."""
        # Simulate monitoring 10 instances
        monitor = VastAIMonitor(list(range(30007010, 30007020)), use_mock=True)
        monitor.update_all_metrics()

        assert len(monitor.metrics) == 10
        assert all(m.status != "" for m in monitor.metrics.values())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
