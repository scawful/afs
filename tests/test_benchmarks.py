"""Tests for benchmark system."""

import tempfile
from pathlib import Path

import pytest


def test_speed_benchmark_imports():
    """Test that speed benchmark modules import correctly."""
    from afs.benchmarks.speed import (
        SpeedBenchmark,
        LatencyMetrics,
        ThroughputMetrics,
        measure_tokens_per_second,
    )

    assert SpeedBenchmark is not None
    assert LatencyMetrics is not None
    assert ThroughputMetrics is not None
    assert measure_tokens_per_second is not None


def test_quality_benchmark_imports():
    """Test that quality benchmark modules import correctly."""
    from afs.benchmarks.quality import (
        QualityBenchmark,
        AccuracyMetrics,
        ConsistencyMetrics,
        CodeCorrectnessChecker,
    )

    assert QualityBenchmark is not None
    assert AccuracyMetrics is not None
    assert ConsistencyMetrics is not None
    assert CodeCorrectnessChecker is not None


def test_resource_benchmark_imports():
    """Test that resource benchmark modules import correctly."""
    from afs.benchmarks.resources import (
        ResourceBenchmark,
        MemoryMonitor,
        VRAMMonitor,
        CPUMonitor,
        PowerMonitor,
    )

    assert ResourceBenchmark is not None
    assert MemoryMonitor is not None
    assert VRAMMonitor is not None
    assert CPUMonitor is not None
    assert PowerMonitor is not None


def test_latency_metrics():
    """Test LatencyMetrics dataclass."""
    from afs.benchmarks.speed import LatencyMetrics

    metrics = LatencyMetrics(
        time_to_first_token_ms=10.0,
        total_latency_ms=100.0,
        tokens_generated=50,
        prompt_tokens=10,
        latency_per_token_ms=2.0,
        prompt_processing_time_ms=10.0,
    )

    assert metrics.time_to_first_token_ms == 10.0
    assert metrics.total_latency_ms == 100.0

    data = metrics.to_dict()
    assert data["time_to_first_token_ms"] == 10.0
    assert data["tokens_generated"] == 50


def test_accuracy_metrics():
    """Test AccuracyMetrics dataclass."""
    from afs.benchmarks.quality import AccuracyMetrics

    metrics = AccuracyMetrics(
        total_tests=100,
        correct=80,
        partially_correct=15,
        incorrect=5,
        accuracy=0.80,
        partial_accuracy=0.95,
    )

    assert metrics.accuracy == 0.80
    assert metrics.partial_accuracy == 0.95

    data = metrics.to_dict()
    assert data["correct"] == 80


def test_code_correctness_checker_python():
    """Test Python code correctness checking."""
    from afs.benchmarks.quality import CodeCorrectnessChecker

    checker = CodeCorrectnessChecker(language="python")

    # Valid Python code
    code = "print('Hello, World!')"
    result = checker.check_python(code)

    assert result["compilable"] is True
    # execution depends on environment

    # Invalid Python code
    invalid_code = "print('Hello, World!"  # Missing closing quote
    result = checker.check_python(invalid_code)

    assert result["compilable"] is False


def test_memory_monitor():
    """Test memory monitoring."""
    import time

    from afs.benchmarks.resources import MemoryMonitor

    monitor = MemoryMonitor(sample_interval=0.05)
    monitor.start()

    # Simulate some work
    time.sleep(0.2)

    metrics = monitor.stop()

    assert metrics.samples > 0
    assert metrics.peak_rss_mb > 0
    assert metrics.average_rss_mb > 0


def test_cpu_monitor():
    """Test CPU monitoring."""
    import time

    from afs.benchmarks.resources import CPUMonitor

    monitor = CPUMonitor(sample_interval=0.05)
    monitor.start()

    # Simulate some CPU work
    for _ in range(1000):
        _ = sum(range(100))

    time.sleep(0.2)
    metrics = monitor.stop()

    assert metrics.samples > 0
    assert metrics.peak_percent >= 0


def test_speed_benchmark_result_serialization():
    """Test speed benchmark result serialization."""
    from afs.benchmarks.speed import (
        SpeedBenchmarkResult,
        LatencyMetrics,
        ThroughputMetrics,
    )

    latency = LatencyMetrics(10.0, 100.0, 50, 10, 2.0, 10.0)
    throughput = ThroughputMetrics(100.0, 50.0, 2, 2, 10.0, 10.0)

    result = SpeedBenchmarkResult(
        model_name="test-model",
        model_path="/path/to/model",
        test_prompts=10,
        latency=latency,
        throughput=throughput,
        context_window_size=8192,
        context_utilization=0.05,
        timestamp="2026-01-14T12:00:00",
        duration_seconds=10.0,
    )

    data = result.to_dict()
    assert data["model_name"] == "test-model"
    assert "latency" in data
    assert "throughput" in data

    summary = result.summary()
    assert "test-model" in summary
    assert "Tokens/Second" in summary


def test_quality_benchmark_result_serialization():
    """Test quality benchmark result serialization."""
    from afs.benchmarks.quality import (
        QualityBenchmarkResult,
        AccuracyMetrics,
        ConsistencyMetrics,
        CodeCorrectnessResult,
    )

    accuracy = AccuracyMetrics(100, 80, 15, 5, 0.80, 0.95)
    consistency = ConsistencyMetrics(3, 10, 0.05, 0.95, False, 0.10)
    code_correctness = CodeCorrectnessResult(50, 45, 40, 35, 5, 5, 0.90, 0.80, 0.70)

    result = QualityBenchmarkResult(
        model_name="test-model",
        model_path="/path/to/model",
        accuracy=accuracy,
        consistency=consistency,
        code_correctness=code_correctness,
        reasoning_score=0.85,
        timestamp="2026-01-14T12:00:00",
        duration_seconds=45.0,
    )

    data = result.to_dict()
    assert data["model_name"] == "test-model"
    assert "accuracy" in data
    assert "code_correctness" in data

    summary = result.summary()
    assert "test-model" in summary
    assert "Accuracy" in summary


def test_resource_benchmark_result_serialization():
    """Test resource benchmark result serialization."""
    from afs.benchmarks.resources import (
        ResourceBenchmarkResult,
        MemoryMetrics,
        CPUMetrics,
        PowerMetrics,
    )

    memory = MemoryMetrics(1000.0, 800.0, 1200.0, 900.0, 20.0, 15.0, 100)
    cpu = CPUMetrics(90.0, 70.0, [95.0, 85.0], [75.0, 65.0], 1000, 100)
    power = PowerMetrics(15.0, 20.0, 675.0, True)

    result = ResourceBenchmarkResult(
        model_name="test-model",
        model_path="/path/to/model",
        memory=memory,
        vram=None,
        cpu=cpu,
        power=power,
        monitoring_duration_seconds=45.0,
        timestamp="2026-01-14T12:00:00",
    )

    data = result.to_dict()
    assert data["model_name"] == "test-model"
    assert "memory" in data
    assert "cpu" in data
    assert "power" in data

    summary = result.summary()
    assert "test-model" in summary
    assert "Memory" in summary
    assert "CPU" in summary


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
