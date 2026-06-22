"""Comprehensive performance benchmarking for trained models.

This module provides automated benchmarking infrastructure across:
- Speed: tokens/second, latency, throughput
- Quality: accuracy, correctness, consistency
- Resources: memory, VRAM, CPU, power
- Comparisons: vs base models, previous versions, other models

Usage:
    from afs.benchmarks import SpeedBenchmark, QualityBenchmark, ResourceBenchmark

    # Run speed tests
    speed = SpeedBenchmark(model_path)
    speed_results = speed.run()

    # Run quality tests
    quality = QualityBenchmark(model_path, test_dataset)
    quality_results = quality.run()

    # Monitor resources
    resources = ResourceBenchmark(model_path)
    resource_results = resources.run()
"""

from .quality import (
    AccuracyMetrics,
    CodeCorrectnessChecker,
    ConsistencyMetrics,
    QualityBenchmark,
)
from .resources import (
    CPUMonitor,
    MemoryMonitor,
    PowerMonitor,
    ResourceBenchmark,
    VRAMMonitor,
)
from .speed import (
    LatencyMetrics,
    SpeedBenchmark,
    ThroughputMetrics,
    measure_tokens_per_second,
)

__all__ = [
    # Speed
    "SpeedBenchmark",
    "LatencyMetrics",
    "ThroughputMetrics",
    "measure_tokens_per_second",
    # Quality
    "QualityBenchmark",
    "AccuracyMetrics",
    "ConsistencyMetrics",
    "CodeCorrectnessChecker",
    # Resources
    "ResourceBenchmark",
    "MemoryMonitor",
    "VRAMMonitor",
    "CPUMonitor",
    "PowerMonitor",
]
