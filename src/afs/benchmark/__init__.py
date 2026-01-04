"""Benchmark suite for AFS expert models.

Provides standardized evaluation across all expert domains:
- Din: Optimization metrics (cycle/byte reduction)
- Nayru: Generation metrics (ASAR validity, entity coverage)
- Farore: Debugging metrics (bug detection, fix accuracy)
- Veran: Explanation metrics (concept coverage, accuracy)
"""

from .base import (
    BenchmarkResult,
    BenchmarkConfig,
    BenchmarkRunner,
    BenchmarkItem,
    load_benchmark_items,
    save_benchmark_items,
)
from .din import DinBenchmark
from .nayru import NayruBenchmark, FaroreBenchmark, VeranBenchmark
from .suite import BenchmarkSuite, run_benchmark
from .leaderboard import LeaderboardManager, LeaderboardEntry, ComparisonResult

__all__ = [
    # Base
    "BenchmarkResult",
    "BenchmarkConfig",
    "BenchmarkRunner",
    "BenchmarkItem",
    "load_benchmark_items",
    "save_benchmark_items",
    # Runners
    "DinBenchmark",
    "NayruBenchmark",
    "FaroreBenchmark",
    "VeranBenchmark",
    # Suite
    "BenchmarkSuite",
    "run_benchmark",
    # Leaderboard
    "LeaderboardManager",
    "LeaderboardEntry",
    "ComparisonResult",
]
