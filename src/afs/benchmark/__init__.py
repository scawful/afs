"""Benchmark suite for AFS models.

Core provides generic benchmark infrastructure (BenchmarkRunner, BenchmarkSuite).
Domain-specific runners (Din, Nayru, Farore, Veran) are extension-owned
and available when a companion extension repo is installed.
"""

from .base import (
    BenchmarkConfig,
    BenchmarkItem,
    BenchmarkResult,
    BenchmarkRunner,
    load_benchmark_items,
    save_benchmark_items,
)
from .leaderboard import ComparisonResult, LeaderboardEntry, LeaderboardManager
from .suite import BenchmarkSuite, run_benchmark

__all__ = [
    # Base
    "BenchmarkResult",
    "BenchmarkConfig",
    "BenchmarkRunner",
    "BenchmarkItem",
    "load_benchmark_items",
    "save_benchmark_items",
    # Suite
    "BenchmarkSuite",
    "run_benchmark",
    # Leaderboard
    "LeaderboardManager",
    "LeaderboardEntry",
    "ComparisonResult",
]

# Domain-specific runners (extension-owned, available with a companion extension repo)
try:
    from .din import DinBenchmark

    __all__.append("DinBenchmark")
except Exception:
    pass

try:
    from .nayru import FaroreBenchmark, NayruBenchmark, VeranBenchmark

    __all__.extend(["NayruBenchmark", "FaroreBenchmark", "VeranBenchmark"])
except Exception:
    pass
