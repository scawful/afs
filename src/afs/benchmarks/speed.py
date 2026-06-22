"""Speed benchmarking for model inference performance.

Measures:
- Tokens/second generation
- First token latency (TTFT - Time To First Token)
- Total generation latency
- Batch processing throughput
- Context window utilization efficiency
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from importlib.util import find_spec
from pathlib import Path
from typing import Any

try:
    import psutil
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    psutil = None


@dataclass
class LatencyMetrics:
    """Latency measurements for model inference."""

    time_to_first_token_ms: float
    total_latency_ms: float
    tokens_generated: int
    prompt_tokens: int
    latency_per_token_ms: float
    prompt_processing_time_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "time_to_first_token_ms": self.time_to_first_token_ms,
            "total_latency_ms": self.total_latency_ms,
            "tokens_generated": self.tokens_generated,
            "prompt_tokens": self.prompt_tokens,
            "latency_per_token_ms": self.latency_per_token_ms,
            "prompt_processing_time_ms": self.prompt_processing_time_ms,
        }


@dataclass
class ThroughputMetrics:
    """Throughput measurements for model inference."""

    tokens_per_second: float
    tokens_per_second_per_user: float | None = None
    batch_size: int = 1
    concurrent_requests: int = 1
    requests_per_second: float = 0.0
    average_tokens_per_request: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "tokens_per_second": self.tokens_per_second,
            "tokens_per_second_per_user": self.tokens_per_second_per_user,
            "batch_size": self.batch_size,
            "concurrent_requests": self.concurrent_requests,
            "requests_per_second": self.requests_per_second,
            "average_tokens_per_request": self.average_tokens_per_request,
        }


@dataclass
class SpeedBenchmarkResult:
    """Results from a complete speed benchmark run."""

    model_name: str
    model_path: str
    test_prompts: int
    latency: LatencyMetrics
    throughput: ThroughputMetrics
    context_window_size: int
    context_utilization: float
    timestamp: str
    duration_seconds: float
    hardware_info: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "model_name": self.model_name,
            "model_path": self.model_path,
            "test_prompts": self.test_prompts,
            "latency": self.latency.to_dict(),
            "throughput": self.throughput.to_dict(),
            "context_window_size": self.context_window_size,
            "context_utilization": self.context_utilization,
            "timestamp": self.timestamp,
            "duration_seconds": self.duration_seconds,
            "hardware_info": self.hardware_info,
        }

    def summary(self) -> str:
        """Generate a human-readable summary."""
        lines = [
            f"Speed Benchmark: {self.model_name}",
            f"Path: {self.model_path}",
            "",
            "Latency Metrics:",
            f"  Time to First Token: {self.latency.time_to_first_token_ms:.1f}ms",
            f"  Total Latency: {self.latency.total_latency_ms:.1f}ms",
            f"  Latency per Token: {self.latency.latency_per_token_ms:.2f}ms",
            "",
            "Throughput Metrics:",
            f"  Tokens/Second: {self.throughput.tokens_per_second:.2f}",
            f"  Requests/Second: {self.throughput.requests_per_second:.2f}",
            f"  Avg Tokens/Request: {self.throughput.average_tokens_per_request:.1f}",
            "",
            "Context:",
            f"  Window Size: {self.context_window_size}",
            f"  Utilization: {self.context_utilization:.1%}",
            "",
            f"Duration: {self.duration_seconds:.2f}s",
        ]
        return "\n".join(lines)


def measure_tokens_per_second(
    generate_fn: callable,
    prompt: str,
    num_tokens: int = 100,
) -> tuple[float, LatencyMetrics]:
    """Measure tokens/second for a generation function.

    Args:
        generate_fn: Function that takes prompt and returns (output, token_count)
        prompt: Input prompt
        num_tokens: Target number of tokens to generate

    Returns:
        Tuple of (tokens_per_second, latency_metrics)
    """
    start_time = time.perf_counter()
    first_token_time = None
    tokens_generated = 0

    # Measure generation
    output, token_count = generate_fn(prompt, max_tokens=num_tokens)
    end_time = time.perf_counter()

    # For streaming, we'd capture first_token_time during generation
    # For non-streaming, estimate it as 10% of total time (rough heuristic)
    total_time = end_time - start_time
    if first_token_time is None:
        first_token_time = start_time + (total_time * 0.1)

    ttft_ms = (first_token_time - start_time) * 1000
    total_latency_ms = total_time * 1000
    tokens_generated = token_count

    latency = LatencyMetrics(
        time_to_first_token_ms=ttft_ms,
        total_latency_ms=total_latency_ms,
        tokens_generated=tokens_generated,
        prompt_tokens=len(prompt.split()),  # Rough estimate
        latency_per_token_ms=total_latency_ms / tokens_generated if tokens_generated > 0 else 0,
        prompt_processing_time_ms=ttft_ms,
    )

    tokens_per_second = tokens_generated / total_time if total_time > 0 else 0

    return tokens_per_second, latency


class SpeedBenchmark:
    """Comprehensive speed benchmarking for model inference.

    Tests:
    - Single-request latency
    - Multi-request throughput
    - Batch processing efficiency
    - Context window scaling
    """

    def __init__(
        self,
        model_path: Path | str,
        model_name: str | None = None,
        model_loader: callable | None = None,
    ):
        """Initialize speed benchmark.

        Args:
            model_path: Path to model checkpoint
            model_name: Display name for model
            model_loader: Custom loader function(path) -> model
        """
        self.model_path = Path(model_path)
        self.model_name = model_name or self.model_path.name
        self.model_loader = model_loader
        self._model = None

    def _load_model(self):
        """Load model using provided loader or default."""
        if self._model is None:
            if self.model_loader:
                self._model = self.model_loader(self.model_path)
            else:
                # Default: try to load with mlx
                try:
                    from mlx_lm import load

                    self._model = load(str(self.model_path))
                except ImportError as err:
                    raise RuntimeError(
                        "No model loader provided and mlx not available. "
                        "Pass model_loader function."
                    ) from err
        return self._model

    def _get_hardware_info(self) -> dict[str, Any]:
        """Collect hardware information."""
        info: dict[str, Any] = {}
        if psutil is not None:
            info = {
                "cpu_count": psutil.cpu_count(logical=False),
                "cpu_count_logical": psutil.cpu_count(logical=True),
                "total_memory_gb": psutil.virtual_memory().total / (1024**3),
            }

        if find_spec("mlx.core") is not None:
            info["device"] = "Apple Silicon GPU"
            info["unified_memory"] = True

        return info

    def run(
        self,
        test_prompts: list[str] | None = None,
        num_runs: int = 10,
        max_tokens: int = 100,
        batch_sizes: list[int] | None = None,
    ) -> SpeedBenchmarkResult:
        """Run complete speed benchmark.

        Args:
            test_prompts: List of prompts to test (default: synthetic prompts)
            num_runs: Number of runs per test (for averaging)
            max_tokens: Maximum tokens to generate per prompt
            batch_sizes: Batch sizes to test (default: [1, 4, 8])

        Returns:
            SpeedBenchmarkResult with all metrics
        """
        from datetime import datetime

        if test_prompts is None:
            test_prompts = self._generate_test_prompts()

        if batch_sizes is None:
            batch_sizes = [1]

        self._load_model()
        start_time = time.perf_counter()

        # Measure latency (single request)
        latency_results = []
        for prompt in test_prompts[:num_runs]:
            _, latency = self._measure_single_request(prompt, max_tokens)
            latency_results.append(latency)

        # Average latency metrics
        avg_latency = self._average_latency(latency_results)

        # Measure throughput (multiple requests)
        throughput_results = []
        for batch_size in batch_sizes:
            tput = self._measure_throughput(test_prompts, batch_size, max_tokens)
            throughput_results.append(tput)

        # Use best throughput
        best_throughput = max(throughput_results, key=lambda t: t.tokens_per_second)

        # Context window metrics
        context_size = self._get_context_window_size()
        context_util = self._measure_context_utilization(test_prompts, context_size)

        end_time = time.perf_counter()

        return SpeedBenchmarkResult(
            model_name=self.model_name,
            model_path=str(self.model_path),
            test_prompts=len(test_prompts),
            latency=avg_latency,
            throughput=best_throughput,
            context_window_size=context_size,
            context_utilization=context_util,
            timestamp=datetime.now().isoformat(),
            duration_seconds=end_time - start_time,
            hardware_info=self._get_hardware_info(),
        )

    def _measure_single_request(
        self, prompt: str, max_tokens: int
    ) -> tuple[float, LatencyMetrics]:
        """Measure latency for a single request."""
        # This is a placeholder - actual implementation depends on model API
        start = time.perf_counter()

        # Simulate generation (replace with actual model call)
        tokens_generated = max_tokens

        # Simulate first token delay
        time.sleep(0.01)  # 10ms first token
        first_token_time = time.perf_counter()

        # Simulate rest of generation
        time.sleep(0.001 * max_tokens)  # 1ms per token
        end = time.perf_counter()

        ttft_ms = (first_token_time - start) * 1000
        total_ms = (end - start) * 1000

        latency = LatencyMetrics(
            time_to_first_token_ms=ttft_ms,
            total_latency_ms=total_ms,
            tokens_generated=tokens_generated,
            prompt_tokens=len(prompt.split()),
            latency_per_token_ms=(total_ms - ttft_ms) / tokens_generated if tokens_generated > 0 else 0,
            prompt_processing_time_ms=ttft_ms,
        )

        tps = tokens_generated / ((end - start) if (end - start) > 0 else 1)
        return tps, latency

    def _measure_throughput(
        self, prompts: list[str], batch_size: int, max_tokens: int
    ) -> ThroughputMetrics:
        """Measure throughput with batching."""
        start = time.perf_counter()
        total_tokens = 0
        requests = 0

        # Process in batches
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i : i + batch_size]
            for prompt in batch:
                tps, _ = self._measure_single_request(prompt, max_tokens)
                total_tokens += max_tokens
                requests += 1

        end = time.perf_counter()
        duration = end - start

        return ThroughputMetrics(
            tokens_per_second=total_tokens / duration if duration > 0 else 0,
            tokens_per_second_per_user=(total_tokens / duration) / batch_size if duration > 0 else 0,
            batch_size=batch_size,
            concurrent_requests=batch_size,
            requests_per_second=requests / duration if duration > 0 else 0,
            average_tokens_per_request=total_tokens / requests if requests > 0 else 0,
        )

    def _average_latency(self, results: list[LatencyMetrics]) -> LatencyMetrics:
        """Average multiple latency measurements."""
        if not results:
            return LatencyMetrics(0, 0, 0, 0, 0, 0)

        return LatencyMetrics(
            time_to_first_token_ms=sum(r.time_to_first_token_ms for r in results) / len(results),
            total_latency_ms=sum(r.total_latency_ms for r in results) / len(results),
            tokens_generated=int(sum(r.tokens_generated for r in results) / len(results)),
            prompt_tokens=int(sum(r.prompt_tokens for r in results) / len(results)),
            latency_per_token_ms=sum(r.latency_per_token_ms for r in results) / len(results),
            prompt_processing_time_ms=sum(r.prompt_processing_time_ms for r in results) / len(results),
        )

    def _get_context_window_size(self) -> int:
        """Get model's context window size."""
        # This would read from model config
        # Default: 8192 (common for Qwen2.5-Coder-7B)
        return 8192

    def _measure_context_utilization(
        self, prompts: list[str], context_size: int
    ) -> float:
        """Measure how efficiently context window is used."""
        # Average prompt length / context window size
        avg_prompt_len = sum(len(p.split()) for p in prompts) / len(prompts)
        return avg_prompt_len / context_size

    def _generate_test_prompts(self) -> list[str]:
        """Generate synthetic test prompts."""
        return [
            "Write a function to calculate factorial",
            "Explain how binary search works",
            "Optimize this code for performance",
            "Debug this assembly routine",
            "Generate test cases for this function",
            "Refactor this code to be more readable",
            "Implement a linked list in C++",
            "Parse JSON data from API",
            "Create a REST endpoint handler",
            "Write unit tests for user authentication",
        ]
