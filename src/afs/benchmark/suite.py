"""Benchmark suite for running all domain benchmarks."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .base import BenchmarkConfig, BenchmarkResult
from .din import DinBenchmark
from .nayru import NayruBenchmark, FaroreBenchmark, VeranBenchmark


class BenchmarkSuite:
    """Suite for running benchmarks across all domains."""

    RUNNERS = {
        "din": DinBenchmark,
        "nayru": NayruBenchmark,
        "farore": FaroreBenchmark,
        "veran": VeranBenchmark,
    }

    def __init__(
        self,
        datasets_root: Path,
        model_name: str,
        model_path: Path | None = None,
        model_type: str = "api",
        api_provider: str = "gemini",
        enable_semantic_eval: bool = False,
        output_dir: Path | None = None,
    ):
        self.datasets_root = Path(datasets_root)
        self.model_name = model_name
        self.model_path = model_path
        self.model_type = model_type
        self.api_provider = api_provider
        self.enable_semantic_eval = enable_semantic_eval
        self.output_dir = output_dir or Path("benchmark_results")
        self._results: dict[str, BenchmarkResult] = {}

    def _get_config(self, domain: str) -> BenchmarkConfig:
        """Create config for a domain."""
        dataset_path = self.datasets_root / domain / "benchmark.jsonl"
        if not dataset_path.exists():
            # Try alternative paths
            for alt in ["basic.jsonl", "all.jsonl", f"{domain}.jsonl"]:
                alt_path = self.datasets_root / domain / alt
                if alt_path.exists():
                    dataset_path = alt_path
                    break

        return BenchmarkConfig(
            dataset_path=dataset_path,
            model_name=self.model_name,
            model_path=self.model_path,
            model_type=self.model_type,
            api_provider=self.api_provider,
            enable_semantic_eval=self.enable_semantic_eval,
            output_dir=self.output_dir / domain,
        )

    def run_domain(self, domain: str) -> BenchmarkResult:
        """Run benchmark for a single domain."""
        if domain not in self.RUNNERS:
            raise ValueError(f"Unknown domain: {domain}. Available: {list(self.RUNNERS.keys())}")

        config = self._get_config(domain)
        runner_cls = self.RUNNERS[domain]
        runner = runner_cls(config)

        result = runner.run()
        self._results[domain] = result

        return result

    def run_all(self, domains: list[str] | None = None) -> dict[str, BenchmarkResult]:
        """Run benchmarks for all (or specified) domains."""
        domains = domains or list(self.RUNNERS.keys())

        for domain in domains:
            config = self._get_config(domain)
            if not config.dataset_path.exists():
                print(f"Skipping {domain}: dataset not found at {config.dataset_path}")
                continue

            try:
                result = self.run_domain(domain)
                print(f"{domain}: {result.pass_rate:.1%} ({result.passed}/{result.total_items})")
            except Exception as e:
                print(f"{domain}: ERROR - {e}")

        return self._results

    def save_results(self, output_path: Path | None = None) -> Path:
        """Save all results to a JSON file."""
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.output_dir / f"results_{self.model_name}_{timestamp}.json"

        output_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "model": self.model_name,
            "timestamp": datetime.now().isoformat(),
            "domains": {
                domain: result.to_dict()
                for domain, result in self._results.items()
            },
            "summary": self.summary_dict(),
        }

        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        return output_path

    def summary_dict(self) -> dict[str, Any]:
        """Generate summary statistics."""
        if not self._results:
            return {}

        total_items = sum(r.total_items for r in self._results.values())
        total_passed = sum(r.passed for r in self._results.values())

        return {
            "total_items": total_items,
            "total_passed": total_passed,
            "overall_pass_rate": total_passed / total_items if total_items > 0 else 0,
            "domains": {
                domain: {
                    "pass_rate": result.pass_rate,
                    "items": result.total_items,
                }
                for domain, result in self._results.items()
            },
        }

    def generate_report(self, output_path: Path | None = None) -> str:
        """Generate a markdown report."""
        lines = [
            f"# Benchmark Report: {self.model_name}",
            "",
            f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}",
            "",
            "## Summary",
            "",
        ]

        summary = self.summary_dict()
        if summary:
            lines.extend([
                f"- **Total Items:** {summary['total_items']}",
                f"- **Total Passed:** {summary['total_passed']}",
                f"- **Overall Pass Rate:** {summary['overall_pass_rate']:.1%}",
                "",
                "## Results by Domain",
                "",
                "| Domain | Pass Rate | Items | Duration |",
                "|--------|-----------|-------|----------|",
            ])

            for domain, result in self._results.items():
                lines.append(
                    f"| {domain.title()} | {result.pass_rate:.1%} | "
                    f"{result.passed}/{result.total_items} | {result.duration_seconds:.1f}s |"
                )

            lines.append("")

            # Detailed metrics
            lines.extend([
                "## Detailed Metrics",
                "",
            ])

            for domain, result in self._results.items():
                lines.extend([
                    f"### {domain.title()}",
                    "",
                ])

                for metric, value in sorted(result.metrics.items()):
                    lines.append(f"- **{metric}:** {value:.3f}")

                lines.append("")

        report = "\n".join(lines)

        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w") as f:
                f.write(report)

        return report


def run_benchmark(
    domain: str,
    dataset_path: Path,
    model_name: str,
    model_type: str = "api",
    api_provider: str = "gemini",
    model_path: Path | None = None,
    enable_semantic_eval: bool = False,
    output_dir: Path | None = None,
) -> BenchmarkResult:
    """Convenience function to run a single benchmark."""
    config = BenchmarkConfig(
        dataset_path=dataset_path,
        model_name=model_name,
        model_path=model_path,
        model_type=model_type,
        api_provider=api_provider,
        enable_semantic_eval=enable_semantic_eval,
        output_dir=output_dir or Path("benchmark_results"),
    )

    runners = {
        "din": DinBenchmark,
        "nayru": NayruBenchmark,
        "farore": FaroreBenchmark,
        "veran": VeranBenchmark,
    }

    if domain not in runners:
        raise ValueError(f"Unknown domain: {domain}")

    runner = runners[domain](config)
    return runner.run()
