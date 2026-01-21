#!/usr/bin/env python3
"""CLI runner for comprehensive model benchmarking.

Usage:
    # Run all benchmarks
    python scripts/run_benchmarks.py --model path/to/model

    # Run specific benchmarks
    python scripts/run_benchmarks.py --model path/to/model --speed --quality

    # Compare with baseline
    python scripts/run_benchmarks.py --model path/to/model --baseline Qwen2.5-Coder-7B

    # Generate HTML report
    python scripts/run_benchmarks.py --model path/to/model --html report.html
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from afs.benchmarks.speed import SpeedBenchmark
from afs.benchmarks.quality import QualityBenchmark
from afs.benchmarks.resources import ResourceBenchmark


def run_speed_benchmark(model_path: Path, model_name: str, args) -> dict:
    """Run speed benchmarking."""
    print(f"\n{'=' * 60}")
    print("SPEED BENCHMARK")
    print(f"{'=' * 60}\n")

    benchmark = SpeedBenchmark(model_path, model_name)

    # Define workload
    def workload():
        return benchmark.run(
            test_prompts=args.test_prompts,
            num_runs=args.num_runs,
            max_tokens=args.max_tokens,
        )

    result = workload()
    print(result.summary())

    return result.to_dict()


def run_quality_benchmark(model_path: Path, model_name: str, args) -> dict:
    """Run quality benchmarking."""
    print(f"\n{'=' * 60}")
    print("QUALITY BENCHMARK")
    print(f"{'=' * 60}\n")

    if not args.test_dataset:
        print("Warning: No test dataset provided, skipping quality benchmark")
        return {}

    benchmark = QualityBenchmark(
        model_path=model_path,
        test_dataset=args.test_dataset,
        model_name=model_name,
    )

    result = benchmark.run(
        check_code_correctness=args.check_code,
        num_consistency_runs=args.consistency_runs,
    )
    print(result.summary())

    return result.to_dict()


def run_resource_benchmark(model_path: Path, model_name: str, args) -> dict:
    """Run resource monitoring benchmark."""
    print(f"\n{'=' * 60}")
    print("RESOURCE BENCHMARK")
    print(f"{'=' * 60}\n")

    benchmark = ResourceBenchmark(model_path, model_name)

    # Define a sample workload
    def workload():
        speed = SpeedBenchmark(model_path, model_name)
        # Run a few generations to monitor resources
        for _ in range(5):
            speed._measure_single_request("Write a function", max_tokens=50)

    result = benchmark.run(
        workload_fn=workload,
        monitor_vram=args.monitor_vram,
        monitor_power=args.monitor_power,
    )
    print(result.summary())

    return result.to_dict()


def compare_with_baseline(results: dict, baseline_name: str, baseline_results: dict) -> dict:
    """Compare results with baseline model."""
    comparison = {
        "baseline": baseline_name,
        "candidate": results["model_name"],
        "improvements": {},
    }

    # Speed comparison
    if "speed" in results and "speed" in baseline_results:
        speed_base = baseline_results["speed"]["throughput"]["tokens_per_second"]
        speed_cand = results["speed"]["throughput"]["tokens_per_second"]
        comparison["improvements"]["tokens_per_second"] = {
            "baseline": speed_base,
            "candidate": speed_cand,
            "improvement_pct": ((speed_cand - speed_base) / speed_base * 100) if speed_base else 0,
        }

    # Quality comparison
    if "quality" in results and "quality" in baseline_results:
        acc_base = baseline_results["quality"]["accuracy"]["accuracy"]
        acc_cand = results["quality"]["accuracy"]["accuracy"]
        comparison["improvements"]["accuracy"] = {
            "baseline": acc_base,
            "candidate": acc_cand,
            "improvement_pct": ((acc_cand - acc_base) / acc_base * 100) if acc_base else 0,
        }

    # Resource comparison
    if "resources" in results and "resources" in baseline_results:
        mem_base = baseline_results["resources"]["memory"]["peak_rss_mb"]
        mem_cand = results["resources"]["memory"]["peak_rss_mb"]
        comparison["improvements"]["memory_efficiency"] = {
            "baseline": mem_base,
            "candidate": mem_cand,
            "improvement_pct": ((mem_base - mem_cand) / mem_base * 100) if mem_base else 0,
        }

    return comparison


def generate_html_report(results: dict, output_path: Path):
    """Generate HTML report with graphs."""
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Model Benchmark Report - {results['model_name']}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background: white;
            padding: 40px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #333;
            border-bottom: 3px solid #007AFF;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #555;
            margin-top: 40px;
        }}
        .metric-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
            gap: 20px;
            margin: 20px 0;
        }}
        .metric-card {{
            background: #f8f9fa;
            padding: 20px;
            border-radius: 6px;
            border-left: 4px solid #007AFF;
        }}
        .metric-label {{
            font-size: 14px;
            color: #666;
            margin-bottom: 5px;
        }}
        .metric-value {{
            font-size: 28px;
            font-weight: bold;
            color: #333;
        }}
        .metric-unit {{
            font-size: 14px;
            color: #999;
        }}
        .chart-container {{
            position: relative;
            height: 300px;
            margin: 30px 0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background: #f8f9fa;
            font-weight: 600;
        }}
        .timestamp {{
            color: #999;
            font-size: 14px;
        }}
        .improvement {{
            color: #28a745;
        }}
        .regression {{
            color: #dc3545;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Model Benchmark Report</h1>
        <p><strong>Model:</strong> {results['model_name']}</p>
        <p><strong>Path:</strong> {results.get('model_path', 'N/A')}</p>
        <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
"""

    # Speed metrics
    if "speed" in results:
        speed = results["speed"]
        html += """
        <h2>Speed Metrics</h2>
        <div class="metric-grid">
"""
        html += f"""
            <div class="metric-card">
                <div class="metric-label">Tokens/Second</div>
                <div class="metric-value">{speed['throughput']['tokens_per_second']:.2f}</div>
            </div>
            <div class="metric-card">
                <div class="metric-label">First Token Latency</div>
                <div class="metric-value">{speed['latency']['time_to_first_token_ms']:.1f} <span class="metric-unit">ms</span></div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Latency per Token</div>
                <div class="metric-value">{speed['latency']['latency_per_token_ms']:.2f} <span class="metric-unit">ms</span></div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Context Utilization</div>
                <div class="metric-value">{speed['context_utilization']*100:.1f} <span class="metric-unit">%</span></div>
            </div>
        </div>
"""

    # Quality metrics
    if "quality" in results:
        quality = results["quality"]
        html += """
        <h2>Quality Metrics</h2>
        <div class="metric-grid">
"""
        html += f"""
            <div class="metric-card">
                <div class="metric-label">Accuracy</div>
                <div class="metric-value">{quality['accuracy']['accuracy']*100:.1f} <span class="metric-unit">%</span></div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Consistency Score</div>
                <div class="metric-value">{quality['consistency']['consistency_score']:.3f}</div>
            </div>
"""
        if quality.get("code_correctness"):
            html += f"""
            <div class="metric-card">
                <div class="metric-label">Code Compilation Rate</div>
                <div class="metric-value">{quality['code_correctness']['compilation_rate']*100:.1f} <span class="metric-unit">%</span></div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Test Pass Rate</div>
                <div class="metric-value">{quality['code_correctness']['test_pass_rate']*100:.1f} <span class="metric-unit">%</span></div>
            </div>
"""
        html += """
        </div>
"""

    # Resource metrics
    if "resources" in results:
        resources = results["resources"]
        html += """
        <h2>Resource Usage</h2>
        <div class="metric-grid">
"""
        html += f"""
            <div class="metric-card">
                <div class="metric-label">Peak Memory</div>
                <div class="metric-value">{resources['memory']['peak_rss_mb']:.1f} <span class="metric-unit">MB</span></div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Average Memory</div>
                <div class="metric-value">{resources['memory']['average_rss_mb']:.1f} <span class="metric-unit">MB</span></div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Peak CPU</div>
                <div class="metric-value">{resources['cpu']['peak_percent']:.1f} <span class="metric-unit">%</span></div>
            </div>
            <div class="metric-card">
                <div class="metric-label">Average CPU</div>
                <div class="metric-value">{resources['cpu']['average_percent']:.1f} <span class="metric-unit">%</span></div>
            </div>
        </div>
"""

    # Comparison table
    if "comparison" in results:
        comp = results["comparison"]
        html += f"""
        <h2>Comparison vs {comp['baseline']}</h2>
        <table>
            <thead>
                <tr>
                    <th>Metric</th>
                    <th>Baseline</th>
                    <th>Candidate</th>
                    <th>Improvement</th>
                </tr>
            </thead>
            <tbody>
"""
        for metric, values in comp["improvements"].items():
            improvement_pct = values["improvement_pct"]
            improvement_class = "improvement" if improvement_pct > 0 else "regression"
            html += f"""
                <tr>
                    <td>{metric.replace('_', ' ').title()}</td>
                    <td>{values['baseline']:.2f}</td>
                    <td>{values['candidate']:.2f}</td>
                    <td class="{improvement_class}">{improvement_pct:+.1f}%</td>
                </tr>
"""
        html += """
            </tbody>
        </table>
"""

    html += """
    </div>
</body>
</html>
"""

    output_path.write_text(html)
    print(f"\nHTML report generated: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Run comprehensive model benchmarks")

    # Model configuration
    parser.add_argument("--model", type=Path, required=True, help="Path to model checkpoint")
    parser.add_argument("--model-name", type=str, help="Display name for model")

    # Benchmark selection
    parser.add_argument("--speed", action="store_true", help="Run speed benchmark")
    parser.add_argument("--quality", action="store_true", help="Run quality benchmark")
    parser.add_argument("--resources", action="store_true", help="Run resource benchmark")
    parser.add_argument("--all", action="store_true", help="Run all benchmarks (default)")

    # Speed benchmark options
    parser.add_argument("--num-runs", type=int, default=10, help="Number of runs for averaging")
    parser.add_argument("--max-tokens", type=int, default=100, help="Max tokens to generate")
    parser.add_argument("--test-prompts", type=str, nargs="*", help="Custom test prompts")

    # Quality benchmark options
    parser.add_argument("--test-dataset", type=Path, help="Test dataset (JSONL)")
    parser.add_argument("--check-code", action="store_true", help="Check code correctness")
    parser.add_argument("--consistency-runs", type=int, default=3, help="Runs for consistency check")

    # Resource benchmark options
    parser.add_argument("--monitor-vram", action="store_true", default=True, help="Monitor VRAM")
    parser.add_argument("--monitor-power", action="store_true", default=True, help="Monitor power")

    # Comparison options
    parser.add_argument("--baseline", type=str, help="Baseline model name for comparison")
    parser.add_argument("--baseline-results", type=Path, help="Path to baseline results JSON")

    # Output options
    parser.add_argument("--output-dir", type=Path, default=Path("benchmarks/results"), help="Output directory")
    parser.add_argument("--html", type=Path, help="Generate HTML report")
    parser.add_argument("--json", type=Path, help="Save JSON results")

    args = parser.parse_args()

    # Default to all benchmarks if none specified
    if not (args.speed or args.quality or args.resources):
        args.all = True

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Model info
    model_name = args.model_name or args.model.name

    print(f"\n{'#' * 60}")
    print(f"MODEL BENCHMARK: {model_name}")
    print(f"{'#' * 60}")

    # Run benchmarks
    results = {
        "model_name": model_name,
        "model_path": str(args.model),
        "timestamp": datetime.now().isoformat(),
    }

    if args.all or args.speed:
        results["speed"] = run_speed_benchmark(args.model, model_name, args)

    if args.all or args.quality:
        results["quality"] = run_quality_benchmark(args.model, model_name, args)

    if args.all or args.resources:
        results["resources"] = run_resource_benchmark(args.model, model_name, args)

    # Compare with baseline
    if args.baseline and args.baseline_results:
        baseline_results = json.loads(args.baseline_results.read_text())
        results["comparison"] = compare_with_baseline(results, args.baseline, baseline_results)

    # Save results
    if args.json:
        args.json.write_text(json.dumps(results, indent=2))
        print(f"\nResults saved to: {args.json}")
    else:
        # Auto-save with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_path = args.output_dir / f"benchmark_{model_name}_{timestamp}.json"
        json_path.write_text(json.dumps(results, indent=2))
        print(f"\nResults saved to: {json_path}")

    # Generate HTML report
    if args.html:
        generate_html_report(results, args.html)

    print(f"\n{'#' * 60}")
    print("BENCHMARK COMPLETE")
    print(f"{'#' * 60}\n")


if __name__ == "__main__":
    main()
