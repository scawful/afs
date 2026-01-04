"""Benchmark CLI commands."""

from __future__ import annotations

import argparse
from pathlib import Path


def benchmark_run_command(args: argparse.Namespace) -> int:
    """Run benchmarks for one or all domains."""
    from ..benchmark import BenchmarkSuite

    datasets_root = Path(args.datasets).expanduser()
    output_dir = Path(args.output).expanduser() if args.output else Path("benchmark_results")

    suite = BenchmarkSuite(
        datasets_root=datasets_root,
        model_name=args.model,
        model_type=args.type,
        api_provider=args.provider,
        enable_semantic_eval=args.semantic,
        output_dir=output_dir,
    )

    if args.domain:
        # Run single domain
        result = suite.run_domain(args.domain)
        print(result.summary())
    else:
        # Run all domains
        suite.run_all()

    # Generate report
    if args.report:
        report_path = output_dir / "report.md"
        report = suite.generate_report(report_path)
        print(f"\nReport saved to: {report_path}")

    # Save results
    results_path = suite.save_results()
    print(f"Results saved to: {results_path}")

    return 0


def benchmark_compare_command(args: argparse.Namespace) -> int:
    """Compare two models on benchmarks."""
    from ..benchmark import LeaderboardManager

    results_dir = Path(args.results_dir).expanduser() if args.results_dir else Path("benchmark_results")
    manager = LeaderboardManager(results_dir)

    result = manager.compare(
        baseline_model=args.baseline,
        candidate_model=args.candidate,
        domain=args.domain,
        metric=args.metric,
    )

    if result is None:
        print(f"Could not compare: models not found in leaderboard for {args.domain}")
        return 1

    print(result.summary())

    if result.is_significant:
        print(f"\n{'Improvement' if result.improvement > 0 else 'Regression'} is statistically significant")

    return 0


def benchmark_leaderboard_command(args: argparse.Namespace) -> int:
    """Show benchmark leaderboard."""
    from ..benchmark import LeaderboardManager

    results_dir = Path(args.results_dir).expanduser() if args.results_dir else Path("benchmark_results")
    manager = LeaderboardManager(results_dir)

    if args.domain:
        # Show single domain
        leaders = manager.get_leaders(args.domain, limit=args.limit)
        if not leaders:
            print(f"No results for domain: {args.domain}")
            return 1

        print(f"\n{args.domain.title()} Leaderboard:")
        print("-" * 50)
        for i, entry in enumerate(leaders, 1):
            print(f"{i:2}. {entry.model:30} {entry.pass_rate:.1%}")
    else:
        # Show all domains
        report = manager.generate_report()
        print(report)

    return 0


def benchmark_trend_command(args: argparse.Namespace) -> int:
    """Show trend data for a model/domain."""
    from ..benchmark import LeaderboardManager

    results_dir = Path(args.results_dir).expanduser() if args.results_dir else Path("benchmark_results")
    manager = LeaderboardManager(results_dir)

    trend_data = manager.trend(
        model=args.model,
        domain=args.domain,
        metric=args.metric,
        days=args.days,
    )

    if not trend_data:
        print(f"No history found for {args.model} on {args.domain}")
        return 1

    print(f"\nTrend: {args.model} on {args.domain}/{args.metric}")
    print("-" * 50)
    for point in trend_data:
        print(f"{point['timestamp'][:10]}  {point['value']:.3f}")

    return 0


def benchmark_create_dataset_command(args: argparse.Namespace) -> int:
    """Create a sample benchmark dataset."""
    import json
    from pathlib import Path

    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Sample items for each domain
    SAMPLE_ITEMS = {
        "din": [
            {
                "id": "din_basic_001",
                "category": "redundant_loads",
                "difficulty": 1,
                "code": "LDA #$00\nSTA $10\nLDA #$00\nSTA $11",
                "expected_output": "STZ $10\nSTZ $11",
                "expected_metrics": {"cycles_baseline": 16, "cycles_optimized": 8},
            },
            {
                "id": "din_basic_002",
                "category": "increment",
                "difficulty": 1,
                "code": "LDA $10\nCLC\nADC #$01\nSTA $10",
                "expected_output": "INC $10",
                "expected_metrics": {"cycles_baseline": 14, "cycles_optimized": 6},
            },
        ],
        "nayru": [
            {
                "id": "nayru_basic_001",
                "category": "joypad",
                "difficulty": 1,
                "code": "read joypad state",
                "metadata": {"task": "read joypad 1 and store in zero page"},
            },
            {
                "id": "nayru_basic_002",
                "category": "memory",
                "difficulty": 1,
                "code": "copy memory block",
                "metadata": {"task": "copy 16 bytes from $1000 to $2000"},
            },
        ],
        "farore": [
            {
                "id": "farore_basic_001",
                "category": "mode_mismatch",
                "difficulty": 1,
                "code": "LDA #$1234\nSTA $10",
                "expected_output": "REP #$20\nLDA #$1234\nSTA $10\nSEP #$20",
                "metadata": {"issue": "16-bit immediate with 8-bit accumulator", "symptom": "only low byte stored"},
            },
        ],
        "veran": [
            {
                "id": "veran_basic_001",
                "category": "load_store",
                "difficulty": 1,
                "code": "LDA #$42\nSTA $7E0100",
                "metadata": {"concepts": ["immediate addressing", "WRAM", "absolute long"]},
            },
        ],
    }

    domain = args.domain
    items = SAMPLE_ITEMS.get(domain, [])

    if not items:
        print(f"No sample items for domain: {domain}")
        return 1

    with open(output_path, "w") as f:
        for item in items:
            f.write(json.dumps(item) + "\n")

    print(f"Created {len(items)} sample items at: {output_path}")
    return 0


def register_parsers(subparsers: argparse._SubParsersAction) -> None:
    """Register benchmark command parsers."""
    bench_parser = subparsers.add_parser(
        "benchmark", help="Run benchmarks and compare models."
    )
    bench_sub = bench_parser.add_subparsers(dest="benchmark_command")

    # benchmark run
    run_parser = bench_sub.add_parser(
        "run", help="Run benchmarks."
    )
    run_parser.add_argument(
        "--domain", "-d",
        choices=["din", "nayru", "farore", "veran"],
        help="Run single domain (default: all).",
    )
    run_parser.add_argument(
        "--datasets", required=True,
        help="Path to benchmark datasets directory.",
    )
    run_parser.add_argument(
        "--model", "-m", required=True,
        help="Model name for tracking.",
    )
    run_parser.add_argument(
        "--type", "-t",
        choices=["api", "mlx", "huggingface"],
        default="api",
        help="Model type (default: api).",
    )
    run_parser.add_argument(
        "--provider", "-p",
        choices=["gemini", "claude", "openai"],
        default="gemini",
        help="API provider (default: gemini).",
    )
    run_parser.add_argument(
        "--semantic",
        action="store_true",
        help="Enable semantic evaluation (requires emulator).",
    )
    run_parser.add_argument(
        "--output", "-o",
        help="Output directory for results.",
    )
    run_parser.add_argument(
        "--report",
        action="store_true",
        help="Generate markdown report.",
    )
    run_parser.set_defaults(func=benchmark_run_command)

    # benchmark compare
    compare_parser = bench_sub.add_parser(
        "compare", help="Compare two models."
    )
    compare_parser.add_argument(
        "--baseline", "-b", required=True,
        help="Baseline model name.",
    )
    compare_parser.add_argument(
        "--candidate", "-c", required=True,
        help="Candidate model name.",
    )
    compare_parser.add_argument(
        "--domain", "-d", required=True,
        choices=["din", "nayru", "farore", "veran"],
        help="Domain to compare.",
    )
    compare_parser.add_argument(
        "--metric",
        default="pass_rate",
        help="Metric to compare (default: pass_rate).",
    )
    compare_parser.add_argument(
        "--results-dir",
        help="Path to results directory.",
    )
    compare_parser.set_defaults(func=benchmark_compare_command)

    # benchmark leaderboard
    leaderboard_parser = bench_sub.add_parser(
        "leaderboard", help="Show benchmark leaderboard."
    )
    leaderboard_parser.add_argument(
        "--domain", "-d",
        choices=["din", "nayru", "farore", "veran"],
        help="Show single domain (default: all).",
    )
    leaderboard_parser.add_argument(
        "--limit", "-n",
        type=int,
        default=10,
        help="Number of entries to show (default: 10).",
    )
    leaderboard_parser.add_argument(
        "--results-dir",
        help="Path to results directory.",
    )
    leaderboard_parser.set_defaults(func=benchmark_leaderboard_command)

    # benchmark trend
    trend_parser = bench_sub.add_parser(
        "trend", help="Show trend data for a model."
    )
    trend_parser.add_argument(
        "--model", "-m", required=True,
        help="Model name.",
    )
    trend_parser.add_argument(
        "--domain", "-d", required=True,
        choices=["din", "nayru", "farore", "veran"],
        help="Domain to show trend for.",
    )
    trend_parser.add_argument(
        "--metric",
        default="pass_rate",
        help="Metric to track (default: pass_rate).",
    )
    trend_parser.add_argument(
        "--days",
        type=int,
        default=30,
        help="Days of history (default: 30).",
    )
    trend_parser.add_argument(
        "--results-dir",
        help="Path to results directory.",
    )
    trend_parser.set_defaults(func=benchmark_trend_command)

    # benchmark create-dataset
    create_parser = bench_sub.add_parser(
        "create-dataset", help="Create sample benchmark dataset."
    )
    create_parser.add_argument(
        "--domain", "-d", required=True,
        choices=["din", "nayru", "farore", "veran"],
        help="Domain to create dataset for.",
    )
    create_parser.add_argument(
        "--output", "-o", required=True,
        help="Output file path.",
    )
    create_parser.set_defaults(func=benchmark_create_dataset_command)
