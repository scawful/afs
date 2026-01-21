"""Model comparison CLI commands.

Examples:
    # Head-to-head comparison
    python3 -m afs comparison compare --models v5,v6 --questions eval/questions.json

    # Tournament mode (all models)
    python3 -m afs comparison tournament --models v5,v6,v7 --questions eval/questions.json

    # Regression testing (historical baseline)
    python3 -m afs comparison regression --baseline v5 --candidate v6 --questions eval/historical.json

    # A/B test analysis
    python3 -m afs comparison ab-test --results ab_test_results.json
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def comparison_compare_command(args: argparse.Namespace) -> int:
    """Run head-to-head comparison between two models.

    Compares two models on identical prompts, scoring on:
    - Correctness
    - Completeness
    - Clarity
    - Efficiency
    - Speed
    """
    from ..comparison import ComparisonMode, ModelComparator
    from ..generators.model_generator import create_generator

    # Parse model specifications
    model_specs = [s.strip() for s in args.models.split(",")]
    if len(model_specs) != 2:
        print("ERROR: head-to-head comparison requires exactly 2 models")
        return 1

    # Load questions
    questions_path = Path(args.questions).expanduser()
    if not questions_path.exists():
        print(f"ERROR: Questions file not found: {questions_path}")
        return 1

    questions = _load_questions(questions_path)
    print(f"Loaded {len(questions)} questions")

    # Create comparator
    comparator = ModelComparator(comparison_mode=ComparisonMode.HEAD_TO_HEAD)

    # Load models
    for i, spec in enumerate(model_specs):
        print(f"Loading model {i+1}/2: {spec}")
        try:
            model = create_generator(
                model_type=args.type or "api",
                model_name=spec,
                api_provider=args.provider or "gemini",
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            comparator.load_model(spec, lambda m=model: m)
        except Exception as e:
            print(f"ERROR: Failed to load model {spec}: {e}")
            return 1

    # Run comparison
    print(f"\nRunning comparison on {len(questions)} questions...")
    gen_config = {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }

    try:
        report = comparator.run_prompts(questions, generation_config=gen_config)
    except Exception as e:
        logger.error(f"Comparison failed: {e}")
        print(f"ERROR: {e}")
        return 1

    # Generate and save reports
    output_dir = Path(args.output).expanduser() if args.output else Path("comparison_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Markdown report
    md_path = output_dir / "comparison_report.md"
    md_content = comparator.generate_markdown_report()
    md_path.write_text(md_content)
    print(f"\nMarkdown report: {md_path}")

    # JSON report
    json_path = output_dir / "comparison_report.json"
    comparator.save_report_json(json_path)
    print(f"JSON report: {json_path}")

    # HTML dashboard
    html_path = output_dir / "comparison_dashboard.html"
    comparator.generate_html_report(html_path)
    print(f"HTML dashboard: {html_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    ranked = report.get_ranked_models()
    for rank, (model_name, stats) in enumerate(ranked, 1):
        winner_mark = "🏆" if rank == 1 else " "
        print(f"{winner_mark} {rank}. {model_name}")
        print(f"   Mean Score: {stats['mean_overall_score']:.3f}")
        print(f"   Win Rate: {stats['win_rate']:.1%}")
        print(f"   Avg Latency: {stats['mean_latency_ms']:.0f}ms")
        print()

    return 0


def comparison_tournament_command(args: argparse.Namespace) -> int:
    """Run tournament comparison across 3+ models.

    Evaluates all models on the same prompts and ranks them.
    """
    from ..comparison import ComparisonMode, ModelComparator
    from ..generators.model_generator import create_generator

    # Parse model specifications
    model_specs = [s.strip() for s in args.models.split(",")]
    if len(model_specs) < 2:
        print("ERROR: tournament mode requires at least 2 models")
        return 1

    if len(model_specs) > 5:
        print("ERROR: tournament mode supports maximum 5 models")
        return 1

    # Load questions
    questions_path = Path(args.questions).expanduser()
    if not questions_path.exists():
        print(f"ERROR: Questions file not found: {questions_path}")
        return 1

    questions = _load_questions(questions_path)
    print(f"Loaded {len(questions)} questions")

    # Create comparator
    comparator = ModelComparator(comparison_mode=ComparisonMode.TOURNAMENT)

    # Load models
    for i, spec in enumerate(model_specs, 1):
        print(f"Loading model {i}/{len(model_specs)}: {spec}")
        try:
            model = create_generator(
                model_type=args.type or "api",
                model_name=spec,
                api_provider=args.provider or "gemini",
                temperature=args.temperature,
                max_tokens=args.max_tokens,
            )
            comparator.load_model(spec, lambda m=model: m)
        except Exception as e:
            print(f"ERROR: Failed to load model {spec}: {e}")
            return 1

    # Run tournament
    print(f"\nRunning tournament on {len(questions)} questions...")
    gen_config = {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }

    try:
        report = comparator.run_prompts(questions, generation_config=gen_config)
    except Exception as e:
        logger.error(f"Tournament failed: {e}")
        print(f"ERROR: {e}")
        return 1

    # Generate reports
    output_dir = Path(args.output).expanduser() if args.output else Path("tournament_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    md_path = output_dir / "tournament_report.md"
    md_path.write_text(comparator.generate_markdown_report())
    print(f"\nMarkdown report: {md_path}")

    json_path = output_dir / "tournament_report.json"
    comparator.save_report_json(json_path)
    print(f"JSON report: {json_path}")

    html_path = output_dir / "tournament_dashboard.html"
    comparator.generate_html_report(html_path)
    print(f"HTML dashboard: {html_path}")

    # Print rankings
    print("\n" + "=" * 60)
    print("TOURNAMENT STANDINGS")
    print("=" * 60)
    ranked = report.get_ranked_models()
    for rank, (model_name, stats) in enumerate(ranked, 1):
        print(f"{rank}. {model_name}")
        print(f"   Score: {stats['mean_overall_score']:.3f} "
              f"(±{stats['stdev_overall_score']:.3f})")
        print(f"   Wins: {stats['win_count']}/{stats['total_comparisons']} "
              f"({stats['win_rate']:.1%})")
        print()

    return 0


def comparison_regression_command(args: argparse.Namespace) -> int:
    """Run regression testing to detect performance degradation.

    Compares a new candidate model against a baseline model on
    historical test questions, looking for regressions.
    """
    from ..comparison import ComparisonMode, ModelComparator, StatisticalTester
    from ..generators.model_generator import create_generator

    baseline_model = args.baseline
    candidate_model = args.candidate

    # Load historical questions
    questions_path = Path(args.questions).expanduser()
    if not questions_path.exists():
        print(f"ERROR: Questions file not found: {questions_path}")
        return 1

    questions = _load_questions(questions_path)
    print(f"Loaded {len(questions)} regression test questions")

    # Create comparator
    comparator = ModelComparator(comparison_mode=ComparisonMode.REGRESSION)

    # Load baseline
    print(f"Loading baseline model: {baseline_model}")
    try:
        baseline = create_generator(
            model_type=args.type or "api",
            model_name=baseline_model,
            api_provider=args.provider or "gemini",
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        comparator.load_model(baseline_model, lambda m=baseline: m)
    except Exception as e:
        print(f"ERROR: Failed to load baseline: {e}")
        return 1

    # Load candidate
    print(f"Loading candidate model: {candidate_model}")
    try:
        candidate = create_generator(
            model_type=args.type or "api",
            model_name=candidate_model,
            api_provider=args.provider or "gemini",
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
        comparator.load_model(candidate_model, lambda m=candidate: m)
    except Exception as e:
        print(f"ERROR: Failed to load candidate: {e}")
        return 1

    # Run regression tests
    print(f"\nRunning regression tests on {len(questions)} questions...")
    gen_config = {
        "temperature": args.temperature,
        "max_tokens": args.max_tokens,
    }

    try:
        report = comparator.run_prompts(questions, generation_config=gen_config)
    except Exception as e:
        logger.error(f"Regression test failed: {e}")
        print(f"ERROR: {e}")
        return 1

    # Analyze results
    baseline_scores = [
        result.responses[baseline_model].overall_score
        for result in report.results
        if baseline_model in result.responses
    ]
    candidate_scores = [
        result.responses[candidate_model].overall_score
        for result in report.results
        if candidate_model in result.responses
    ]

    # Statistical testing
    t_stat, is_significant = StatisticalTester.t_test(candidate_scores, baseline_scores)
    effect_size = StatisticalTester.effect_size(candidate_scores, baseline_scores)

    # Generate report
    output_dir = Path(args.output).expanduser() if args.output else Path("regression_results")
    output_dir.mkdir(parents=True, exist_ok=True)

    md_path = output_dir / "regression_report.md"
    md_path.write_text(comparator.generate_markdown_report())
    print(f"\nMarkdown report: {md_path}")

    json_path = output_dir / "regression_report.json"
    comparator.save_report_json(json_path)
    print(f"JSON report: {json_path}")

    # Print analysis
    print("\n" + "=" * 60)
    print("REGRESSION TEST ANALYSIS")
    print("=" * 60)
    print(f"Baseline Model: {baseline_model}")
    print(f"  Mean Score: {sum(baseline_scores)/len(baseline_scores):.3f}")
    print()
    print(f"Candidate Model: {candidate_model}")
    print(f"  Mean Score: {sum(candidate_scores)/len(candidate_scores):.3f}")
    print()
    print("Statistical Test Results:")
    print(f"  t-statistic: {t_stat:.3f}")
    print(f"  Significant: {'YES ⚠️' if is_significant else 'NO ✓'}")
    print(f"  Effect Size (Cohen's d): {effect_size:.3f}")
    print()

    if effect_size > 0:
        print(f"✓ Candidate improves on baseline (d={effect_size:.3f})")
        return 0
    elif abs(effect_size) < 0.2:
        print(f"≈ No meaningful difference between models (d={effect_size:.3f})")
        return 0
    else:
        print(f"⚠️ Candidate regresses on baseline (d={effect_size:.3f})")
        return 1


def comparison_ab_test_command(args: argparse.Namespace) -> int:
    """Analyze A/B test results from production traffic split.

    Analyzes pre-recorded A/B test results showing which model
    performed better on real user traffic.
    """
    results_path = Path(args.results).expanduser()
    if not results_path.exists():
        print(f"ERROR: Results file not found: {results_path}")
        return 1

    # Load results
    try:
        with open(results_path) as f:
            ab_results = json.load(f)
    except json.JSONDecodeError as e:
        print(f"ERROR: Invalid JSON in results file: {e}")
        return 1

    # Extract data
    model_a = ab_results.get("model_a", "model_a")
    model_b = ab_results.get("model_b", "model_b")

    results_a = ab_results.get("results_a", [])
    results_b = ab_results.get("results_b", [])

    if not results_a or not results_b:
        print("ERROR: Results missing for one or both models")
        return 1

    # Compute statistics
    from ..comparison import StatisticalTester

    scores_a = [r.get("score", 0.0) for r in results_a]
    scores_b = [r.get("score", 0.0) for r in results_b]

    t_stat, is_significant = StatisticalTester.t_test(scores_a, scores_b)
    effect_size = StatisticalTester.effect_size(scores_a, scores_b)

    # Print results
    print("\n" + "=" * 60)
    print("A/B TEST ANALYSIS")
    print("=" * 60)
    print(f"Model A: {model_a}")
    print(f"  Samples: {len(results_a)}")
    print(f"  Mean Score: {sum(scores_a)/len(scores_a):.3f}")
    print()
    print(f"Model B: {model_b}")
    print(f"  Samples: {len(results_b)}")
    print(f"  Mean Score: {sum(scores_b)/len(scores_b):.3f}")
    print()
    print("Statistical Results:")
    print(f"  t-statistic: {t_stat:.3f}")
    print(f"  Significant (α=0.05): {'YES' if is_significant else 'NO'}")
    print(f"  Effect Size (Cohen's d): {effect_size:.3f}")
    print()

    winner = model_a if effect_size > 0 else model_b
    print(f"Winner: {winner}")

    return 0


def _load_questions(path: Path) -> list[str]:
    """Load questions from JSON file.

    File format: either array of strings or array of objects with 'prompt' field.
    """
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, list):
        questions = []
        for item in data:
            if isinstance(item, str):
                questions.append(item)
            elif isinstance(item, dict) and "prompt" in item:
                questions.append(item["prompt"])
        return questions

    raise ValueError("Questions file must be JSON array of strings or objects with 'prompt' field")


def register_parsers(subparsers: argparse._SubParsersAction) -> None:
    """Register comparison CLI commands."""
    comparison_parser = subparsers.add_parser(
        "comparison",
        help="Model comparison and evaluation",
        description="Compare multiple model versions side-by-side",
    )

    comparison_subparsers = comparison_parser.add_subparsers(
        dest="comparison_command",
        help="Comparison modes",
    )

    # Head-to-head comparison
    compare_parser = comparison_subparsers.add_parser(
        "compare",
        help="Head-to-head comparison (2 models)",
        description="Compare two models on identical prompts",
    )
    compare_parser.add_argument(
        "--models",
        required=True,
        help="Two models to compare (comma-separated)",
    )
    compare_parser.add_argument(
        "--questions",
        required=True,
        help="Path to questions JSON file",
    )
    compare_parser.add_argument(
        "--type",
        default="api",
        choices=["api", "mlx", "huggingface", "llama_cpp"],
        help="Model type",
    )
    compare_parser.add_argument(
        "--provider",
        default="gemini",
        choices=["gemini", "claude", "openai"],
        help="API provider (for API models)",
    )
    compare_parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Generation temperature",
    )
    compare_parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum tokens to generate",
    )
    compare_parser.add_argument(
        "--output",
        help="Output directory for reports",
    )
    compare_parser.set_defaults(func=comparison_compare_command)

    # Tournament mode
    tournament_parser = comparison_subparsers.add_parser(
        "tournament",
        help="Tournament comparison (3-5 models)",
        description="Compare multiple models and rank them",
    )
    tournament_parser.add_argument(
        "--models",
        required=True,
        help="Models to compare (comma-separated, 2-5 models)",
    )
    tournament_parser.add_argument(
        "--questions",
        required=True,
        help="Path to questions JSON file",
    )
    tournament_parser.add_argument(
        "--type",
        default="api",
        choices=["api", "mlx", "huggingface", "llama_cpp"],
        help="Model type",
    )
    tournament_parser.add_argument(
        "--provider",
        default="gemini",
        choices=["gemini", "claude", "openai"],
        help="API provider (for API models)",
    )
    tournament_parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Generation temperature",
    )
    tournament_parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum tokens to generate",
    )
    tournament_parser.add_argument(
        "--output",
        help="Output directory for reports",
    )
    tournament_parser.set_defaults(func=comparison_tournament_command)

    # Regression testing
    regression_parser = comparison_subparsers.add_parser(
        "regression",
        help="Regression testing (new vs baseline)",
        description="Test for performance degradation",
    )
    regression_parser.add_argument(
        "--baseline",
        required=True,
        help="Baseline model name/path",
    )
    regression_parser.add_argument(
        "--candidate",
        required=True,
        help="Candidate model name/path",
    )
    regression_parser.add_argument(
        "--questions",
        required=True,
        help="Path to historical questions JSON file",
    )
    regression_parser.add_argument(
        "--type",
        default="api",
        choices=["api", "mlx", "huggingface", "llama_cpp"],
        help="Model type",
    )
    regression_parser.add_argument(
        "--provider",
        default="gemini",
        choices=["gemini", "claude", "openai"],
        help="API provider (for API models)",
    )
    regression_parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Generation temperature",
    )
    regression_parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum tokens to generate",
    )
    regression_parser.add_argument(
        "--output",
        help="Output directory for reports",
    )
    regression_parser.set_defaults(func=comparison_regression_command)

    # A/B test analysis
    ab_parser = comparison_subparsers.add_parser(
        "ab-test",
        help="A/B test analysis",
        description="Analyze A/B test results",
    )
    ab_parser.add_argument(
        "--results",
        required=True,
        help="Path to A/B test results JSON file",
    )
    ab_parser.set_defaults(func=comparison_ab_test_command)

    # Set default function for comparison command group
    comparison_parser.set_defaults(func=lambda args: comparison_parser.print_help())
