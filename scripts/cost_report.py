#!/usr/bin/env python3
"""Generate cost reports and optimization recommendations.

Usage:
    python scripts/cost_report.py --help
    python scripts/cost_report.py --summary
    python scripts/cost_report.py --model mymodel
    python scripts/cost_report.py --fetch-prices
    python scripts/cost_report.py --recommendations
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from afs.cost import CostAnalyzer, CostOptimizer, GPUPriceTracker


def print_header(title: str) -> None:
    """Print a formatted header."""
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def print_price_summary(tracker: GPUPriceTracker) -> None:
    """Print GPU price summary."""
    print_header("GPU Price Summary")

    stats = tracker.get_price_statistics()

    print(f"Total GPU types tracked: {stats['total_tracked']}")
    print(f"Available price points: {len(stats['current_prices'])}\n")

    # Cheapest options
    if stats["current_prices"]:
        sorted_prices = sorted(
            stats["current_prices"].items(), key=lambda x: x[1]
        )
        print("Cheapest GPU options:")
        for i, (key, price) in enumerate(sorted_prices[:10], 1):
            avg_24h = stats["average_prices_24h"].get(key, 0.0)
            trend = stats["price_trends_24h"].get(key, 0.0)
            print(
                f"  {i}. {key:30s} ${price:.4f}/hr "
                f"(24h avg: ${avg_24h:.4f}, trend: {trend:+.1f}%)"
            )

    # Recent alerts
    recent_alerts = tracker.get_alerts(hours=24)
    if recent_alerts:
        print(f"\nRecent price alerts (24h):")
        for alert in recent_alerts[:5]:
            print(f"  - {alert.message}")


def print_cost_analysis(analyzer: CostAnalyzer, model_name: Optional[str] = None) -> None:
    """Print cost analysis."""
    print_header("Cost Analysis")

    if model_name:
        # Single model analysis
        model_reports = [r for r in analyzer.reports.values() if r.model_name == model_name]

        if not model_reports:
            print(f"No training reports found for model: {model_name}")
            return

        print(f"Model: {model_name}")
        print(f"Training runs: {len(model_reports)}\n")

        total_cost = sum(r.total_cost for r in model_reports)
        total_samples = sum(r.metrics.num_samples for r in model_reports if r.metrics)
        total_hours = sum(r.gpu_hours for r in model_reports)

        print(f"Total cost: ${total_cost:.2f}")
        print(f"Total GPU hours: {total_hours:.2f}")
        print(f"Cost per GPU hour: ${total_cost / total_hours:.3f}" if total_hours > 0 else "N/A")
        print(f"Cost per sample: ${total_cost / total_samples:.6f}" if total_samples > 0 else "N/A")

        # ROI analysis
        roi = analyzer.get_roi_analysis(model_name)
        print(f"\nROI Analysis:")
        print(f"  Investment: ${roi['total_investment']:.2f}")
        print(f"  Initial accuracy: {roi.get('first_accuracy', 0)*100:.2f}%")
        print(f"  Current accuracy: {roi.get('current_accuracy', 0)*100:.2f}%")
        print(f"  Improvement: {roi.get('accuracy_improvement', 0)*100:+.2f}%")
        print(f"  Per dollar: {roi.get('improvement_per_dollar', 0):.6f}")

    else:
        # Overall analysis
        comparison = analyzer.get_cost_comparison()

        if not comparison:
            print("No training reports found")
            return

        print(f"Models analyzed: {len(comparison)}\n")

        for model, stats in sorted(
            comparison.items(), key=lambda x: x[1]["total_cost"], reverse=True
        ):
            print(f"{model}:")
            print(f"  Runs: {stats['runs']}")
            print(f"  Total cost: ${stats['total_cost']:.2f}")
            print(f"  Avg per run: ${stats['avg_cost_per_run']:.2f}")
            print(f"  Best efficiency: {stats['best_efficiency_score']:.2f}")
            print()


def print_recommendations(optimizer: CostOptimizer) -> None:
    """Print optimization recommendations."""
    print_header("Cost Optimization Recommendations")

    recent_recs = optimizer.get_all_recommendations(hours=24)

    if not recent_recs:
        print("No recent recommendations generated")
        return

    # Group by category
    by_category = {}
    for rec in recent_recs:
        if rec.category not in by_category:
            by_category[rec.category] = []
        by_category[rec.category].append(rec)

    total_savings = optimizer.get_total_potential_savings()
    print(f"Total potential savings: ${total_savings:.2f}\n")

    for category in sorted(by_category.keys()):
        print(f"{category.upper()}")
        print("-" * 70)

        for rec in by_category[category]:
            print(f"\n{rec.title}")
            print(f"  Category: {rec.category}")
            print(f"  Estimated savings: ${rec.estimated_savings:.2f}")
            if rec.estimated_time_saved:
                print(f"  Time saved: {rec.estimated_time_saved:.2f} hours")
            print(f"  Confidence: {rec.confidence:.0%}")
            print(f"  Action: {rec.action}")

            if rec.risks:
                print(f"  Risks:")
                for risk in rec.risks:
                    print(f"    - {risk}")

        print()


def print_budget_status(analyzer: CostAnalyzer) -> None:
    """Print budget status for all models."""
    print_header("Budget Status")

    if not analyzer.budgets:
        print("No budgets configured")
        return

    for model, budget in sorted(analyzer.budgets.items()):
        # Calculate current cost
        model_reports = [r for r in analyzer.reports.values() if r.model_name == model]
        current_cost = sum(r.total_cost for r in model_reports)

        percent_used = (current_cost / budget) * 100 if budget > 0 else 0

        status = "✓ OK"
        if percent_used >= 100:
            status = "✗ EXCEEDED"
        elif percent_used >= 90:
            status = "⚠ 90%"
        elif percent_used >= 75:
            status = "⚠ 75%"
        elif percent_used >= 50:
            status = "→ 50%"

        print(f"{model:20s} {status:12s} ${current_cost:8.2f} / ${budget:8.2f} ({percent_used:5.1f}%)")


def forecast_costs(analyzer: CostAnalyzer, model: str, runs: int) -> None:
    """Print cost forecast."""
    print_header(f"Cost Forecast: {model}")

    forecast = analyzer.forecast_cost(model, runs)

    print(f"Model: {forecast['model_name']}")
    print(f"Planned runs: {forecast['planned_runs']}")
    print(f"Average cost per run: ${forecast['avg_cost_per_run']:.2f}")
    print(f"Estimated total: ${forecast['estimated_total']:.2f}")
    print(f"Status: {forecast['budget_status']}")

    if forecast["budget_status"] == "OVER":
        overage = forecast.get("overage", 0)
        print(f"Budget overage: ${overage:.2f}")


def main() -> None:
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Generate cost reports and optimization recommendations"
    )
    parser.add_argument(
        "--summary", action="store_true", help="Print cost summary"
    )
    parser.add_argument(
        "--model", type=str, help="Analyze specific model"
    )
    parser.add_argument(
        "--fetch-prices", action="store_true", help="Fetch latest GPU prices"
    )
    parser.add_argument(
        "--prices", action="store_true", help="Show GPU price summary"
    )
    parser.add_argument(
        "--recommendations", action="store_true", help="Show optimization recommendations"
    )
    parser.add_argument(
        "--budget", action="store_true", help="Show budget status"
    )
    parser.add_argument(
        "--forecast", type=str, nargs=2, metavar=("MODEL", "RUNS"),
        help="Forecast costs for N planned runs"
    )
    parser.add_argument(
        "--set-budget", type=str, nargs=2, metavar=("MODEL", "BUDGET"),
        help="Set budget limit for model"
    )
    parser.add_argument(
        "--json", action="store_true", help="Output as JSON"
    )

    args = parser.parse_args()

    # Initialize tools
    tracker = GPUPriceTracker()
    analyzer = CostAnalyzer()
    optimizer = CostOptimizer()

    # Handle set-budget
    if args.set_budget:
        model_name, budget_str = args.set_budget
        try:
            budget = float(budget_str)
            analyzer.set_budget(model_name, budget)
            print(f"Set budget for {model_name}: ${budget:.2f}")
            return
        except ValueError:
            print(f"Invalid budget amount: {budget_str}")
            sys.exit(1)

    # Handle fetch-prices
    if args.fetch_prices:
        print_header("Fetching GPU Prices")
        prices = tracker.fetch_vastai_prices()
        print(f"Fetched {len(prices)} prices from vast.ai")

        for price in prices:
            tracker.track_price(price)

        print("Price tracking updated")

    # Default to showing summary if no specific report requested
    if not any([
        args.summary,
        args.model,
        args.prices,
        args.recommendations,
        args.budget,
        args.forecast,
        args.fetch_prices,
    ]):
        args.summary = True

    # Generate reports
    if args.summary:
        print_cost_analysis(analyzer)
        print_price_summary(tracker)
        print_budget_status(analyzer)

    if args.prices:
        print_price_summary(tracker)

    if args.model:
        print_cost_analysis(analyzer, args.model)

    if args.recommendations:
        print_recommendations(optimizer)

    if args.budget:
        print_budget_status(analyzer)

    if args.forecast:
        model_name, runs_str = args.forecast
        try:
            runs = int(runs_str)
            forecast_costs(analyzer, model_name, runs)
        except ValueError:
            print(f"Invalid number of runs: {runs_str}")
            sys.exit(1)


if __name__ == "__main__":
    main()
