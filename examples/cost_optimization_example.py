#!/usr/bin/env python3
"""
Example: Cost Optimization and Analysis

This example demonstrates how to use the cost optimization system
to track GPU prices, analyze training costs, and generate optimization
recommendations.
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from afs.cost import (
    CostAnalyzer,
    CostOptimizer,
    GPUPrice,
    GPUPriceTracker,
    TrainingMetrics,
)


def example_1_gpu_price_tracking():
    """Example 1: Track GPU prices from various providers."""
    print("\n" + "=" * 70)
    print("EXAMPLE 1: GPU Price Tracking")
    print("=" * 70)

    tracker = GPUPriceTracker()

    # Track some GPU prices
    gpus = [
        GPUPrice(gpu_name="A100", provider="vast.ai", price_per_hour=1.50, vram_gb=40),
        GPUPrice(
            gpu_name="RTX 4090", provider="vast.ai", price_per_hour=0.45, vram_gb=24
        ),
        GPUPrice(
            gpu_name="H100", provider="lambda", price_per_hour=2.50, vram_gb=80
        ),
        GPUPrice(
            gpu_name="V100", provider="vast.ai", price_per_hour=0.29, vram_gb=32
        ),
    ]

    for gpu_price in gpus:
        tracker.track_price(gpu_price)

    # Get statistics
    stats = tracker.get_price_statistics()
    print(f"\nTracked {stats['total_tracked']} GPU type(s)")
    print(f"\nCurrent prices:")
    for key, price in sorted(
        stats["current_prices"].items(), key=lambda x: x[1]
    )[:5]:
        print(f"  {key:30s} ${price:.4f}/hr")

    # Get cheapest
    cheapest = tracker.get_cheapest_gpu()
    if cheapest:
        print(f"\nCheapest option: {cheapest[0].provider} {cheapest[0].gpu_name}")
        print(f"  Price: ${cheapest[0].price_per_hour:.4f}/hr")
        print(f"  24h average: ${cheapest[1]:.4f}/hr")

    # Get recommendations for specific GPU
    print(f"\nRecommendations for A100 under $2/hr:")
    recs = tracker.get_price_recommendations("A100", max_price=2.0)
    for gpu, reason in recs:
        print(f"  {gpu.provider:15s} ${gpu.price_per_hour:.4f}/hr - {reason}")


def example_2_cost_analysis():
    """Example 2: Analyze training costs."""
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Training Cost Analysis")
    print("=" * 70)

    analyzer = CostAnalyzer()

    # Simulate multiple training runs
    training_runs = [
        TrainingMetrics(
            run_id="bert_base_run_1",
            model_name="bert-base",
            num_samples=100000,
            num_epochs=3,
            batch_size=32,
            learning_rate=2e-5,
            total_duration_hours=24.5,
            gpu_name="A100",
            gpu_price_per_hour=1.50,
            test_accuracy=0.88,
            tokens_processed=50_000_000,
            timestamp=datetime.utcnow() - timedelta(days=3),
        ),
        TrainingMetrics(
            run_id="bert_base_run_2",
            model_name="bert-base",
            num_samples=150000,
            num_epochs=3,
            batch_size=32,
            learning_rate=2e-5,
            total_duration_hours=28.2,
            gpu_name="A100",
            gpu_price_per_hour=1.50,
            test_accuracy=0.91,
            tokens_processed=75_000_000,
            timestamp=datetime.utcnow() - timedelta(days=1),
        ),
        TrainingMetrics(
            run_id="gpt2_small_run_1",
            model_name="gpt2-small",
            num_samples=200000,
            num_epochs=2,
            batch_size=64,
            learning_rate=5e-5,
            total_duration_hours=16.3,
            gpu_name="RTX 4090",
            gpu_price_per_hour=0.45,
            test_accuracy=0.85,
            tokens_processed=100_000_000,
        ),
    ]

    print("\nAnalyzing training runs...")
    for metrics in training_runs:
        report = analyzer.analyze_training_run(metrics)
        print(
            f"\n{metrics.run_id}:"
            f"\n  Total cost: ${report.total_cost:.2f}"
            f"\n  Cost per sample: ${report.cost_per_sample:.6f}"
            f"\n  Cost per epoch: ${report.cost_per_epoch:.2f}"
            f"\n  Efficiency score: {report.efficiency_score:.3f}"
        )

    # Cost comparison
    print("\n" + "-" * 70)
    print("Cost Comparison by Model:")
    comparison = analyzer.get_cost_comparison()
    for model, stats in comparison.items():
        print(
            f"\n{model}:"
            f"\n  Runs: {stats['runs']}"
            f"\n  Total cost: ${stats['total_cost']:.2f}"
            f"\n  Avg per run: ${stats['avg_cost_per_run']:.2f}"
            f"\n  Best efficiency: {stats['best_efficiency_score']:.3f}"
        )

    # ROI analysis
    print("\n" + "-" * 70)
    print("ROI Analysis:")
    for model in ["bert-base", "gpt2-small"]:
        roi = analyzer.get_roi_analysis(model)
        print(
            f"\n{model}:"
            f"\n  Investment: ${roi['total_investment']:.2f}"
            f"\n  Accuracy improvement: {roi['accuracy_improvement']*100:+.2f}%"
            f"\n  Improvement per dollar: {roi['improvement_per_dollar']:.6f}"
        )


def example_3_budget_management():
    """Example 3: Manage training budgets."""
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Budget Management")
    print("=" * 70)

    analyzer = CostAnalyzer()

    # Set budgets for different models
    budgets = {
        "bert-base": 500.0,
        "gpt2-small": 300.0,
        "roberta-large": 1000.0,
    }

    print("\nSetting budgets:")
    for model, budget in budgets.items():
        analyzer.set_budget(model, budget)
        print(f"  {model:20s}: ${budget:8.2f}")

    # Simulate spending
    print("\n" + "-" * 70)
    print("Budget Status (simulated spending):")
    spending = {
        "bert-base": 250.0,
        "gpt2-small": 280.0,
        "roberta-large": 850.0,
    }

    for model, spent in spending.items():
        alert = analyzer.check_budget(model, spent)
        budget = budgets[model]
        percent = (spent / budget) * 100

        status = "✓ OK"
        if alert:
            status = f"⚠ {alert.alert_type}"

        print(f"  {model:20s} {status:15s} ${spent:8.2f} / ${budget:8.2f} ({percent:5.1f}%)")


def example_4_optimization_recommendations():
    """Example 4: Generate optimization recommendations."""
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Optimization Recommendations")
    print("=" * 70)

    optimizer = CostOptimizer()

    # Batch size recommendation
    print("\n1. Batch Size Optimization:")
    rec = optimizer.recommend_batch_size(
        current_batch_size=32,
        gpu_vram_gb=40,
        model_param_count=12_000_000,
        current_throughput=100,
        gpu_price_per_hour=1.50,
        epoch_hours=2.5,
    )
    if rec:
        print(f"  {rec.title}")
        print(f"  Estimated savings: ${rec.estimated_savings:.2f}")
        print(f"  Confidence: {rec.confidence:.0%}")
        print(f"  Action: {rec.action}")

    # Early stopping recommendation
    print("\n2. Early Stopping:")
    val_loss_history = [0.50, 0.35, 0.25, 0.20, 0.18, 0.1801, 0.1799, 0.1798]
    rec = optimizer.recommend_early_stopping(
        validation_loss_history=val_loss_history,
        gpu_price_per_hour=1.50,
        hours_per_epoch=2.5,
        patience=3,
    )
    if rec:
        print(f"  {rec.title}")
        print(f"  Estimated savings: ${rec.estimated_savings:.2f}")
        print(f"  Time saved: {rec.estimated_time_saved:.1f} hours")
        print(f"  Action: {rec.action}")

    # Epoch count optimization
    print("\n3. Epoch Count Optimization:")
    accuracy_history = [0.70, 0.80, 0.88, 0.90, 0.91, 0.9101, 0.9102]
    rec = optimizer.recommend_epoch_count(
        validation_scores=accuracy_history,
        gpu_price_per_hour=1.50,
        hours_per_epoch=2.5,
        score_type="accuracy",
    )
    if rec:
        print(f"  {rec.title}")
        print(f"  Estimated savings: ${rec.estimated_savings:.2f}")
        print(f"  Action: {rec.action}")

    # Dataset size optimization
    print("\n4. Dataset Size Optimization:")
    accuracy_curve = {
        1000: 0.75,
        5000: 0.85,
        10000: 0.90,
        50000: 0.92,
        100000: 0.925,
    }
    rec = optimizer.recommend_dataset_size(
        dataset_size=100000,
        validation_accuracy_curve=accuracy_curve,
        gpu_price_per_hour=1.50,
        hours_per_epoch=2.5,
        num_epochs=3,
    )
    if rec:
        print(f"  {rec.title}")
        print(f"  Estimated savings: ${rec.estimated_savings:.2f}")
        print(f"  Action: {rec.action}")

    # Summary
    print("\n" + "-" * 70)
    print("Recommendation Summary:")
    all_recs = optimizer.get_all_recommendations(hours=24)
    print(f"  Total recommendations: {len(all_recs)}")
    print(f"  Total potential savings: ${optimizer.get_total_potential_savings():.2f}")
    high_conf = optimizer.get_high_confidence_recommendations(confidence_threshold=0.8)
    print(f"  High-confidence (>80%): {len(high_conf)}")


def example_5_cost_forecasting():
    """Example 5: Forecast future training costs."""
    print("\n" + "=" * 70)
    print("EXAMPLE 5: Cost Forecasting")
    print("=" * 70)

    analyzer = CostAnalyzer()

    # Add sample training run
    metrics = TrainingMetrics(
        run_id="forecast_example",
        model_name="test_model",
        num_samples=100000,
        num_epochs=3,
        batch_size=32,
        learning_rate=2e-5,
        total_duration_hours=10.0,
        gpu_name="A100",
        gpu_price_per_hour=1.50,
        test_accuracy=0.90,
    )
    analyzer.analyze_training_run(metrics)

    # Set budget
    analyzer.set_budget("test_model", 100.0)

    # Forecast for different run counts
    print("\nCost Forecasts:")
    for runs in [1, 5, 10, 20]:
        forecast = analyzer.forecast_cost("test_model", runs)
        print(
            f"  {runs:2d} runs: ${forecast['estimated_total']:8.2f} - {forecast['budget_status']}"
        )

    # Show budget impact
    print("\nBudget Analysis:")
    budget = analyzer.budgets.get("test_model", 0)
    for runs in [1, 5, 10]:
        forecast = analyzer.forecast_cost("test_model", runs)
        remaining = budget - forecast["estimated_total"]
        print(
            f"  {runs} runs: ${remaining:+8.2f} remaining"
            f" ({(remaining/budget)*100:+6.1f}%)"
        )


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("COST OPTIMIZATION SYSTEM - COMPLETE EXAMPLES")
    print("=" * 70)

    example_1_gpu_price_tracking()
    example_2_cost_analysis()
    example_3_budget_management()
    example_4_optimization_recommendations()
    example_5_cost_forecasting()

    print("\n" + "=" * 70)
    print("✓ All examples completed successfully!")
    print("=" * 70)
    print(
        "\nFor more information, see:"
        "\n  - /Users/scawful/src/lab/afs/docs/COST_OPTIMIZATION.md"
        "\n  - /Users/scawful/src/lab/afs/src/afs/cost/README.md"
        "\n"
    )


if __name__ == "__main__":
    main()
