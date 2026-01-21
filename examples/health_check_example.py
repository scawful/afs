#!/usr/bin/env python3
"""Example: Using the AFS Health Check System.

Demonstrates all major features of the health monitoring system.
"""

import asyncio
import json
from pathlib import Path

from afs.health import EnhancedHealthChecker, HealthCheckLevel, HealthStatus
from afs.health.daemon import HealthMonitoringDaemon


def example_basic_check():
    """Example 1: Run a basic health check."""
    print("\n" + "=" * 70)
    print("Example 1: Basic Health Check (System Health Only)")
    print("=" * 70)

    checker = EnhancedHealthChecker()
    result = checker.check(
        level=HealthCheckLevel.BASIC,
        auto_heal=False,
        save_report=False,
    )

    print(result.summary())

    # Access individual scores
    print("Individual Metrics:")
    for score in result.scores:
        status_emoji = "✓" if score.status == HealthStatus.EXCELLENT else "⚠"
        print(f"  {status_emoji} {score.category}/{score.metric}: {score.score:.2f} - {score.message}")


def example_comprehensive_check():
    """Example 2: Run a comprehensive health check."""
    print("\n" + "=" * 70)
    print("Example 2: Comprehensive Health Check (Full Diagnostics)")
    print("=" * 70)

    checker = EnhancedHealthChecker()
    result = checker.check(
        level=HealthCheckLevel.COMPREHENSIVE,
        auto_heal=False,
        save_report=False,
    )

    print(f"Overall Health: {result.overall_status.value.upper()} ({result.overall_score:.2f})")
    print(f"Total Duration: {result.duration_ms:.2f}ms")
    print(f"Metrics Checked: {len(result.scores)}")
    print(f"Detailed Checks: {len(result.checks)}")

    # Group scores by category
    categories = {}
    for score in result.scores:
        if score.category not in categories:
            categories[score.category] = []
        categories[score.category].append(score)

    print("\nMetrics by Category:")
    for category in sorted(categories.keys()):
        scores = categories[category]
        avg_score = sum(s.score for s in scores) / len(scores)
        status = "✓" if avg_score >= 0.7 else "⚠" if avg_score >= 0.5 else "✗"
        print(f"  {status} {category.upper()}: {avg_score:.2f}")

        for score in scores:
            indent = "    "
            print(f"{indent}• {score.metric}: {score.score:.2f} - {score.message}")


def example_json_output():
    """Example 3: JSON output for logging/integration."""
    print("\n" + "=" * 70)
    print("Example 3: JSON Output for Integration")
    print("=" * 70)

    checker = EnhancedHealthChecker()
    result = checker.check(
        level=HealthCheckLevel.BASIC,
        auto_heal=False,
        save_report=False,
    )

    result_json = result.to_dict()

    print("Sample JSON output (first 500 chars):")
    json_str = json.dumps(result_json, indent=2)
    print(json_str[:500] + "\n...")

    # Could be sent to logging system
    print(f"\nFull JSON size: {len(json_str)} bytes")
    print("Ready for: Splunk, DataDog, CloudWatch, or custom dashboards")


def example_custom_config():
    """Example 4: Custom configuration."""
    print("\n" + "=" * 70)
    print("Example 4: Custom Configuration")
    print("=" * 70)

    custom_config = {
        "cpu_threshold": 75.0,  # More aggressive
        "memory_threshold": 70.0,
        "test_queries": 10,  # More thorough
        "api_timeout_s": 5.0,  # Faster timeout
    }

    checker = EnhancedHealthChecker(config=custom_config)

    print("Custom config applied:")
    for key, value in sorted(custom_config.items()):
        print(f"  {key}: {value}")

    result = checker.check(
        level=HealthCheckLevel.STANDARD,
        auto_heal=False,
        save_report=False,
    )

    print(f"\nCheck completed with custom config: {result.overall_status.value}")


def example_auto_healing():
    """Example 5: Auto-healing (simulated)."""
    print("\n" + "=" * 70)
    print("Example 5: Auto-Healing (Safety Measures Enabled)")
    print("=" * 70)

    checker = EnhancedHealthChecker()

    print("Running health check with auto-healing enabled...")
    result = checker.check(
        level=HealthCheckLevel.BASIC,
        auto_heal=True,
        save_report=False,
    )

    print(f"\nHealing actions taken: {len(result.healing_actions)}")
    for action in result.healing_actions:
        print(f"  → {action}")

    if not result.healing_actions:
        print("  (No healing needed - system is healthy)")


def example_reporting():
    """Example 6: Report persistence and trends."""
    print("\n" + "=" * 70)
    print("Example 6: Report Persistence & Trends")
    print("=" * 70)

    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        context_root = Path(tmpdir)
        checker = EnhancedHealthChecker(context_root=context_root)

        # Run multiple checks to build trends
        print("Running 3 health checks to build trend data...")
        for i in range(3):
            result = checker.check(
                level=HealthCheckLevel.BASIC,
                auto_heal=False,
                save_report=True,
            )
            print(f"  Check {i+1}: {result.overall_status.value} ({result.overall_score:.2f})")

        # Check saved files
        health_dir = context_root / "health"
        reports = list(health_dir.glob("report-*.json"))

        print(f"\nReports saved: {len(reports)}")
        for report in sorted(reports):
            print(f"  • {report.name}")

        # Load and analyze trends
        trends_file = health_dir / "trends.json"
        if trends_file.exists():
            with open(trends_file) as f:
                trends = json.load(f)
                print(f"\nTrend data collected for {len(trends)} metrics")


async def example_continuous_monitoring():
    """Example 7: Continuous monitoring daemon (demo)."""
    print("\n" + "=" * 70)
    print("Example 7: Continuous Monitoring Daemon (30 seconds demo)")
    print("=" * 70)

    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        context_root = Path(tmpdir)
        daemon = HealthMonitoringDaemon(
            context_root=context_root,
            check_interval_s=10,  # 10 seconds for demo
            alert_threshold=0.2,
            auto_heal=False,
        )

        print("Starting monitoring daemon (will run 30 seconds)...")
        print("  Interval: 10 seconds")
        print("  Alert threshold: 0.2")
        print("  Check level: Standard")

        # Run for 30 seconds
        await daemon.start(
            check_level=HealthCheckLevel.STANDARD,
            duration_s=30,
        )

        # Get trend analysis
        trend = daemon.get_trend(hours=1)
        print(f"\nTrend Analysis:")
        print(f"  Current score: {trend.get('current', 'N/A'):.2f}")
        print(f"  Average score: {trend.get('average', 'N/A'):.2f}")
        print(f"  Data points: {trend.get('data_points', 0)}")
        print(f"  Trend: {trend.get('trend', 'N/A')}")


def example_status_command():
    """Example 8: Quick status check (like `afs health status`)."""
    print("\n" + "=" * 70)
    print("Example 8: Quick Status Check")
    print("=" * 70)

    checker = EnhancedHealthChecker()
    result = checker.check(
        level=HealthCheckLevel.BASIC,
        auto_heal=False,
        save_report=False,
    )

    status_summary = checker.get_status_summary()
    print(status_summary)


def main():
    """Run all examples."""
    print("\n" + "🏥 " * 20)
    print("AFS HEALTH CHECK SYSTEM - EXAMPLES")
    print("🏥 " * 20)

    # Run synchronous examples
    example_basic_check()
    example_comprehensive_check()
    example_json_output()
    example_custom_config()
    example_auto_healing()
    example_reporting()
    example_status_command()

    # Run async example
    print("\n" + "=" * 70)
    print("Running async example (continuous monitoring daemon)...")
    print("=" * 70)

    try:
        asyncio.run(example_continuous_monitoring())
    except KeyboardInterrupt:
        print("\n[Interrupted by user]")

    print("\n" + "=" * 70)
    print("All examples completed!")
    print("=" * 70)
    print("\nFor more information:")
    print("  • See: src/afs/health/README.md")
    print("  • CLI:  afs health --help")
    print("  • API:  from afs.health import EnhancedHealthChecker")


if __name__ == "__main__":
    main()
