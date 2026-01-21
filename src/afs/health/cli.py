"""CLI commands for health monitoring."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from afs.logging_config import get_logger

from .daemon import run_daemon_cli
from .enhanced_checks import EnhancedHealthChecker, HealthCheckLevel

logger = get_logger(__name__)


def register_parsers(subparsers: argparse._SubParsersAction) -> None:
    """Register health check commands."""
    health_parser = subparsers.add_parser("health", help="System health checks and monitoring")
    health_subparsers = health_parser.add_subparsers(dest="health_command")

    # health check
    check_parser = health_subparsers.add_parser(
        "check",
        help="Run health checks at specified level",
    )
    check_parser.add_argument(
        "--level",
        choices=["basic", "standard", "comprehensive", "stress"],
        default="standard",
        help="Check depth level (default: standard)",
    )
    check_parser.add_argument(
        "--auto-heal",
        action="store_true",
        help="Enable automatic healing of detected issues",
    )
    check_parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON",
    )
    check_parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save report to disk",
    )
    check_parser.set_defaults(func=handle_check)

    # health monitor
    monitor_parser = health_subparsers.add_parser(
        "monitor",
        help="Start continuous health monitoring daemon",
    )
    monitor_parser.add_argument(
        "--interval",
        type=int,
        default=60,
        help="Check interval in seconds (default: 60)",
    )
    monitor_parser.add_argument(
        "--level",
        choices=["basic", "standard", "comprehensive", "stress"],
        default="standard",
        help="Check depth level (default: standard)",
    )
    monitor_parser.add_argument(
        "--duration",
        type=int,
        help="Run for N seconds (default: infinite)",
    )
    monitor_parser.add_argument(
        "--auto-heal",
        action="store_true",
        help="Enable automatic healing",
    )
    monitor_parser.set_defaults(func=handle_monitor)

    # health status
    status_parser = health_subparsers.add_parser(
        "status",
        help="Get current health status",
    )
    status_parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    status_parser.set_defaults(func=handle_status)

    # health trend
    trend_parser = health_subparsers.add_parser(
        "trend",
        help="Show health trends",
    )
    trend_parser.add_argument(
        "--hours",
        type=int,
        default=24,
        help="Show trend for N hours (default: 24)",
    )
    trend_parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    trend_parser.set_defaults(func=handle_trend)

    # health history
    history_parser = health_subparsers.add_parser(
        "history",
        help="List recent health check reports",
    )
    history_parser.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Show last N reports (default: 10)",
    )
    history_parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    history_parser.set_defaults(func=handle_history)

    # Set default if no subcommand
    health_parser.set_defaults(func=lambda args: health_parser.print_help())


def handle_check(args: argparse.Namespace) -> None:
    """Execute health check command."""
    checker = EnhancedHealthChecker()

    result = checker.check(
        level=args.level,
        auto_heal=args.auto_heal,
        save_report=not args.no_save,
    )

    if args.json:
        # Output JSON
        print(json.dumps(result.to_dict(), indent=2, default=str))
    else:
        # Output human-readable
        print(result.summary())
        if args.auto_heal and result.healing_actions:
            print("\nHealing Actions Taken:")
            for action in result.healing_actions:
                print(f"  • {action}")

    # Exit with status based on health
    sys.exit(0 if result.overall_score >= 0.5 else 1)


def handle_monitor(args: argparse.Namespace) -> None:
    """Execute monitor command."""
    logger.info(
        f"Starting health monitor: interval={args.interval}s, level={args.level}, "
        f"auto_heal={args.auto_heal}"
    )

    run_daemon_cli(
        interval=args.interval,
        level=args.level,
        duration=args.duration,
        auto_heal=args.auto_heal,
    )


def handle_status(args: argparse.Namespace) -> None:
    """Execute status command."""
    checker = EnhancedHealthChecker()

    # Quick health check
    result = checker.check(level=HealthCheckLevel.BASIC, save_report=False)

    if args.json:
        print(json.dumps(result.to_dict(), indent=2, default=str))
    else:
        print(checker.get_status_summary())
        print(f"\nLast updated: {result.timestamp.isoformat()}")


def handle_trend(args: argparse.Namespace) -> None:
    """Execute trend command."""
    from .daemon import HealthMonitoringDaemon

    daemon = HealthMonitoringDaemon()
    trend = daemon.get_trend(hours=args.hours)

    if args.json:
        print(json.dumps(trend, indent=2, default=str))
    else:
        print(f"\nHealth Trend (Last {args.hours} hours)")
        print("=" * 50)

        if "error" in trend:
            print(f"  {trend['error']}")
        else:
            print(f"  Current Score: {trend.get('current', 'N/A'):.2f}" if trend.get('current') else "  Current Score: N/A")
            print(f"  Average Score: {trend.get('average', 'N/A'):.2f}" if trend.get('average') else "  Average Score: N/A")
            print(f"  Min Score: {trend.get('min', 'N/A'):.2f}" if trend.get('min') else "  Min Score: N/A")
            print(f"  Max Score: {trend.get('max', 'N/A'):.2f}" if trend.get('max') else "  Max Score: N/A")
            print(f"  Trend: {trend.get('trend', 'N/A')}")
            print(f"  Data Points: {trend.get('data_points', 0)}")


def handle_history(args: argparse.Namespace) -> None:
    """Execute history command."""
    context_root = Path.home() / ".context"
    health_dir = context_root / "health"

    if not health_dir.exists():
        print("No health check history found")
        return

    # Get recent reports
    reports = sorted(health_dir.glob("report-*.json"), reverse=True)[: args.limit]

    if not reports:
        print("No health check reports found")
        return

    if args.json:
        history = []
        for report_file in reports:
            with open(report_file) as f:
                history.append(json.load(f))
        print(json.dumps(history, indent=2, default=str))
    else:
        print(f"\nRecent Health Check Reports (Last {len(reports)})")
        print("=" * 70)
        print(f"{'Timestamp':<30} {'Level':<15} {'Score':<8} {'Status':<10}")
        print("-" * 70)

        for report_file in reports:
            try:
                with open(report_file) as f:
                    data = json.load(f)
                    timestamp = data.get("timestamp", "Unknown")[:19]
                    level = data.get("check_level", "Unknown")
                    score = data.get("overall_score", 0)
                    status = data.get("overall_status", "Unknown")
                    print(f"{timestamp:<30} {level:<15} {score:<8.2f} {status:<10}")
            except Exception as e:
                logger.debug(f"Error reading report {report_file}: {e}")
