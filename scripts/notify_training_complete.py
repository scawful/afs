#!/usr/bin/env python3
"""CLI tool for sending training completion notifications.

Examples:
    # Send training completion notification
    python notify_training_complete.py complete \\
        --model "Qwen2.5-Coder-7B" \\
        --run-id "training_123" \\
        --duration 3600 \\
        --loss 0.1234 \\
        --metrics "accuracy=0.95,f1=0.92"

    # Send training error notification
    python notify_training_complete.py error \\
        --model "Qwen2.5-Coder-7B" \\
        --run-id "training_123" \\
        --error "Out of memory error"

    # Send cost alert
    python notify_training_complete.py cost-alert \\
        --cost 12.50 \\
        --threshold 10.00

    # Send evaluation complete
    python notify_training_complete.py evaluation \\
        --model "Qwen2.5-Coder-7B" \\
        --run-id "training_123" \\
        --metrics "accuracy=0.95,f1=0.92,perplexity=1.23"

    # Check notification system status
    python notify_training_complete.py status
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from afs.logging_config import get_logger
from afs.notifications import (
    NotificationManager,
    DesktopNotifier,
    EmailNotifier,
    SlackNotifier,
    DiscordNotifier,
    EventType,
    NotificationLevel
)

logger = get_logger(__name__)


def setup_notifications() -> NotificationManager:
    """Set up notification manager with all configured handlers.

    Returns:
        Configured NotificationManager instance
    """
    manager = NotificationManager()

    # Desktop notifications (macOS)
    if os.getenv("DESKTOP_NOTIFICATIONS_ENABLED", "true").lower() == "true":
        desktop = DesktopNotifier(
            enable_sound=os.getenv("DESKTOP_NOTIFICATIONS_SOUND", "true").lower() == "true"
        )
        manager.register_handler("desktop", desktop)

    # Email notifications
    if os.getenv("SMTP_HOST"):
        email = EmailNotifier()
        manager.register_handler("email", email)

    # Slack notifications
    if os.getenv("SLACK_WEBHOOK_URL"):
        slack = SlackNotifier()
        manager.register_handler("slack", slack)

    # Discord notifications
    if os.getenv("DISCORD_WEBHOOK_URL"):
        discord = DiscordNotifier()
        manager.register_handler("discord", discord)

    return manager


def parse_metrics(metrics_str: str) -> dict:
    """Parse metrics from comma-separated key=value format.

    Args:
        metrics_str: "key1=value1,key2=value2"

    Returns:
        Dictionary of metrics
    """
    metrics = {}
    if not metrics_str:
        return metrics

    for item in metrics_str.split(","):
        if "=" not in item:
            continue
        key, value = item.split("=", 1)
        key = key.strip()
        value = value.strip()

        # Try to convert to float
        try:
            metrics[key] = float(value)
        except ValueError:
            metrics[key] = value

    return metrics


def cmd_status(args):
    """Show notification system status."""
    manager = setup_notifications()
    status = manager.get_status()

    print("\nNotification System Status")
    print("=" * 50)
    print(f"\nRegistered Handlers: {', '.join(status['registered_handlers'])}")
    print(f"Enabled Channels: {', '.join(status['enabled_channels']) or 'None'}")

    print("\nChannel Details:")
    for name, info in status["channel_status"].items():
        status_str = "✓ Configured" if info["configured"] else "✗ Not configured"
        print(f"  {name:15} {status_str:20} ({info['handler_type']})")

    if not status["enabled_channels"]:
        print("\n⚠️  No notification channels are enabled!")
        print("Check your .env configuration and ensure required credentials are set.")


def cmd_complete(args):
    """Send training completion notification."""
    manager = setup_notifications()

    if not manager.enabled_channels:
        print("Error: No notification channels enabled")
        return 1

    metrics = parse_metrics(args.metrics)

    manager.notify_training_completed(
        model_name=args.model,
        run_id=args.run_id,
        duration=args.duration,
        final_loss=args.loss,
        eval_metrics=metrics if metrics else None
    )

    print(f"✓ Training completion notification sent for {args.model}")
    return 0


def cmd_error(args):
    """Send training error notification."""
    manager = setup_notifications()

    if not manager.enabled_channels:
        print("Error: No notification channels enabled")
        return 1

    manager.notify_training_failed(
        model_name=args.model,
        run_id=args.run_id,
        error=args.error
    )

    print(f"✓ Training error notification sent for {args.model}")
    return 0


def cmd_started(args):
    """Send training started notification."""
    manager = setup_notifications()

    if not manager.enabled_channels:
        print("Error: No notification channels enabled")
        return 1

    manager.notify_training_started(
        model_name=args.model,
        run_id=args.run_id,
        epochs=args.epochs,
        batch_size=args.batch_size
    )

    print(f"✓ Training started notification sent for {args.model}")
    return 0


def cmd_cost_alert(args):
    """Send cost threshold alert."""
    manager = setup_notifications()

    if not manager.enabled_channels:
        print("Error: No notification channels enabled")
        return 1

    manager.notify_cost_alert(
        current_cost=args.cost,
        threshold=args.threshold,
        model_name=args.model
    )

    print(f"✓ Cost alert notification sent (${args.cost:.2f}/${args.threshold:.2f})")
    return 0


def cmd_evaluation(args):
    """Send evaluation complete notification."""
    manager = setup_notifications()

    if not manager.enabled_channels:
        print("Error: No notification channels enabled")
        return 1

    metrics = parse_metrics(args.metrics)

    if not metrics:
        print("Error: Must provide metrics with --metrics")
        return 1

    manager.notify_evaluation_completed(
        model_name=args.model,
        run_id=args.run_id,
        metrics=metrics
    )

    print(f"✓ Evaluation complete notification sent for {args.model}")
    return 0


def cmd_custom(args):
    """Send custom notification."""
    manager = setup_notifications()

    if not manager.enabled_channels:
        print("Error: No notification channels enabled")
        return 1

    # Parse event type
    try:
        event_type = EventType[args.event_type.upper()]
    except KeyError:
        print(f"Invalid event type: {args.event_type}")
        return 1

    # Parse level
    try:
        level = NotificationLevel[args.level.upper()]
    except KeyError:
        print(f"Invalid notification level: {args.level}")
        return 1

    kwargs = {}
    if args.model:
        kwargs["model_name"] = args.model
    if args.run_id:
        kwargs["run_id"] = args.run_id

    manager.notify(
        title=args.title,
        message=args.message,
        event_type=event_type,
        level=level,
        **kwargs
    )

    print(f"✓ Custom notification sent: {args.title}")
    return 0


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="AFS Training Notification CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )

    subparsers = parser.add_subparsers(dest="command", help="Command to execute")

    # Status command
    subparsers.add_parser("status", help="Show notification system status")

    # Complete command
    complete = subparsers.add_parser("complete", help="Send training completion notification")
    complete.add_argument("--model", required=True, help="Model name")
    complete.add_argument("--run-id", required=True, help="Training run ID")
    complete.add_argument("--duration", type=float, required=True, help="Training duration in seconds")
    complete.add_argument("--loss", type=float, required=True, help="Final training loss")
    complete.add_argument("--metrics", default="", help="Evaluation metrics (key1=value1,key2=value2)")

    # Error command
    error = subparsers.add_parser("error", help="Send training error notification")
    error.add_argument("--model", required=True, help="Model name")
    error.add_argument("--run-id", required=True, help="Training run ID")
    error.add_argument("--error", required=True, help="Error message")

    # Started command
    started = subparsers.add_parser("started", help="Send training started notification")
    started.add_argument("--model", required=True, help="Model name")
    started.add_argument("--run-id", required=True, help="Training run ID")
    started.add_argument("--epochs", type=int, required=True, help="Number of epochs")
    started.add_argument("--batch-size", type=int, required=True, help="Batch size")

    # Cost alert command
    cost = subparsers.add_parser("cost-alert", help="Send cost threshold alert")
    cost.add_argument("--cost", type=float, required=True, help="Current cost in USD")
    cost.add_argument("--threshold", type=float, required=True, help="Alert threshold in USD")
    cost.add_argument("--model", help="Model name (optional)")

    # Evaluation command
    evaluation = subparsers.add_parser("evaluation", help="Send evaluation complete notification")
    evaluation.add_argument("--model", required=True, help="Model name")
    evaluation.add_argument("--run-id", required=True, help="Training run ID")
    evaluation.add_argument("--metrics", required=True, help="Evaluation metrics (key1=value1,key2=value2)")

    # Custom command
    custom = subparsers.add_parser("custom", help="Send custom notification")
    custom.add_argument("--title", required=True, help="Notification title")
    custom.add_argument("--message", required=True, help="Notification message")
    custom.add_argument("--event-type", default="error_occurred", help="Event type")
    custom.add_argument("--level", default="info", help="Notification level (info/success/warning/error/critical)")
    custom.add_argument("--model", help="Model name (optional)")
    custom.add_argument("--run-id", help="Run ID (optional)")

    args = parser.parse_args()

    # Load environment variables from .env if present
    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        from dotenv import load_dotenv
        load_dotenv(env_path)

    # Route to appropriate command
    if args.command == "status":
        cmd_status(args)
        return 0
    elif args.command == "complete":
        return cmd_complete(args)
    elif args.command == "error":
        return cmd_error(args)
    elif args.command == "started":
        return cmd_started(args)
    elif args.command == "cost-alert":
        return cmd_cost_alert(args)
    elif args.command == "evaluation":
        return cmd_evaluation(args)
    elif args.command == "custom":
        return cmd_custom(args)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
