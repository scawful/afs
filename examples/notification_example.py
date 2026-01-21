#!/usr/bin/env python3
"""Example: Using the notification system for training events.

This example demonstrates how to integrate the notification system
into a training pipeline with multiple notification channels.
"""

import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from afs.notifications import (
    NotificationManager,
    DesktopNotifier,
    EmailNotifier,
    SlackNotifier,
    DiscordNotifier,
    EventType,
    NotificationLevel,
)


def setup_notifications() -> NotificationManager:
    """Set up notification manager with available channels.

    Returns:
        Configured NotificationManager instance
    """
    manager = NotificationManager()

    # Desktop notifications (macOS)
    desktop = DesktopNotifier(enable_sound=True)
    manager.register_handler("desktop", desktop)

    # Email notifications (configure in .env)
    email = EmailNotifier()
    manager.register_handler("email", email)

    # Slack notifications (configure in .env)
    slack = SlackNotifier()
    manager.register_handler("slack", slack)

    # Discord notifications (configure in .env)
    discord = DiscordNotifier()
    manager.register_handler("discord", discord)

    return manager


def simulate_training():
    """Simulate training loop with notifications."""
    print("=" * 60)
    print("Training Simulation with Notifications")
    print("=" * 60)

    # Setup
    manager = setup_notifications()
    print("\nNotification System Status:")
    status = manager.get_status()
    print(f"  Registered handlers: {', '.join(status['registered_handlers'])}")
    print(f"  Enabled channels: {', '.join(status['enabled_channels']) or 'None'}")

    # Training parameters
    model_name = "Qwen2.5-Coder-7B"
    run_id = f"training_{int(time.time())}"
    num_epochs = 3
    batch_size = 4
    cost_per_epoch = 5.0

    print(f"\n{'='*60}")
    print("STARTING TRAINING")
    print(f"{'='*60}")

    # Notify training started
    print(f"\n[1/6] Notifying training start...")
    manager.notify_training_started(
        model_name=model_name,
        run_id=run_id,
        epochs=num_epochs,
        batch_size=batch_size
    )
    print("✓ Training started notification sent")

    # Simulate training epochs
    try:
        total_cost = 0.0
        final_loss = None

        for epoch in range(1, num_epochs + 1):
            print(f"\n[Epoch {epoch}/{num_epochs}]")
            print("  Training...")
            time.sleep(1)

            # Simulate loss values
            loss = 0.5 * (0.8 ** epoch)
            final_loss = loss
            total_cost += cost_per_epoch

            print(f"  Loss: {loss:.4f}")
            print(f"  Cumulative cost: ${total_cost:.2f}")

            # Check cost threshold
            if epoch == 2 and total_cost > 5.0:
                print(f"\n[2/6] Notifying cost alert...")
                manager.notify_cost_alert(
                    current_cost=total_cost,
                    threshold=5.0,
                    model_name=model_name
                )
                print("✓ Cost alert notification sent")

        print(f"\n{'='*60}")
        print("TRAINING COMPLETED")
        print(f"{'='*60}")

        # Notify training completed
        print(f"\n[3/6] Notifying training completion...")
        duration = num_epochs * 60.0  # Simulated duration
        manager.notify_training_completed(
            model_name=model_name,
            run_id=run_id,
            duration=duration,
            final_loss=final_loss,
            eval_metrics={
                "accuracy": 0.9543,
                "f1_score": 0.9201,
                "perplexity": 1.234,
                "total_cost": total_cost
            }
        )
        print("✓ Training completion notification sent")

        # Simulate evaluation
        print(f"\n[4/6] Running evaluation...")
        time.sleep(1)
        eval_metrics = {
            "accuracy": 0.9543,
            "f1": 0.9201,
            "precision": 0.9312,
            "recall": 0.9095,
            "perplexity": 1.234
        }

        print(f"\n[5/6] Notifying evaluation results...")
        manager.notify_evaluation_completed(
            model_name=model_name,
            run_id=run_id,
            metrics=eval_metrics
        )
        print("✓ Evaluation notification sent")

        # Custom milestone notification
        print(f"\n[6/6] Notifying loss improvement...")
        manager.notify(
            title="Loss Improved",
            message=f"Loss improved to {final_loss:.4f}",
            event_type=EventType.LOSS_IMPROVED,
            level=NotificationLevel.SUCCESS,
            model_name=model_name,
            run_id=run_id,
            metrics={"previous_loss": 0.4, "current_loss": final_loss}
        )
        print("✓ Loss improvement notification sent")

    except Exception as e:
        print(f"\n{'='*60}")
        print("TRAINING FAILED")
        print(f"{'='*60}")

        print(f"\nNotifying training failure...")
        manager.notify_training_failed(
            model_name=model_name,
            run_id=run_id,
            error=str(e)
        )
        print("✓ Training failure notification sent")
        raise

    print(f"\n{'='*60}")
    print("ALL NOTIFICATIONS SENT")
    print(f"{'='*60}\n")


def example_batch_notifications():
    """Example: Sending batch notifications."""
    print("\n" + "=" * 60)
    print("BATCH NOTIFICATION EXAMPLE")
    print("=" * 60)

    from afs.notifications import NotificationEvent

    manager = NotificationManager()

    # Create multiple events
    events = [
        NotificationEvent(
            event_type=EventType.CHECKPOINT_SAVED,
            title="Checkpoint Saved",
            message="Model checkpoint saved at step 1000",
            model_name="test-model",
            metrics={"step": 1000}
        ),
        NotificationEvent(
            event_type=EventType.CHECKPOINT_SAVED,
            title="Checkpoint Saved",
            message="Model checkpoint saved at step 2000",
            model_name="test-model",
            metrics={"step": 2000}
        ),
        NotificationEvent(
            event_type=EventType.LOSS_IMPROVED,
            title="Loss Improved",
            message="Loss decreased from 0.15 to 0.12",
            model_name="test-model",
            level=NotificationLevel.SUCCESS,
            metrics={"old_loss": 0.15, "new_loss": 0.12}
        ),
    ]

    # Send batch to email
    email = EmailNotifier()
    if email.is_configured():
        count = email.send_batch(events)
        print(f"\n✓ Sent {count} batch notifications via email")
    else:
        print("\n⚠️  Email notifier not configured (configure in .env)")


def example_conditional_notifications():
    """Example: Conditional notifications based on thresholds."""
    print("\n" + "=" * 60)
    print("CONDITIONAL NOTIFICATION EXAMPLE")
    print("=" * 60)

    manager = NotificationManager()
    desktop = DesktopNotifier()
    manager.register_handler("desktop", desktop)

    # Simulate metrics monitoring
    metrics = {
        "loss": 0.08,
        "accuracy": 0.96,
        "gpu_memory_usage": 0.92,
        "cost_so_far": 12.50,
        "epoch": 5
    }

    print("\nMonitoring metrics with thresholds:")
    print(f"  Loss: {metrics['loss']:.4f}")
    print(f"  Accuracy: {metrics['accuracy']:.2%}")
    print(f"  GPU Memory: {metrics['gpu_memory_usage']:.0%}")
    print(f"  Cost: ${metrics['cost_so_far']:.2f}")

    # Notify only if thresholds exceeded
    if metrics["gpu_memory_usage"] > 0.85:
        print("\n✓ GPU memory threshold exceeded")
        manager.notify(
            title="GPU Memory Warning",
            message=f"GPU memory usage: {metrics['gpu_memory_usage']:.0%}",
            event_type=EventType.GPU_MEMORY_WARNING,
            level=NotificationLevel.WARNING,
            metrics={"gpu_memory": metrics["gpu_memory_usage"]}
        )

    if metrics["cost_so_far"] > 10.0:
        print("✓ Cost threshold exceeded")
        manager.notify_cost_alert(
            current_cost=metrics["cost_so_far"],
            threshold=10.0
        )

    if metrics["loss"] < 0.1:
        print("✓ Loss below excellent threshold")
        manager.notify(
            title="Excellent Loss Value",
            message=f"Loss is {metrics['loss']:.4f} - below 0.1!",
            event_type=EventType.LOSS_IMPROVED,
            level=NotificationLevel.SUCCESS,
            metrics={"loss": metrics["loss"]}
        )


def main():
    """Main entry point."""
    # Load environment variables
    from dotenv import load_dotenv

    env_path = Path(__file__).parent.parent / ".env"
    if env_path.exists():
        load_dotenv(env_path)
        print(f"Loaded configuration from {env_path}")
    else:
        print(f"⚠️  No .env file found at {env_path}")
        print("Copy .env.example to .env and configure notification channels")

    # Run examples
    simulate_training()
    example_batch_notifications()
    example_conditional_notifications()

    print("\n" + "=" * 60)
    print("Examples completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()
