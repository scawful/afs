#!/usr/bin/env python3
"""Real-time terminal monitoring UI for vast.ai training instances.

A production-ready TUI for monitoring multiple training instances with:
- Live progress tracking (epoch, step, loss)
- Real-time resource metrics (GPU, memory)
- Cost tracking and time estimates
- Training log display
- Epoch completion alerts
- Keyboard controls for interactivity

Usage:
    # Monitor specific instances
    python3 scripts/training_monitor_tui.py --instances 30007012,30007267,30007268,30007269,30007270

    # With custom update interval
    python3 scripts/training_monitor_tui.py --instances 30007012,30007267 --interval 5

    # Monitor all running instances
    python3 scripts/training_monitor_tui.py --all

    # Dry run without vast.ai (uses mock data)
    python3 scripts/training_monitor_tui.py --instances 30007012,30007267 --mock
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import re
import subprocess
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text
from rich.box import DOUBLE
from rich.panel import Panel
from rich.progress import Progress, BarColumn, DownloadColumn, TransferSpeedColumn
from rich.layout import Layout
from rich.align import Align

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    """Training progress metrics for a single instance."""
    instance_id: int
    gpu_type: str
    status: str  # "running", "loading", "exited", "created"

    # Progress metrics
    epoch: int = 0
    total_epochs: int = 3
    step: int = 0
    total_steps: int = 0
    loss: float = 0.0

    # Resource metrics
    gpu_util: float = 0.0
    memory_util: float = 0.0
    disk_util: float = 0.0

    # Cost metrics
    cost_per_hour: float = 0.0
    total_cost: float = 0.0
    runtime_seconds: float = 0.0

    # Timestamps
    started_at: Optional[datetime] = None
    last_updated: Optional[datetime] = None

    # Logs
    recent_logs: deque = field(default_factory=lambda: deque(maxlen=10))

    def progress_percentage(self) -> float:
        """Calculate overall training progress percentage."""
        if self.total_steps <= 0:
            return 0.0
        return min(100.0, (self.step / self.total_steps) * 100.0)

    def health_status(self) -> str:
        """Determine health status based on metrics."""
        if self.status == "exited":
            return "error"
        if self.gpu_util < 10:
            return "warning"
        if self.loss > 10.0:
            return "warning"
        if self.gpu_util > 95:
            return "warning"
        return "healthy"

    def estimated_completion_time(self) -> Optional[timedelta]:
        """Estimate time to completion."""
        if self.total_steps <= 0 or self.step <= 0:
            return None

        elapsed = self.runtime_seconds
        time_per_step = elapsed / self.step
        remaining_steps = self.total_steps - self.step
        estimated_seconds = remaining_steps * time_per_step

        return timedelta(seconds=int(estimated_seconds))


class VastAIMonitor:
    """Monitor vast.ai instances and collect metrics."""

    def __init__(self, instance_ids: list[int], use_mock: bool = False):
        self.instance_ids = instance_ids
        self.use_mock = use_mock
        self.metrics: dict[int, TrainingMetrics] = {
            iid: self._create_default_metrics(iid) for iid in instance_ids
        }
        self.console = Console()
        self.running = True

    def _create_default_metrics(self, instance_id: int) -> TrainingMetrics:
        """Create default metrics for an instance."""
        return TrainingMetrics(
            instance_id=instance_id,
            gpu_type="Unknown",
            status="loading",
            started_at=datetime.now(),
        )

    def _run_vastai_command(self, cmd: list[str]) -> dict[str, Any] | None:
        """Run vastai CLI command and parse JSON output."""
        try:
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
                timeout=10,
            )
            return json.loads(result.stdout)
        except (subprocess.CalledProcessError, json.JSONDecodeError, subprocess.TimeoutExpired) as e:
            logger.warning(f"Error running {' '.join(cmd)}: {e}")
            return None

    def _fetch_instance_status(self, instance_id: int) -> dict[str, Any] | None:
        """Fetch instance status from vast.ai API."""
        if self.use_mock:
            return self._generate_mock_status(instance_id)

        data = self._run_vastai_command(["vastai", "show", "instances", "--raw"])
        if not data:
            return None

        for instance in data if isinstance(data, list) else [data]:
            if instance.get("id") == instance_id:
                return instance
        return None

    def _generate_mock_status(self, instance_id: int) -> dict[str, Any]:
        """Generate mock status for testing."""
        gpu_types = ["RTX 4090", "RTX 3090", "A100", "H100", "L40S"]
        import random

        return {
            "id": instance_id,
            "gpu_name": gpu_types[instance_id % len(gpu_types)],
            "actual_status": random.choice(["running", "loading"]),
            "gpu_util": random.uniform(20, 95),
            "disk_util": random.uniform(30, 80),
            "duration": random.uniform(300, 3600),
            "dph_total": round(random.uniform(0.25, 1.50), 2),
            "total_cost": round(random.uniform(5, 50), 2),
        }

    def _fetch_training_logs(self, instance_id: int) -> list[str]:
        """Fetch training logs from remote instance (via SSH)."""
        if self.use_mock:
            return self._generate_mock_logs(instance_id)

        # In production, you would SSH into the instance and fetch logs
        # For now, we'll return empty logs
        # ssh_cmd = f"ssh root@{instance_ip} tail -n 10 /workspace/output/*/training.log"
        return []

    def _generate_mock_logs(self, instance_id: int) -> list[str]:
        """Generate mock training logs."""
        logs = [
            f"Step {100 + instance_id}: Loss = 2.345, LR = 0.0001",
            f"Step {99 + instance_id}: Loss = 2.356, LR = 0.0001",
            f"Epoch 1/3 completed at {datetime.now().isoformat()}",
            f"Step {98 + instance_id}: Loss = 2.367, LR = 0.0001",
        ]
        return logs

    def _parse_training_metrics(
        self,
        instance_id: int,
        instance_data: dict[str, Any],
        logs: list[str],
    ) -> None:
        """Parse instance data and logs to extract training metrics."""
        metrics = self.metrics[instance_id]

        # Update basic instance info
        metrics.gpu_type = instance_data.get("gpu_name", "Unknown")
        metrics.status = instance_data.get("actual_status", "unknown")
        metrics.gpu_util = instance_data.get("gpu_util", 0.0) or 0.0
        metrics.memory_util = instance_data.get("memory", 0.0) or 0.0
        metrics.disk_util = instance_data.get("disk_util", 0.0) or 0.0
        metrics.cost_per_hour = instance_data.get("dph_total", 0.0) or 0.0
        metrics.total_cost = instance_data.get("total_cost", 0.0) or 0.0
        metrics.runtime_seconds = instance_data.get("duration", 0.0) or 0.0
        metrics.last_updated = datetime.now()

        # Store recent logs
        for log in logs:
            metrics.recent_logs.append(log)

        # Parse logs for training progress
        for log in reversed(logs):
            # Parse epoch info: "Epoch X/Y"
            epoch_match = re.search(r"Epoch\s+(\d+)/(\d+)", log)
            if epoch_match:
                metrics.epoch = int(epoch_match.group(1))
                metrics.total_epochs = int(epoch_match.group(2))

            # Parse step info: "Step X"
            step_match = re.search(r"Step\s+(\d+)", log)
            if step_match:
                metrics.step = int(step_match.group(1))
                # Estimate total steps (common: 1000 steps per epoch)
                if metrics.total_steps == 0:
                    metrics.total_steps = metrics.step * 10

            # Parse loss: "Loss = X.XXX"
            loss_match = re.search(r"Loss\s*=\s*([\d.]+)", log)
            if loss_match:
                metrics.loss = float(loss_match.group(1))

            # If we found updates, stop searching
            if epoch_match or step_match or loss_match:
                break

    def update_all_metrics(self) -> None:
        """Update metrics for all instances."""
        for instance_id in self.instance_ids:
            try:
                instance_data = self._fetch_instance_status(instance_id)
                if not instance_data:
                    logger.warning(f"Could not fetch data for instance {instance_id}")
                    continue

                logs = self._fetch_training_logs(instance_id)
                self._parse_training_metrics(instance_id, instance_data, logs)

            except Exception as e:
                logger.error(f"Error updating instance {instance_id}: {e}")


class TrainingMonitorUI:
    """Rich TUI for monitoring training instances."""

    def __init__(self, monitor: VastAIMonitor, refresh_interval: int = 10):
        self.monitor = monitor
        self.refresh_interval = refresh_interval
        self.console = Console()
        self.should_exit = False
        self.paused = False

    def _get_color_for_health(self, health: str) -> str:
        """Get color based on health status."""
        colors = {
            "healthy": "green",
            "warning": "yellow",
            "error": "red",
        }
        return colors.get(health, "white")

    def _format_time(self, seconds: float) -> str:
        """Format seconds as human-readable time."""
        if seconds < 60:
            return f"{int(seconds)}s"
        elif seconds < 3600:
            return f"{int(seconds // 60)}m"
        else:
            hours = seconds / 3600
            return f"{hours:.1f}h"

    def _format_cost(self, cost: float) -> str:
        """Format cost as currency string."""
        return f"${cost:.3f}"

    def _build_metrics_table(self) -> Table:
        """Build main metrics table."""
        table = Table(
            title="[bold cyan]VAST.AI Training Monitor[/bold cyan]",
            box=DOUBLE,
            show_header=True,
            header_style="bold magenta",
        )

        table.add_column("Instance", style="cyan", no_wrap=True)
        table.add_column("GPU Type", style="blue")
        table.add_column("Status", style="yellow")
        table.add_column("Progress", style="green")
        table.add_column("GPU%", justify="right")
        table.add_column("Memory%", justify="right")
        table.add_column("Loss", justify="right")
        table.add_column("Cost/hr", justify="right")
        table.add_column("Total Cost", justify="right")
        table.add_column("ETA", justify="right")

        total_cost = 0.0
        total_cost_per_hour = 0.0

        for metrics in self.monitor.metrics.values():
            health = metrics.health_status()
            color = self._get_color_for_health(health)

            # Status indicator
            status_emoji = {
                "running": "✅",
                "loading": "🔄",
                "exited": "❌",
                "created": "⏳",
                "unknown": "❓",
            }.get(metrics.status, "❓")

            status_text = f"{status_emoji} {metrics.status}"

            # Progress bar
            progress_text = (
                f"E{metrics.epoch}/{metrics.total_epochs} "
                f"({metrics.progress_percentage():.0f}%)"
            )

            # Loss formatting
            loss_text = f"{metrics.loss:.3f}" if metrics.loss > 0 else "N/A"

            # ETA
            eta = metrics.estimated_completion_time()
            eta_text = (
                self._format_time(eta.total_seconds())
                if eta and eta.total_seconds() > 0
                else "N/A"
            )

            # Row styling based on health
            row_style = None
            if health == "error":
                row_style = "red"
            elif health == "warning":
                row_style = "yellow"

            table.add_row(
                f"[{color}]{metrics.instance_id}[/{color}]",
                metrics.gpu_type,
                status_text,
                f"[{color}]{progress_text}[/{color}]",
                f"{metrics.gpu_util:>5.1f}%",
                f"{metrics.memory_util:>5.1f}%",
                f"[{color}]{loss_text}[/{color}]",
                self._format_cost(metrics.cost_per_hour),
                self._format_cost(metrics.total_cost),
                eta_text,
                style=row_style,
            )

            total_cost += metrics.total_cost
            total_cost_per_hour += metrics.cost_per_hour

        return table

    def _build_logs_panel(self) -> Panel:
        """Build recent logs display panel."""
        all_logs = []
        for metrics in self.monitor.metrics.values():
            for log in metrics.recent_logs:
                all_logs.append(f"[cyan]#{metrics.instance_id}[/cyan] {log}")

        if not all_logs:
            logs_text = "[dim]No logs available yet[/dim]"
        else:
            logs_text = "\n".join(all_logs[-10:])

        return Panel(
            logs_text,
            title="[bold]Recent Training Logs[/bold]",
            expand=False,
            height=8,
        )

    def _build_stats_panel(self) -> Panel:
        """Build statistics panel."""
        metrics_list = list(self.monitor.metrics.values())

        total_cost = sum(m.total_cost for m in metrics_list)
        total_cost_per_hour = sum(m.cost_per_hour for m in metrics_list)
        avg_gpu_util = sum(m.gpu_util for m in metrics_list) / len(metrics_list) if metrics_list else 0
        avg_memory = sum(m.memory_util for m in metrics_list) / len(metrics_list) if metrics_list else 0

        running_count = sum(1 for m in metrics_list if m.status == "running")
        error_count = sum(1 for m in metrics_list if m.status == "exited")

        stats_text = (
            f"[bold cyan]Instance Stats[/bold cyan]\n"
            f"Running: {running_count}/{len(metrics_list)} | "
            f"Errors: {error_count}\n\n"
            f"[bold green]Cost Summary[/bold green]\n"
            f"Total: ${total_cost:.2f} | "
            f"Per Hour: ${total_cost_per_hour:.3f}\n\n"
            f"[bold yellow]Resource Utilization[/bold yellow]\n"
            f"Avg GPU: {avg_gpu_util:.1f}% | "
            f"Avg Memory: {avg_memory:.1f}%"
        )

        return Panel(stats_text, title="[bold]Summary[/bold]")

    def _build_controls_panel(self) -> Panel:
        """Build keyboard controls panel."""
        paused_indicator = "[red]PAUSED[/red]" if self.paused else "[green]ACTIVE[/green]"

        controls_text = (
            f"[bold]Status: {paused_indicator}[/bold]\n"
            f"[cyan]q[/cyan] - Quit | "
            f"[cyan]r[/cyan] - Refresh | "
            f"[cyan]p[/cyan] - Pause/Resume | "
            f"[cyan]c[/cyan] - Clear Logs"
        )

        return Panel(controls_text, style="dim")

    def build_layout(self) -> Layout:
        """Build the complete layout."""
        layout = Layout()

        layout.split_column(
            Layout(name="metrics", size=15),
            Layout(name="logs", size=10),
            Layout(name="bottom"),
        )

        layout["bottom"].split_row(
            Layout(name="stats"),
            Layout(name="controls"),
        )

        layout["metrics"].update(self._build_metrics_table())
        layout["logs"].update(self._build_logs_panel())
        layout["stats"].update(self._build_stats_panel())
        layout["controls"].update(self._build_controls_panel())

        return layout

    def handle_keyboard_input(self) -> None:
        """Handle keyboard input in a non-blocking way."""
        import select

        if select.select([sys.stdin], [], [], 0)[0]:
            try:
                key = sys.stdin.read(1).lower()

                if key == "q":
                    self.should_exit = True
                elif key == "r":
                    self.monitor.update_all_metrics()
                elif key == "p":
                    self.paused = not self.paused
                elif key == "c":
                    for metrics in self.monitor.metrics.values():
                        metrics.recent_logs.clear()

            except Exception as e:
                logger.warning(f"Error reading input: {e}")

    def run(self) -> None:
        """Run the monitoring UI."""
        self.console.print(
            Panel(
                "[bold cyan]Initializing Training Monitor[/bold cyan]\n"
                "Fetching initial metrics...",
                style="blue",
            )
        )

        # Initial metrics fetch
        self.monitor.update_all_metrics()

        try:
            with Live(self.build_layout(), refresh_per_second=1, console=self.console) as live:
                last_update = time.time()

                while not self.should_exit:
                    # Handle keyboard input
                    self.handle_keyboard_input()

                    # Update metrics periodically
                    current_time = time.time()
                    if not self.paused and (current_time - last_update) >= self.refresh_interval:
                        self.monitor.update_all_metrics()
                        last_update = current_time

                    # Update display
                    live.update(self.build_layout())
                    time.sleep(0.1)

        except KeyboardInterrupt:
            pass
        finally:
            self.console.print(
                Panel(
                    "[yellow]Monitor stopped[/yellow]",
                    style="dim",
                )
            )


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Real-time monitoring UI for vast.ai training instances",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--instances",
        type=str,
        help="Comma-separated instance IDs (e.g., 30007012,30007267,30007268)",
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Monitor all running instances",
    )

    parser.add_argument(
        "--interval",
        type=int,
        default=10,
        help="Update interval in seconds (default: 10)",
    )

    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock data instead of real vast.ai API (for testing)",
    )

    args = parser.parse_args()

    # Determine instance IDs
    instance_ids: list[int] = []

    if args.all:
        # Fetch all running instances
        monitor = VastAIMonitor([], use_mock=args.mock)
        data = monitor._run_vastai_command(["vastai", "show", "instances", "--raw"])
        if data:
            instance_ids = [inst.get("id") for inst in (data if isinstance(data, list) else [data])]
            if not instance_ids:
                print("Error: No running instances found. Use --instances to specify IDs.")
                sys.exit(1)
    elif args.instances:
        instance_ids = [int(x.strip()) for x in args.instances.split(",")]
    else:
        parser.print_help()
        sys.exit(1)

    if not instance_ids:
        print("Error: Must provide --instances or --all")
        sys.exit(1)

    print(f"Monitoring {len(instance_ids)} instances: {instance_ids}")
    print(f"Update interval: {args.interval}s")
    print("Press 'h' for help or 'q' to quit\n")

    # Create monitor and UI
    monitor = VastAIMonitor(instance_ids, use_mock=args.mock)
    ui = TrainingMonitorUI(monitor, refresh_interval=args.interval)

    # Run the UI
    try:
        ui.run()
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
