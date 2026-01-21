#!/usr/bin/env python3
"""Monitor multiple vast.ai training instances in parallel.

Usage:
    python3 scripts/monitor_training_swarm.py --instances 30007012,30007267,30007268,30007269,30007270
"""

import argparse
import json
import subprocess
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any


def run_vastai_command(cmd: list[str]) -> dict[str, Any] | None:
    """Run vastai CLI command and parse JSON output."""
    try:
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True
        )
        return json.loads(result.stdout)
    except (subprocess.CalledProcessError, json.JSONDecodeError) as e:
        print(f"Error running command {' '.join(cmd)}: {e}")
        return None


def get_instance_status(instance_id: int) -> dict[str, Any] | None:
    """Get status of a single instance."""
    data = run_vastai_command(["vastai", "show", "instances", "--raw"])
    if not data:
        return None

    for instance in data:
        if instance.get("id") == instance_id:
            return instance
    return None


def format_time(seconds: float) -> str:
    """Format seconds as human-readable time."""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.0f}m"
    else:
        hours = seconds / 3600
        return f"{hours:.1f}h"


def estimate_completion(instance: dict[str, Any]) -> str:
    """Estimate training completion time (rough guess based on 3-epoch standard)."""
    # Assume 3 epochs total, estimate based on runtime
    runtime = instance.get("duration", 0)

    # Very rough estimate: 3 hours for most models
    estimated_total = 3 * 3600  # 3 hours in seconds
    remaining = max(0, estimated_total - runtime)

    if remaining < 60:
        return "Complete soon"
    return f"~{format_time(remaining)} remaining"


def print_status_table(instances: list[dict[str, Any]]):
    """Print formatted status table."""
    print("\n" + "="*100)
    print(f"Training Swarm Status - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*100)

    # Table header
    print(f"{'Instance':<10} {'GPU':<18} {'Status':<10} {'GPU%':<6} {'Mem%':<6} {'Runtime':<10} {'Cost/hr':<8} {'Estimate'}")
    print("-"*100)

    total_cost = 0.0
    for inst in instances:
        instance_id = inst.get("id", "?")
        gpu_name = inst.get("gpu_name", "?")
        status = inst.get("actual_status", "?")
        gpu_util = inst.get("gpu_util") or 0
        mem_util = inst.get("disk_util") or 0
        runtime = inst.get("duration") or 0
        cost_per_hour = inst.get("dph_total") or 0

        estimate = estimate_completion(inst)

        # Status emoji
        status_emoji = {
            "running": "✅",
            "loading": "🔄",
            "exited": "❌",
            "created": "⏳"
        }.get(status, "❓")

        print(
            f"{instance_id:<10} {gpu_name:<18} {status_emoji} {status:<8} "
            f"{gpu_util:>5.1f}% {mem_util:>5.1f}% {format_time(runtime):<10} "
            f"${cost_per_hour:<7.3f} {estimate}"
        )

        total_cost += cost_per_hour

    print("-"*100)
    print(f"Total Cost: ${total_cost:.3f}/hr (~${total_cost * 3:.2f} for 3 hours)")
    print("="*100)


def check_for_errors(instance_id: int) -> list[str]:
    """Check instance logs for errors (via SSH)."""
    # This would require SSH access - placeholder for now
    # In practice, you'd SSH in and check /workspace/output/*/logs
    return []


def main():
    parser = argparse.ArgumentParser(description="Monitor vast.ai training swarm")
    parser.add_argument(
        "--instances",
        required=True,
        help="Comma-separated list of instance IDs (e.g., 30007012,30007267)"
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Update interval in seconds (default: 300 = 5 minutes)"
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run once and exit (don't loop)"
    )

    args = parser.parse_args()
    instance_ids = [int(x.strip()) for x in args.instances.split(",")]

    print(f"Monitoring {len(instance_ids)} instances: {instance_ids}")
    print(f"Update interval: {args.interval}s")

    while True:
        instances = []
        for instance_id in instance_ids:
            inst = get_instance_status(instance_id)
            if inst:
                instances.append(inst)

        if instances:
            print_status_table(instances)
        else:
            print("No instances found")

        if args.once:
            break

        print(f"\nNext update in {args.interval}s... (Ctrl+C to stop)")
        time.sleep(args.interval)


if __name__ == "__main__":
    main()
