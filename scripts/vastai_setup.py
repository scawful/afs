#!/usr/bin/env python3
"""Aggressive vast.ai training setup for multiple models.

Launches multiple vast.ai instances in parallel for:
- Majora v1 training
- Veran v5 retraining
- Experimental model training
- Hyperparameter search

Budget: $100
Strategy: Max parallelization, cost-effective GPU selection

Prerequisites:
    - vast.ai CLI installed: pip install vastai
    - API key configured: vastai set api-key YOUR_KEY
    - Training data uploaded to storage

Usage:
    # Launch Majora v1 training
    python3 scripts/vastai_setup.py --model majora --data models/majora_v1_training.jsonl

    # Launch multiple models in parallel
    python3 scripts/vastai_setup.py --all-models --budget 100

    # Monitor all instances
    python3 scripts/vastai_setup.py --monitor

    # Cleanup instances
    python3 scripts/vastai_setup.py --cleanup
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


@dataclass
class GPUConfig:
    """GPU configuration for vast.ai instances."""
    name: str
    gpu_name: str
    num_gpus: int
    disk_space: int  # GB
    min_download: float  # Mbps
    max_price: float  # $/hour


# GPU configurations by use case
GPU_CONFIGS = {
    "budget": GPUConfig(
        name="Budget Training",
        gpu_name="RTX 3090",
        num_gpus=1,
        disk_space=50,
        min_download=100,
        max_price=0.30,  # $0.30/hour
    ),
    "balanced": GPUConfig(
        name="Balanced Training",
        gpu_name="RTX 4090",
        num_gpus=1,
        disk_space=100,
        min_download=200,
        max_price=0.50,  # $0.50/hour
    ),
    "performance": GPUConfig(
        name="Performance Training",
        gpu_name="A100",
        num_gpus=1,
        disk_space=200,
        min_download=500,
        max_price=1.50,  # $1.50/hour
    ),
}


@dataclass
class TrainingJob:
    """Training job configuration."""
    model_name: str
    training_script: str
    training_data: Path
    output_path: Path
    epochs: int
    gpu_config: str  # "budget", "balanced", "performance"
    estimated_hours: float


# Training jobs configuration
TRAINING_JOBS = {
    "majora": TrainingJob(
        model_name="majora-v1",
        training_script="scripts/train_majora_v1.py",
        training_data=Path("models/majora_v1_training.jsonl"),
        output_path=Path("/workspace/output/majora-v1-lora"),
        epochs=3,
        gpu_config="balanced",  # RTX 4090
        estimated_hours=4.0,
    ),
    "veran": TrainingJob(
        model_name="veran-v5",
        training_script="scripts/train_veran_v5.py",
        training_data=Path("models/veran_v5_training.jsonl"),
        output_path=Path("/workspace/output/veran-v5-lora"),
        epochs=3,
        gpu_config="budget",  # RTX 3090
        estimated_hours=3.0,
    ),
}


class VastAIManager:
    """Manage vast.ai instances for training."""

    def __init__(self, dry_run: bool = False):
        self.dry_run = dry_run
        self.active_instances = []

    def search_offers(self, gpu_config: GPUConfig) -> list[dict]:
        """Search for available vast.ai offers matching config."""
        search_cmd = [
            "vastai", "search", "offers",
            f"gpu_name={gpu_config.gpu_name}",
            f"num_gpus={gpu_config.num_gpus}",
            f"disk_space>={gpu_config.disk_space}",
            f"inet_down>={gpu_config.min_download}",
            f"dph<={gpu_config.max_price}",
            "verified=true",
            "order=dph",  # Sort by price (cheapest first)
            "--raw",
        ]

        print(f"\nSearching for {gpu_config.name} instances...")
        print(f"  GPU: {gpu_config.gpu_name} x{gpu_config.num_gpus}")
        print(f"  Max price: ${gpu_config.max_price}/hour")

        if self.dry_run:
            print("  [DRY RUN] Would search vast.ai")
            return []

        try:
            result = subprocess.run(
                search_cmd,
                capture_output=True,
                text=True,
                check=True
            )
            offers = json.loads(result.stdout)
            print(f"  Found {len(offers)} offers")
            return offers
        except subprocess.CalledProcessError as e:
            print(f"  Error searching: {e}")
            return []

    def create_instance(
        self,
        job: TrainingJob,
        offer_id: int,
        budget: float
    ) -> int | None:
        """Create vast.ai instance for training job."""
        config = GPU_CONFIGS[job.gpu_config]

        # Calculate max runtime based on budget
        cost_per_hour = config.max_price
        max_hours = min(budget / cost_per_hour, job.estimated_hours * 1.5)

        print(f"\nLaunching {job.model_name} training...")
        print(f"  Offer ID: {offer_id}")
        print(f"  GPU: {config.gpu_name}")
        print(f"  Estimated: {job.estimated_hours}h @ ${cost_per_hour}/h")
        print(f"  Max runtime: {max_hours:.1f}h")

        # Build docker command
        docker_cmd = self._build_training_command(job)

        # Create instance
        create_cmd = [
            "vastai", "create", "instance", str(offer_id),
            "--image", "unslothai/unsloth:latest",
            "--disk", str(GPU_CONFIGS[job.gpu_config].disk_space),
            "--onstart-cmd", docker_cmd,
        ]

        if self.dry_run:
            print("  [DRY RUN] Would run:")
            print(f"    {' '.join(create_cmd)}")
            return None

        try:
            result = subprocess.run(
                create_cmd,
                capture_output=True,
                text=True,
                check=True
            )
            instance_data = json.loads(result.stdout)
            instance_id = instance_data.get("new_contract")

            print(f"  ✓ Created instance {instance_id}")

            self.active_instances.append({
                "instance_id": instance_id,
                "model_name": job.model_name,
                "offer_id": offer_id,
                "started": datetime.now().isoformat(),
                "max_hours": max_hours,
            })

            return instance_id

        except subprocess.CalledProcessError as e:
            print(f"  Error creating instance: {e}")
            return None

    def _build_training_command(self, job: TrainingJob) -> str:
        """Build training command for docker container."""
        # Commands to run on instance startup
        commands = [
            "cd /workspace",
            "git clone https://github.com/scawful/afs.git || true",
            "cd afs",
            "git pull",
            f"pip install -e .",
            f"python3 {job.training_script} --data {job.training_data} --output {job.output_path} --epochs {job.epochs}",
        ]

        return " && ".join(commands)

    def monitor_instances(self):
        """Monitor active training instances."""
        print("\n" + "=" * 60)
        print("Active Training Instances")
        print("=" * 60)

        if not self.active_instances:
            print("\nNo active instances")
            return

        for instance in self.active_instances:
            instance_id = instance["instance_id"]

            # Get instance status
            status_cmd = ["vastai", "show", "instance", str(instance_id), "--raw"]

            try:
                result = subprocess.run(
                    status_cmd,
                    capture_output=True,
                    text=True,
                    check=True
                )
                status = json.loads(result.stdout)

                print(f"\nInstance {instance_id} ({instance['model_name']})")
                print(f"  Status: {status.get('actual_status', 'unknown')}")
                print(f"  Runtime: {status.get('duration', 0)/3600:.1f}h")
                print(f"  Cost: ${status.get('total_cost', 0):.2f}")

            except subprocess.CalledProcessError:
                print(f"\nInstance {instance_id}: Error getting status")

    def cleanup_instances(self, instance_ids: list[int] | None = None):
        """Destroy specified instances or all active instances."""
        if instance_ids is None:
            instance_ids = [i["instance_id"] for i in self.active_instances]

        print("\n" + "=" * 60)
        print("Cleaning Up Instances")
        print("=" * 60)

        for instance_id in instance_ids:
            print(f"\nDestroying instance {instance_id}...")

            destroy_cmd = ["vastai", "destroy", "instance", str(instance_id)]

            if self.dry_run:
                print("  [DRY RUN] Would destroy instance")
                continue

            try:
                subprocess.run(destroy_cmd, check=True)
                print(f"  ✓ Destroyed instance {instance_id}")

                # Remove from active list
                self.active_instances = [
                    i for i in self.active_instances
                    if i["instance_id"] != instance_id
                ]

            except subprocess.CalledProcessError as e:
                print(f"  Error destroying instance: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="Aggressive vast.ai training setup",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--model",
        choices=list(TRAINING_JOBS.keys()),
        help="Train specific model",
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Train all models in parallel",
    )
    parser.add_argument(
        "--budget",
        type=float,
        default=100.0,
        help="Total budget in USD (default: $100)",
    )
    parser.add_argument(
        "--monitor",
        action="store_true",
        help="Monitor active instances",
    )
    parser.add_argument(
        "--cleanup",
        action="store_true",
        help="Cleanup all instances",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing",
    )

    args = parser.parse_args()

    manager = VastAIManager(dry_run=args.dry_run)

    # Monitor mode
    if args.monitor:
        manager.monitor_instances()
        return 0

    # Cleanup mode
    if args.cleanup:
        manager.cleanup_instances()
        return 0

    # Training mode
    jobs_to_run = []

    if args.all_models:
        jobs_to_run = list(TRAINING_JOBS.values())
    elif args.model:
        jobs_to_run = [TRAINING_JOBS[args.model]]
    else:
        print("Error: Specify --model or --all-models")
        return 1

    # Calculate budget allocation
    total_estimated_cost = sum(
        GPU_CONFIGS[job.gpu_config].max_price * job.estimated_hours
        for job in jobs_to_run
    )

    print("=" * 60)
    print("Vast.ai Training Setup")
    print("=" * 60)
    print(f"\nBudget: ${args.budget:.2f}")
    print(f"Jobs: {len(jobs_to_run)}")
    print(f"Estimated cost: ${total_estimated_cost:.2f}")

    if total_estimated_cost > args.budget:
        print("\nWarning: Estimated cost exceeds budget")
        print("Jobs will be limited to budget constraints")

    # Launch jobs
    remaining_budget = args.budget

    for job in jobs_to_run:
        if remaining_budget <= 0:
            print(f"\nSkipping {job.model_name}: Budget exhausted")
            continue

        # Search for offers
        config = GPU_CONFIGS[job.gpu_config]
        offers = manager.search_offers(config)

        if not offers:
            print(f"\nNo offers found for {job.model_name}")
            continue

        # Select cheapest offer
        best_offer = offers[0]
        offer_id = best_offer["id"]

        # Create instance
        instance_id = manager.create_instance(job, offer_id, remaining_budget)

        if instance_id:
            estimated_cost = config.max_price * job.estimated_hours
            remaining_budget -= estimated_cost

    print("\n" + "=" * 60)
    print("Setup Complete")
    print("=" * 60)
    print(f"\nLaunched {len(manager.active_instances)} instances")
    print(f"Remaining budget: ${remaining_budget:.2f}")
    print("\nMonitor with: python3 scripts/vastai_setup.py --monitor")
    print("Cleanup with: python3 scripts/vastai_setup.py --cleanup")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
