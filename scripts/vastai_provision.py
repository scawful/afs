#!/usr/bin/env python3
"""
vast.ai GPU instance provisioning for AFS inference.

Spins up a GPU instance with Ollama, loads models, sets up SSH tunnel.

Usage:
    python vastai_provision.py up [--gpu RTX_4090] [--disk 50]
    python vastai_provision.py down
    python vastai_provision.py status
    python vastai_provision.py ssh
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

# State file location
STATE_FILE = Path.home() / ".config" / "afs" / "vastai_state.json"


@dataclass
class InstanceState:
    """Current instance state."""
    instance_id: int | None = None
    ip_address: str | None = None
    ssh_port: int | None = None
    status: str = "stopped"
    gpu_type: str | None = None
    created_at: float | None = None
    cost_per_hour: float = 0.0

    def save(self):
        STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
        STATE_FILE.write_text(json.dumps(self.__dict__, indent=2))

    @classmethod
    def load(cls) -> "InstanceState":
        if STATE_FILE.exists():
            data = json.loads(STATE_FILE.read_text())
            return cls(**data)
        return cls()


def run_vastai(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    """Run vastai CLI command."""
    cmd = ["vastai"] + list(args)
    return subprocess.run(cmd, capture_output=True, text=True, check=check)


def find_best_offer(gpu_type: str = "RTX_4090", disk_gb: int = 50) -> dict | None:
    """Find best available GPU offer."""
    print(f"Searching for {gpu_type} offers...")

    # Search for offers
    result = run_vastai(
        "search", "offers",
        f"gpu_name={gpu_type}",
        f"disk_space>={disk_gb}",
        "reliability>0.95",
        "num_gpus=1",
        "--order", "dph_total",
        "--limit", "5",
        "-o", "json",
    )

    if result.returncode != 0:
        print(f"Search failed: {result.stderr}")
        return None

    offers = json.loads(result.stdout)
    if not offers:
        print(f"No {gpu_type} instances available")
        return None

    # Return cheapest offer
    return offers[0]


def provision_instance(gpu_type: str = "RTX_4090", disk_gb: int = 50) -> InstanceState:
    """Provision a new instance."""
    state = InstanceState.load()

    if state.instance_id and state.status == "running":
        print(f"Instance {state.instance_id} already running")
        return state

    # Find offer
    offer = find_best_offer(gpu_type, disk_gb)
    if not offer:
        raise RuntimeError("No suitable GPU offer found")

    print(f"Found offer: ${offer['dph_total']:.2f}/hr on {offer['gpu_name']}")

    # Onstart script to install and run Ollama
    onstart = """#!/bin/bash
set -e
# Install Ollama
curl -fsSL https://ollama.ai/install.sh | sh
# Start Ollama server
ollama serve &
sleep 10
# Pull base models
ollama pull qwen2.5-coder:7b
echo "Ollama ready"
"""

    # Create instance
    result = run_vastai(
        "create", "instance",
        str(offer["id"]),
        "--image", "nvidia/cuda:12.1-devel-ubuntu22.04",
        "--disk", str(disk_gb),
        "--onstart-cmd", onstart,
        "-o", "json",
    )

    if result.returncode != 0:
        raise RuntimeError(f"Failed to create instance: {result.stderr}")

    data = json.loads(result.stdout)
    state.instance_id = data.get("new_contract")
    state.gpu_type = gpu_type
    state.cost_per_hour = offer["dph_total"]
    state.created_at = time.time()
    state.status = "starting"
    state.save()

    print(f"Created instance {state.instance_id}")
    print("Waiting for instance to start...")

    # Wait for instance to be ready
    for _ in range(60):  # 5 minute timeout
        time.sleep(5)
        info = get_instance_info(state.instance_id)
        if info and info.get("actual_status") == "running":
            state.ip_address = info.get("public_ipaddr")
            state.ssh_port = info.get("ssh_port")
            state.status = "running"
            state.save()
            print(f"Instance ready: {state.ip_address}:{state.ssh_port}")
            return state

    raise RuntimeError("Timeout waiting for instance to start")


def get_instance_info(instance_id: int) -> dict | None:
    """Get instance info."""
    result = run_vastai("show", "instance", str(instance_id), "-o", "json", check=False)
    if result.returncode == 0:
        return json.loads(result.stdout)
    return None


def teardown_instance() -> bool:
    """Teardown current instance."""
    state = InstanceState.load()

    if not state.instance_id:
        print("No instance to teardown")
        return False

    print(f"Destroying instance {state.instance_id}...")
    result = run_vastai("destroy", "instance", str(state.instance_id), check=False)

    if result.returncode == 0:
        # Calculate cost
        if state.created_at:
            hours = (time.time() - state.created_at) / 3600
            cost = hours * state.cost_per_hour
            print(f"Instance ran for {hours:.2f} hours, estimated cost: ${cost:.2f}")

        state = InstanceState()
        state.save()
        print("Instance destroyed")
        return True

    print(f"Failed to destroy: {result.stderr}")
    return False


def show_status():
    """Show current instance status."""
    state = InstanceState.load()

    if not state.instance_id:
        print("No active instance")
        return

    info = get_instance_info(state.instance_id)

    print(f"Instance ID: {state.instance_id}")
    print(f"GPU: {state.gpu_type}")
    print(f"Status: {info.get('actual_status') if info else state.status}")
    print(f"IP: {state.ip_address}:{state.ssh_port}")
    print(f"Cost: ${state.cost_per_hour:.2f}/hr")

    if state.created_at:
        hours = (time.time() - state.created_at) / 3600
        print(f"Running: {hours:.2f} hours")
        print(f"Estimated cost: ${hours * state.cost_per_hour:.2f}")


def setup_tunnel(local_port: int = 11436):
    """Set up SSH tunnel to instance."""
    state = InstanceState.load()

    if not state.instance_id or not state.ip_address:
        print("No running instance")
        return

    print(f"Setting up SSH tunnel: localhost:{local_port} -> {state.ip_address}:11434")

    # This will block until interrupted
    subprocess.run([
        "ssh", "-N", "-L",
        f"{local_port}:localhost:11434",
        f"root@{state.ip_address}",
        "-p", str(state.ssh_port),
    ])


def connect_ssh():
    """SSH into the instance."""
    state = InstanceState.load()

    if not state.instance_id or not state.ip_address:
        print("No running instance")
        return

    subprocess.run([
        "ssh",
        f"root@{state.ip_address}",
        "-p", str(state.ssh_port),
    ])


def main():
    parser = argparse.ArgumentParser(description="vast.ai GPU provisioning for AFS")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # up command
    up_parser = subparsers.add_parser("up", help="Provision GPU instance")
    up_parser.add_argument("--gpu", default="RTX_4090", help="GPU type")
    up_parser.add_argument("--disk", type=int, default=50, help="Disk size in GB")

    # down command
    subparsers.add_parser("down", help="Teardown instance")

    # status command
    subparsers.add_parser("status", help="Show instance status")

    # tunnel command
    tunnel_parser = subparsers.add_parser("tunnel", help="Set up SSH tunnel")
    tunnel_parser.add_argument("--port", type=int, default=11436, help="Local port")

    # ssh command
    subparsers.add_parser("ssh", help="SSH into instance")

    args = parser.parse_args()

    try:
        if args.command == "up":
            provision_instance(args.gpu, args.disk)
        elif args.command == "down":
            teardown_instance()
        elif args.command == "status":
            show_status()
        elif args.command == "tunnel":
            setup_tunnel(args.port)
        elif args.command == "ssh":
            connect_ssh()
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
