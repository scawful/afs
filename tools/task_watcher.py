#!/usr/bin/env python3
"""
AFS Task Watcher - Monitors vast.ai instances and local downloads.
Prevents false idle notifications by tracking all active work.

Usage:
    python3 task_watcher.py status          # Show all active tasks
    python3 task_watcher.py downloads       # Show active downloads
    python3 task_watcher.py instances       # Show vast.ai instances
    python3 task_watcher.py watch           # Continuous monitoring
"""

import subprocess
import json
import time
import argparse
from datetime import datetime
from pathlib import Path


def get_active_downloads():
    """Find active scp/rsync downloads."""
    downloads = []

    # Check for scp processes
    try:
        result = subprocess.run(
            ["pgrep", "-lf", "scp.*vast"],
            capture_output=True, text=True
        )
        for line in result.stdout.strip().split('\n'):
            if line:
                downloads.append({"type": "scp", "process": line})
    except Exception:
        pass

    # Check for rsync processes
    try:
        result = subprocess.run(
            ["pgrep", "-lf", "rsync.*vast"],
            capture_output=True, text=True
        )
        for line in result.stdout.strip().split('\n'):
            if line:
                downloads.append({"type": "rsync", "process": line})
    except Exception:
        pass

    # Check for active downloads by looking at recently modified large files
    gguf_dir = Path.home() / "src/lab/afs/models/gguf"
    if gguf_dir.exists():
        for f in gguf_dir.glob("*.gguf"):
            try:
                stat = f.stat()
                # If file was modified in last 60 seconds and is growing
                if time.time() - stat.st_mtime < 60:
                    size_gb = stat.st_size / (1024**3)
                    downloads.append({
                        "type": "gguf_download",
                        "file": f.name,
                        "size_gb": f"{size_gb:.2f}",
                        "status": "active" if size_gb < 7.5 else "complete"
                    })
            except Exception:
                pass

    return downloads


def get_vast_instances():
    """Get vast.ai instance status."""
    instances = []

    try:
        result = subprocess.run(
            ["vastai", "show", "instances", "--raw"],
            capture_output=True, text=True, timeout=30
        )
        if result.returncode == 0:
            data = json.loads(result.stdout)
            for inst in data:
                instances.append({
                    "id": inst.get("id"),
                    "label": inst.get("label", "unlabeled"),
                    "status": inst.get("actual_status", inst.get("cur_state")),
                    "gpu": inst.get("gpu_name", inst.get("model", "unknown")),
                    "gpu_util": inst.get("gpu_util", 0),
                    "ssh": f"{inst.get('ssh_host', 'unknown')}:{inst.get('ssh_port', '?')}",
                    "cost_hr": inst.get("dph_total", 0),
                })
    except subprocess.TimeoutExpired:
        pass
    except json.JSONDecodeError:
        pass
    except FileNotFoundError:
        pass

    return instances


def get_ssh_connections():
    """Check active SSH connections to vast.ai."""
    connections = []

    try:
        result = subprocess.run(
            ["pgrep", "-lf", "ssh.*vast"],
            capture_output=True, text=True
        )
        for line in result.stdout.strip().split('\n'):
            if line and 'ssh' in line:
                connections.append(line)
    except Exception:
        pass

    return connections


def print_status():
    """Print comprehensive status."""
    print(f"\n{'='*60}")
    print(f"AFS Task Watcher - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    # Downloads
    downloads = get_active_downloads()
    print("ACTIVE DOWNLOADS:")
    if downloads:
        for d in downloads:
            if d["type"] == "gguf_download":
                print(f"  [{d['status']}] {d['file']} - {d['size_gb']} GB")
            else:
                print(f"  [{d['type']}] {d['process'][:60]}...")
    else:
        print("  None")

    # SSH connections
    ssh_conns = get_ssh_connections()
    print(f"\nACTIVE SSH CONNECTIONS: {len(ssh_conns)}")
    for conn in ssh_conns[:5]:
        print(f"  {conn[:70]}...")

    # Vast instances
    instances = get_vast_instances()
    print(f"\nVAST.AI INSTANCES:")
    if instances:
        for inst in instances:
            status_icon = "🟢" if inst["status"] == "running" else "🔴"
            gpu_info = f"GPU: {inst['gpu_util']}%" if inst["gpu_util"] else "GPU: idle"
            print(f"  {status_icon} [{inst['label']}] {inst['status']} - {gpu_info} - ${inst['cost_hr']:.2f}/hr")
    else:
        print("  None or vastai CLI not available")

    # Summary
    active_work = len(downloads) + len(ssh_conns)
    print(f"\n{'='*60}")
    if active_work > 0:
        print(f"STATUS: BUSY ({active_work} active tasks)")
        print("Instances should NOT be terminated.")
    else:
        running = [i for i in instances if i["status"] == "running"]
        idle = [i for i in running if i["gpu_util"] == 0]
        if idle:
            print(f"STATUS: IDLE ({len(idle)} instances with 0% GPU)")
            print("Consider terminating idle instances to save costs.")
        else:
            print("STATUS: OK")
    print(f"{'='*60}\n")


def watch_loop(interval=30):
    """Continuous monitoring."""
    print(f"Starting watch mode (refresh every {interval}s). Ctrl+C to stop.\n")
    try:
        while True:
            subprocess.run(["clear"])
            print_status()
            time.sleep(interval)
    except KeyboardInterrupt:
        print("\nStopped.")


def main():
    parser = argparse.ArgumentParser(description="AFS Task Watcher")
    parser.add_argument(
        "command",
        choices=["status", "downloads", "instances", "watch"],
        default="status",
        nargs="?",
        help="Command to run"
    )
    parser.add_argument(
        "--interval", "-i",
        type=int,
        default=30,
        help="Watch interval in seconds"
    )

    args = parser.parse_args()

    if args.command == "status":
        print_status()
    elif args.command == "downloads":
        downloads = get_active_downloads()
        print(json.dumps(downloads, indent=2))
    elif args.command == "instances":
        instances = get_vast_instances()
        print(json.dumps(instances, indent=2))
    elif args.command == "watch":
        watch_loop(args.interval)


if __name__ == "__main__":
    main()
