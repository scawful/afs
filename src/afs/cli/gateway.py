"""Gateway CLI commands for AFS."""

from __future__ import annotations

import argparse
import asyncio
import sys


def register_parsers(subparsers: argparse._SubParsersAction) -> None:
    """Register gateway-related subparsers."""

    # gateway command group
    gateway_parser = subparsers.add_parser(
        "gateway",
        help="AFS Gateway API for chat interfaces",
    )
    gateway_subs = gateway_parser.add_subparsers(dest="gateway_command")

    # gateway serve - run the API server
    serve = gateway_subs.add_parser("serve", help="Run the gateway API server")
    serve.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    serve.add_argument("--port", type=int, default=8000, help="Port to bind to")
    serve.add_argument("--reload", action="store_true", help="Enable auto-reload")
    serve.set_defaults(func=cmd_serve)

    # gateway health - check backend health
    health = gateway_subs.add_parser("health", help="Check backend health")
    health.set_defaults(func=cmd_health)

    # gateway backends - list/manage backends
    backends = gateway_subs.add_parser("backends", help="List available backends")
    backends.add_argument("--activate", metavar="NAME", help="Activate a backend")
    backends.set_defaults(func=cmd_backends)

    # gateway chat - quick chat test
    chat = gateway_subs.add_parser("chat", help="Quick chat test")
    chat.add_argument("message", nargs="?", help="Message to send")
    chat.add_argument("--model", "-m", default="din", help="Model/persona to use")
    chat.add_argument("--stream", "-s", action="store_true", help="Stream response")
    chat.set_defaults(func=cmd_chat)

    # gateway docker - docker compose shortcuts
    docker = gateway_subs.add_parser("docker", help="Docker compose management")
    docker.add_argument("action", choices=["up", "down", "logs", "simple-up"],
                       help="Docker action")
    docker.set_defaults(func=cmd_docker)

    # vastai command group
    vastai_parser = subparsers.add_parser(
        "vastai",
        help="vast.ai GPU instance management",
    )
    vastai_subs = vastai_parser.add_subparsers(dest="vastai_command")

    # vastai up
    up = vastai_subs.add_parser("up", help="Provision GPU instance")
    up.add_argument("--gpu", default="RTX_4090", help="GPU type")
    up.add_argument("--disk", type=int, default=50, help="Disk size in GB")
    up.set_defaults(func=cmd_vastai_up)

    # vastai down
    down = vastai_subs.add_parser("down", help="Teardown instance")
    down.set_defaults(func=cmd_vastai_down)

    # vastai status
    status = vastai_subs.add_parser("status", help="Show instance status")
    status.set_defaults(func=cmd_vastai_status)

    # vastai tunnel
    tunnel = vastai_subs.add_parser("tunnel", help="Set up SSH tunnel")
    tunnel.add_argument("--port", type=int, default=11436, help="Local port")
    tunnel.set_defaults(func=cmd_vastai_tunnel)


def cmd_serve(args: argparse.Namespace) -> int:
    """Run the gateway server."""
    from afs.gateway.server import run_server
    run_server(host=args.host, port=args.port, reload=args.reload)
    return 0


def cmd_health(args: argparse.Namespace) -> int:
    """Check backend health."""
    from afs.gateway.backends import BackendManager

    async def check():
        async with BackendManager() as manager:
            await manager.check_all()

            print("Backend Status:")
            print("-" * 50)
            for name, status in manager.status.items():
                symbol = "✓" if status.healthy else "✗"
                print(f"  {symbol} {name}: {'healthy' if status.healthy else status.error}")
                if status.models:
                    print(f"      Models: {', '.join(status.models[:5])}")

            print("-" * 50)
            if manager.active:
                print(f"Active: {manager.active.name}")
            else:
                print("Active: none")

    asyncio.run(check())
    return 0


def cmd_backends(args: argparse.Namespace) -> int:
    """List/manage backends."""
    from afs.gateway.backends import BackendManager

    async def manage():
        async with BackendManager() as manager:
            if args.activate:
                if manager.set_active(args.activate):
                    print(f"Activated: {args.activate}")
                else:
                    print(f"Failed to activate: {args.activate}")
                    return 1
            else:
                print("Backends:")
                for b in manager.backends:
                    status = manager.status.get(b.name)
                    healthy = status.healthy if status else False
                    active = "(*)" if b.name == manager._active_backend else ""
                    print(f"  {b.name}{active}: {b.type.value} @ {b.base_url} "
                          f"[{'healthy' if healthy else 'unhealthy'}]")
        return 0

    return asyncio.run(manage())


def cmd_chat(args: argparse.Namespace) -> int:
    """Quick chat test."""
    from afs.gateway.backends import BackendManager
    from afs.gateway.server import PERSONAS

    message = args.message
    if not message:
        print("Enter message (Ctrl+D to send):")
        message = sys.stdin.read().strip()
        if not message:
            print("No message provided")
            return 1

    async def chat():
        async with BackendManager() as manager:
            if not manager.active:
                print("No backend available")
                return 1

            # Get persona
            persona = PERSONAS.get(args.model, PERSONAS["din"])
            model_id = {
                "din": "din-v2:latest",
                "nayru": "nayru-v5:latest",
                "farore": "farore-v1:latest",
                "veran": "qwen2.5-coder:7b",
                "scribe": "qwen2.5-coder:7b",
            }.get(args.model, args.model)

            messages = [
                {"role": "system", "content": persona["system_prompt"]},
                {"role": "user", "content": message},
            ]

            print(f"\n[{args.model}]:")

            if args.stream:
                async for token in await manager.chat(
                    model=model_id,
                    messages=messages,
                    stream=True,
                ):
                    print(token, end="", flush=True)
                print()
            else:
                result = await manager.chat(
                    model=model_id,
                    messages=messages,
                )
                print(result.get("message", {}).get("content", ""))

        return 0

    return asyncio.run(chat())


def cmd_docker(args: argparse.Namespace) -> int:
    """Docker compose management."""
    import subprocess
    from pathlib import Path

    docker_dir = Path(__file__).parents[3] / "docker"

    if args.action == "simple-up":
        compose_file = docker_dir / "docker-compose.simple.yml"
    else:
        compose_file = docker_dir / "docker-compose.yml"

    if args.action in ("up", "simple-up"):
        cmd = ["docker", "compose", "-f", str(compose_file), "up", "-d"]
    elif args.action == "down":
        cmd = ["docker", "compose", "-f", str(compose_file), "down"]
    elif args.action == "logs":
        cmd = ["docker", "compose", "-f", str(compose_file), "logs", "-f"]
    else:
        print(f"Unknown action: {args.action}")
        return 1

    return subprocess.run(cmd).returncode


def cmd_vastai_up(args: argparse.Namespace) -> int:
    """Provision vast.ai instance."""
    from pathlib import Path
    import subprocess

    script = Path(__file__).parents[3] / "scripts" / "vastai_provision.py"
    return subprocess.run([
        sys.executable, str(script), "up",
        "--gpu", args.gpu,
        "--disk", str(args.disk),
    ]).returncode


def cmd_vastai_down(args: argparse.Namespace) -> int:
    """Teardown vast.ai instance."""
    from pathlib import Path
    import subprocess

    script = Path(__file__).parents[3] / "scripts" / "vastai_provision.py"
    return subprocess.run([sys.executable, str(script), "down"]).returncode


def cmd_vastai_status(args: argparse.Namespace) -> int:
    """Show vast.ai status."""
    from pathlib import Path
    import subprocess

    script = Path(__file__).parents[3] / "scripts" / "vastai_provision.py"
    return subprocess.run([sys.executable, str(script), "status"]).returncode


def cmd_vastai_tunnel(args: argparse.Namespace) -> int:
    """Set up SSH tunnel."""
    from pathlib import Path
    import subprocess

    script = Path(__file__).parents[3] / "scripts" / "vastai_provision.py"
    return subprocess.run([
        sys.executable, str(script), "tunnel",
        "--port", str(args.port),
    ]).returncode
