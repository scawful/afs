"""Profile CLI commands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from ..config import load_config_model
from ..profiles import resolve_active_profile
from ..schema import AFSConfig
from ..workspace_sync import resolve_config_output
from ._utils import write_config


def _load_writable_config(path: Path) -> AFSConfig:
    if path.exists():
        return load_config_model(config_path=path, merge_user=False)
    return AFSConfig()


def profile_current_command(args: argparse.Namespace) -> int:
    """Show current profile resolution."""
    config_path = Path(args.config).expanduser().resolve() if args.config else None
    config = load_config_model(config_path=config_path, merge_user=True)
    resolved = resolve_active_profile(config)

    payload = {
        "resolved_profile": resolved.name,
        "configured_active_profile": config.profiles.active_profile,
        "available_profiles": sorted(config.profiles.profiles.keys()),
        "enabled_extensions": resolved.enabled_extensions,
        "policies": resolved.policies,
    }

    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    available = ", ".join(payload["available_profiles"]) or "(none)"
    extensions = ", ".join(payload["enabled_extensions"]) or "(none)"
    policies = ", ".join(payload["policies"]) or "(none)"
    print(f"current: {payload['resolved_profile']}")
    print(f"configured: {payload['configured_active_profile']}")
    print(f"available: {available}")
    print(f"extensions: {extensions}")
    print(f"policies: {policies}")
    return 0


def profile_list_command(args: argparse.Namespace) -> int:
    """List configured profiles."""
    config_path = Path(args.config).expanduser().resolve() if args.config else None
    config = load_config_model(config_path=config_path, merge_user=True)
    resolved = resolve_active_profile(config)

    names = sorted(config.profiles.profiles.keys())
    if not names:
        print("(no profiles)")
        return 0

    for name in names:
        marker = "*" if name == resolved.name else " "
        print(f"{marker} {name}")
    return 0


def profile_switch_command(args: argparse.Namespace) -> int:
    """Persist active profile selection."""
    config_path = Path(args.config).expanduser().resolve() if args.config else None
    output = resolve_config_output(config_path)
    config = _load_writable_config(output)

    available = config.profiles.profiles
    if args.name not in available:
        configured = ", ".join(sorted(available.keys())) or "(none)"
        print(f"unknown profile: {args.name}")
        print(f"configured profiles: {configured}")
        return 1

    config.profiles.active_profile = args.name
    output.parent.mkdir(parents=True, exist_ok=True)
    write_config(output, config)

    payload = {
        "active_profile": args.name,
        "config_path": str(output),
    }
    if args.json:
        print(json.dumps(payload, indent=2))
        return 0

    print(f"active_profile: {args.name}")
    print(f"config_path: {output}")
    return 0


def register_parsers(subparsers: argparse._SubParsersAction) -> None:
    """Register profile parser tree."""
    parser = subparsers.add_parser("profile", help="Inspect and switch active profile.")
    sub = parser.add_subparsers(dest="profile_command")

    current = sub.add_parser("current", help="Show current active profile.")
    current.add_argument("--config", help="Config path.")
    current.add_argument("--json", action="store_true", help="Output JSON.")
    current.set_defaults(func=profile_current_command)

    listing = sub.add_parser("list", help="List available profiles.")
    listing.add_argument("--config", help="Config path.")
    listing.set_defaults(func=profile_list_command)

    switch = sub.add_parser("switch", help="Switch active profile.")
    switch.add_argument("name", help="Profile name.")
    switch.add_argument("--config", help="Config path.")
    switch.add_argument("--json", action="store_true", help="Output JSON.")
    switch.set_defaults(func=profile_switch_command)

    parser.set_defaults(func=lambda _args, p=parser: p.print_help())
