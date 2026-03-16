"""CLI commands for profile bundle management."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def pack_command(args: argparse.Namespace) -> int:
    """Pack a profile into a bundle."""
    from ..bundler import pack_bundle

    output = Path(args.output).expanduser().resolve() if args.output else Path.cwd()
    try:
        result = pack_bundle(args.profile, output_path=output)
    except Exception as exc:
        print(f"error: {exc}")
        return 1

    print(f"packed: {result.path}")
    print(f"  files: {result.file_count}")
    print(f"  size:  {result.size_bytes} bytes")
    return 0


def install_command(args: argparse.Namespace) -> int:
    """Install a bundle as an extension."""
    from ..bundler import install_bundle

    bundle_path = Path(args.path).expanduser().resolve()
    name = args.name if args.name else None
    try:
        result = install_bundle(bundle_path, name_override=name)
    except FileNotFoundError as exc:
        print(f"error: {exc}")
        return 1

    print(f"installed: {result.extension_path}")
    print(f"  profile: {result.profile_name}")
    return 0


def inspect_command(args: argparse.Namespace) -> int:
    """Inspect a bundle."""
    from ..bundler import inspect_bundle

    bundle_path = Path(args.path).expanduser().resolve()
    try:
        result = inspect_bundle(bundle_path)
    except FileNotFoundError as exc:
        print(f"error: {exc}")
        return 1

    if args.json:
        payload = {
            "name": result.manifest.name,
            "version": result.manifest.version,
            "description": result.manifest.description,
            "author": result.manifest.author,
            "resource_counts": result.resource_counts,
        }
        print(json.dumps(payload, indent=2))
        return 0

    print(f"name:        {result.manifest.name}")
    print(f"version:     {result.manifest.version}")
    print(f"description: {result.manifest.description}")
    if result.manifest.author:
        print(f"author:      {result.manifest.author}")
    if result.resource_counts:
        print("resources:")
        for dir_name, count in sorted(result.resource_counts.items()):
            print(f"  {dir_name}: {count} files")
    return 0


def list_command(args: argparse.Namespace) -> int:
    """List installed bundles."""
    from ..bundler import list_bundles

    bundles = list_bundles()
    if not bundles:
        print("no bundles installed")
        return 0

    for bundle in bundles:
        desc = f"\t{bundle['description']}" if bundle.get("description") else ""
        print(f"{bundle['name']}\t{bundle['version']}{desc}")
    return 0


def register_parsers(subparsers: argparse._SubParsersAction) -> None:
    """Register bundle CLI commands."""
    bundle_parser = subparsers.add_parser("bundle", help="Profile bundle management.")
    bundle_sub = bundle_parser.add_subparsers(dest="bundle_command")

    pack_p = bundle_sub.add_parser("pack", help="Pack a profile into a bundle.")
    pack_p.add_argument("profile", help="Profile name to pack.")
    pack_p.add_argument("--output", help="Output directory.")
    pack_p.set_defaults(func=pack_command)

    install_p = bundle_sub.add_parser("install", help="Install a bundle.")
    install_p.add_argument("path", help="Path to bundle directory.")
    install_p.add_argument("--name", help="Override bundle name.")
    install_p.set_defaults(func=install_command)

    inspect_p = bundle_sub.add_parser("inspect", help="Inspect a bundle.")
    inspect_p.add_argument("path", help="Path to bundle directory.")
    inspect_p.add_argument("--json", action="store_true", help="Output JSON.")
    inspect_p.set_defaults(func=inspect_command)

    list_p = bundle_sub.add_parser("list", help="List installed bundles.")
    list_p.set_defaults(func=list_command)
