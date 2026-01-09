"""Example AFS plugin."""

from __future__ import annotations

import argparse


def register_cli(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("hello", help="Hello from AFS plugin.")
    parser.set_defaults(func=_hello_command)

def _hello_command(_args: argparse.Namespace) -> int:
    print("hello from afs_plugin_hello")
    return 0
