"""Example CLI registration for an AFS extension."""

from __future__ import annotations

from . import greeting


def register_cli(subparsers) -> None:
    """Register a tiny extension command if the host CLI exposes subparsers."""
    parser = subparsers.add_parser("hello-extension", help="Hello extension example")
    parser.set_defaults(func=_run)


def _run(_args) -> int:
    print(greeting())
    return 0
