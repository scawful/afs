"""MCP CLI commands."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from ..mcp_server import MCP_TOOL_CATALOG_ENV, serve


def mcp_serve_command(args: argparse.Namespace) -> int:
    config_path = Path(args.config).expanduser().resolve() if args.config else None
    if args.tool_catalog:
        os.environ[MCP_TOOL_CATALOG_ENV] = args.tool_catalog
    return serve(config_path=config_path)


def register_parsers(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("mcp", help="MCP server operations.")
    sub = parser.add_subparsers(dest="mcp_command")

    serve_parser = sub.add_parser("serve", help="Run the AFS MCP stdio server.")
    serve_parser.add_argument("--config", help="Config path.")
    serve_parser.add_argument(
        "--tool-catalog",
        choices=("slim", "full"),
        help=(
            "Tool catalog exposed by tools/list. Defaults to slim; use full for "
            "legacy/debug clients that need every registered tool."
        ),
    )
    serve_parser.set_defaults(func=mcp_serve_command)
