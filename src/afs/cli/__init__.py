"""AFS command-line interface package.

This package provides a modular CLI structure for AFS commands.
Commands are organized into logical groups:
- core: init, plugins, status, services, agents, orchestrator, studio
- context: context management, graph, workspace
- training: training data, discriminator, tokenizer, encoder
- pipeline: pipeline, evaluation, scoring, entity, active learning, generator
"""

from __future__ import annotations

import argparse
from typing import Iterable

from . import core
from . import context


def build_parser() -> argparse.ArgumentParser:
    """Build the main argument parser."""
    parser = argparse.ArgumentParser(prog="afs")
    subparsers = parser.add_subparsers(dest="command")

    # Register core commands (init, plugins, status, services, agents, orchestrator, studio)
    core.register_parsers(subparsers)

    # Register context commands (context, graph, workspace)
    context.register_parsers(subparsers)

    # Import remaining commands from legacy cli module
    # These will be migrated to separate modules over time
    from .._cli_legacy import register_remaining_parsers
    register_remaining_parsers(subparsers)

    return parser


def main(argv: Iterable[str] | None = None) -> int:
    """Main CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not getattr(args, "command", None):
        parser.print_help()
        return 1

    # Check for subcommands that require a subcommand
    subcommand_checks = [
        ("workspace", "workspace_command"),
        ("context", "context_command"),
        ("graph", "graph_command"),
        ("services", "services_command"),
        ("agents", "agents_command"),
        ("orchestrator", "orchestrator_command"),
        ("studio", "studio_command"),
        ("generators", "generators_command"),
        ("training", "training_command"),
        ("discriminator", "discriminator_command"),
        ("tokenizer", "tokenizer_command"),
        ("encoder", "encoder_command"),
        ("entity", "entity_command"),
        ("scoring", "scoring_command"),
        ("pipeline", "pipeline_command"),
        ("evaluation", "evaluation_command"),
        ("active-learning", "active_learning_command"),
        ("generator", "generator_command"),
    ]

    for cmd, subcmd_attr in subcommand_checks:
        if args.command == cmd and not getattr(args, subcmd_attr, None):
            parser.print_help()
            return 1

    return args.func(args)


__all__ = ["build_parser", "main"]
