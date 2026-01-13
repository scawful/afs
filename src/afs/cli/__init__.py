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
import sys
from typing import Iterable

from . import core
from . import context
from . import training
from . import generators
from . import tokenizer
from . import encoder
from . import entity
from . import pipeline
from . import active_learning
from . import generator
from . import gateway
from . import benchmark
from . import distillation
from . import fs
from . import embeddings
from . import review
from ..history import log_cli_invocation
from ._help import render_default_help, render_topic_help


def build_parser() -> argparse.ArgumentParser:
    """Build the main argument parser."""
    parser = argparse.ArgumentParser(prog="afs")
    subparsers = parser.add_subparsers(dest="command")

    # Register core commands (init, plugins, status, services, agents, orchestrator, studio)
    core.register_parsers(subparsers)

    # Register context commands (context, graph, workspace)
    context.register_parsers(subparsers)

    # Register training commands (training, discriminator)
    training.register_parsers(subparsers)

    # Register generator commands (generators - data augmentation)
    generators.register_parsers(subparsers)

    # Register tokenizer commands
    tokenizer.register_parsers(subparsers)

    # Register encoder commands
    encoder.register_parsers(subparsers)

    # Register entity commands
    entity.register_parsers(subparsers)

    # Register pipeline commands (scoring, pipeline, evaluation)
    pipeline.register_parsers(subparsers)

    # Register active learning commands
    active_learning.register_parsers(subparsers)

    # Register generator commands (model-based generation)
    generator.register_parsers(subparsers)

    # Register gateway commands (API server, backends, vast.ai)
    gateway.register_parsers(subparsers)

    # Register benchmark commands
    benchmark.register_parsers(subparsers)

    # Register distillation commands
    distillation.register_parsers(subparsers)

    # Register filesystem commands
    fs.register_parsers(subparsers)

    # Register embedding commands
    embeddings.register_parsers(subparsers)

    # Register review commands
    review.register_parsers(subparsers)

    # help
    help_parser = subparsers.add_parser("help", help="Show help for commands.")
    help_parser.add_argument(
        "topic",
        nargs=argparse.REMAINDER,
        help="Command path to show help for (e.g., context init).",
    )
    help_parser.set_defaults(
        func=lambda args, root_parser=parser: render_topic_help(
            root_parser, args.topic
        )
    )

    # Allow plugins to extend the CLI surface.
    try:
        from ..plugins import call_plugin_hook, load_enabled_plugins

        plugins = load_enabled_plugins()
        call_plugin_hook("register_cli", subparsers, plugins=plugins.values())
        call_plugin_hook("register_parsers", subparsers, plugins=plugins.values())
    except Exception:
        pass

    return parser


def main(argv: Iterable[str] | None = None) -> int:
    """Main CLI entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if not getattr(args, "command", None):
        render_default_help(parser)
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
        ("gateway", "gateway_command"),
        ("vastai", "vastai_command"),
        ("benchmark", "benchmark_command"),
        ("distill", "distill_command"),
        ("fs", "fs_command"),
        ("embeddings", "embeddings_command"),
    ]

    for cmd, subcmd_attr in subcommand_checks:
        if args.command == cmd and not getattr(args, subcmd_attr, None):
            parser.print_help()
            return 1

    exit_code = args.func(args)
    try:
        log_cli_invocation(argv or sys.argv[1:], exit_code)
    except Exception:
        pass
    return exit_code


__all__ = ["build_parser", "main"]
