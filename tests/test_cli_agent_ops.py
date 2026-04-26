from __future__ import annotations

import argparse

from afs.cli import build_parser
from afs.cli.agent_ops import register_parsers


def test_agent_ops_parsers_register() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command")
    register_parsers(subparsers)

    manifest = parser.parse_args(["agent-manifest", "validate"])
    assert manifest.command == "agent-manifest"
    assert hasattr(manifest, "func")

    run = parser.parse_args(["agent-runs", "start", "Fix issue"])
    assert run.command == "agent-runs"
    assert hasattr(run, "func")

    finish = parser.parse_args(["agent-runs", "finish", "run-1", "--command", "pytest"])
    assert finish.command == "agent-runs"
    assert finish.ran_command == ["pytest"]
    assert hasattr(finish, "func")

    job = parser.parse_args(["agent-jobs", "create", "Review docs", "--prompt", "scan", "--allow-destructive"])
    assert job.command == "agent-jobs"
    assert job.allow_destructive is True
    assert hasattr(job, "func")

    work = parser.parse_args(["agent-jobs", "work", "--agent", "worker", "--dry-run", "--allow-destructive"])
    assert work.command == "agent-jobs"
    assert work.agent == "worker"
    assert work.job_command is None
    assert work.dry_run is True
    assert work.allow_destructive is True
    assert hasattr(work, "func")


def test_build_parser_includes_agent_ops_commands() -> None:
    parser = build_parser()
    args = parser.parse_args(["agent-manifest", "export", "codex"])
    assert args.command == "agent-manifest"
    assert hasattr(args, "func")

    sync = parser.parse_args(["agent-manifest", "sync", "--apply", "--harness", "claude"])
    assert sync.command == "agent-manifest"
    assert sync.apply is True
    assert sync.harness == ["claude"]

    hooks = parser.parse_args(["agent-hooks", "install-shell", "--apply"])
    assert hooks.command == "agent-hooks"
    assert hooks.apply is True
    assert hasattr(hooks, "func")

    worker = parser.parse_args(["agent-hooks", "install-worker", "--apply", "--load"])
    assert worker.command == "agent-hooks"
    assert worker.apply is True
    assert worker.load is True
