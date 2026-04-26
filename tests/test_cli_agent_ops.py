from __future__ import annotations

import argparse

from afs.cli import build_parser
from afs.cli.agent_ops import hooks_status_command, register_parsers


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

    status = parser.parse_args(["agent-jobs", "status", "--strict", "--stale-after", "60"])
    assert status.command == "agent-jobs"
    assert status.strict is True
    assert status.stale_after == 60
    assert hasattr(status, "func")

    inbox = parser.parse_args(["agent-jobs", "inbox", "--strict", "--stale-after", "60"])
    assert inbox.command == "agent-jobs"
    assert inbox.strict is True
    assert inbox.stale_after == 60
    assert hasattr(inbox, "func")

    review = parser.parse_args(["agent-jobs", "review", "job-1"])
    assert review.command == "agent-jobs"
    assert review.job_id == "job-1"
    assert hasattr(review, "func")

    archive = parser.parse_args(["agent-jobs", "archive", "job-1"])
    assert archive.command == "agent-jobs"
    assert archive.job_id == "job-1"
    assert hasattr(archive, "func")

    promote = parser.parse_args(["agent-jobs", "promote", "job-1", "--to-handoff", "--archive"])
    assert promote.command == "agent-jobs"
    assert promote.to_handoff is True
    assert promote.archive is True
    assert hasattr(promote, "func")

    seed = parser.parse_args(["agent-jobs", "seed", "--profile", "repo-maintenance", "--dry-run"])
    assert seed.command == "agent-jobs"
    assert seed.profile == "repo-maintenance"
    assert seed.dry_run is True
    assert hasattr(seed, "func")

    hooks_status = parser.parse_args(["agent-hooks", "status", "--path", "/tmp/repo"])
    assert hooks_status.command == "agent-hooks"
    assert hooks_status.path == "/tmp/repo"
    assert hasattr(hooks_status, "func")


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


def test_agent_hooks_status_prints_next_commands(tmp_path, capsys) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    profile = tmp_path / ".zshrc"

    rc = hooks_status_command(
        argparse.Namespace(
            profile=str(profile),
            path=str(workspace),
            label="com.afs.test-missing-worker",
            json=False,
        )
    )

    assert rc == 0
    output = capsys.readouterr().out
    assert "next_commands:" in output
    assert "afs agent-hooks install-shell --apply" in output
    assert f"afs agent-hooks install-worker --path {workspace.resolve()} --apply --load" in output
    assert f"afs agent-jobs status --path {workspace.resolve()}" in output
    assert f"afs agent-jobs inbox --path {workspace.resolve()}" in output
