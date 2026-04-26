from __future__ import annotations

import plistlib
from pathlib import Path

from afs.agent_hooks import (
    SHELL_BLOCK_BEGIN,
    install_shell_profile_hooks,
    install_worker_launchd,
    render_launchd_plist,
    render_shell_profile_block,
    shell_hooks_installed,
)


def test_shell_profile_hook_install_is_idempotent(tmp_path: Path) -> None:
    profile = tmp_path / ".zshrc"
    profile.write_text("export PATH=/usr/bin:$PATH\n", encoding="utf-8")
    root = tmp_path / "afs"

    planned = install_shell_profile_hooks(afs_root=root, profile_path=profile)
    assert planned.changed is True
    assert SHELL_BLOCK_BEGIN not in profile.read_text(encoding="utf-8")

    applied = install_shell_profile_hooks(afs_root=root, profile_path=profile, apply=True)
    assert applied.applied is True
    assert shell_hooks_installed(profile) is True

    current = install_shell_profile_hooks(afs_root=root, profile_path=profile, apply=True)
    assert current.changed is False
    assert profile.read_text(encoding="utf-8").count(SHELL_BLOCK_BEGIN) == 1


def test_render_shell_profile_block_sources_expected_scripts(tmp_path: Path) -> None:
    block = render_shell_profile_block(tmp_path / "afs")
    assert "afs-shell-init.sh" in block
    assert "afs-agent-hooks.sh" in block


def test_render_launchd_plist_runs_agent_job_worker(tmp_path: Path) -> None:
    payload = plistlib.loads(
        render_launchd_plist(
            afs_root=tmp_path / "afs",
            context_path=tmp_path / "repo",
            agent_name="worker",
            command="echo hi",
            poll_seconds=12,
        )
    )

    assert payload["Label"] == "com.afs.agent-jobs"
    assert payload["RunAtLoad"] is True
    assert payload["KeepAlive"] is True
    args = payload["ProgramArguments"]
    assert args[1] == "agent-jobs"
    assert "work" in args
    assert "--loop" in args
    assert "echo hi" in args
    assert "12" in args


def test_install_worker_launchd_dry_run_does_not_write(tmp_path: Path) -> None:
    target = tmp_path / "com.afs.agent-jobs.plist"
    result = install_worker_launchd(
        afs_root=tmp_path / "afs",
        context_path=tmp_path / "repo",
        plist_path=target,
    )

    assert result.changed is True
    assert result.applied is False
    assert not target.exists()
