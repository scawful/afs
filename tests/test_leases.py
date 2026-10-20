from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest

from afs import leases


def _dead_pid() -> int:
    pid = 999_999
    while leases._pid_alive(pid):
        pid -= 1
    return pid


def test_lease_path_maps_names_to_tmp() -> None:
    assert leases.lease_path("oos-heavy") == Path("/tmp/oos-heavy.lock")
    assert leases.lease_path("/var/run/x.lock") == Path("/var/run/x.lock")
    with pytest.raises(ValueError):
        leases.lease_path("bad name")


def test_acquire_writes_compatible_metadata_and_release_requires_token(tmp_path: Path) -> None:
    path = tmp_path / "heavy.lock"
    meta = leases.acquire(path, "codex-lyra", task="build", wait_seconds=0)

    assert (path / "owner").read_text(encoding="utf-8") == "codex-lyra\n"
    on_disk = json.loads((path / "metadata.json").read_text(encoding="utf-8"))
    assert on_disk == meta
    assert {"schema", "token", "pid", "owner", "started_at"} <= set(on_disk)
    assert on_disk["schema"] == 1 and on_disk["pid"] == os.getpid()
    assert leases.inspect(path)["status"] == "held"

    assert leases.release(path, token="0" * 32)["released"] is False
    assert leases.release(path, owner="someone-else")["released"] is False
    assert leases.release(path, token=meta["token"])["released"] is True
    assert leases.inspect(path)["status"] == "free"


def test_acquire_waits_then_raises_busy(tmp_path: Path) -> None:
    path = tmp_path / "heavy.lock"
    leases.acquire(path, "first", wait_seconds=0)
    seen: list[str] = []
    with pytest.raises(leases.LeaseBusy) as excinfo:
        leases.acquire(
            path,
            "second",
            wait_seconds=0.2,
            poll_seconds=0.1,
            on_wait=lambda state: seen.append(state["status"]),
            sleep=lambda _seconds: time.sleep(0.11),
        )
    assert excinfo.value.state["owner"] == "first"
    assert seen and seen[0] == "held"


def test_dead_owner_is_stale_and_reclaimable(tmp_path: Path) -> None:
    path = tmp_path / "heavy.lock"
    leases.acquire(path, "gone", pid=_dead_pid(), wait_seconds=0)
    assert leases.inspect(path)["status"] == "stale"

    dry = leases.reclaim(path)
    assert dry["reclaimed"] is False and "reason" not in dry
    assert path.exists()
    assert leases.reclaim(path, apply=True)["reclaimed"] is True
    assert not path.exists()


def test_unexpired_ttl_keeps_dead_owner_held(tmp_path: Path) -> None:
    path = tmp_path / "heavy.lock"
    leases.acquire(path, "batches", pid=_dead_pid(), ttl_minutes=30, wait_seconds=0)
    state = leases.inspect(path)
    assert state["status"] == "held" and "held_until" in state
    assert leases.reclaim(path, apply=True)["reason"] == "lease is held"


def test_workers_under_source_keep_lease_busy(tmp_path: Path, monkeypatch) -> None:
    path = tmp_path / "heavy.lock"
    source = tmp_path / "tree"
    source.mkdir()
    leases.acquire(path, "build", pid=_dead_pid(), source=str(source), wait_seconds=0)
    monkeypatch.setattr(leases, "_processes_under", lambda _source: [4242])
    monkeypatch.setattr(leases, "_command_names", lambda pids: {4242: "ninja"})
    state = leases.inspect(path)
    assert state["status"] == "owner-exited"
    assert state["workers"][0]["command"] == "ninja"
    assert leases.reclaim(path, apply=True)["reclaimed"] is False


def test_legacy_lock_is_never_reclaimed_or_released(tmp_path: Path) -> None:
    path = tmp_path / "heavy.lock"
    path.mkdir()
    (path / "owner").write_text("claude-rails task 2026-10-18T19:46Z\n", encoding="utf-8")
    assert leases.inspect(path)["status"] == "legacy"
    assert leases.reclaim(path, apply=True)["reclaimed"] is False
    assert leases.release(path, owner="claude-rails")["released"] is False
    assert path.exists()


def test_run_holds_lease_for_command_and_releases(tmp_path: Path) -> None:
    path = tmp_path / "heavy.lock"
    probe = tmp_path / "probe.json"
    code = (
        "import json,sys,pathlib;"
        f"p=pathlib.Path({str(path)!r});"
        f"pathlib.Path({str(probe)!r}).write_text((p/'metadata.json').read_text());"
        "sys.exit(3)"
    )
    returncode = leases.run(path, "runner", [sys.executable, "-c", code], wait_seconds=0)
    assert returncode == 3
    seen = json.loads(probe.read_text(encoding="utf-8"))
    assert seen["owner"] == "runner" and isinstance(seen["child_pgid"], int)
    assert not path.exists()
