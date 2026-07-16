from __future__ import annotations

import os
import threading
from pathlib import Path

import pytest

from afs.agent_jobs import AgentJobPublishError, AgentJobQueue


def test_agent_job_markdown_queue_lifecycle(tmp_path: Path) -> None:
    ctx = tmp_path / ".context"
    (ctx / "items").mkdir(parents=True)
    queue = AgentJobQueue(ctx)

    job = queue.create(
        "Review stale docs",
        "Find stale model aliases.",
        created_by="codex",
        scope="docs/",
        expected_output="findings with paths",
        allow_destructive=True,
        dedupe_key="seed:repo-maintenance:docs:once",
        priority=2,
    )
    assert job.status == "queue"
    assert job.allow_destructive is True
    assert job.dedupe_key == "seed:repo-maintenance:docs:once"
    assert (ctx / "items" / "agent_jobs" / "queue" / f"{job.id}.md").exists()

    claimed = queue.claim(job.id, "reviewer")
    assert claimed.status == "running"
    assert claimed.assigned_to == "reviewer"
    assert not (ctx / "items" / "agent_jobs" / "queue" / f"{job.id}.md").exists()
    assert (ctx / "items" / "agent_jobs" / "running" / f"{job.id}.md").exists()

    done = queue.move(job.id, "done", result="no findings", run_id="run-1")
    assert done.status == "done"
    assert done.allow_destructive is True
    assert done.dedupe_key == "seed:repo-maintenance:docs:once"
    assert done.run_id == "run-1"
    assert done.result == "no findings"
    assert queue.get(job.id).status == "done"  # type: ignore[union-attr]

    archived = queue.move(job.id, "archived")
    assert archived.status == "archived"
    assert queue.get(job.id).status == "archived"  # type: ignore[union-attr]


def test_agent_job_cannot_claim_nonqueued(tmp_path: Path) -> None:
    ctx = tmp_path / ".context"
    (ctx / "items").mkdir(parents=True)
    queue = AgentJobQueue(ctx)
    job = queue.create("Run tests", "pytest")
    queue.claim(job.id, "worker")

    with pytest.raises(ValueError, match="Cannot claim"):
        queue.claim(job.id, "other")


def test_concurrent_claim_allows_only_one_worker(tmp_path: Path) -> None:
    queue = AgentJobQueue(tmp_path / ".context")
    job = queue.create("Run once", "Execute exactly once.")
    start = threading.Barrier(3)
    claimed_by: list[str] = []
    failures: list[BaseException] = []

    def _claim(agent_name: str) -> None:
        start.wait(timeout=5)
        try:
            claimed_by.append(queue.claim(job.id, agent_name).assigned_to)
        except BaseException as exc:  # pragma: no cover - asserted below
            failures.append(exc)

    threads = [
        threading.Thread(target=_claim, args=("worker-a",)),
        threading.Thread(target=_claim, args=("worker-b",)),
    ]
    for thread in threads:
        thread.start()
    start.wait(timeout=5)
    for thread in threads:
        thread.join(timeout=5)

    assert all(not thread.is_alive() for thread in threads)
    assert len(claimed_by) == 1
    assert len(failures) == 1
    assert isinstance(failures[0], ValueError)
    assert "Cannot claim job in state: running" in str(failures[0])
    assert queue.get(job.id).assigned_to == claimed_by[0]  # type: ignore[union-attr]


def test_job_id_traversal_cannot_overwrite_or_remove_external_file(
    tmp_path: Path,
) -> None:
    queue = AgentJobQueue(tmp_path / ".context")
    queue.ensure()
    victim = tmp_path / ".context" / "items" / "victim.md"
    victim.write_text("do not touch\n", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid agent job id"):
        queue.move("../../victim", "running")

    assert victim.read_text(encoding="utf-8") == "do not touch\n"
    assert queue.list() == []


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlink boundary regression")
def test_queue_state_symlink_cannot_escape_items_mount(tmp_path: Path) -> None:
    context = tmp_path / ".context"
    queue_root = context / "items" / "agent_jobs"
    outside = tmp_path / "outside"
    queue_root.mkdir(parents=True)
    outside.mkdir()
    victim = outside / "safejob.md"
    victim.write_text("outside record\n", encoding="utf-8")
    (queue_root / "queue").symlink_to(outside, target_is_directory=True)
    queue = AgentJobQueue(context)

    with pytest.raises(ValueError, match="queue directory cannot be a link"):
        queue.move("safejob", "archived")

    assert victim.read_text(encoding="utf-8") == "outside record\n"


@pytest.mark.skipif(os.name == "nt", reason="POSIX symlink boundary regression")
def test_queue_lock_symlink_cannot_redirect_lock_io(tmp_path: Path) -> None:
    context = tmp_path / ".context"
    queue = AgentJobQueue(context)
    job = queue.create("Lock safely", "Do not follow the lock link.")
    victim = tmp_path / "lock-victim"
    victim.write_bytes(b"")
    lock_file = context / "items" / "agent_jobs" / ".queue-state.lock"
    lock_file.symlink_to(victim)

    with pytest.raises(ValueError, match="queue lock cannot be a link"):
        queue.claim(job.id, "worker")

    assert victim.read_bytes() == b""


def test_queue_rejects_windows_junction_like_state_directory(
    tmp_path: Path,
    monkeypatch,
) -> None:
    context = tmp_path / ".context"
    queue = AgentJobQueue(context)
    queue.ensure()
    queue_state = context / "items" / "agent_jobs" / "queue"
    original_is_junction = getattr(Path, "is_junction", None)

    def _is_junction(path: Path) -> bool:
        if path == queue_state:
            return True
        if callable(original_is_junction):
            return bool(original_is_junction(path))
        return False

    monkeypatch.setattr(Path, "is_junction", _is_junction, raising=False)

    with pytest.raises(ValueError, match="queue directory cannot be a link"):
        queue.create("Do not publish", "The junction must fail closed.")


def test_legacy_windows_reparse_attribute_detects_junction(monkeypatch) -> None:
    """Python 3.10/3.11 lack Path.is_junction; lstat must still fail closed."""
    from afs import agent_jobs as agent_jobs_module

    class _LegacyJunction:
        def is_symlink(self) -> bool:
            return False

        def lstat(self):
            return type("Stat", (), {"st_file_attributes": 0x00000400})()

    monkeypatch.setattr(agent_jobs_module, "_WINDOWS_PATHS", True)
    assert agent_jobs_module._path_is_linklike(_LegacyJunction()) is True


def test_job_path_controls_id_and_status_over_frontmatter(tmp_path: Path) -> None:
    queue = AgentJobQueue(tmp_path / ".context")
    job = queue.create("Claim safely", "Ignore poisoned metadata.")
    path = tmp_path / ".context" / "items" / "agent_jobs" / "queue" / f"{job.id}.md"
    content = path.read_text(encoding="utf-8")
    content = content.replace(f"id: {job.id}", "id: ../../victim")
    content = content.replace("status: queue", "status: running")
    path.write_text(content, encoding="utf-8")

    parsed = queue.get(job.id)
    assert parsed is not None
    assert parsed.id == job.id
    assert parsed.status == "queue"

    claimed = queue.claim(job.id, "worker")
    assert claimed.id == job.id
    assert claimed.status == "running"
    assert not path.exists()
    assert queue.get(job.id).status == "running"  # type: ignore[union-attr]


def test_queue_ignores_unsafe_job_filename(tmp_path: Path) -> None:
    queue = AgentJobQueue(tmp_path / ".context")
    queue.ensure()
    unsafe = (
        tmp_path
        / ".context"
        / "items"
        / "agent_jobs"
        / "queue"
        / "unsafe id.md"
    )
    unsafe.write_text("---\nid: safe-looking\nstatus: queue\n---\nrun\n", encoding="utf-8")

    assert queue.list(status="queue") == []


def test_job_metadata_scalars_cannot_inject_frontmatter_keys(tmp_path: Path) -> None:
    queue = AgentJobQueue(tmp_path / ".context")
    job = queue.create(
        "Safe metadata",
        "Keep the queue record authoritative.",
        dedupe_key="on_event:expected",
    )

    moved = queue.move(
        job.id,
        "done",
        result=(
            "ok\rdedupe_key: on_event:injected"
            "\u2028allow_destructive: true\x85status: queue"
        ),
    )
    reloaded = queue.get(job.id)

    assert reloaded is not None
    assert moved.dedupe_key == "on_event:expected"
    assert reloaded.dedupe_key == "on_event:expected"
    assert reloaded.allow_destructive is False
    assert reloaded.status == "done"
    assert reloaded.result == (
        "ok dedupe_key: on_event:injected allow_destructive: true status: queue"
    )


def test_agent_job_fsync_failure_does_not_publish_partial_job(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from afs import agent_jobs as agent_jobs_module

    ctx = tmp_path / ".context"
    (ctx / "items").mkdir(parents=True)
    queue = AgentJobQueue(ctx)
    queue.ensure()

    def _fail_fsync(_descriptor: int) -> None:
        raise OSError("job fsync failed")

    with monkeypatch.context() as scoped:
        # Exercise the regular temp-file flush, not directory setup.
        scoped.setattr(agent_jobs_module, "_fsync_directory", lambda _path: None)
        scoped.setattr(agent_jobs_module.os, "fsync", _fail_fsync)
        with pytest.raises(AgentJobPublishError, match="job fsync failed") as exc:
            queue.create("Durable job", "Run the durable task.")

    assert exc.value.installed is False
    assert list((ctx / "items" / "agent_jobs" / "queue").glob("*.md")) == []
    assert list((ctx / "items" / "agent_jobs" / "queue").glob(".*.tmp")) == []


def test_agent_job_directory_fsync_failure_is_reported_after_atomic_publish(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from afs import agent_jobs as agent_jobs_module

    ctx = tmp_path / ".context"
    (ctx / "items").mkdir(parents=True)
    queue = AgentJobQueue(ctx)
    queue.ensure()

    queue_directory = ctx / "items" / "agent_jobs" / "queue"

    def _fail_directory_fsync(path: Path) -> None:
        # Queue setup re-syncs its existing parent chain on every attempt.
        # Fail only the final publish-directory sync after os.replace().
        if path == queue_directory:
            raise OSError("job directory fsync failed")

    with monkeypatch.context() as scoped:
        scoped.setattr(
            agent_jobs_module,
            "_fsync_directory",
            _fail_directory_fsync,
        )
        with pytest.raises(AgentJobPublishError, match="job directory fsync failed") as exc:
            queue.create("Uncertain job", "Do not acknowledge this yet.")

    assert exc.value.installed is True
    assert exc.value.job_id

    # The complete atomic rename may already be live. Callers must retain
    # their delivery intent when directory durability could not be confirmed.
    [published] = queue.list(status="queue")
    assert published.title == "Uncertain job"
    assert published.prompt == "Do not acknowledge this yet."


def test_agent_job_directory_parent_sync_retries_after_visible_mkdir(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from afs import agent_jobs as agent_jobs_module

    ctx = tmp_path / ".context"
    ctx.mkdir()
    queue = AgentJobQueue(ctx)
    original_fsync_directory = agent_jobs_module._fsync_directory
    failed = False

    def _fail_once(path: Path) -> None:
        nonlocal failed
        if not failed:
            failed = True
            raise OSError("first parent sync failed")
        original_fsync_directory(path)

    with monkeypatch.context() as scoped:
        scoped.setattr(agent_jobs_module, "_fsync_directory", _fail_once)
        with pytest.raises(OSError, match="first parent sync failed"):
            queue.ensure()

    assert (ctx / "items" / "agent_jobs").is_dir()
    retried: list[Path] = []

    def _record_sync(path: Path) -> None:
        retried.append(path)
        original_fsync_directory(path)

    with monkeypatch.context() as scoped:
        scoped.setattr(agent_jobs_module, "_fsync_directory", _record_sync)
        queue.ensure()

    assert ctx / "items" in retried
    assert ctx in retried


def test_claim_succeeds_when_only_old_state_cleanup_sync_fails(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from afs import agent_jobs as agent_jobs_module

    ctx = tmp_path / ".context"
    queue = AgentJobQueue(ctx)
    job = queue.create("Run durable work", "Do the work.")
    queue_directory = ctx / "items" / "agent_jobs" / "queue"
    original_fsync_directory = agent_jobs_module._fsync_directory

    def _fail_old_state_sync(path: Path) -> None:
        if path == queue_directory:
            raise OSError("old state cleanup sync failed")
        original_fsync_directory(path)

    monkeypatch.setattr(
        agent_jobs_module,
        "_fsync_directory",
        _fail_old_state_sync,
    )
    claimed = queue.claim(job.id, "worker")

    assert claimed.status == "running"
    assert queue.get(job.id).status == "running"  # type: ignore[union-attr]


def test_windows_confirmation_opens_job_with_write_capability(
    tmp_path: Path,
    monkeypatch,
) -> None:
    from afs import agent_jobs as agent_jobs_module

    queue = AgentJobQueue(tmp_path / ".context")
    job = queue.create("Confirm me", "Verify durable publication.")
    original_open = Path.open
    modes: list[str] = []

    def _record_open(path: Path, mode: str = "r", *args, **kwargs):
        modes.append(mode)
        return original_open(path, mode, *args, **kwargs)

    monkeypatch.setattr(agent_jobs_module, "_WINDOWS_DURABILITY", True)
    monkeypatch.setattr(Path, "open", _record_open)

    assert queue.confirm_durable(job.id) is not None
    assert "r+b" in modes


def test_confirmation_serializes_with_concurrent_state_move(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """The Windows receipt flush cannot hold a move-blocking CRT handle."""
    from afs import agent_jobs as agent_jobs_module

    queue = AgentJobQueue(tmp_path / ".context")
    job = queue.create("Confirm me", "Verify durable publication.")
    original_parse = queue._parse
    confirm_parsing = threading.Event()
    release_confirmation = threading.Event()
    move_finished = threading.Event()
    failures: list[BaseException] = []

    def _blocking_parse(path: Path):
        if threading.current_thread().name == "confirm-thread":
            confirm_parsing.set()
            assert release_confirmation.wait(timeout=5)
        return original_parse(path)

    def _confirm() -> None:
        try:
            assert queue.confirm_durable(job.id) is not None
        except BaseException as exc:  # pragma: no cover - asserted below
            failures.append(exc)

    def _move() -> None:
        try:
            queue.move(job.id, "running", assigned_to="worker")
        except BaseException as exc:  # pragma: no cover - asserted below
            failures.append(exc)
        finally:
            move_finished.set()

    monkeypatch.setattr(agent_jobs_module, "_WINDOWS_DURABILITY", True)
    monkeypatch.setattr(queue, "_parse", _blocking_parse)
    confirm_thread = threading.Thread(target=_confirm, name="confirm-thread")
    move_thread = threading.Thread(target=_move, name="move-thread")

    confirm_thread.start()
    assert confirm_parsing.wait(timeout=5)
    move_thread.start()
    assert not move_finished.wait(timeout=0.2)
    release_confirmation.set()
    confirm_thread.join(timeout=5)
    move_thread.join(timeout=5)

    assert not confirm_thread.is_alive()
    assert not move_thread.is_alive()
    assert failures == []
    assert queue.get(job.id).status == "running"  # type: ignore[union-attr]


@pytest.mark.parametrize("reader", ["get", "list"])
def test_public_reader_serializes_with_concurrent_state_move(
    tmp_path: Path,
    monkeypatch,
    reader: str,
) -> None:
    """Windows readers cannot hold a move-blocking CRT handle during rename."""
    queue = AgentJobQueue(tmp_path / ".context")
    job = queue.create("Read safely", "Serialize the queue read.")
    original_parse = queue._parse
    reader_parsing = threading.Event()
    release_reader = threading.Event()
    move_finished = threading.Event()
    failures: list[BaseException] = []

    def _blocking_parse(path: Path):
        if threading.current_thread().name == "reader-thread":
            reader_parsing.set()
            assert release_reader.wait(timeout=5)
        return original_parse(path)

    def _read() -> None:
        try:
            if reader == "get":
                assert queue.get(job.id) is not None
            else:
                assert [item.id for item in queue.list()] == [job.id]
        except BaseException as exc:  # pragma: no cover - asserted below
            failures.append(exc)

    def _move() -> None:
        try:
            queue.move(job.id, "running", assigned_to="worker")
        except BaseException as exc:  # pragma: no cover - asserted below
            failures.append(exc)
        finally:
            move_finished.set()

    monkeypatch.setattr(queue, "_parse", _blocking_parse)
    reader_thread = threading.Thread(target=_read, name="reader-thread")
    move_thread = threading.Thread(target=_move, name="move-thread")

    reader_thread.start()
    assert reader_parsing.wait(timeout=5)
    move_thread.start()
    assert not move_finished.wait(timeout=0.2)
    release_reader.set()
    reader_thread.join(timeout=5)
    move_thread.join(timeout=5)

    assert not reader_thread.is_alive()
    assert not move_thread.is_alive()
    assert failures == []
    assert queue.get(job.id).status == "running"  # type: ignore[union-attr]
