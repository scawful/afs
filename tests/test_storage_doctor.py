from __future__ import annotations

import errno
import hashlib
import json
import os
import subprocess
from pathlib import Path
from typing import Any

import pytest

import afs.storage_doctor as storage_doctor
from afs.human_provenance import HumanAuthorization, _broker_for_reader
from afs.storage_doctor import (
    StorageApplyError,
    StorageSafetyError,
    _canonical_json,
    apply_storage_plan,
    audit_storage,
    build_storage_plan,
    default_remove_candidate,
    load_storage_plan,
    storage_authorization_scope,
    write_storage_plan,
)

DAY_NS = 24 * 60 * 60 * 1_000_000_000
NOW_NS = 1_800_000_000_000_000_000


def clear_open(_path: Path, _kind: str) -> tuple[str, str]:
    return "clear", ""


def blocked_open(_path: Path, _kind: str) -> tuple[str, str]:
    return "open", "test_process_has_file_open"


def age(path: Path, *, days: int = 45) -> None:
    timestamp = NOW_NS - days * DAY_NS
    os.utime(path, ns=(timestamp, timestamp), follow_symlinks=False)


def make_fixture(home: Path, *, include_recent: bool = True) -> dict[str, Path]:
    build = home / ".yaze" / "nightly" / "job-a" / "build-nightly-debug"
    build.mkdir(parents=True)
    (build / "artifact.o").write_bytes(b"object")
    age(build / "artifact.o")
    age(build)

    pack = home / ".local" / "share" / "opencode" / "snapshot" / ("a" * 40) / "objects" / "pack"
    pack.mkdir(parents=True)
    temporary_pack = pack / "tmp_pack_deadbeef"
    temporary_pack.write_bytes(b"pack")
    age(temporary_pack)

    logs = home / ".lmstudio" / "server-logs" / "2026-01"
    logs.mkdir(parents=True)
    old_log = logs / "2026-01-01.1.log"
    old_log.write_text("old\n", encoding="utf-8")
    age(old_log)
    if include_recent:
        recent_log = logs / "recent.log"
        recent_log.write_text("recent\n", encoding="utf-8")
        os.utime(recent_log, ns=(NOW_NS, NOW_NS))

    models = home / ".lmstudio" / "models"
    models.mkdir(parents=True)
    broken_link = models / "retired.gguf"
    broken_link.symlink_to(home / "models" / "missing.gguf")
    age(broken_link)

    protected_model = home / "models" / "important.gguf"
    protected_model.parent.mkdir()
    protected_model.write_bytes(b"never delete")
    for protected in (
        home / "Archives" / "archive.bin",
        home / ".Trash" / "personal.bin",
        home / ".context" / "memory.md",
    ):
        protected.parent.mkdir(parents=True, exist_ok=True)
        protected.write_bytes(b"protected")

    decoy = home / "models" / "tmp_pack_not_a_candidate"
    decoy.write_bytes(b"decoy")
    age(decoy)
    return {
        "build": build,
        "temporary_pack": temporary_pack,
        "old_log": old_log,
        "broken_link": broken_link,
        "protected_model": protected_model,
        "decoy": decoy,
    }


def valid_rehash(plan: dict[str, Any]) -> dict[str, Any]:
    base = {key: value for key, value in plan.items() if key not in {"plan_sha256", "transaction"}}
    digest = hashlib.sha256(_canonical_json(base)).hexdigest()
    return base | {
        "plan_sha256": digest,
        "transaction": f"storage_{digest[:32]}",
    }


def write_raw_plan(path: Path, plan: dict[str, Any]) -> None:
    path.write_text(json.dumps(plan, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def authorize(plan: dict[str, Any], rationale: str) -> HumanAuthorization:
    scope = storage_authorization_scope(
        plan["plan_sha256"],
        plan["transaction"],
        rationale,
    )
    authorization = _broker_for_reader(lambda _prompt: plan["transaction"]).confirm_token(
        plan["transaction"],
        "confirm",
        scope=scope,
    )
    assert authorization is not None
    return authorization


def remove_with_home(
    candidate: Any,
    *,
    home: Path,
    quarantine_dir: Path | None = None,
) -> None:
    default_remove_candidate(
        candidate,
        quarantine_dir=quarantine_dir,
        trusted_home=home,
        home_mount_identity=storage_doctor.default_mount_identity_reader(home),
    )


def test_audit_is_bounded_and_protected_footprints_are_informational(
    tmp_path: Path,
) -> None:
    paths = make_fixture(tmp_path)
    payload = audit_storage(
        tmp_path,
        stale_days=30,
        now_ns=NOW_NS,
        open_checker=clear_open,  # type: ignore[arg-type]
        snapshot_reader=lambda: ("com.apple.TimeMachine.example",),
    )

    candidate_paths = {record["path"] for record in payload["candidates"]}
    assert str(paths["build"]) in candidate_paths
    assert str(paths["temporary_pack"]) in candidate_paths
    assert str(paths["old_log"]) in candidate_paths
    assert str(paths["broken_link"]) in candidate_paths
    assert str(paths["decoy"]) not in candidate_paths
    assert str(paths["protected_model"]) not in candidate_paths
    assert any(
        record["blocked_reasons"] == ("newer_than_30_days",) for record in payload["candidates"]
    )
    assert payload["local_snapshots"] == ["com.apple.TimeMachine.example"]
    assert payload["safety"]["processes_stopped"] is False
    assert "apfs_snapshots" in payload["safety"]["protected_categories"]
    assert {item["name"] for item in payload["footprints"]} >= {
        "local_models",
        "archives",
        "trash",
        "afs_context",
    }


def test_plan_is_hash_bound_and_duplicate_json_keys_are_rejected(tmp_path: Path) -> None:
    make_fixture(tmp_path, include_recent=False)
    plan = build_storage_plan(
        tmp_path,
        stale_days=30,
        now_ns=NOW_NS,
        open_checker=clear_open,  # type: ignore[arg-type]
        snapshot_reader=lambda: (),
    )
    plan_path = write_storage_plan(tmp_path / "plans" / "cleanup.json", plan)
    assert load_storage_plan(plan_path) == plan
    assert plan["transaction"].startswith("storage_")
    assert len(plan["candidates"]) == 4

    tampered = json.loads(plan_path.read_text(encoding="utf-8"))
    tampered["estimated_reclaim_bytes"] += 1
    write_raw_plan(plan_path, tampered)
    with pytest.raises(StorageSafetyError, match="hash"):
        load_storage_plan(plan_path)

    plan_path.write_text('{"schema":"one","schema":"two"}', encoding="utf-8")
    with pytest.raises(StorageSafetyError, match="duplicate"):
        load_storage_plan(plan_path)


def test_apply_revalidates_then_deletes_without_stopping_processes(
    tmp_path: Path,
) -> None:
    paths = make_fixture(tmp_path, include_recent=False)
    plan = build_storage_plan(
        tmp_path,
        stale_days=30,
        now_ns=NOW_NS,
        open_checker=clear_open,  # type: ignore[arg-type]
        snapshot_reader=lambda: (),
    )
    plan_path = write_storage_plan(tmp_path / "cleanup.json", plan)
    rationale = "Remove reviewed rebuildable artifacts while agents remain online."
    receipt = apply_storage_plan(
        plan_path,
        confirm=plan["transaction"],
        because=rationale,
        now_ns=NOW_NS,
        open_checker=clear_open,  # type: ignore[arg-type]
        authorization=authorize(plan, rationale),
    )

    assert receipt["status"] == "applied"
    assert receipt["processes_stopped"] is False
    assert len(receipt["deleted"]) == 4
    assert receipt["durability_uncertain"] == []
    for name in ("build", "temporary_pack", "old_log", "broken_link"):
        assert not os.path.lexists(paths[name])
    assert paths["protected_model"].exists()
    assert paths["decoy"].exists()
    assert Path(receipt["receipt_path"]).is_file()
    assert Path(receipt["manifest_path"]).is_file()
    assert Path(receipt["claim_path"]).is_file()
    journal = Path(receipt["journal_path"])
    assert len(tuple(journal.glob("*.intent.json"))) == 4
    assert len(tuple(journal.glob("*.deleted.json"))) == 4
    quarantine = Path(receipt["quarantine_path"])
    assert quarantine.is_dir()
    assert list(quarantine.iterdir()) == []
    assert receipt["receipt_written"] is True

    with pytest.raises(StorageSafetyError, match="already claimed"):
        apply_storage_plan(
            plan_path,
            confirm=plan["transaction"],
            because="A replay must fail closed.",
            now_ns=NOW_NS,
            open_checker=clear_open,  # type: ignore[arg-type]
        )


def test_apply_rejects_active_drift_missing_and_symlink_substitution(
    tmp_path: Path,
) -> None:
    home = tmp_path / "active"
    home.mkdir()
    paths = make_fixture(home, include_recent=False)
    plan = build_storage_plan(
        home,
        now_ns=NOW_NS,
        open_checker=clear_open,  # type: ignore[arg-type]
        snapshot_reader=lambda: (),
    )
    plan_path = write_storage_plan(home / "active-plan.json", plan)
    with pytest.raises(StorageSafetyError, match="drifted|eligible"):
        apply_storage_plan(
            plan_path,
            confirm=plan["transaction"],
            because="This should be blocked because a process opened a candidate.",
            now_ns=NOW_NS,
            open_checker=blocked_open,  # type: ignore[arg-type]
        )
    assert paths["build"].exists()

    drift_home = tmp_path / "drift"
    drift_home.mkdir()
    drift_paths = make_fixture(drift_home, include_recent=False)
    drift_plan = build_storage_plan(
        drift_home,
        now_ns=NOW_NS,
        open_checker=clear_open,  # type: ignore[arg-type]
        snapshot_reader=lambda: (),
    )
    drift_plan_path = write_storage_plan(drift_home / "drift-plan.json", drift_plan)
    drift_paths["old_log"].write_text("changed\n", encoding="utf-8")
    age(drift_paths["old_log"])
    with pytest.raises(StorageSafetyError, match="ID|drifted"):
        apply_storage_plan(
            drift_plan_path,
            confirm=drift_plan["transaction"],
            because="This should be blocked because content drifted.",
            now_ns=NOW_NS,
            open_checker=clear_open,  # type: ignore[arg-type]
        )

    link_home = tmp_path / "link"
    link_home.mkdir()
    link_paths = make_fixture(link_home, include_recent=False)
    link_plan = build_storage_plan(
        link_home,
        now_ns=NOW_NS,
        open_checker=clear_open,  # type: ignore[arg-type]
        snapshot_reader=lambda: (),
    )
    link_plan_path = write_storage_plan(link_home / "link-plan.json", link_plan)
    link_paths["old_log"].unlink()
    link_paths["old_log"].symlink_to(link_paths["protected_model"])
    with pytest.raises(StorageSafetyError, match="missing|bounded"):
        apply_storage_plan(
            link_plan_path,
            confirm=link_plan["transaction"],
            because="This should be blocked after a link substitution.",
            now_ns=NOW_NS,
            open_checker=clear_open,  # type: ignore[arg-type]
        )
    assert link_paths["protected_model"].read_bytes() == b"never delete"


def test_validly_rehashed_arbitrary_path_plan_cannot_escape_discovery(
    tmp_path: Path,
) -> None:
    paths = make_fixture(tmp_path, include_recent=False)
    plan = build_storage_plan(
        tmp_path,
        now_ns=NOW_NS,
        open_checker=clear_open,  # type: ignore[arg-type]
        snapshot_reader=lambda: (),
    )
    plan["candidates"][0]["path"] = str(paths["protected_model"])
    plan = valid_rehash(plan)
    plan_path = tmp_path / "forged.json"
    write_raw_plan(plan_path, plan)

    with pytest.raises(StorageSafetyError, match="ID|drifted"):
        apply_storage_plan(
            plan_path,
            confirm=plan["transaction"],
            because="A hash is not authority to escape bounded discovery.",
            now_ns=NOW_NS,
            open_checker=clear_open,  # type: ignore[arg-type]
        )
    assert paths["protected_model"].exists()


def test_live_opencode_gc_pid_is_blocked_even_without_an_open_file(
    tmp_path: Path,
) -> None:
    pack = tmp_path / ".local" / "share" / "opencode" / "snapshot" / ("b" * 40) / "objects" / "pack"
    pack.mkdir(parents=True)
    marker = pack / "gc.pid"
    marker.write_text(f"{os.getpid()} localhost\n", encoding="ascii")
    age(marker)
    payload = audit_storage(
        tmp_path,
        now_ns=NOW_NS,
        open_checker=clear_open,  # type: ignore[arg-type]
        snapshot_reader=lambda: (),
    )
    candidate = next(record for record in payload["candidates"] if record["path"] == str(marker))
    assert candidate["eligible"] is False
    assert candidate["blocked_reasons"] == (f"gc_pid_{os.getpid()}_is_running",)


@pytest.mark.parametrize(
    ("payload", "reason"),
    [
        (b"", "gc_pid_marker_empty"),
        (b"not-a-pid\n", "gc_pid_marker_malformed"),
        (b"1234\n", "gc_pid_marker_malformed"),
        (b"\xff\n", "gc_pid_marker_non_ascii"),
        (b"1 " + b"x" * 64, "gc_pid_marker_truncated"),
        (b"9999999999 host\n", "gc_pid_marker_out_of_range"),
    ],
)
def test_invalid_opencode_gc_pid_markers_fail_closed(
    tmp_path: Path,
    payload: bytes,
    reason: str,
) -> None:
    marker = tmp_path / "gc.pid"
    marker.write_bytes(payload)
    assert storage_doctor._pid_marker_block(marker) == reason


def test_lsof_checks_files_in_bounded_batches_and_directories_separately(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    files = tuple(tmp_path / f"file-{index:03d}.log" for index in range(202))
    unsafe = tmp_path / "unsafe\nname.log"
    directory = tmp_path / "build-nightly"
    specs = tuple(("lmstudio_log", path, "file") for path in (*files, unsafe)) + (
        ("yaze_build", directory, "directory"),
    )
    calls: list[list[str]] = []
    file_calls = 0

    monkeypatch.setattr(storage_doctor.shutil, "which", lambda _name: "/fake/lsof")
    monkeypatch.setattr(storage_doctor, "_lsof_payload_budget", lambda _prefix: 10**9)

    def fake_run(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        nonlocal file_calls
        calls.append(command)
        assert kwargs["timeout"] == 10
        if "+D" in command:
            raise subprocess.TimeoutExpired(command, 10)
        file_calls += 1
        stderr = "permission denied" if file_calls == 1 else ""
        return subprocess.CompletedProcess(command, 1, stdout="", stderr=stderr)

    monkeypatch.setattr(storage_doctor.subprocess, "run", fake_run)
    checker = storage_doctor._batched_open_checker(specs)  # type: ignore[arg-type]

    selected_calls = [command for command in calls if "--" in command]
    assert [len(command[command.index("--") + 1 :]) for command in selected_calls] == [
        100,
        100,
        2,
    ]
    assert all(command != ["/fake/lsof", "-Fn"] for command in calls)
    assert checker(files[0], "file") == ("unknown", "lsof_reported_errors")
    assert checker(files[100], "file") == ("clear", "")
    assert checker(directory, "directory") == ("unknown", "lsof_timed_out")
    assert checker(unsafe, "file") == (
        "unknown",
        "path_contains_control_or_formatting_characters",
    )
    assert all(os.fspath(unsafe) not in command for command in calls)


def test_lsof_directory_checks_share_one_operation_deadline(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    directories = tuple(tmp_path / f"build-nightly-{index}" for index in range(3))
    specs: tuple[tuple[str, Path, storage_doctor.CandidateKind], ...] = tuple(
        ("yaze_build", path, "directory") for path in directories
    )
    clock = [0.0]
    deadline = storage_doctor._OperationDeadline(
        expires_at=15.0,
        clock=lambda: clock[0],
    )
    timeouts: list[float] = []

    monkeypatch.setattr(storage_doctor.shutil, "which", lambda _name: "/fake/lsof")
    monkeypatch.setattr(storage_doctor, "_lsof_payload_budget", lambda _prefix: 10**9)

    def hang(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        timeout = kwargs["timeout"]
        timeouts.append(timeout)
        clock[0] += timeout
        raise subprocess.TimeoutExpired(command, timeout)

    monkeypatch.setattr(storage_doctor.subprocess, "run", hang)
    checker = storage_doctor._batched_open_checker(
        specs,
        deadline=deadline,
    )

    assert timeouts == [10.0, 5.0]
    assert checker(directories[0], "directory") == (
        "unknown",
        "lsof_timed_out",
    )
    assert checker(directories[1], "directory") == (
        "unknown",
        "lsof_operation_deadline_exceeded",
    )
    assert checker(directories[2], "directory") == (
        "unknown",
        "lsof_operation_deadline_exceeded",
    )


def test_apply_reuses_one_lsof_deadline_across_preflight_and_candidates(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logs = tmp_path / ".lmstudio" / "server-logs"
    logs.mkdir(parents=True)
    first = logs / "a.log"
    second = logs / "b.log"
    first.write_text("first\n", encoding="utf-8")
    second.write_text("second\n", encoding="utf-8")
    age(first)
    age(second)
    plan = build_storage_plan(
        tmp_path,
        now_ns=NOW_NS,
        open_checker=clear_open,  # type: ignore[arg-type]
        snapshot_reader=lambda: (),
        mount_reader=lambda: frozenset(),
    )
    plan_path = write_storage_plan(tmp_path / "deadline-plan.json", plan)
    rationale = "Exercise the bounded lsof deadline without stopping active agents."
    clock = [0.0]
    deadline = storage_doctor._OperationDeadline(
        expires_at=15.0,
        clock=lambda: clock[0],
    )
    timeouts: list[float] = []
    removed: list[str] = []

    monkeypatch.setattr(storage_doctor, "_new_lsof_deadline", lambda: deadline)
    monkeypatch.setattr(storage_doctor.shutil, "which", lambda _name: "/fake/lsof")
    monkeypatch.setattr(storage_doctor, "_lsof_payload_budget", lambda _prefix: 10**9)

    def check(command: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        timeout = kwargs["timeout"]
        timeouts.append(timeout)
        if len(timeouts) == 1:
            clock[0] += 9.0
            return subprocess.CompletedProcess(command, 1, stdout="", stderr="")
        if len(timeouts) == 2:
            clock[0] += 5.0
            return subprocess.CompletedProcess(command, 1, stdout="", stderr="")
        clock[0] += timeout
        raise subprocess.TimeoutExpired(command, timeout)

    def remove(candidate: Any) -> None:
        removed.append(candidate.path)
        Path(candidate.path).unlink()

    monkeypatch.setattr(storage_doctor.subprocess, "run", check)
    with pytest.raises(StorageApplyError) as raised:
        apply_storage_plan(
            plan_path,
            confirm=plan["transaction"],
            because=rationale,
            now_ns=NOW_NS,
            authorization=authorize(plan, rationale),
            mount_reader=lambda: frozenset(),
            remove_candidate=remove,
        )

    assert timeouts == [10.0, 6.0, 1.0]
    assert removed == [str(first)]
    assert not first.exists()
    assert second.exists()
    assert raised.value.receipt["status"] == "partial_failure"


def test_default_open_checker_fails_closed_on_partial_or_malformed_lsof(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    candidate = tmp_path / "old.log"
    candidate.write_text("old\n", encoding="utf-8")
    calls: list[list[str]] = []

    monkeypatch.setattr(storage_doctor.shutil, "which", lambda _name: "/fake/lsof")
    monkeypatch.setattr(storage_doctor, "_lsof_payload_budget", lambda _prefix: 10**9)

    def respond(
        returncode: int,
        *,
        stdout: str = "",
        stderr: str = "",
    ) -> None:
        def fake_run(command: list[str], **_kwargs: Any) -> subprocess.CompletedProcess[str]:
            calls.append(command)
            return subprocess.CompletedProcess(
                command,
                returncode,
                stdout=stdout,
                stderr=stderr,
            )

        monkeypatch.setattr(storage_doctor.subprocess, "run", fake_run)

    respond(1, stderr="permission denied")
    assert storage_doctor.default_open_checker(candidate, "file") == (
        "unknown",
        "lsof_reported_errors",
    )
    assert calls[-1] == ["/fake/lsof", "-Fn", "--", str(candidate)]

    respond(1)
    assert storage_doctor.default_open_checker(candidate, "file") == ("clear", "")

    respond(0, stdout=f"n{tmp_path / 'different.log'}\n")
    assert storage_doctor.default_open_checker(candidate, "file") == (
        "unknown",
        "lsof_output_missing_candidate",
    )

    respond(0, stdout=f"n{candidate}\n")
    assert storage_doctor.default_open_checker(candidate, "file") == (
        "open",
        "open_files_detected",
    )

    def fail_decode(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid")

    monkeypatch.setattr(storage_doctor.subprocess, "run", fail_decode)
    assert storage_doctor.default_open_checker(candidate, "file") == (
        "unknown",
        "lsof_failed:UnicodeDecodeError",
    )


def test_control_character_candidate_path_is_never_eligible(tmp_path: Path) -> None:
    logs = tmp_path / ".lmstudio" / "server-logs"
    logs.mkdir(parents=True)
    candidate_path = logs / "unsafe\nname.log"
    candidate_path.write_text("old\n", encoding="utf-8")
    age(candidate_path)

    payload = audit_storage(
        tmp_path,
        now_ns=NOW_NS,
        open_checker=clear_open,  # type: ignore[arg-type]
        snapshot_reader=lambda: (),
    )
    candidate = next(
        record for record in payload["candidates"] if record["path"] == str(candidate_path)
    )
    assert candidate["eligible"] is False
    assert "path_contains_control_or_formatting_characters" in candidate["blocked_reasons"]


def test_tmutil_reader_excludes_heading_and_unrelated_output(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(storage_doctor.sys, "platform", "darwin")
    monkeypatch.setattr(storage_doctor.shutil, "which", lambda _name: "/usr/bin/tmutil")
    monkeypatch.setattr(
        storage_doctor.subprocess,
        "run",
        lambda *_args, **_kwargs: subprocess.CompletedProcess(
            [],
            0,
            stdout=(
                "Snapshots for disk /:\n"
                "com.apple.TimeMachine.2026-07-24-010101.local\n"
                "unrelated text\n"
                "com.apple.os.update-ABC123\n"
            ),
            stderr="",
        ),
    )

    assert storage_doctor.default_snapshot_reader() == (
        "com.apple.TimeMachine.2026-07-24-010101.local",
        "com.apple.os.update-ABC123",
    )


def test_external_huggingface_hub_cache_precedence_and_deduplication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    home = tmp_path / "home"
    modern = tmp_path / "modern-hub"
    legacy = tmp_path / "legacy-hub"
    hf_home = tmp_path / "hf-home"
    for path in (home, modern, legacy, hf_home):
        path.mkdir()
    monkeypatch.setenv("HF_HOME", str(hf_home))
    monkeypatch.setenv("HF_HUB_CACHE", str(modern))
    monkeypatch.setenv("HUGGINGFACE_HUB_CACHE", str(legacy))

    payload = audit_storage(
        home,
        now_ns=NOW_NS,
        open_checker=clear_open,  # type: ignore[arg-type]
        snapshot_reader=lambda: (),
    )
    paths = {item["path"] for item in payload["footprints"]}
    assert str(modern) in paths
    assert str(legacy) in paths

    monkeypatch.delenv("HF_HUB_CACHE")
    payload = audit_storage(
        home,
        now_ns=NOW_NS,
        open_checker=clear_open,  # type: ignore[arg-type]
        snapshot_reader=lambda: (),
    )
    paths = {item["path"] for item in payload["footprints"]}
    assert str(legacy) in paths

    nested = hf_home / "hub"
    nested.mkdir()
    monkeypatch.setenv("HF_HUB_CACHE", str(nested))
    monkeypatch.delenv("HUGGINGFACE_HUB_CACHE")
    payload = audit_storage(
        home,
        now_ns=NOW_NS,
        open_checker=clear_open,  # type: ignore[arg-type]
        snapshot_reader=lambda: (),
    )
    hf_paths = {
        item["path"] for item in payload["footprints"] if item["name"].startswith("huggingface")
    }
    assert hf_paths == {str(hf_home)}


def test_informational_footprints_stop_at_nested_mounts_but_measure_mounted_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models = tmp_path / "models"
    nested_mount = models / "mounted"
    nested_mount.mkdir(parents=True)
    (nested_mount / "large.bin").write_bytes(b"mounted")
    for variable in (
        "HF_HOME",
        "HF_HUB_CACHE",
        "HUGGINGFACE_HUB_CACHE",
        "OLLAMA_MODELS",
        "OLLAMA_MODELS_DIR",
    ):
        monkeypatch.delenv(variable, raising=False)

    nested_payload = audit_storage(
        tmp_path,
        now_ns=NOW_NS,
        open_checker=clear_open,  # type: ignore[arg-type]
        snapshot_reader=lambda: (),
        mount_reader=lambda: frozenset({nested_mount}),
    )
    nested_footprint = next(
        item for item in nested_payload["footprints"] if item["name"] == "local_models"
    )
    assert nested_footprint["entry_count"] == 2

    root_payload = audit_storage(
        tmp_path,
        now_ns=NOW_NS,
        open_checker=clear_open,  # type: ignore[arg-type]
        snapshot_reader=lambda: (),
        mount_reader=lambda: frozenset({models}),
    )
    root_footprint = next(
        item for item in root_payload["footprints"] if item["name"] == "local_models"
    )
    assert root_footprint["entry_count"] == 3


def test_informational_footprints_do_not_traverse_without_mount_inventory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    models = tmp_path / "models"
    models.mkdir()
    (models / "large.bin").write_bytes(b"protected")
    real_measure_tree = storage_doctor.measure_tree

    def refuse_footprint_traversal(
        path: Path,
        *,
        mount_points: frozenset[Path] = frozenset(),
    ) -> storage_doctor.TreeMeasurement:
        if path == models:
            raise AssertionError("footprint traversed without a mount inventory")
        return real_measure_tree(path, mount_points=mount_points)

    monkeypatch.setattr(storage_doctor, "measure_tree", refuse_footprint_traversal)
    payload = audit_storage(
        tmp_path,
        now_ns=NOW_NS,
        open_checker=clear_open,  # type: ignore[arg-type]
        snapshot_reader=lambda: (),
        mount_reader=lambda: None,
    )

    local_models = next(item for item in payload["footprints"] if item["name"] == "local_models")
    assert local_models["entry_count"] == 0
    assert local_models["allocated_bytes"] == 0
    assert local_models["issue"] == ("mount inventory unavailable; footprint was not traversed")


def test_ollama_models_uses_canonical_override_before_legacy_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    canonical = tmp_path / "canonical-ollama"
    legacy = tmp_path / "legacy-ollama"
    canonical.mkdir()
    legacy.mkdir()
    monkeypatch.setenv("OLLAMA_MODELS", str(canonical))
    monkeypatch.setenv("OLLAMA_MODELS_DIR", str(legacy))

    payload = audit_storage(
        tmp_path,
        now_ns=NOW_NS,
        open_checker=clear_open,  # type: ignore[arg-type]
        snapshot_reader=lambda: (),
        mount_reader=lambda: frozenset(),
    )
    ollama = next(item for item in payload["footprints"] if item["name"] == "ollama_models")
    assert ollama["path"] == str(canonical)

    monkeypatch.delenv("OLLAMA_MODELS")
    payload = audit_storage(
        tmp_path,
        now_ns=NOW_NS,
        open_checker=clear_open,  # type: ignore[arg-type]
        snapshot_reader=lambda: (),
        mount_reader=lambda: frozenset(),
    )
    ollama = next(item for item in payload["footprints"] if item["name"] == "ollama_models")
    assert ollama["path"] == str(legacy)


def test_linked_candidate_root_is_not_traversed(tmp_path: Path) -> None:
    home = tmp_path / "home"
    outside = tmp_path / "outside"
    outside_logs = outside / "server-logs"
    outside_logs.mkdir(parents=True)
    outside_log = outside_logs / "old.log"
    outside_log.write_text("outside\n", encoding="utf-8")
    age(outside_log)
    home.mkdir()
    (home / ".lmstudio").symlink_to(outside)

    payload = audit_storage(
        home,
        now_ns=NOW_NS,
        open_checker=clear_open,  # type: ignore[arg-type]
        snapshot_reader=lambda: (),
    )
    assert str(outside_log) not in {record["path"] for record in payload["candidates"]}
    assert outside_log.exists()


def test_partial_failure_writes_receipt_and_claim_blocks_retry(tmp_path: Path) -> None:
    make_fixture(tmp_path, include_recent=False)
    plan = build_storage_plan(
        tmp_path,
        now_ns=NOW_NS,
        open_checker=clear_open,  # type: ignore[arg-type]
        snapshot_reader=lambda: (),
    )
    plan_path = write_storage_plan(tmp_path / "partial.json", plan)
    quarantine = tmp_path / "custom-quarantine"
    quarantine.mkdir()
    calls = 0

    def fail_second(candidate: Any) -> None:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("injected deletion failure")
        remove_with_home(
            candidate,
            home=tmp_path,
            quarantine_dir=quarantine,
        )

    rationale = "Exercise partial-failure accounting."
    with pytest.raises(StorageApplyError) as raised:
        apply_storage_plan(
            plan_path,
            confirm=plan["transaction"],
            because=rationale,
            now_ns=NOW_NS,
            open_checker=clear_open,  # type: ignore[arg-type]
            remove_candidate=fail_second,
            authorization=authorize(plan, rationale),
        )
    receipt = raised.value.receipt
    assert receipt["status"] == "partial_failure"
    assert len(receipt["deleted"]) == 1
    assert receipt["failure"]["error"].endswith("injected deletion failure")
    assert Path(receipt["receipt_path"]).is_file()
    journal = Path(receipt["journal_path"])
    assert len(tuple(journal.glob("*.intent.json"))) == 2
    assert len(tuple(journal.glob("*.deleted.json"))) == 1
    assert len(tuple(journal.glob("*.failed.json"))) == 1

    with pytest.raises(StorageSafetyError, match="already claimed"):
        apply_storage_plan(
            plan_path,
            confirm=plan["transaction"],
            because="A partially applied transaction cannot be replayed.",
            now_ns=NOW_NS,
            open_checker=clear_open,  # type: ignore[arg-type]
        )


def test_post_unlink_fsync_failure_is_recorded_as_durability_uncertain(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logs = tmp_path / ".lmstudio" / "server-logs"
    logs.mkdir(parents=True)
    old_log = logs / "old.log"
    old_log.write_text("old\n", encoding="utf-8")
    age(old_log)
    plan = build_storage_plan(
        tmp_path,
        now_ns=NOW_NS,
        open_checker=clear_open,  # type: ignore[arg-type]
        snapshot_reader=lambda: (),
    )
    assert len(plan["candidates"]) == 1
    plan_path = write_storage_plan(tmp_path / "durability-uncertain.json", plan)

    real_fsync = storage_doctor._strict_fsync_directory_fd
    calls = 0

    def fail_final_quarantine_fsync(descriptor: int) -> None:
        nonlocal calls
        calls += 1
        if calls == 3:
            raise OSError("injected final quarantine fsync failure")
        real_fsync(descriptor)

    monkeypatch.setattr(
        storage_doctor,
        "_strict_fsync_directory_fd",
        fail_final_quarantine_fsync,
    )
    rationale = "Exercise honest post-unlink durability accounting."
    with pytest.raises(StorageApplyError) as raised:
        apply_storage_plan(
            plan_path,
            confirm=plan["transaction"],
            because=rationale,
            now_ns=NOW_NS,
            open_checker=clear_open,  # type: ignore[arg-type]
            authorization=authorize(plan, rationale),
        )

    receipt = raised.value.receipt
    assert receipt["status"] == "durability_uncertain"
    assert receipt["failure"] is None
    assert not old_log.exists()
    assert len(receipt["deleted"]) == 1
    assert receipt["deleted"][0]["outcome"] == "deleted_durability_uncertain"
    assert receipt["deleted"][0]["durability_error"].endswith(
        "injected final quarantine fsync failure"
    )
    assert receipt["durability_uncertain"] == receipt["deleted"]
    quarantine = Path(receipt["quarantine_path"])
    assert list(quarantine.iterdir()) == []
    journal = Path(receipt["journal_path"])
    outcomes = tuple(journal.glob("*.deleted.json"))
    assert len(outcomes) == 1
    assert json.loads(outcomes[0].read_text(encoding="utf-8"))["status"] == (
        "deleted_durability_uncertain"
    )
    assert not tuple(journal.glob("*.failed.json"))

    with pytest.raises(StorageSafetyError, match="already claimed"):
        apply_storage_plan(
            plan_path,
            confirm=plan["transaction"],
            because="An uncertain transaction cannot be replayed.",
            now_ns=NOW_NS,
            open_checker=clear_open,  # type: ignore[arg-type]
        )


def test_receipt_write_failure_reports_changed_state_and_blocks_replay(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logs = tmp_path / ".lmstudio" / "server-logs"
    logs.mkdir(parents=True)
    old_log = logs / "old.log"
    old_log.write_text("old\n", encoding="utf-8")
    age(old_log)
    plan = build_storage_plan(
        tmp_path,
        now_ns=NOW_NS,
        open_checker=clear_open,  # type: ignore[arg-type]
        snapshot_reader=lambda: (),
    )
    plan_path = write_storage_plan(tmp_path / "receipt-failure.json", plan)

    def fail_receipt(_path: Path, _receipt: dict[str, Any]) -> None:
        raise OSError("injected receipt failure")

    monkeypatch.setattr(storage_doctor, "_write_receipt", fail_receipt)
    rationale = "Exercise receipt failure reporting."
    with pytest.raises(StorageApplyError) as raised:
        apply_storage_plan(
            plan_path,
            confirm=plan["transaction"],
            because=rationale,
            now_ns=NOW_NS,
            open_checker=clear_open,  # type: ignore[arg-type]
            authorization=authorize(plan, rationale),
        )
    assert raised.value.receipt["status"] == "receipt_failure"
    assert raised.value.receipt["receipt_written"] is False
    assert Path(raised.value.receipt["claim_path"]).is_file()
    assert not Path(raised.value.receipt["receipt_path"]).exists()
    assert not old_log.exists()

    with pytest.raises(StorageSafetyError, match="already claimed"):
        apply_storage_plan(
            plan_path,
            confirm=plan["transaction"],
            because="A missing receipt must not allow replay.",
            now_ns=NOW_NS,
            open_checker=clear_open,  # type: ignore[arg-type]
        )


def test_empty_expired_and_bad_confirmation_plans_refuse_apply(tmp_path: Path) -> None:
    plan = build_storage_plan(
        tmp_path,
        now_ns=NOW_NS,
        open_checker=clear_open,  # type: ignore[arg-type]
        snapshot_reader=lambda: (),
    )
    plan_path = write_storage_plan(tmp_path / "empty.json", plan)
    with pytest.raises(StorageSafetyError, match="confirmation"):
        apply_storage_plan(
            plan_path,
            confirm="wrong",
            because="Wrong transaction.",
            now_ns=NOW_NS,
            open_checker=clear_open,  # type: ignore[arg-type]
        )
    with pytest.raises(StorageSafetyError, match="no eligible"):
        apply_storage_plan(
            plan_path,
            confirm=plan["transaction"],
            because="Empty plans must not create cleanup state.",
            now_ns=NOW_NS,
            open_checker=clear_open,  # type: ignore[arg-type]
        )

    make_fixture(tmp_path, include_recent=False)
    expired = build_storage_plan(
        tmp_path,
        now_ns=NOW_NS,
        open_checker=clear_open,  # type: ignore[arg-type]
        snapshot_reader=lambda: (),
    )
    expired_path = write_storage_plan(tmp_path / "expired.json", expired)
    with pytest.raises(StorageSafetyError, match="expired"):
        apply_storage_plan(
            expired_path,
            confirm=expired["transaction"],
            because="Expired plans require a new audit.",
            now_ns=expired["expires_at_ns"] + 1,
            open_checker=clear_open,  # type: ignore[arg-type]
        )


def test_apply_requires_fresh_controlling_terminal_authorization(
    tmp_path: Path,
) -> None:
    paths = make_fixture(tmp_path, include_recent=False)
    plan = build_storage_plan(
        tmp_path,
        now_ns=NOW_NS,
        open_checker=clear_open,  # type: ignore[arg-type]
        snapshot_reader=lambda: (),
    )
    plan_path = write_storage_plan(tmp_path / "authorization.json", plan)
    rationale = "Only a person at the terminal can authorize cleanup."

    with pytest.raises(StorageSafetyError, match="controlling-terminal"):
        apply_storage_plan(
            plan_path,
            confirm=plan["transaction"],
            because=rationale,
            now_ns=NOW_NS,
            open_checker=clear_open,  # type: ignore[arg-type]
        )

    assert paths["build"].exists()
    assert not (tmp_path / ".afs" / "storage" / "transactions" / plan["transaction"]).exists()


def test_symlink_home_is_rejected_before_discovery_or_apply(tmp_path: Path) -> None:
    actual_home = tmp_path / "actual-home"
    actual_home.mkdir()
    paths = make_fixture(actual_home, include_recent=False)
    linked_home = tmp_path / "linked-home"
    linked_home.symlink_to(actual_home, target_is_directory=True)

    with pytest.raises((StorageSafetyError, ValueError), match="symbolic link"):
        audit_storage(
            linked_home,
            now_ns=NOW_NS,
            open_checker=clear_open,  # type: ignore[arg-type]
            snapshot_reader=lambda: (),
        )

    plan = build_storage_plan(
        actual_home,
        now_ns=NOW_NS,
        open_checker=clear_open,  # type: ignore[arg-type]
        snapshot_reader=lambda: (),
    )
    plan["home"] = str(linked_home)
    for record in plan["candidates"]:
        original = Path(record["path"])
        forged_path = linked_home / original.relative_to(actual_home)
        record["path"] = str(forged_path)
        record["candidate_id"] = storage_doctor._candidate_id(
            record["category"],
            linked_home,
            forged_path,
        )
    forged = valid_rehash(plan)
    plan_path = tmp_path / "linked-home-plan.json"
    write_raw_plan(plan_path, forged)

    with pytest.raises((StorageSafetyError, ValueError), match="symbolic link"):
        apply_storage_plan(
            plan_path,
            confirm=forged["transaction"],
            because="A linked home must never redirect deletion.",
            now_ns=NOW_NS,
            open_checker=clear_open,  # type: ignore[arg-type]
        )
    assert paths["protected_model"].read_bytes() == b"never delete"
    assert paths["old_log"].exists()


def test_plan_timestamps_and_nonfinite_numbers_fail_closed(tmp_path: Path) -> None:
    plan = build_storage_plan(
        tmp_path,
        now_ns=NOW_NS,
        open_checker=clear_open,  # type: ignore[arg-type]
        snapshot_reader=lambda: (),
    )
    plan_path = tmp_path / "invalid-time.json"
    plan["created_at_ns"] = storage_doctor.MAX_TIMESTAMP_NS + 1
    write_raw_plan(plan_path, plan)
    with pytest.raises(StorageSafetyError, match="supported 2000-2100 range"):
        load_storage_plan(plan_path)

    rendered = json.dumps(
        valid_rehash(
            build_storage_plan(
                tmp_path,
                now_ns=NOW_NS,
                open_checker=clear_open,  # type: ignore[arg-type]
                snapshot_reader=lambda: (),
            )
        )
    )
    rendered = rendered.replace(
        '"estimated_reclaim_bytes": 0',
        '"estimated_reclaim_bytes": NaN',
        1,
    )
    plan_path.write_text(rendered, encoding="utf-8")
    with pytest.raises(StorageSafetyError, match="non-finite"):
        load_storage_plan(plan_path)

    with pytest.raises(StorageSafetyError, match="plan expiry"):
        build_storage_plan(
            tmp_path,
            now_ns=storage_doctor.MAX_TIMESTAMP_NS,
            open_checker=clear_open,  # type: ignore[arg-type]
            snapshot_reader=lambda: (),
        )


def test_plan_size_and_candidate_limits_apply_before_publication(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    make_fixture(tmp_path, include_recent=False)
    plan = build_storage_plan(
        tmp_path,
        now_ns=NOW_NS,
        open_checker=clear_open,  # type: ignore[arg-type]
        snapshot_reader=lambda: (),
    )

    monkeypatch.setattr(storage_doctor, "MAX_PLAN_CANDIDATES", 1)
    with pytest.raises(StorageSafetyError, match="candidate limit"):
        build_storage_plan(
            tmp_path,
            now_ns=NOW_NS,
            open_checker=clear_open,  # type: ignore[arg-type]
            snapshot_reader=lambda: (),
        )
    with pytest.raises(StorageSafetyError, match="invalid or excessive"):
        write_storage_plan(tmp_path / "too-many.json", plan)
    assert not (tmp_path / "too-many.json").exists()

    monkeypatch.setattr(storage_doctor, "MAX_PLAN_CANDIDATES", 50_000)
    monkeypatch.setattr(storage_doctor, "MAX_PLAN_BYTES", 128)
    with pytest.raises(StorageSafetyError, match="8 MiB limit"):
        write_storage_plan(tmp_path / "too-large.json", plan)
    assert not (tmp_path / "too-large.json").exists()


def test_non_string_rationale_is_rejected_without_cleanup(tmp_path: Path) -> None:
    make_fixture(tmp_path, include_recent=False)
    plan = build_storage_plan(
        tmp_path,
        now_ns=NOW_NS,
        open_checker=clear_open,  # type: ignore[arg-type]
        snapshot_reader=lambda: (),
    )
    plan_path = write_storage_plan(tmp_path / "bad-rationale.json", plan)
    with pytest.raises(StorageSafetyError, match="string"):
        apply_storage_plan(
            plan_path,
            confirm=plan["transaction"],
            because=123,  # type: ignore[arg-type]
            now_ns=NOW_NS,
            open_checker=clear_open,  # type: ignore[arg-type]
        )


def test_mount_inventory_parsers_decode_paths() -> None:
    linux = storage_doctor._parse_linux_mountinfo(
        "36 25 0:32 / / rw,relatime - ext4 /dev/root rw\n"
        "37 36 0:33 / /mnt/with\\040space rw - tmpfs tmpfs rw\n"
    )
    assert linux == frozenset({Path("/"), Path("/mnt/with space")})

    darwin = storage_doctor._parse_darwin_mount_output(
        "/dev/disk3s1s1 on / (apfs, sealed)\n"
        "map auto_home on /System/Volumes/Data/home (autofs, automounted)\n"
    )
    assert darwin == frozenset({Path("/"), Path("/System/Volumes/Data/home")})


def test_directory_candidate_is_blocked_by_mount_inventory(tmp_path: Path) -> None:
    paths = make_fixture(tmp_path, include_recent=False)
    mounted = paths["build"] / "mounted"
    mounted.mkdir()
    protected = mounted / "protected.bin"
    protected.write_bytes(b"mounted data")
    age(protected)
    age(mounted)
    age(paths["build"])

    unknown_candidates = storage_doctor.discover_candidates(
        tmp_path,
        now_ns=NOW_NS,
        open_checker=clear_open,  # type: ignore[arg-type]
        mount_reader=lambda: None,
    )
    unknown_build = next(
        candidate for candidate in unknown_candidates if candidate.path == str(paths["build"])
    )
    assert "mount_inventory_unavailable" in unknown_build.blocked_reasons

    candidates = storage_doctor.discover_candidates(
        tmp_path,
        now_ns=NOW_NS,
        open_checker=clear_open,  # type: ignore[arg-type]
        mount_reader=lambda: frozenset({mounted}),
    )
    build = next(candidate for candidate in candidates if candidate.path == str(paths["build"]))
    assert "contains_mount_point" in build.blocked_reasons
    quarantine = tmp_path / "quarantine"
    quarantine.mkdir()

    with pytest.raises(StorageSafetyError, match="mount point"):
        default_remove_candidate(
            build,
            mount_reader=lambda: frozenset({mounted}),
            quarantine_dir=quarantine,
            trusted_home=tmp_path,
            home_mount_identity=storage_doctor.default_mount_identity_reader(tmp_path),
        )
    assert protected.read_bytes() == b"mounted data"


def test_directory_remover_unlinks_links_without_following_them(tmp_path: Path) -> None:
    paths = make_fixture(tmp_path, include_recent=False)
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"keep")
    escape = paths["build"] / "escape"
    escape.symlink_to(outside)
    age(escape)
    age(paths["build"])

    candidates = storage_doctor.discover_candidates(
        tmp_path,
        now_ns=NOW_NS,
        open_checker=clear_open,  # type: ignore[arg-type]
        mount_reader=lambda: frozenset(),
    )
    build = next(candidate for candidate in candidates if candidate.path == str(paths["build"]))
    assert build.eligible
    quarantine = tmp_path / "quarantine"
    quarantine.mkdir()

    default_remove_candidate(
        build,
        mount_reader=lambda: frozenset(),
        quarantine_dir=quarantine,
        trusted_home=tmp_path,
        home_mount_identity=storage_doctor.default_mount_identity_reader(tmp_path),
    )
    assert not paths["build"].exists()
    assert outside.read_bytes() == b"keep"
    assert list(quarantine.iterdir()) == []


def test_mounted_ancestor_between_home_and_candidate_is_blocked(
    tmp_path: Path,
) -> None:
    paths = make_fixture(tmp_path, include_recent=False)
    mounted_ancestor = tmp_path / ".lmstudio"
    mounted_yaze = tmp_path / ".yaze"
    candidates = storage_doctor.discover_candidates(
        tmp_path,
        now_ns=NOW_NS,
        open_checker=clear_open,  # type: ignore[arg-type]
        mount_reader=lambda: frozenset({tmp_path, mounted_ancestor, mounted_yaze}),
    )

    old_log = next(candidate for candidate in candidates if candidate.path == str(paths["old_log"]))
    build = next(candidate for candidate in candidates if candidate.path == str(paths["build"]))
    assert "mounted_ancestor_between_home_and_candidate" in old_log.blocked_reasons
    assert "mounted_ancestor_between_home_and_candidate" in build.blocked_reasons


def test_plan_pins_home_mount_identity_and_rejects_mount_change(
    tmp_path: Path,
) -> None:
    make_fixture(tmp_path, include_recent=False)
    planned_identity = (123, 456)
    plan = build_storage_plan(
        tmp_path,
        now_ns=NOW_NS,
        open_checker=clear_open,  # type: ignore[arg-type]
        snapshot_reader=lambda: (),
        mount_reader=lambda: frozenset({tmp_path}),
        mount_identity_reader=lambda _path: planned_identity,
    )
    assert plan["home_mount_identity"] == {"device": 123, "mount_id": 456}
    plan_path = write_storage_plan(tmp_path / "mount-pinned.json", plan)
    assert load_storage_plan(plan_path)["home_mount_identity"] == {
        "device": 123,
        "mount_id": 456,
    }

    with pytest.raises(StorageSafetyError, match="home mount identity changed"):
        apply_storage_plan(
            plan_path,
            confirm=plan["transaction"],
            because="A remounted home invalidates the reviewed cleanup plan.",
            now_ns=NOW_NS,
            open_checker=clear_open,  # type: ignore[arg-type]
            mount_reader=lambda: frozenset({tmp_path}),
            mount_identity_reader=lambda _path: (123, 457),
        )


def test_candidate_parent_fd_must_match_pinned_home_mount(tmp_path: Path) -> None:
    paths = make_fixture(tmp_path, include_recent=False)
    candidate = next(
        item
        for item in storage_doctor.discover_candidates(
            tmp_path,
            now_ns=NOW_NS,
            open_checker=clear_open,  # type: ignore[arg-type]
            mount_reader=lambda: frozenset({tmp_path}),
        )
        if item.path == str(paths["old_log"])
    )
    actual = storage_doctor.default_mount_identity_reader(tmp_path)
    wrong = (actual[0] + 1, actual[1])

    with pytest.raises(StorageSafetyError, match="pinned home mount"):
        storage_doctor._assert_candidate_parent_on_home_mount(
            candidate,
            home=tmp_path,
            home_mount_identity=wrong,
        )
    quarantine = tmp_path / "quarantine"
    quarantine.mkdir()
    with pytest.raises(StorageSafetyError, match="pinned home mount"):
        default_remove_candidate(
            candidate,
            trusted_home=tmp_path,
            home_mount_identity=wrong,
            quarantine_dir=quarantine,
        )
    assert paths["old_log"].exists()


def test_file_and_broken_link_removal_require_and_use_quarantine(tmp_path: Path) -> None:
    paths = make_fixture(tmp_path, include_recent=False)
    candidates = storage_doctor.discover_candidates(
        tmp_path,
        now_ns=NOW_NS,
        open_checker=clear_open,  # type: ignore[arg-type]
        mount_reader=lambda: frozenset(),
    )
    old_log = next(candidate for candidate in candidates if candidate.path == str(paths["old_log"]))
    broken_link = next(
        candidate for candidate in candidates if candidate.path == str(paths["broken_link"])
    )
    quarantine = tmp_path / "quarantine"
    quarantine.mkdir()

    with pytest.raises(StorageSafetyError, match="requires a transaction quarantine"):
        remove_with_home(old_log, home=tmp_path)
    assert paths["old_log"].read_text(encoding="utf-8") == "old\n"

    remove_with_home(old_log, home=tmp_path, quarantine_dir=quarantine)
    remove_with_home(broken_link, home=tmp_path, quarantine_dir=quarantine)
    assert not paths["old_log"].exists()
    assert not os.path.lexists(paths["broken_link"])
    assert list(quarantine.iterdir()) == []


def test_file_swap_during_quarantine_preserves_replacement_for_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = make_fixture(tmp_path, include_recent=False)
    candidate = next(
        item
        for item in storage_doctor.discover_candidates(
            tmp_path,
            now_ns=NOW_NS,
            open_checker=clear_open,  # type: ignore[arg-type]
            mount_reader=lambda: frozenset(),
        )
        if item.path == str(paths["old_log"])
    )
    quarantine = tmp_path / "quarantine"
    quarantine.mkdir()
    displaced = tmp_path / "displaced-original.log"
    real_rename = storage_doctor._renameat_noreplace

    def swap_then_rename(
        source_descriptor: int,
        source_name: str,
        destination_descriptor: int,
        destination_name: str,
    ) -> None:
        paths["old_log"].rename(displaced)
        paths["old_log"].write_text("replacement\n", encoding="utf-8")
        real_rename(
            source_descriptor,
            source_name,
            destination_descriptor,
            destination_name,
        )

    monkeypatch.setattr(storage_doctor, "_renameat_noreplace", swap_then_rename)
    with pytest.raises(StorageSafetyError, match="recover it at"):
        remove_with_home(candidate, home=tmp_path, quarantine_dir=quarantine)

    quarantined = quarantine / f"{candidate.candidate_id}.file"
    assert quarantined.read_text(encoding="utf-8") == "replacement\n"
    assert displaced.read_text(encoding="utf-8") == "old\n"


def test_file_in_place_drift_during_quarantine_is_left_for_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = make_fixture(tmp_path, include_recent=False)
    candidate = next(
        item
        for item in storage_doctor.discover_candidates(
            tmp_path,
            now_ns=NOW_NS,
            open_checker=clear_open,  # type: ignore[arg-type]
            mount_reader=lambda: frozenset(),
        )
        if item.path == str(paths["old_log"])
    )
    quarantine = tmp_path / "quarantine"
    quarantine.mkdir()
    real_rename = storage_doctor._renameat_noreplace

    def mutate_then_rename(
        source_descriptor: int,
        source_name: str,
        destination_descriptor: int,
        destination_name: str,
    ) -> None:
        paths["old_log"].write_text(
            "changed while the inode was pinned\n",
            encoding="utf-8",
        )
        real_rename(
            source_descriptor,
            source_name,
            destination_descriptor,
            destination_name,
        )

    monkeypatch.setattr(
        storage_doctor,
        "_renameat_noreplace",
        mutate_then_rename,
    )
    with pytest.raises(StorageSafetyError, match="quarantined file changed; recover it at"):
        remove_with_home(candidate, home=tmp_path, quarantine_dir=quarantine)

    recovery = quarantine / f"{candidate.candidate_id}.file"
    assert not paths["old_log"].exists()
    assert recovery.read_text(encoding="utf-8") == ("changed while the inode was pinned\n")


def test_broken_link_timestamp_drift_during_quarantine_is_left_for_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = make_fixture(tmp_path, include_recent=False)
    candidate = next(
        item
        for item in storage_doctor.discover_candidates(
            tmp_path,
            now_ns=NOW_NS,
            open_checker=clear_open,  # type: ignore[arg-type]
            mount_reader=lambda: frozenset(),
        )
        if item.path == str(paths["broken_link"])
    )
    quarantine = tmp_path / "quarantine"
    quarantine.mkdir()
    expected_target = os.readlink(paths["broken_link"])
    real_rename = storage_doctor._renameat_noreplace

    def touch_then_rename(
        source_descriptor: int,
        source_name: str,
        destination_descriptor: int,
        destination_name: str,
    ) -> None:
        os.utime(
            paths["broken_link"],
            ns=(NOW_NS, NOW_NS),
            follow_symlinks=False,
        )
        real_rename(
            source_descriptor,
            source_name,
            destination_descriptor,
            destination_name,
        )

    monkeypatch.setattr(
        storage_doctor,
        "_renameat_noreplace",
        touch_then_rename,
    )
    with pytest.raises(StorageSafetyError, match="quarantined link changed; recover it at"):
        remove_with_home(candidate, home=tmp_path, quarantine_dir=quarantine)

    recovery = quarantine / f"{candidate.candidate_id}.broken_symlink"
    assert not os.path.lexists(paths["broken_link"])
    assert recovery.is_symlink()
    assert os.readlink(recovery) == expected_target


def test_file_recreated_after_atomic_quarantine_is_not_deleted(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = make_fixture(tmp_path, include_recent=False)
    candidate = next(
        item
        for item in storage_doctor.discover_candidates(
            tmp_path,
            now_ns=NOW_NS,
            open_checker=clear_open,  # type: ignore[arg-type]
            mount_reader=lambda: frozenset(),
        )
        if item.path == str(paths["old_log"])
    )
    quarantine = tmp_path / "quarantine"
    quarantine.mkdir()
    real_rename = storage_doctor._renameat_noreplace

    def rename_then_recreate(
        source_descriptor: int,
        source_name: str,
        destination_descriptor: int,
        destination_name: str,
    ) -> None:
        real_rename(
            source_descriptor,
            source_name,
            destination_descriptor,
            destination_name,
        )
        paths["old_log"].write_text("replacement\n", encoding="utf-8")

    monkeypatch.setattr(storage_doctor, "_renameat_noreplace", rename_then_recreate)
    remove_with_home(candidate, home=tmp_path, quarantine_dir=quarantine)

    assert paths["old_log"].read_text(encoding="utf-8") == "replacement\n"
    assert list(quarantine.iterdir()) == []


def test_parent_component_swaps_cannot_redirect_pinned_quarantine_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = make_fixture(tmp_path, include_recent=False)
    candidate = next(
        item
        for item in storage_doctor.discover_candidates(
            tmp_path,
            now_ns=NOW_NS,
            open_checker=clear_open,  # type: ignore[arg-type]
            mount_reader=lambda: frozenset(),
        )
        if item.path == str(paths["old_log"])
    )
    source_parent = paths["old_log"].parent
    displaced_source_parent = source_parent.with_name("2026-01-pinned")
    quarantine = tmp_path / "quarantine"
    quarantine.mkdir()
    displaced_quarantine = tmp_path / "quarantine-pinned"
    real_rename = storage_doctor._renameat_noreplace

    def swap_parents_then_rename(
        source_descriptor: int,
        source_name: str,
        destination_descriptor: int,
        destination_name: str,
    ) -> None:
        source_parent.rename(displaced_source_parent)
        source_parent.mkdir()
        paths["old_log"].write_text("protected replacement\n", encoding="utf-8")
        quarantine.rename(displaced_quarantine)
        quarantine.mkdir()
        (quarantine / "sentinel").write_text("protected\n", encoding="utf-8")
        real_rename(
            source_descriptor,
            source_name,
            destination_descriptor,
            destination_name,
        )

    monkeypatch.setattr(
        storage_doctor,
        "_renameat_noreplace",
        swap_parents_then_rename,
    )
    remove_with_home(candidate, home=tmp_path, quarantine_dir=quarantine)

    assert paths["old_log"].read_text(encoding="utf-8") == "protected replacement\n"
    assert not (displaced_source_parent / paths["old_log"].name).exists()
    assert (quarantine / "sentinel").read_text(encoding="utf-8") == "protected\n"
    assert list(displaced_quarantine.iterdir()) == []


def test_directory_child_swap_at_unlink_cannot_escape_root_quarantine(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = make_fixture(tmp_path, include_recent=False)
    candidate = next(
        item
        for item in storage_doctor.discover_candidates(
            tmp_path,
            now_ns=NOW_NS,
            open_checker=clear_open,  # type: ignore[arg-type]
            mount_reader=lambda: frozenset(),
        )
        if item.path == str(paths["build"])
    )
    quarantine = tmp_path / "quarantine"
    quarantine.mkdir()
    original_unlink = storage_doctor.os.unlink
    recreated = False

    def unlink_after_recreating_source(
        path: str | bytes | os.PathLike[str] | os.PathLike[bytes],
        *,
        dir_fd: int | None = None,
    ) -> None:
        nonlocal recreated
        if path == "artifact.o" and dir_fd is not None and not recreated:
            recreated = True
            paths["build"].mkdir(parents=True)
            (paths["build"] / "artifact.o").write_text(
                "protected replacement\n",
                encoding="utf-8",
            )
        original_unlink(path, dir_fd=dir_fd)

    monkeypatch.setattr(storage_doctor.os, "unlink", unlink_after_recreating_source)
    remove_with_home(candidate, home=tmp_path, quarantine_dir=quarantine)

    assert recreated
    assert (paths["build"] / "artifact.o").read_text(encoding="utf-8") == (
        "protected replacement\n"
    )
    assert list(quarantine.iterdir()) == []


def test_directory_tree_drift_after_quarantine_is_left_for_recovery(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = make_fixture(tmp_path, include_recent=False)
    candidate = next(
        item
        for item in storage_doctor.discover_candidates(
            tmp_path,
            now_ns=NOW_NS,
            open_checker=clear_open,  # type: ignore[arg-type]
            mount_reader=lambda: frozenset(),
        )
        if item.path == str(paths["build"])
    )
    quarantine = tmp_path / "quarantine"
    quarantine.mkdir()
    real_rename = storage_doctor._renameat_noreplace

    def rename_then_mutate_tree(
        source_descriptor: int,
        source_name: str,
        destination_descriptor: int,
        destination_name: str,
    ) -> None:
        real_rename(
            source_descriptor,
            source_name,
            destination_descriptor,
            destination_name,
        )
        quarantined_file = quarantine / destination_name / "artifact.o"
        quarantined_file.write_text("changed after rename\n", encoding="utf-8")

    monkeypatch.setattr(
        storage_doctor,
        "_renameat_noreplace",
        rename_then_mutate_tree,
    )
    with pytest.raises(StorageSafetyError, match="tree changed; recover it at"):
        remove_with_home(candidate, home=tmp_path, quarantine_dir=quarantine)

    recovery = quarantine / f"{candidate.candidate_id}.directory"
    assert not paths["build"].exists()
    assert (recovery / "artifact.o").read_text(encoding="utf-8") == ("changed after rename\n")


def test_quarantine_rename_never_replaces_planted_target(tmp_path: Path) -> None:
    paths = make_fixture(tmp_path, include_recent=False)
    candidate = next(
        item
        for item in storage_doctor.discover_candidates(
            tmp_path,
            now_ns=NOW_NS,
            open_checker=clear_open,  # type: ignore[arg-type]
            mount_reader=lambda: frozenset(),
        )
        if item.path == str(paths["old_log"])
    )
    quarantine = tmp_path / "quarantine"
    quarantine.mkdir()
    planted = quarantine / f"{candidate.candidate_id}.file"
    planted.write_text("planted\n", encoding="utf-8")

    with pytest.raises(OSError):
        remove_with_home(candidate, home=tmp_path, quarantine_dir=quarantine)
    assert paths["old_log"].read_text(encoding="utf-8") == "old\n"
    assert planted.read_text(encoding="utf-8") == "planted\n"


def test_cross_device_quarantine_failure_keeps_source(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    paths = make_fixture(tmp_path, include_recent=False)
    candidate = next(
        item
        for item in storage_doctor.discover_candidates(
            tmp_path,
            now_ns=NOW_NS,
            open_checker=clear_open,  # type: ignore[arg-type]
            mount_reader=lambda: frozenset(),
        )
        if item.path == str(paths["old_log"])
    )
    quarantine = tmp_path / "quarantine"
    quarantine.mkdir()

    def fail_cross_device(
        _source_descriptor: int,
        _source_name: str,
        _destination_descriptor: int,
        destination_name: str,
    ) -> None:
        raise OSError(errno.EXDEV, "cross-device rename", destination_name)

    monkeypatch.setattr(storage_doctor, "_renameat_noreplace", fail_cross_device)
    with pytest.raises(OSError) as raised:
        remove_with_home(candidate, home=tmp_path, quarantine_dir=quarantine)
    assert raised.value.errno == errno.EXDEV
    assert paths["old_log"].read_text(encoding="utf-8") == "old\n"
    assert list(quarantine.iterdir()) == []
