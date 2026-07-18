from __future__ import annotations

import errno
import json
import os
import stat
import threading
import time
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

import afs.layout_migration as migration
from afs.context_layout import (
    LAYOUT_FILE,
    MigrationPlan,
    MigrationRetention,
    build_migration_plan,
    detect_layout_version,
    write_manifest,
)
from afs.human_provenance import HumanAuthorization, _broker_for_reader


def _source(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    (source / "history").mkdir(parents=True)
    (source / "history" / "event.jsonl").write_text('{"kind":"test"}\n', encoding="utf-8")
    (source / "scratchpad").mkdir()
    (source / "scratchpad" / "run.sh").write_text("#!/bin/sh\ntrue\n", encoding="utf-8")
    os.chmod(source / "scratchpad" / "run.sh", 0o775)
    return source


def _authorization(
    plan: MigrationPlan,
    rationale: str = "I reviewed the separate copy",
) -> HumanAuthorization:
    scope = migration.layout_migration_authorization_scope(
        plan.plan_sha256,
        plan.transaction_id,
        rationale,
    )
    authorization = _broker_for_reader(lambda _prompt: "MIGRATE").confirm_token(
        "MIGRATE",
        "confirm",
        scope=scope,
    )
    assert authorization is not None
    return authorization


def test_preflight_is_strictly_read_only(tmp_path: Path) -> None:
    source = _source(tmp_path)
    destination = tmp_path / "destination"
    plan = build_migration_plan(source, destination)
    before = {path.name for path in tmp_path.iterdir()}

    result = migration.preflight_migration(plan)

    assert result.status == "ready"
    assert result.to_dict()["operations"]
    assert {path.name for path in tmp_path.iterdir()} == before
    assert not destination.exists()
    assert not (tmp_path / ".destination.migration.lock").exists()


def test_migration_fails_closed_without_private_destination_permissions(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(tmp_path)
    destination = tmp_path / "destination"
    plan = build_migration_plan(source, destination)
    monkeypatch.setattr(
        migration,
        "_supports_private_destination_permissions",
        lambda: False,
    )

    with pytest.raises(migration.MigrationPreflightError, match="private destination DACL"):
        migration.preflight_migration(plan)
    with pytest.raises(migration.MigrationPreflightError, match="private destination DACL"):
        migration.apply_migration(
            plan,
            rationale="I reviewed the unsupported-platform test",
            authorization=_authorization(plan, "I reviewed the unsupported-platform test"),
        )
    assert not destination.exists()


def test_apply_copies_to_private_v2_and_never_modifies_source(tmp_path: Path) -> None:
    source = _source(tmp_path)
    destination = tmp_path / "destination"
    plan = build_migration_plan(source, destination)
    before = plan.source_fingerprint
    rationale = "I reviewed the separate copy"

    result = migration.apply_migration(
        plan,
        rationale=rationale,
        authorization=_authorization(plan, rationale),
    )

    assert result.status == "applied"
    assert result.to_dict()["source_unchanged"] is True
    assert detect_layout_version(destination) == 2
    assert (destination / "history/common/event.jsonl").read_text(encoding="utf-8") == (
        source / "history/event.jsonl"
    ).read_text(encoding="utf-8")
    copied_script = destination / "scratchpad/common/run.sh"
    assert stat.S_IMODE(copied_script.stat().st_mode) == 0o700
    assert stat.S_IMODE(destination.stat().st_mode) == 0o700
    assert stat.S_IMODE(result.receipt_path.stat().st_mode) == 0o600
    assert migration._tree_fingerprint(source)[0] == before
    receipt = json.loads(result.receipt_path.read_text(encoding="utf-8"))
    assert receipt["rationale"] == rationale
    assert receipt["candidate_sha256"]


def test_preflight_rejects_in_place_nested_existing_and_dangling_destinations(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)
    destination = tmp_path / "destination"
    plan = build_migration_plan(source, destination)

    in_place = replace(
        plan,
        destination_root=str(source),
        plan_sha256="",
    ).with_canonical_hash()
    with pytest.raises(migration.MigrationPreflightError):
        migration.preflight_migration(in_place)

    nested = replace(
        plan,
        destination_root=str(source / "nested"),
        plan_sha256="",
    ).with_canonical_hash()
    with pytest.raises(migration.MigrationPreflightError, match="contain"):
        migration.preflight_migration(nested)

    destination.mkdir()
    with pytest.raises(migration.MigrationPreflightError, match="already exists"):
        migration.preflight_migration(plan)
    destination.rmdir()
    destination.symlink_to(tmp_path / "missing-target")
    with pytest.raises(migration.MigrationPreflightError, match="symbolic link|already exists"):
        migration.preflight_migration(plan)


def test_preflight_rejects_source_and_destination_parent_links(tmp_path: Path) -> None:
    source = _source(tmp_path)
    destination = tmp_path / "destination"
    plan = build_migration_plan(source, destination)

    source_alias = tmp_path / "source-alias"
    source_alias.symlink_to(source, target_is_directory=True)
    linked_source_plan = replace(
        plan,
        source_root=str(source_alias),
        plan_sha256="",
    ).with_canonical_hash()
    with pytest.raises(migration.MigrationPreflightError, match="symbolic link"):
        migration.preflight_migration(linked_source_plan)

    real_parent = tmp_path / "real-parent"
    real_parent.mkdir()
    parent_alias = tmp_path / "parent-alias"
    parent_alias.symlink_to(real_parent, target_is_directory=True)
    linked_destination_plan = replace(
        plan,
        destination_root=str(parent_alias / "destination"),
        operations=tuple(
            replace(
                operation,
                destination=str(
                    parent_alias
                    / "destination"
                    / Path(operation.destination).relative_to(destination)
                ),
            )
            for operation in plan.operations
        ),
        plan_sha256="",
    ).with_canonical_hash()
    with pytest.raises(migration.MigrationPreflightError, match="symbolic link"):
        migration.preflight_migration(linked_destination_plan)


def test_preflight_rejects_nested_links_hardlinks_and_special_files(tmp_path: Path) -> None:
    source = _source(tmp_path)
    destination = tmp_path / "destination"
    plan = build_migration_plan(source, destination)
    nested_link = source / "history" / "outside"
    nested_link.symlink_to(tmp_path, target_is_directory=True)
    with pytest.raises(migration.MigrationPreflightError, match="symbolic link"):
        migration.preflight_migration(plan)

    nested_link.unlink()
    plan = build_migration_plan(source, destination)
    hardlink = source / "history" / "event-copy.jsonl"
    os.link(source / "history" / "event.jsonl", hardlink)
    with pytest.raises(migration.MigrationPreflightError, match="hard-linked"):
        migration.preflight_migration(plan)
    hardlink.unlink()

    plan = build_migration_plan(source, destination)
    regular = source / "history" / "event.jsonl"
    regular.unlink()
    os.mkfifo(regular)
    with pytest.raises(migration.MigrationPreflightError, match="special"):
        migration.preflight_migration(plan)


@pytest.mark.parametrize("name", ["CON.txt", "bad:name", "control\x01name", "format\u200ename"])
def test_planning_rejects_nonportable_names(tmp_path: Path, name: str) -> None:
    source = _source(tmp_path)
    (source / "history" / name).write_text("unsafe", encoding="utf-8")
    with pytest.raises(ValueError, match="path name|character|reserved"):
        build_migration_plan(source, tmp_path / "destination")


def test_preflight_rejects_casefold_collisions(tmp_path: Path) -> None:
    source = _source(tmp_path)
    (source / "history" / "A.txt").write_text("one", encoding="utf-8")
    (source / "history" / "Ａ.txt").write_text("two", encoding="utf-8")
    plan = build_migration_plan(source, tmp_path / "destination")

    with pytest.raises(migration.MigrationPreflightError, match="collision"):
        migration.preflight_migration(plan)


def test_preflight_rejects_insufficient_destination_space(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(tmp_path)
    plan = build_migration_plan(source, tmp_path / "destination")
    monkeypatch.setattr(migration.shutil, "disk_usage", lambda _path: SimpleNamespace(free=1))

    with pytest.raises(migration.MigrationPreflightError, match="insufficient"):
        migration.preflight_migration(plan)


def test_loaded_plan_rejects_hash_tampering(tmp_path: Path) -> None:
    source = _source(tmp_path)
    plan_path = tmp_path / "plan.json"
    write_manifest(plan_path, build_migration_plan(source, tmp_path / "destination"))
    payload = json.loads(plan_path.read_text(encoding="utf-8"))
    payload["source_bytes"] += 1
    plan_path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(migration.MigrationPreflightError, match="canonical SHA-256"):
        migration.preflight_migration(plan_path)


def test_preflight_recomputes_exact_operation_semantics(tmp_path: Path) -> None:
    source = _source(tmp_path)
    destination = tmp_path / "destination"
    plan = build_migration_plan(source, destination)
    first = plan.operations[0]
    tampered = replace(
        plan,
        operations=(
            replace(first, destination=str(destination / "memory/common/not-history")),
            *plan.operations[1:],
        ),
        plan_sha256="",
    ).with_canonical_hash()

    with pytest.raises(migration.MigrationPreflightError, match="semantically tampered"):
        migration.preflight_migration(tampered)


def test_explicit_unknown_mapping_is_recomputed_and_copied(tmp_path: Path) -> None:
    source = _source(tmp_path)
    (source / "research").mkdir()
    (source / "research" / "finding.md").write_text("evidence", encoding="utf-8")
    destination = tmp_path / "destination"
    plan = build_migration_plan(
        source,
        destination,
        explicit_mappings={"research": "knowledge/common/research"},
    )

    migration.preflight_migration(plan)
    migration.apply_migration(
        plan,
        rationale="I reviewed the explicit mapping",
        authorization=_authorization(plan, "I reviewed the explicit mapping"),
    )

    assert (destination / "knowledge/common/research/finding.md").read_text() == "evidence"


def test_empty_source_creates_verified_scaffold_with_zero_operations(tmp_path: Path) -> None:
    source = tmp_path / "empty-source"
    source.mkdir()
    destination = tmp_path / "destination"
    plan = build_migration_plan(source, destination)
    assert plan.ready is True
    assert plan.operations == ()

    preview = migration.preflight_migration(plan)
    assert preview.mapping_count == 0
    result = migration.apply_migration(
        plan,
        rationale="I reviewed the empty source scaffold",
        authorization=_authorization(plan, "I reviewed the empty source scaffold"),
    )

    assert result.status == "applied"
    assert detect_layout_version(destination) == 2
    assert migration.preflight_migration(plan).status == "already_applied"


def test_disjoint_nested_builtin_destinations_merge_and_retry(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "memory").mkdir(parents=True)
    (source / "memory" / "note.md").write_text("memory", encoding="utf-8")
    (source / "missions").mkdir()
    (source / "missions" / "active.json").write_text("mission", encoding="utf-8")
    destination = tmp_path / "destination"
    plan = build_migration_plan(source, destination)

    migration.preflight_migration(plan)
    result = migration.apply_migration(
        plan,
        rationale="I reviewed the disjoint built-in merge",
        authorization=_authorization(plan, "I reviewed the disjoint built-in merge"),
    )

    assert result.status == "applied"
    assert (destination / "memory/common/note.md").read_text() == "memory"
    assert (destination / "memory/common/missions/active.json").read_text() == "mission"
    assert migration.preflight_migration(plan).status == "already_applied"


def test_overlapping_builtin_content_collision_is_rejected(tmp_path: Path) -> None:
    source = tmp_path / "source"
    (source / "memory" / "missions").mkdir(parents=True)
    (source / "memory" / "missions" / "from-memory.json").write_text("one")
    (source / "missions").mkdir()
    (source / "missions" / "from-top.json").write_text("two")
    plan = build_migration_plan(source, tmp_path / "destination")

    with pytest.raises(migration.MigrationPreflightError, match="collision"):
        migration.preflight_migration(plan)


def test_apply_requires_exact_fresh_authorization_and_safe_rationale(tmp_path: Path) -> None:
    source = _source(tmp_path)
    plan = build_migration_plan(source, tmp_path / "destination")
    wrong = _authorization(plan, "different rationale")
    with pytest.raises(migration.MigrationApplyError, match="fresh HumanDecisionBroker"):
        migration.apply_migration(
            plan,
            rationale="reviewed rationale",
            authorization=wrong,
        )
    assert not Path(plan.destination_root).exists()

    with pytest.raises(ValueError, match="control"):
        migration.layout_migration_authorization_scope(
            plan.plan_sha256,
            plan.transaction_id,
            "unsafe\nreason",
        )
    with pytest.raises(ValueError, match="exceeds"):
        migration.layout_migration_authorization_scope(
            plan.plan_sha256,
            plan.transaction_id,
            "x" * 4097,
        )


def test_copy_fault_quarantines_partial_destination_without_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(tmp_path)
    destination = tmp_path / "destination"
    plan = build_migration_plan(source, destination)

    def fail_copy(_source: Path, _destination: Path) -> tuple[str, int]:
        raise OSError("injected copy failure")

    monkeypatch.setattr(migration, "_copy_regular_file", fail_copy)
    with pytest.raises(migration.MigrationApplyError) as failure:
        migration.apply_migration(
            plan,
            rationale="I reviewed the fault test",
            authorization=_authorization(plan, "I reviewed the fault test"),
        )

    assert not destination.exists()
    assert failure.value.failed_destination is not None
    assert failure.value.failed_destination.is_dir()
    assert not (failure.value.failed_destination / ".afs" / LAYOUT_FILE).exists()


def test_marker_publication_failure_quarantines_unmarked_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(tmp_path)
    destination = tmp_path / "destination"
    plan = build_migration_plan(source, destination)
    real_atomic = migration.atomic_write_text

    def fail_marker(
        path: Path,
        text: str,
        *,
        encoding: str = "utf-8",
        mode: int | None = None,
        durable: bool = False,
    ) -> None:
        if path.name == LAYOUT_FILE:
            raise OSError("injected marker publication failure")
        real_atomic(path, text, encoding=encoding, mode=mode, durable=durable)

    monkeypatch.setattr(migration, "atomic_write_text", fail_marker)
    with pytest.raises(migration.MigrationApplyError) as failure:
        migration.apply_migration(
            plan,
            rationale="I reviewed the audit test",
            authorization=_authorization(plan, "I reviewed the audit test"),
        )

    assert not destination.exists()
    assert failure.value.failed_destination is not None
    assert not (failure.value.failed_destination / ".afs" / LAYOUT_FILE).exists()


def test_marker_revocation_failure_reports_marked_candidate_in_place(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(tmp_path)
    destination = tmp_path / "destination"
    plan = build_migration_plan(source, destination)
    marker = destination / ".afs" / LAYOUT_FILE
    real_atomic = migration.atomic_write_text
    real_unlink = Path.unlink

    def publish_then_fail(
        path: Path,
        text: str,
        *,
        encoding: str = "utf-8",
        mode: int | None = None,
        durable: bool = False,
    ) -> None:
        real_atomic(path, text, encoding=encoding, mode=mode, durable=durable)
        if path == marker:
            raise OSError("injected post-publication failure")

    def refuse_marker_unlink(path: Path, missing_ok: bool = False) -> None:
        if path == marker:
            raise PermissionError("injected marker revocation failure")
        real_unlink(path, missing_ok=missing_ok)

    monkeypatch.setattr(migration, "atomic_write_text", publish_then_fail)
    monkeypatch.setattr(Path, "unlink", refuse_marker_unlink)
    with pytest.raises(migration.MigrationApplyError, match="revocation failed") as failure:
        migration.apply_migration(
            plan,
            rationale="I reviewed marker revocation handling",
            authorization=_authorization(plan, "I reviewed marker revocation handling"),
        )

    assert failure.value.failed_destination == destination
    assert marker.exists()
    assert destination.exists()


def test_failed_quarantine_rename_reports_original_partial_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(tmp_path)
    destination = tmp_path / "destination"
    plan = build_migration_plan(source, destination)
    monkeypatch.setattr(
        migration,
        "_copy_regular_file",
        lambda _source, _destination: (_ for _ in ()).throw(OSError("copy failed")),
    )
    monkeypatch.setattr(
        migration,
        "_quarantine_destination",
        lambda *_args: (None, False),
    )

    with pytest.raises(migration.MigrationApplyError, match="rename failed") as failure:
        migration.apply_migration(
            plan,
            rationale="I reviewed quarantine failure handling",
            authorization=_authorization(plan, "I reviewed quarantine failure handling"),
        )

    assert failure.value.failed_destination == destination
    assert destination.exists()
    assert not (destination / ".afs" / LAYOUT_FILE).exists()


def test_quarantine_parent_fsync_failure_reports_retained_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(tmp_path)
    destination = tmp_path / "destination"
    plan = build_migration_plan(source, destination)
    real_fsync_directory = migration.fsync_directory

    def fail_copy(_source: Path, _destination: Path) -> tuple[str, int]:
        raise OSError("injected copy failure")

    def fail_quarantine_sync(path: Path) -> None:
        quarantined = list(tmp_path.glob("destination.failed-layout_*"))
        if path == destination.parent and not destination.exists() and quarantined:
            raise OSError("injected quarantine durability failure")
        real_fsync_directory(path)

    monkeypatch.setattr(migration, "_copy_regular_file", fail_copy)
    monkeypatch.setattr(migration, "fsync_directory", fail_quarantine_sync)

    with pytest.raises(
        migration.MigrationApplyError,
        match="parent-directory durability sync failed",
    ) as failure:
        migration.apply_migration(
            plan,
            rationale="I reviewed quarantine durability handling",
            authorization=_authorization(plan, "I reviewed quarantine durability handling"),
        )

    assert not destination.exists()
    assert failure.value.failed_destination is not None
    assert failure.value.failed_destination.is_dir()
    assert not (failure.value.failed_destination / ".afs" / LAYOUT_FILE).exists()


def test_interrupt_still_quarantines_partial_candidate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(tmp_path)
    destination = tmp_path / "destination"
    plan = build_migration_plan(source, destination)

    def interrupt(_source: Path, _destination: Path) -> tuple[str, int]:
        raise KeyboardInterrupt

    monkeypatch.setattr(migration, "_copy_regular_file", interrupt)
    with pytest.raises(KeyboardInterrupt):
        migration.apply_migration(
            plan,
            rationale="I reviewed interrupt cleanup",
            authorization=_authorization(plan, "I reviewed interrupt cleanup"),
        )

    assert not destination.exists()
    failed = list(tmp_path.glob("destination.failed-layout_*"))
    assert len(failed) == 1
    assert not (failed[0] / ".afs" / LAYOUT_FILE).exists()


def test_final_source_mode_change_before_marker_is_detected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(tmp_path)
    destination = tmp_path / "destination"
    plan = build_migration_plan(source, destination)
    real_digest = migration._verified_operations_digest
    calls = 0

    def mutate_after_final_operation_check(
        plan_value: object,
        source_root: Path,
        destination_root: Path,
    ) -> str:
        nonlocal calls
        value = real_digest(plan_value, source_root, destination_root)
        calls += 1
        if calls == 2:
            os.chmod(source / "history", 0o711)
        return value

    monkeypatch.setattr(
        migration,
        "_verified_operations_digest",
        mutate_after_final_operation_check,
    )
    with pytest.raises(migration.MigrationApplyError, match="source changed") as failure:
        migration.apply_migration(
            plan,
            rationale="I reviewed final source drift detection",
            authorization=_authorization(plan, "I reviewed final source drift detection"),
        )

    assert not destination.exists()
    assert failure.value.failed_destination is not None
    assert not (failure.value.failed_destination / ".afs" / LAYOUT_FILE).exists()


def test_completed_transaction_is_idempotent_but_candidate_tamper_is_not(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)
    destination = tmp_path / "destination"
    plan = build_migration_plan(source, destination)
    rationale = "I reviewed idempotency"
    authorization = _authorization(plan, rationale)
    first = migration.apply_migration(plan, rationale=rationale, authorization=authorization)

    assert migration.preflight_migration(plan).status == "already_applied"
    assert (
        migration.apply_migration(
            plan,
            rationale=rationale,
            authorization=authorization,
        ).status
        == "already_applied"
    )

    copied = destination / "history/common/event.jsonl"
    original = copied.read_text(encoding="utf-8")
    copied.write_text("tampered", encoding="utf-8")
    with pytest.raises(migration.MigrationPreflightError, match="already exists"):
        migration.preflight_migration(plan)
    copied.write_text(original, encoding="utf-8")
    (destination / "human" / "unexpected.txt").write_text("extra", encoding="utf-8")
    with pytest.raises(migration.MigrationPreflightError, match="candidate"):
        migration.preflight_migration(plan)
    assert first.receipt_path.exists()


def test_completed_recognition_rejects_receipt_and_marker_tampering(tmp_path: Path) -> None:
    source = _source(tmp_path)
    destination = tmp_path / "destination"
    plan = build_migration_plan(source, destination)
    result = migration.apply_migration(
        plan,
        rationale="I reviewed integrity metadata",
        authorization=_authorization(plan, "I reviewed integrity metadata"),
    )
    receipt_original = result.receipt_path.read_bytes()
    marker = destination / ".afs" / LAYOUT_FILE
    marker_original = marker.read_bytes()

    receipt = json.loads(receipt_original)
    receipt["rationale"] = "altered after authorization"
    result.receipt_path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(migration.MigrationPreflightError, match="candidate"):
        migration.preflight_migration(plan)

    result.receipt_path.write_bytes(receipt_original)
    marker.write_text(
        marker_original.decode("utf-8").replace(
            'namespace = "central"',
            'namespace = "altered"',
        ),
        encoding="utf-8",
    )
    with pytest.raises(migration.MigrationPreflightError, match="candidate"):
        migration.preflight_migration(plan)


@pytest.mark.parametrize(
    "relative",
    [
        Path("README.md"),
        Path(".afs") / LAYOUT_FILE,
        Path(".afs/migrations") / "TRANSACTION" / "receipt.json",
    ],
)
def test_completed_recognition_rejects_hardlinked_metadata(
    tmp_path: Path,
    relative: Path,
) -> None:
    source = _source(tmp_path)
    destination = tmp_path / "destination"
    plan = build_migration_plan(source, destination)
    result = migration.apply_migration(
        plan,
        rationale="I reviewed metadata link safety",
        authorization=_authorization(plan, "I reviewed metadata link safety"),
    )
    artifact = result.receipt_path if "TRANSACTION" in relative.parts else destination / relative
    os.link(artifact, tmp_path / f"hardlink-{artifact.name}")

    with pytest.raises(migration.MigrationPreflightError, match="candidate"):
        migration.preflight_migration(plan)


def test_completed_recognition_rejects_fifo_receipt_without_opening_it(tmp_path: Path) -> None:
    source = _source(tmp_path)
    destination = tmp_path / "destination"
    plan = build_migration_plan(source, destination)
    result = migration.apply_migration(
        plan,
        rationale="I reviewed bounded receipt reads",
        authorization=_authorization(plan, "I reviewed bounded receipt reads"),
    )
    result.receipt_path.unlink()
    os.mkfifo(result.receipt_path)

    with pytest.raises(migration.MigrationPreflightError, match="candidate"):
        migration.preflight_migration(plan)


def test_permission_contention_retries_cooperative_lock(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    destination = tmp_path / "destination"
    attempts = 0

    def contend_once(_descriptor: int) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise PermissionError(errno.EACCES, "simulated Windows lock contention")

    monkeypatch.setattr(migration, "_lock_descriptor", contend_once)
    monkeypatch.setattr(migration, "_unlock_descriptor", lambda _descriptor: None)
    with migration._sibling_lock(destination, timeout=1):
        pass

    assert attempts == 2


def test_inventory_oserror_is_a_preflight_block(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(tmp_path)
    plan = build_migration_plan(source, tmp_path / "destination")
    monkeypatch.setattr(
        migration.context_layout,
        "_tree_fingerprint",
        lambda _root: (_ for _ in ()).throw(OSError("inventory unavailable")),
    )

    with pytest.raises(migration.MigrationPreflightError, match="inventory"):
        migration.preflight_migration(plan)


def test_copy_uses_path_chmod_fallback_without_fchmod(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(tmp_path)
    destination = tmp_path / "destination"
    plan = build_migration_plan(source, destination)
    monkeypatch.delattr(migration.os, "fchmod")

    result = migration.apply_migration(
        plan,
        rationale="I reviewed the cross-platform chmod fallback",
        authorization=_authorization(plan, "I reviewed the cross-platform chmod fallback"),
    )

    assert result.status == "applied"
    assert migration.preflight_migration(plan).status == "already_applied"


def test_copied_entries_and_directories_are_fsynced_before_marker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(tmp_path)
    destination = tmp_path / "destination"
    plan = build_migration_plan(source, destination)
    events: list[tuple[str, str]] = []
    real_fsync = migration.fsync_directory
    real_atomic = migration.atomic_write_text

    def tracked_fsync(path: Path) -> None:
        events.append(("fsync", str(path)))
        real_fsync(path)

    def tracked_atomic(
        path: Path,
        text: str,
        *,
        encoding: str = "utf-8",
        mode: int | None = None,
        durable: bool = False,
    ) -> None:
        if path.name == LAYOUT_FILE:
            events.append(("marker", str(path)))
        real_atomic(path, text, encoding=encoding, mode=mode, durable=durable)

    monkeypatch.setattr(migration, "fsync_directory", tracked_fsync)
    monkeypatch.setattr(migration, "atomic_write_text", tracked_atomic)
    migration.apply_migration(
        plan,
        rationale="I reviewed durability ordering",
        authorization=_authorization(plan, "I reviewed durability ordering"),
    )

    marker_index = next(index for index, event in enumerate(events) if event[0] == "marker")
    required_directories = {
        str(tmp_path),
        str(destination),
        str(destination / "history"),
        str(destination / "history/common"),
        str(destination / "scratchpad/common"),
    }
    fsynced_before_marker = {value for kind, value in events[:marker_index] if kind == "fsync"}
    assert required_directories <= fsynced_before_marker


def test_concurrent_apply_serializes_and_second_observes_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(tmp_path)
    destination = tmp_path / "destination"
    plan = build_migration_plan(source, destination)
    original_copy = migration._copy_regular_file
    copy_started = threading.Event()
    release_copy = threading.Event()
    first_call = True

    def slow_first_copy(source_path: Path, destination_path: Path) -> tuple[str, int]:
        nonlocal first_call
        if first_call:
            first_call = False
            copy_started.set()
            assert release_copy.wait(5)
        return original_copy(source_path, destination_path)

    monkeypatch.setattr(migration, "_copy_regular_file", slow_first_copy)
    results: list[str] = []
    failures: list[BaseException] = []

    def run(rationale: str) -> None:
        try:
            result = migration.apply_migration(
                plan,
                rationale=rationale,
                authorization=_authorization(plan, rationale),
                lock_timeout=5,
            )
            results.append(result.status)
        except (AssertionError, OSError, RuntimeError, ValueError) as exc:
            failures.append(exc)

    first = threading.Thread(target=run, args=("I reviewed concurrent one",))
    second = threading.Thread(target=run, args=("I reviewed concurrent two",))
    first.start()
    assert copy_started.wait(5)
    second.start()
    time.sleep(0.1)
    release_copy.set()
    first.join(5)
    second.join(5)

    assert not failures
    assert sorted(results) == ["already_applied", "applied"]
    assert detect_layout_version(destination) == 2


def test_schema_v3_apply_excludes_retained_sources_and_paths_and_is_idempotent(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)
    destination = tmp_path / "destination"
    outside = tmp_path / "outside-skills"
    outside.mkdir()
    (outside / "canary.md").write_text("must not be copied", encoding="utf-8")
    (source / "models").mkdir()
    (source / "models" / "model.bin").write_bytes(b"legacy-model")
    (source / "health").mkdir()
    (source / "health" / "current.json").write_text("{}", encoding="utf-8")
    (source / "health" / "archive:2026.json").write_text("old", encoding="utf-8")
    (source / "knowledge").mkdir()
    try:
        (source / "knowledge" / "skills").symlink_to(outside, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")

    plan = build_migration_plan(
        source,
        destination,
        retained_sources={
            "models": "Model data requires a dedicated importer before producer cutover.",
        },
        retained_paths={
            "health/archive:2026.json": "The historical name is not portable to v2.",
            "knowledge/skills": "The legacy external skill link remains available from v1.",
        },
    )

    preview = migration.preflight_migration(plan)
    rationale = "I reviewed the source-retained candidate"
    result = migration.apply_migration(
        plan,
        rationale=rationale,
        authorization=_authorization(plan, rationale),
    )

    assert plan.schema_version == 3
    assert preview.source_file_count == plan.source_file_count
    assert preview.source_bytes == plan.source_bytes
    assert preview.copy_file_count == plan.copy_file_count
    assert preview.copy_bytes == plan.copy_bytes
    assert (destination / ".afs/health/current.json").read_text(encoding="utf-8") == "{}"
    assert not (destination / ".afs/health/archive:2026.json").exists()
    assert not (destination / "knowledge/common/skills").exists()
    assert not (destination / "models").exists()
    assert not any(path.name == "canary.md" for path in destination.rglob("*"))

    receipt = json.loads(result.receipt_path.read_text(encoding="utf-8"))
    assert receipt["schema_version"] == 2
    assert receipt["plan_schema_version"] == 3
    assert receipt["copy_file_count"] == plan.copy_file_count
    assert receipt["copy_bytes"] == plan.copy_bytes
    assert receipt["retained_sources"] == [item.to_dict() for item in plan.retained_sources]
    assert receipt["retained_paths"] == [item.to_dict() for item in plan.retained_paths]
    assert migration.preflight_migration(plan).status == "already_applied"
    assert (
        migration.apply_migration(
            plan,
            rationale=rationale,
            authorization=_authorization(plan, rationale),
        ).status
        == "already_applied"
    )


@pytest.mark.parametrize("mutation", ["retained_content", "retained_symlink"])
def test_schema_v3_retained_source_mutation_invalidates_plan(
    tmp_path: Path,
    mutation: str,
) -> None:
    source = _source(tmp_path)
    destination = tmp_path / "destination"
    outside_one = tmp_path / "outside-one"
    outside_two = tmp_path / "outside-two"
    outside_one.mkdir()
    outside_two.mkdir()
    (source / "models").mkdir()
    retained_file = source / "models" / "model.bin"
    retained_file.write_bytes(b"original")
    (source / "knowledge").mkdir()
    retained_link = source / "knowledge" / "skills"
    try:
        retained_link.symlink_to(outside_one, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlinks unavailable: {exc}")
    plan = build_migration_plan(
        source,
        destination,
        retained_sources={"models": "Model data requires a dedicated importer."},
        retained_paths={"knowledge/skills": "Retain the legacy external skill link."},
    )

    if mutation == "retained_content":
        retained_file.write_bytes(b"changed")
    else:
        retained_link.unlink()
        retained_link.symlink_to(outside_two, target_is_directory=True)

    with pytest.raises(migration.MigrationPreflightError, match="source changed"):
        migration.preflight_migration(plan)


def test_schema_v3_capacity_uses_copy_bytes_not_retained_source_bytes(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _source(tmp_path)
    destination = tmp_path / "destination"
    (source / "models").mkdir()
    (source / "models" / "model.bin").write_bytes(b"x" * 4096)
    plan = build_migration_plan(
        source,
        destination,
        retained_sources={"models": "Model data requires a dedicated importer."},
    )
    monkeypatch.setattr(migration, "_SPACE_FLOOR_BYTES", 1)
    copy_required = plan.copy_bytes + max(1, plan.copy_bytes // 20)
    whole_source_required = plan.source_bytes + max(1, plan.source_bytes // 20)
    assert copy_required < whole_source_required
    monkeypatch.setattr(
        migration.shutil,
        "disk_usage",
        lambda _path: SimpleNamespace(free=copy_required),
    )

    preview = migration.preflight_migration(plan)

    assert preview.required_bytes == copy_required
    assert preview.available_bytes == copy_required
    assert preview.copy_bytes == plan.copy_bytes
    assert preview.source_bytes == plan.source_bytes


def test_schema_v3_apply_rejects_case_mismatched_retention_spelling(
    tmp_path: Path,
) -> None:
    source = _source(tmp_path)
    destination = tmp_path / "destination"
    retained = source / "knowledge" / "lower"
    retained.mkdir(parents=True)
    (retained / "secret.txt").write_text("source-only", encoding="utf-8")
    reason = "This subtree must remain only in the v1 source."
    plan = build_migration_plan(
        source,
        destination,
        retained_paths={"knowledge/lower": reason},
    )
    mismatched = replace(
        plan,
        retained_paths=(MigrationRetention("knowledge/LOWER", reason),),
        plan_sha256="",
    ).with_canonical_hash()

    with pytest.raises(
        migration.MigrationPreflightError,
        match="exact on-disk spelling|does not exist",
    ):
        migration.apply_migration(
            mismatched,
            rationale="I reviewed the case-spelling regression",
            authorization=_authorization(
                mismatched,
                "I reviewed the case-spelling regression",
            ),
        )
    assert not destination.exists()
