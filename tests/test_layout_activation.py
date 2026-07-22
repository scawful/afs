from __future__ import annotations

import json
import os
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path

import pytest

import afs.layout_activation as activation
import afs.layout_migration as migration
from afs.context_layout import (
    LAYOUT_FILE,
    MigrationPlan,
    build_migration_plan,
    detect_layout_version,
)
from afs.human_provenance import HumanAuthorization, _broker_for_reader


@dataclass(frozen=True)
class _PreparedLayout:
    source: Path
    candidate: Path
    state_dir: Path
    plan: MigrationPlan
    migration_receipt: Path
    migration_receipt_bytes: bytes


def _source(tmp_path: Path) -> Path:
    source = tmp_path / "context"
    (source / "history").mkdir(parents=True)
    (source / "history" / "event.jsonl").write_text(
        '{"kind":"test"}\n',
        encoding="utf-8",
    )
    (source / "scratchpad").mkdir()
    (source / "scratchpad" / "run.sh").write_text(
        "#!/bin/sh\ntrue\n",
        encoding="utf-8",
    )
    os.chmod(source / "scratchpad" / "run.sh", 0o775)
    return source


def _migration_authorization(
    plan: MigrationPlan,
    rationale: str,
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


def _prepared_layout(
    tmp_path: Path,
    *,
    retained_sources: dict[str, str] | None = None,
) -> _PreparedLayout:
    source = _source(tmp_path)
    candidate = tmp_path / "context-v2-candidate"
    if retained_sources:
        for relative in retained_sources:
            retained = source / relative
            retained.mkdir(parents=True)
            (retained / "preserved.txt").write_text("source only\n", encoding="utf-8")
    plan = build_migration_plan(
        source,
        candidate,
        retained_sources=retained_sources,
    )
    rationale = "Create a verified candidate for activation tests"
    migrated = migration.apply_migration(
        plan,
        rationale=rationale,
        authorization=_migration_authorization(plan, rationale),
    )
    return _PreparedLayout(
        source=source,
        candidate=candidate,
        state_dir=tmp_path / "activation-state",
        plan=plan,
        migration_receipt=migrated.receipt_path,
        migration_receipt_bytes=migrated.receipt_path.read_bytes(),
    )


def _authorization(scope: str, token: str) -> HumanAuthorization:
    authorization = _broker_for_reader(lambda _prompt: token).confirm_token(
        token,
        "confirm",
        scope=scope,
    )
    assert authorization is not None
    return authorization


def _activation_authorization(
    preflight: activation.ActivationPreflight,
    rationale: str,
) -> HumanAuthorization:
    return _authorization(
        activation.layout_activation_authorization_scope(preflight, rationale),
        "ACTIVATE",
    )


def _rollback_authorization(
    preflight: activation.RollbackPreflight,
    rationale: str,
) -> HumanAuthorization:
    return _authorization(
        activation.layout_rollback_authorization_scope(preflight, rationale),
        "ROLLBACK",
    )


@pytest.fixture
def controlled_exchange(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep unit tests independent of host support for atomic directory exchange."""

    monkeypatch.delenv("AFS_CONTEXT_ROOT", raising=False)
    monkeypatch.setattr(activation, "_assert_quiescent", lambda *_roots: None)

    def exchange(left: Path, right: Path) -> None:
        assert left.parent == right.parent
        temporary = left.parent / f".{left.name}.test-exchange"
        assert not temporary.exists()
        os.replace(left, temporary)
        try:
            os.replace(right, left)
            os.replace(temporary, right)
        except BaseException:
            if not left.exists() and temporary.exists():
                os.replace(temporary, left)
            raise

    monkeypatch.setattr(activation, "_atomic_exchange", exchange)


def test_activate_and_rollback_exchange_roots_without_rewriting_migration_receipt(
    tmp_path: Path,
    controlled_exchange: None,
) -> None:
    prepared = _prepared_layout(tmp_path)
    original_source_fingerprint = migration.tree_fingerprint(prepared.source)

    preview = activation.preflight_activation(
        prepared.plan,
        state_dir=prepared.state_dir,
        configured_context_root=prepared.source,
    )
    assert preview.status == "ready"
    assert not prepared.state_dir.exists(), "preflight must be strictly read-only"

    activation_rationale = "Make the reviewed v2 candidate the stable active context"
    activated = activation.activate_layout(
        prepared.plan,
        state_dir=prepared.state_dir,
        configured_context_root=prepared.source,
        rationale=activation_rationale,
        authorization=_activation_authorization(preview, activation_rationale),
    )

    assert activated.status == "activated"
    assert detect_layout_version(prepared.source) == 2
    assert detect_layout_version(prepared.candidate) == 1
    assert migration.tree_fingerprint(prepared.candidate) == original_source_fingerprint
    active_migration_receipt = (
        prepared.source / ".afs" / "migrations" / prepared.plan.transaction_id / "receipt.json"
    )
    assert active_migration_receipt.read_bytes() == prepared.migration_receipt_bytes
    assert not (prepared.candidate / ".afs" / LAYOUT_FILE).exists(), (
        "the exchanged inactive root must remain v1"
    )

    rollback_preview = activation.preflight_rollback(
        state_dir=prepared.state_dir,
        configured_context_root=prepared.source,
    )
    assert rollback_preview.status == "ready"
    rollback_rationale = "Restore the preserved v1 root after validating rollback"
    rolled_back = activation.rollback_layout(
        state_dir=prepared.state_dir,
        configured_context_root=prepared.source,
        rationale=rollback_rationale,
        authorization=_rollback_authorization(rollback_preview, rollback_rationale),
    )

    assert rolled_back.status == "rolled_back"
    assert detect_layout_version(prepared.source) == 1
    assert detect_layout_version(prepared.candidate) == 2
    assert migration.tree_fingerprint(prepared.source) == original_source_fingerprint
    assert prepared.migration_receipt.read_bytes() == prepared.migration_receipt_bytes


def test_activation_rejects_source_changes_after_candidate_creation(
    tmp_path: Path,
    controlled_exchange: None,
) -> None:
    prepared = _prepared_layout(tmp_path)
    (prepared.source / "history" / "event.jsonl").write_text(
        '{"kind":"changed-after-copy"}\n',
        encoding="utf-8",
    )

    with pytest.raises(activation.ActivationPreflightError, match="source|stale|changed"):
        activation.preflight_activation(
            prepared.plan,
            state_dir=prepared.state_dir,
            configured_context_root=prepared.source,
        )

    assert detect_layout_version(prepared.source) == 1
    assert detect_layout_version(prepared.candidate) == 2
    assert not prepared.state_dir.exists()


def test_activation_blocks_plans_with_source_only_retentions(
    tmp_path: Path,
    controlled_exchange: None,
) -> None:
    prepared = _prepared_layout(
        tmp_path,
        retained_sources={
            "models": "Model data needs a dedicated importer before producer cutover.",
        },
    )

    with pytest.raises(
        activation.ActivationPreflightError,
        match="retained|source-only|exclusion",
    ):
        activation.preflight_activation(
            prepared.plan,
            state_dir=prepared.state_dir,
            configured_context_root=prepared.source,
        )

    assert (prepared.source / "models" / "preserved.txt").read_text(
        encoding="utf-8"
    ) == "source only\n"
    assert detect_layout_version(prepared.source) == 1
    assert detect_layout_version(prepared.candidate) == 2
    assert not prepared.state_dir.exists()


def test_activation_and_rollback_authorizations_are_scope_separated(
    tmp_path: Path,
    controlled_exchange: None,
) -> None:
    prepared = _prepared_layout(tmp_path)
    preview = activation.preflight_activation(
        prepared.plan,
        state_dir=prepared.state_dir,
        configured_context_root=prepared.source,
    )
    activation_rationale = "Activate the exact reviewed roots"
    migration_rationale = "This only authorized candidate creation"

    with pytest.raises(activation.ActivationApplyError, match="fresh|authorization"):
        activation.activate_layout(
            prepared.plan,
            state_dir=prepared.state_dir,
            configured_context_root=prepared.source,
            rationale=activation_rationale,
            authorization=_migration_authorization(prepared.plan, migration_rationale),
        )
    assert detect_layout_version(prepared.source) == 1
    assert detect_layout_version(prepared.candidate) == 2

    activation_result = activation.activate_layout(
        prepared.plan,
        state_dir=prepared.state_dir,
        configured_context_root=prepared.source,
        rationale=activation_rationale,
        authorization=_activation_authorization(preview, activation_rationale),
    )
    assert activation_result.status == "activated"

    rollback_preview = activation.preflight_rollback(
        state_dir=prepared.state_dir,
        configured_context_root=prepared.source,
    )
    rollback_rationale = "Rollback needs its own human decision"
    active_again = activation.preflight_activation(
        prepared.plan,
        state_dir=prepared.state_dir,
        configured_context_root=prepared.source,
    )
    assert active_again.status == "already_active"

    with pytest.raises(activation.RollbackApplyError, match="fresh|authorization"):
        activation.rollback_layout(
            state_dir=prepared.state_dir,
            configured_context_root=prepared.source,
            rationale=rollback_rationale,
            authorization=_activation_authorization(active_again, activation_rationale),
        )
    assert detect_layout_version(prepared.source) == 2
    assert detect_layout_version(prepared.candidate) == 1

    rolled_back = activation.rollback_layout(
        state_dir=prepared.state_dir,
        configured_context_root=prepared.source,
        rationale=rollback_rationale,
        authorization=_rollback_authorization(rollback_preview, rollback_rationale),
    )
    assert rolled_back.status == "rolled_back"


def test_activation_requires_configured_context_to_match_stable_source(
    tmp_path: Path,
    controlled_exchange: None,
) -> None:
    prepared = _prepared_layout(tmp_path)
    unrelated = tmp_path / "other-context"
    unrelated.mkdir()

    with pytest.raises(activation.ActivationPreflightError, match="configured|context root"):
        activation.preflight_activation(
            prepared.plan,
            state_dir=prepared.state_dir,
            configured_context_root=unrelated,
        )

    assert detect_layout_version(prepared.source) == 1
    assert detect_layout_version(prepared.candidate) == 2
    assert not prepared.state_dir.exists()


def test_activation_publishes_private_external_journal_and_receipt(
    tmp_path: Path,
    controlled_exchange: None,
) -> None:
    prepared = _prepared_layout(tmp_path)
    preview = activation.preflight_activation(
        prepared.plan,
        state_dir=prepared.state_dir,
        configured_context_root=prepared.source,
    )
    rationale = "Record an independently recoverable activation decision"

    result = activation.activate_layout(
        prepared.plan,
        state_dir=prepared.state_dir,
        configured_context_root=prepared.source,
        rationale=rationale,
        authorization=_activation_authorization(preview, rationale),
    )

    for artifact in (result.journal_path, result.receipt_path):
        assert artifact.is_relative_to(prepared.state_dir)
        artifact_stat = os.lstat(artifact)
        assert stat.S_ISREG(artifact_stat.st_mode)
        assert artifact_stat.st_nlink == 1
        assert stat.S_IMODE(artifact_stat.st_mode) == 0o600
    assert stat.S_IMODE(prepared.state_dir.stat().st_mode) == 0o700
    assert not result.receipt_path.is_relative_to(prepared.source)
    assert not result.receipt_path.is_relative_to(prepared.candidate)

    journal = json.loads(result.journal_path.read_text(encoding="utf-8"))
    receipt = json.loads(result.receipt_path.read_text(encoding="utf-8"))
    assert journal["migration_transaction_id"] == prepared.plan.transaction_id
    assert journal["status"] == "active"
    assert receipt["migration_transaction_id"] == prepared.plan.transaction_id
    assert receipt["status"] == "active"
    assert receipt["plan_sha256"] == prepared.plan.plan_sha256
    assert receipt["active_root"] == str(prepared.source)
    assert receipt["inactive_root"] == str(prepared.candidate)
    assert receipt["rationale"] == rationale
    assert receipt["authorized_via"] == "controlling_terminal"
    assert receipt["receipt_sha256"]


def test_activation_exchange_failure_is_recoverable_with_fresh_authorization(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    controlled_exchange: None,
) -> None:
    prepared = _prepared_layout(tmp_path)
    preview = activation.preflight_activation(
        prepared.plan,
        state_dir=prepared.state_dir,
        configured_context_root=prepared.source,
    )
    rationale = "Exercise a recoverable exchange failure"
    real_exchange = activation._atomic_exchange
    monkeypatch.setattr(
        activation,
        "_atomic_exchange",
        lambda _left, _right: (_ for _ in ()).throw(OSError("injected exchange failure")),
    )

    with pytest.raises(activation.ActivationApplyError, match="exchange failure"):
        activation.activate_layout(
            prepared.plan,
            state_dir=prepared.state_dir,
            configured_context_root=prepared.source,
            rationale=rationale,
            authorization=_activation_authorization(preview, rationale),
        )
    assert detect_layout_version(prepared.source) == 1
    assert detect_layout_version(prepared.candidate) == 2
    journal = json.loads(
        (prepared.state_dir / "activation-journal.json").read_text(encoding="utf-8")
    )
    assert journal["status"] == "activate_prepared"

    retry = activation.preflight_activation(
        prepared.plan,
        state_dir=prepared.state_dir,
        configured_context_root=prepared.source,
    )
    assert retry.status == "ready"
    monkeypatch.setattr(activation, "_atomic_exchange", real_exchange)
    result = activation.activate_layout(
        prepared.plan,
        state_dir=prepared.state_dir,
        configured_context_root=prepared.source,
        rationale=rationale,
        authorization=_activation_authorization(retry, rationale),
    )
    assert result.status == "activated"


def test_prepared_activation_retry_reverifies_both_roots(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    controlled_exchange: None,
) -> None:
    prepared = _prepared_layout(tmp_path)
    preview = activation.preflight_activation(
        prepared.plan,
        prepared.state_dir,
        prepared.source,
    )
    rationale = "Prepare then fail before exchange"
    monkeypatch.setattr(
        activation,
        "_atomic_exchange",
        lambda _left, _right: (_ for _ in ()).throw(OSError("stop before exchange")),
    )
    with pytest.raises(activation.ActivationApplyError):
        activation.activate_layout(
            prepared.plan,
            prepared.state_dir,
            prepared.source,
            rationale=rationale,
            authorization=_activation_authorization(preview, rationale),
        )
    (prepared.source / "history" / "event.jsonl").write_text(
        "changed after prepare\n",
        encoding="utf-8",
    )
    with pytest.raises(activation.ActivationPreflightError, match="stale|verified|changed"):
        activation.preflight_activation(
            prepared.plan,
            prepared.state_dir,
            prepared.source,
        )


def test_activation_fsync_failure_recovers_by_finalizing_without_second_exchange(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    controlled_exchange: None,
) -> None:
    prepared = _prepared_layout(tmp_path)
    preview = activation.preflight_activation(
        prepared.plan,
        state_dir=prepared.state_dir,
        configured_context_root=prepared.source,
    )
    rationale = "Exercise receipt-pending recovery"
    real_fsync = activation.strict_fsync_directory
    calls = 0

    def fail_parent_sync(path: Path) -> None:
        nonlocal calls
        calls += 1
        # State-directory creation and prepared-journal publication happen
        # first; fail the exchange-parent durability sync.
        if path == prepared.source.parent and calls > 1:
            raise OSError("injected parent fsync failure")
        real_fsync(path)

    monkeypatch.setattr(activation, "strict_fsync_directory", fail_parent_sync)
    with pytest.raises(activation.ActivationApplyError, match="fsync failure"):
        activation.activate_layout(
            prepared.plan,
            state_dir=prepared.state_dir,
            configured_context_root=prepared.source,
            rationale=rationale,
            authorization=_activation_authorization(preview, rationale),
        )
    assert detect_layout_version(prepared.source) == 2
    assert detect_layout_version(prepared.candidate) == 1
    pending = activation.preflight_activation(
        prepared.plan,
        state_dir=prepared.state_dir,
        configured_context_root=prepared.source,
    )
    assert pending.status == "receipt_pending"

    monkeypatch.setattr(activation, "strict_fsync_directory", real_fsync)

    def must_not_exchange(_left: Path, _right: Path) -> None:
        raise AssertionError("receipt finalization must not exchange roots again")

    monkeypatch.setattr(activation, "_atomic_exchange", must_not_exchange)
    result = activation.activate_layout(
        prepared.plan,
        state_dir=prepared.state_dir,
        configured_context_root=prepared.source,
        rationale=rationale,
        authorization=_activation_authorization(pending, rationale),
    )
    assert result.status == "activated"
    assert detect_layout_version(prepared.source) == 2


def test_post_activation_verification_failure_compensates_to_v1(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    controlled_exchange: None,
) -> None:
    prepared = _prepared_layout(tmp_path)
    preview = activation.preflight_activation(
        prepared.plan,
        prepared.state_dir,
        prepared.source,
    )
    rationale = "Verify compensating exchange"
    monkeypatch.setattr(
        activation,
        "_verify_active_candidate",
        lambda _preflight: (_ for _ in ()).throw(ValueError("injected postcheck failure")),
    )
    with pytest.raises(activation.ActivationApplyError, match="original topology restored"):
        activation.activate_layout(
            prepared.plan,
            prepared.state_dir,
            prepared.source,
            rationale=rationale,
            authorization=_activation_authorization(preview, rationale),
        )
    assert detect_layout_version(prepared.source) == 1
    assert detect_layout_version(prepared.candidate) == 2
    retry = activation.preflight_activation(
        prepared.plan,
        prepared.state_dir,
        prepared.source,
    )
    assert retry.status == "ready"


def test_receipt_pending_finalization_rejects_moved_migration_receipt_tamper(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    controlled_exchange: None,
) -> None:
    prepared = _prepared_layout(tmp_path)
    preview = activation.preflight_activation(
        prepared.plan,
        prepared.state_dir,
        prepared.source,
    )
    rationale = "Create receipt-pending topology for tamper test"
    real_fsync = activation.strict_fsync_directory

    def fail_exchange_parent(path: Path) -> None:
        if path == prepared.source.parent and (prepared.source / ".afs" / LAYOUT_FILE).exists():
            raise OSError("stop after exchange")
        real_fsync(path)

    monkeypatch.setattr(activation, "strict_fsync_directory", fail_exchange_parent)
    with pytest.raises(activation.ActivationApplyError):
        activation.activate_layout(
            prepared.plan,
            prepared.state_dir,
            prepared.source,
            rationale=rationale,
            authorization=_activation_authorization(preview, rationale),
        )
    monkeypatch.setattr(activation, "strict_fsync_directory", real_fsync)
    moved_receipt = (
        prepared.source / ".afs" / "migrations" / prepared.plan.transaction_id / "receipt.json"
    )
    moved_receipt.write_text("{}\n", encoding="utf-8")
    pending = activation.preflight_activation(
        prepared.plan,
        prepared.state_dir,
        prepared.source,
    )
    with pytest.raises(activation.ActivationApplyError, match="original topology restored"):
        activation.activate_layout(
            prepared.plan,
            prepared.state_dir,
            prepared.source,
            rationale=rationale,
            authorization=_activation_authorization(pending, rationale),
        )
    assert detect_layout_version(prepared.source) == 1


def test_rollback_preserves_v2_writes_and_blocks_inactive_v1_drift(
    tmp_path: Path,
    controlled_exchange: None,
) -> None:
    prepared = _prepared_layout(tmp_path)
    preview = activation.preflight_activation(
        prepared.plan,
        state_dir=prepared.state_dir,
        configured_context_root=prepared.source,
    )
    rationale = "Activate before rollback preservation test"
    activation.activate_layout(
        prepared.plan,
        state_dir=prepared.state_dir,
        configured_context_root=prepared.source,
        rationale=rationale,
        authorization=_activation_authorization(preview, rationale),
    )
    v2_write = prepared.source / "memory" / "after-activation.md"
    v2_write.write_text("preserve me\n", encoding="utf-8")
    rollback_preview = activation.preflight_rollback(
        prepared.state_dir,
        prepared.source,
    )
    rollback_rationale = "Restore v1 and preserve v2 writes"
    activation.rollback_layout(
        prepared.state_dir,
        prepared.source,
        rationale=rollback_rationale,
        authorization=_rollback_authorization(rollback_preview, rollback_rationale),
    )
    assert (prepared.candidate / "memory" / "after-activation.md").read_text(
        encoding="utf-8"
    ) == "preserve me\n"

    second = _prepared_layout(tmp_path / "drift")
    second_preview = activation.preflight_activation(
        second.plan,
        second.state_dir,
        second.source,
    )
    activation.activate_layout(
        second.plan,
        second.state_dir,
        second.source,
        rationale=rationale,
        authorization=_activation_authorization(second_preview, rationale),
    )
    (second.candidate / "history" / "event.jsonl").write_text(
        "drifted inactive v1\n",
        encoding="utf-8",
    )
    with pytest.raises(activation.RollbackPreflightError, match="inactive v1 source changed"):
        activation.preflight_rollback(second.state_dir, second.source)


def test_rollback_parent_fsync_failure_finalizes_without_second_exchange(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    controlled_exchange: None,
) -> None:
    prepared = _prepared_layout(tmp_path)
    preview = activation.preflight_activation(
        prepared.plan,
        prepared.state_dir,
        prepared.source,
    )
    activation_rationale = "Activate before rollback recovery"
    activation.activate_layout(
        prepared.plan,
        prepared.state_dir,
        prepared.source,
        rationale=activation_rationale,
        authorization=_activation_authorization(preview, activation_rationale),
    )
    rollback_preview = activation.preflight_rollback(prepared.state_dir, prepared.source)
    rollback_rationale = "Exercise rollback receipt recovery"
    real_fsync = activation.strict_fsync_directory

    def fail_parent(path: Path) -> None:
        if path == prepared.source.parent:
            raise OSError("injected rollback parent fsync failure")
        real_fsync(path)

    monkeypatch.setattr(activation, "strict_fsync_directory", fail_parent)
    with pytest.raises(activation.RollbackApplyError, match="fsync failure"):
        activation.rollback_layout(
            prepared.state_dir,
            prepared.source,
            rationale=rollback_rationale,
            authorization=_rollback_authorization(rollback_preview, rollback_rationale),
        )
    assert detect_layout_version(prepared.source) == 1
    pending = activation.preflight_rollback(prepared.state_dir, prepared.source)
    assert pending.status == "receipt_pending"
    monkeypatch.setattr(activation, "strict_fsync_directory", real_fsync)

    def must_not_exchange(_left: Path, _right: Path) -> None:
        raise AssertionError("rollback receipt finalization must not exchange again")

    monkeypatch.setattr(activation, "_atomic_exchange", must_not_exchange)
    result = activation.rollback_layout(
        prepared.state_dir,
        prepared.source,
        rationale=rollback_rationale,
        authorization=_rollback_authorization(pending, rollback_rationale),
    )
    assert result.status == "rolled_back"


def test_activation_fails_closed_without_atomic_exchange_support(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    controlled_exchange: None,
) -> None:
    prepared = _prepared_layout(tmp_path)
    monkeypatch.setattr(activation, "_atomic_exchange_backend", lambda: "unsupported")
    with pytest.raises(activation.ActivationPreflightError, match="unavailable"):
        activation.preflight_activation(
            prepared.plan,
            prepared.state_dir,
            prepared.source,
        )
    assert not prepared.state_dir.exists()


def test_host_atomic_exchange_adapter_swaps_two_directories(tmp_path: Path) -> None:
    if activation._atomic_exchange_backend() == "unsupported":
        pytest.skip("host has no supported atomic directory exchange")
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    (left / "identity").write_text("left\n", encoding="utf-8")
    (right / "identity").write_text("right\n", encoding="utf-8")
    left_identity = os.lstat(left).st_ino
    right_identity = os.lstat(right).st_ino

    activation._atomic_exchange(left, right)

    assert (left / "identity").read_text(encoding="utf-8") == "right\n"
    assert (right / "identity").read_text(encoding="utf-8") == "left\n"
    assert os.lstat(left).st_ino == right_identity
    assert os.lstat(right).st_ino == left_identity


def test_lsof_warning_cannot_be_treated_as_quiescent(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    completed = subprocess.CompletedProcess(
        args=["lsof"],
        returncode=1,
        stdout="",
        stderr="lsof: WARNING: cannot stat() example\n",
    )
    monkeypatch.setattr(subprocess, "run", lambda *_args, **_kwargs: completed)
    with pytest.raises(ValueError, match="incomplete"):
        activation._open_processes(tmp_path)
