from __future__ import annotations

import json
import shutil
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from pathlib import Path

import pytest

from afs.agents.guardrails import ApprovalGate
from afs.artifacts import NoteStore
from afs.context_layout import scaffold_v2
from afs.history import append_history_event, resolve_history_root
from afs.human_provenance import _broker_for_reader
from afs.insights import (
    MAX_INSIGHT_BODY_BYTES,
    MAX_INSIGHT_CANDIDATE_BYTES,
    InsightContentChangedError,
    InsightEvidencePacket,
    InsightReview,
    InsightStore,
    _packet_digest,
    insight_review_gate_binding,
    reflect_evidence,
)
from afs.project_registry import ProjectRecord, ProjectRegistry, ScopeAuthorizationError
from afs.response_schemas import SchemaValidationResult, validate_structured_response


def _scoped_stores(
    tmp_path: Path,
) -> tuple[Path, Path, Path, ProjectRecord, ProjectRecord, InsightStore, InsightStore]:
    context = tmp_path / ".context"
    scaffold_v2(context)
    alpha = tmp_path / "alpha"
    beta = tmp_path / "beta"
    alpha.mkdir()
    beta.mkdir()
    registry = ProjectRegistry(context)
    alpha_record = registry.register(alpha)
    beta_record = registry.register(beta)
    alpha_store = InsightStore(
        context,
        scope_id=alpha_record.scope_id,
        requester_path=alpha,
    )
    beta_store = InsightStore(
        context,
        scope_id=beta_record.scope_id,
        requester_path=beta,
    )
    return (
        context,
        alpha,
        beta,
        alpha_record,
        beta_record,
        alpha_store,
        beta_store,
    )


def _append(
    context: Path,
    *,
    event_id: str,
    timestamp: str,
    scope_id: str | None,
    project_id: str = "",
    source: str = "agent.worker",
    op: str = "failed",
    extra_metadata: dict[str, object] | None = None,
) -> None:
    metadata: dict[str, object] = {
        "status": op,
        "ok": op not in {"failed", "error"},
        "prompt_preview": "must not enter evidence",
        "summary": "private task text",
        "payload": {"secret": "metadata payload"},
    }
    if scope_id is not None:
        metadata["scope_id"] = scope_id
    if scope_id == "common":
        metadata["scope_attribution"] = "common"
    if project_id:
        metadata["project_id"] = project_id
        metadata["scope_attribution"] = "registry"
    if extra_metadata:
        metadata.update(extra_metadata)
    append_history_event(
        resolve_history_root(context),
        "agent_lifecycle",
        source,
        op=op,
        metadata=metadata,
        payload={"secret": "history payload"},
        timestamp=timestamp,
        context_root=context,
        event_id=event_id,
        include_payloads=True,
    )


def _failed_alpha_packet(tmp_path: Path):
    context, _alpha, _beta, alpha_record, _beta_record, store, _beta_store = _scoped_stores(
        tmp_path
    )
    _append(
        context,
        event_id="alpha-failure-1",
        timestamp="2026-07-17T01:00:00+00:00",
        scope_id=alpha_record.scope_id,
        project_id=alpha_record.project_id,
    )
    _append(
        context,
        event_id="alpha-failure-2",
        timestamp="2026-07-17T02:00:00+00:00",
        scope_id=alpha_record.scope_id,
        project_id=alpha_record.project_id,
    )
    return context, alpha_record, store, store.build_evidence_packet()


def _approve_review_operation(
    store: InsightStore,
    identifier: str,
    *,
    decision: str,
    rationale: str,
    gate_path: Path,
    detail_override: str | None = None,
) -> tuple[ApprovalGate, str]:
    record = store.show(identifier)
    assert record is not None
    assert decision in {"accept", "reject"}
    action, detail = insight_review_gate_binding(
        store,
        record,
        decision=decision,  # type: ignore[arg-type]
        rationale=rationale,
    )
    gate = ApprovalGate(gate_path)
    assert gate.check("insights", action, detail_override or detail) is False
    request = gate.find_pending("insights", action)
    assert request is not None
    scope = gate.human_authorization_scope("approve", request.request_id, rationale)
    authorization = _broker_for_reader(lambda _prompt: "confirm").confirm_token(
        "confirm",
        "confirm",
        scope=scope,
    )
    assert authorization is not None
    assert gate.approve_human(
        "insights",
        action,
        rationale=rationale,
        authorization=authorization,
    )
    return gate, request.request_id


def test_evidence_packet_is_deterministic_payload_free_and_exact_scope(
    tmp_path: Path,
) -> None:
    (
        context,
        _alpha,
        _beta,
        alpha_record,
        beta_record,
        alpha_store,
        _beta_store,
    ) = _scoped_stores(tmp_path)
    _append(
        context,
        event_id="common-event",
        timestamp="2026-07-17T00:00:00+00:00",
        scope_id="common",
    )
    _append(
        context,
        event_id="alpha-event",
        timestamp="2026-07-17T01:00:00+00:00",
        scope_id=alpha_record.scope_id,
        project_id=alpha_record.project_id,
    )
    _append(
        context,
        event_id="beta-event",
        timestamp="2026-07-17T02:00:00+00:00",
        scope_id=beta_record.scope_id,
        project_id=beta_record.project_id,
    )
    _append(
        context,
        event_id="unscoped-event",
        timestamp="2026-07-17T03:00:00+00:00",
        scope_id=None,
    )
    _append(
        context,
        event_id="insight-self-event",
        timestamp="2026-07-17T04:00:00+00:00",
        scope_id=alpha_record.scope_id,
        project_id=alpha_record.project_id,
        source="afs.insights.accept",
    )
    _append(
        context,
        event_id="agent-self-event",
        timestamp="2026-07-17T05:00:00+00:00",
        scope_id=alpha_record.scope_id,
        project_id=alpha_record.project_id,
        source="agent.insights.reflect",
    )
    _append(
        context,
        event_id="prefix-self-event",
        timestamp="2026-07-17T06:00:00+00:00",
        scope_id=alpha_record.scope_id,
        project_id=alpha_record.project_id,
        source="afs.insights-reflect",
    )

    first = alpha_store.build_evidence_packet()
    second = alpha_store.build_evidence_packet()

    assert first.to_dict() == second.to_dict()
    assert first.evidence_ids == ("alpha-event",)
    assert len(first.evidence_digest) == 64
    rendered = json.dumps(first.to_dict(), sort_keys=True)
    assert "history payload" not in rendered
    assert "metadata payload" not in rendered
    assert "prompt_preview" not in rendered
    assert "private task text" not in rendered
    assert first.events[0]["metadata"] == {
        "ok": False,
        "project_id": alpha_record.project_id,
        "scope_attribution": "registry",
        "scope_id": alpha_record.scope_id,
        "status": "failed",
    }

    common = InsightStore(context).build_evidence_packet()
    assert common.evidence_ids == ("common-event",)


def test_project_evidence_requires_registry_attribution_and_matching_project_id(
    tmp_path: Path,
) -> None:
    context, _alpha, _beta, alpha_record, _beta_record, store, _beta_store = _scoped_stores(
        tmp_path
    )
    _append(
        context,
        event_id="missing-attribution",
        timestamp="2026-07-17T01:00:00+00:00",
        scope_id=alpha_record.scope_id,
        extra_metadata={"project_id": alpha_record.project_id},
    )
    _append(
        context,
        event_id="wrong-project",
        timestamp="2026-07-17T02:00:00+00:00",
        scope_id=alpha_record.scope_id,
        project_id="prj_other",
    )
    _append(
        context,
        event_id="valid",
        timestamp="2026-07-17T03:00:00+00:00",
        scope_id=alpha_record.scope_id,
        project_id=alpha_record.project_id,
    )

    assert store.build_evidence_packet().evidence_ids == ("valid",)

    top_level_only = {
        "id": "top-level-only",
        "timestamp": "2026-07-17T04:00:00+00:00",
        "type": "agent_lifecycle",
        "source": "agent.worker",
        "op": "failed",
        "scope_id": alpha_record.scope_id,
        "metadata": {
            "project_id": alpha_record.project_id,
            "scope_attribution": "registry",
            "status": "failed",
        },
    }
    top_level_packet = InsightEvidencePacket(
        schema_version="1",
        scope_id=alpha_record.scope_id,
        evidence_ids=("top-level-only",),
        evidence_digest=_packet_digest(alpha_record.scope_id, [top_level_only]),
        events=(top_level_only,),
    )
    with pytest.raises(ValueError, match="unscoped"):
        top_level_packet.assert_valid()


def test_common_evidence_requires_explicit_common_attribution(tmp_path: Path) -> None:
    context = tmp_path / ".context"
    scaffold_v2(context)
    _append(
        context,
        event_id="missing-attribution",
        timestamp="2026-07-17T01:00:00+00:00",
        scope_id="common",
        extra_metadata={"scope_attribution": None},
    )
    _append(
        context,
        event_id="registry-attribution",
        timestamp="2026-07-17T02:00:00+00:00",
        scope_id="common",
        extra_metadata={"scope_attribution": "registry"},
    )
    _append(
        context,
        event_id="valid",
        timestamp="2026-07-17T03:00:00+00:00",
        scope_id="common",
    )

    assert InsightStore(context).build_evidence_packet().evidence_ids == ("valid",)


def test_empty_packet_is_valid_and_reflection_returns_none(tmp_path: Path) -> None:
    _context, _alpha, _beta, _alpha_record, _beta_record, store, _beta_store = _scoped_stores(
        tmp_path
    )

    packet = store.build_evidence_packet()

    packet.assert_valid()
    assert packet.evidence_ids == ()
    assert reflect_evidence(packet) is None


def test_reflection_prefers_repeated_failures_and_validates_schema(tmp_path: Path) -> None:
    _context, _alpha_record, _store, packet = _failed_alpha_packet(tmp_path)

    candidate = reflect_evidence(packet)

    assert candidate is not None
    assert candidate["title"].startswith("Repeated failure:")
    assert candidate["evidence_ids"] == ["alpha-failure-1", "alpha-failure-2"]
    assert candidate["evidence_digest"] == packet.evidence_digest
    assert validate_structured_response("insight-candidate", candidate).valid


def test_completion_only_rolling_history_never_creates_candidate_spam(
    tmp_path: Path,
) -> None:
    context, _alpha, _beta, alpha_record, _beta_record, store, _beta_store = _scoped_stores(
        tmp_path
    )
    for index in range(1, 6):
        _append(
            context,
            event_id=f"healthy-completion-{index}",
            timestamp=f"2026-07-17T0{index}:00:00+00:00",
            scope_id=alpha_record.scope_id,
            project_id=alpha_record.project_id,
            source="agent.verifier",
            op="completed",
        )
        packet = store.build_evidence_packet()
        assert reflect_evidence(packet) is None

    assert store.list() == []


def test_rolling_failure_pattern_keeps_one_stable_candidate(tmp_path: Path) -> None:
    context, _alpha, _beta, alpha_record, _beta_record, store, _beta_store = _scoped_stores(
        tmp_path
    )
    paths: list[Path] = []
    for index in range(1, 6):
        _append(
            context,
            event_id=f"rolling-failure-{index}",
            timestamp=f"2026-07-17T0{index}:00:00+00:00",
            scope_id=alpha_record.scope_id,
            project_id=alpha_record.project_id,
        )
        packet = store.build_evidence_packet()
        payload = reflect_evidence(packet)
        if index == 1:
            assert payload is None
            continue
        assert payload is not None
        assert payload["evidence_ids"] == ["rolling-failure-1", "rolling-failure-2"]
        paths.append(store.create_candidate(payload, evidence=packet).path)

    assert len(set(paths)) == 1
    assert len(store.list()) == 1


def test_candidate_names_are_readable_scoped_and_creation_is_idempotent(
    tmp_path: Path,
) -> None:
    _context, alpha_record, store, packet = _failed_alpha_packet(tmp_path)
    payload = reflect_evidence(packet)
    assert payload is not None

    first_result = store.create_candidate_result(
        payload,
        evidence=packet,
        agent_name="reflector",
    )
    repeated_result = store.create_candidate_result(
        payload,
        evidence=packet,
        agent_name="reflector",
    )
    first = first_result.artifact
    repeated = repeated_result.artifact

    assert first_result.created is True
    assert repeated_result.created is False
    assert first_result.bound_evidence_digest == packet.evidence_digest
    assert first.path == repeated.path
    assert "repeated-failure" in first.path.name
    assert (
        first.path.parent
        == (
            store.context_path
            / "scratchpad"
            / "projects"
            / alpha_record.project_id
            / "insights"
            / "candidates"
        ).resolve()
    )
    record = store.show(first.metadata.artifact_id)
    assert record is not None
    assert record.status == "pending"
    assert store.list() == [record]


def test_unrelated_new_event_does_not_duplicate_existing_candidate(tmp_path: Path) -> None:
    context, alpha_record, store, first_packet = _failed_alpha_packet(tmp_path)
    first_payload = reflect_evidence(first_packet)
    assert first_payload is not None
    first_result = store.create_candidate_result(first_payload, evidence=first_packet)
    first = first_result.artifact
    _append(
        context,
        event_id="unrelated-completion",
        timestamp="2026-07-17T03:00:00+00:00",
        scope_id=alpha_record.scope_id,
        project_id=alpha_record.project_id,
        source="agent.verifier",
        op="completed",
    )

    later_packet = store.build_evidence_packet()
    later_payload = reflect_evidence(later_packet)
    assert later_payload is not None
    repeated_result = store.create_candidate_result(later_payload, evidence=later_packet)
    repeated = repeated_result.artifact

    assert first_result.created is True
    assert repeated_result.created is False
    assert later_packet.evidence_digest != first_packet.evidence_digest
    assert later_payload["evidence_ids"] == first_payload["evidence_ids"]
    assert repeated_result.bound_evidence_digest == first_packet.evidence_digest
    assert repeated_result.bound_evidence_digest != later_packet.evidence_digest
    assert repeated.path == first.path
    assert len(store.list()) == 1


def test_candidate_rejects_forged_or_mutated_evidence(tmp_path: Path) -> None:
    _context, _alpha_record, store, packet = _failed_alpha_packet(tmp_path)
    payload = reflect_evidence(packet)
    assert payload is not None

    forged = {**payload, "evidence_ids": ["outside-packet"]}
    with pytest.raises(ValueError, match="outside the packet"):
        store.create_candidate(forged, evidence=packet)

    wrong_digest = {**payload, "evidence_digest": "0" * 64}
    with pytest.raises(ValueError, match="does not match"):
        store.create_candidate(wrong_digest, evidence=packet)

    mutated = replace(packet, evidence_digest="0" * 64)
    with pytest.raises(ValueError, match="digest"):
        store.create_candidate(payload, evidence=mutated)


def test_candidate_rejects_self_computable_event_absent_from_local_history(
    tmp_path: Path,
) -> None:
    _context, _alpha_record, store, packet = _failed_alpha_packet(tmp_path)
    forged_events = [dict(event) for event in packet.events]
    forged_events[-1]["id"] = "never-existed"
    forged_tuple = tuple(forged_events)
    forged = InsightEvidencePacket(
        schema_version=packet.schema_version,
        scope_id=packet.scope_id,
        evidence_ids=tuple(str(event["id"]) for event in forged_tuple),
        evidence_digest=_packet_digest(packet.scope_id, forged_tuple),
        events=forged_tuple,
    )
    forged.assert_valid()
    payload = reflect_evidence(forged)
    assert payload is not None

    with pytest.raises(ValueError, match="not present.*local history"):
        store.create_candidate(payload, evidence=forged)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("title", "t" * 241),
        ("insight", "i" * 32769),
        ("limitations", ["l" * 2049]),
        ("next_step", "n" * 8193),
    ],
)
def test_candidate_schema_bounds_human_authored_text(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
) -> None:
    import afs.response_schemas as response_schemas_module

    monkeypatch.setattr(
        response_schemas_module,
        "_collect_schema_errors",
        response_schemas_module._builtin_schema_errors,
    )
    _context, _alpha_record, store, packet = _failed_alpha_packet(tmp_path)
    payload = reflect_evidence(packet)
    assert payload is not None
    payload[field] = value

    with pytest.raises(ValueError, match="invalid insight candidate"):
        store.create_candidate(payload, evidence=packet)


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        ("title", "unsafe\u202etitle", "candidate title"),
        ("insight", "unsafe\x1binsight", "candidate insight"),
        ("limitations", ["unsafe\u2066limitation"], "candidate limitations\\[0\\]"),
        ("next_step", "unsafe\nnext step", "candidate next_step"),
    ],
)
def test_candidate_rejects_terminal_unsafe_human_readable_fields(
    tmp_path: Path,
    field: str,
    value: object,
    expected: str,
) -> None:
    _context, _alpha_record, store, packet = _failed_alpha_packet(tmp_path)
    payload = reflect_evidence(packet)
    assert payload is not None
    payload[field] = value

    with pytest.raises(ValueError, match=expected):
        store.create_candidate(payload, evidence=packet)


def test_accept_fails_closed_for_tampered_terminal_unsafe_candidate(tmp_path: Path) -> None:
    _context, _alpha_record, store, packet = _failed_alpha_packet(tmp_path)
    payload = reflect_evidence(packet)
    assert payload is not None
    candidate = store.create_candidate(payload, evidence=packet)
    candidate.path.write_text(
        candidate.path.read_text(encoding="utf-8") + "unsafe\u202econtent\n",
        encoding="utf-8",
    )
    current = store.show(candidate.metadata.artifact_id)
    assert current is not None
    rationale = "Unsafe hidden content must not be promoted."
    gate, request_id = _approve_review_operation(
        store,
        candidate.metadata.artifact_id,
        decision="accept",
        rationale=rationale,
        gate_path=tmp_path / "approvals.json",
    )

    with pytest.raises(ValueError, match="cannot be rendered exactly"):
        store.accept(
            candidate.metadata.artifact_id,
            expected_digest=current.content_digest,
            approval_gate=gate,
            approval_request_id=request_id,
        )


def test_maximum_candidate_title_remains_promotable(tmp_path: Path) -> None:
    _context, _alpha_record, store, packet = _failed_alpha_packet(tmp_path)
    payload = reflect_evidence(packet)
    assert payload is not None
    payload["title"] = "t" * 240
    candidate = store.create_candidate(payload, evidence=packet)
    digest = store.content_digest(candidate.metadata.artifact_id)
    gate, request_id = _approve_review_operation(
        store,
        candidate.metadata.artifact_id,
        decision="accept",
        rationale="The maximum-length readable title is still valid.",
        gate_path=tmp_path / "approvals.json",
    )

    note = store.accept(
        candidate.metadata.artifact_id,
        expected_digest=digest,
        approval_gate=gate,
        approval_request_id=request_id,
    )

    assert len(note.metadata.title) == 240
    decision_files = list(store.decisions_root.glob("*.md"))
    assert len(decision_files) == 1


def test_candidate_enforces_canonical_and_rendered_byte_caps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _context, _alpha_record, store, packet = _failed_alpha_packet(tmp_path)
    payload = reflect_evidence(packet)
    assert payload is not None

    from afs import insights as insights_module

    oversized = {**payload, "insight": "x" * (MAX_INSIGHT_CANDIDATE_BYTES + 1)}

    def permissive_validation(_name: str, data: object) -> SchemaValidationResult:
        return SchemaValidationResult(
            valid=True,
            schema="insight-candidate",
            parsed=data,
        )

    monkeypatch.setattr(
        insights_module,
        "validate_structured_response",
        permissive_validation,
    )
    with pytest.raises(ValueError, match="canonical payload limit"):
        store.create_candidate(oversized, evidence=packet)

    monkeypatch.setattr(insights_module, "MAX_INSIGHT_BODY_BYTES", 64)
    with pytest.raises(ValueError, match="rendered body limit"):
        store.create_candidate(payload, evidence=packet)

    assert MAX_INSIGHT_CANDIDATE_BYTES == 64 * 1024
    assert MAX_INSIGHT_BODY_BYTES == 64 * 1024


def test_concurrent_candidate_creation_is_idempotent(tmp_path: Path) -> None:
    _context, _alpha_record, store, packet = _failed_alpha_packet(tmp_path)
    payload = reflect_evidence(packet)
    assert payload is not None

    with ThreadPoolExecutor(max_workers=8) as executor:
        artifacts = list(
            executor.map(
                lambda _index: store.create_candidate(payload, evidence=packet),
                range(8),
            )
        )

    assert len({artifact.path for artifact in artifacts}) == 1
    assert len(store.list()) == 1


def test_accept_is_idempotent_archives_candidate_and_creates_one_memory_note(
    tmp_path: Path,
) -> None:
    context, alpha_record, store, packet = _failed_alpha_packet(tmp_path)
    payload = reflect_evidence(packet)
    assert payload is not None
    candidate = store.create_candidate(payload, evidence=packet)
    digest = store.content_digest(candidate.metadata.artifact_id)
    rationale = "The same attributed failure occurred twice."
    gate, request_id = _approve_review_operation(
        store,
        candidate.metadata.artifact_id,
        decision="accept",
        rationale=rationale,
        gate_path=tmp_path / "approvals.json",
    )

    note = store.accept(
        candidate.metadata.artifact_id,
        expected_digest=digest,
        approval_gate=gate,
        approval_request_id=request_id,
    )
    repeated = store.accept(
        candidate.metadata.artifact_id,
        expected_digest=digest,
        approval_gate=gate,
        approval_request_id=request_id,
    )

    assert note.path == repeated.path
    assert not candidate.path.exists()
    record = store.show(candidate.metadata.artifact_id)
    assert record is not None
    assert record.status == "accepted"
    assert record.review is not None
    assert record.review["decision"] == "accepted"
    assert record.review["rationale"] == rationale
    assert record.review["human_confirmed"] is True
    assert record.artifact.path.parent == store.accepted_root
    assert (
        note.path.parent
        == (context / "memory" / "projects" / alpha_record.project_id / "notes").resolve()
    )
    assert note.metadata.provenance is not None
    assert note.metadata.provenance["source_artifact_id"] == candidate.metadata.artifact_id
    assert note.metadata.provenance["evidence_digest"] == packet.evidence_digest
    assert note.metadata.provenance["request_id"] == request_id
    assert note.metadata.provenance["reviewer"]
    assert note.metadata.provenance["via"] == "controlling_terminal"
    assert note.metadata.provenance["authenticated"] is True
    assert note.metadata.provenance["human_confirmed"] is True
    notes = NoteStore(context, scope_id=alpha_record.scope_id).list(limit=10)
    assert [item.path for item in notes] == [note.path]
    with pytest.raises(ValueError, match="cannot be rejected"):
        store.reject(candidate.metadata.artifact_id, expected_digest=digest)


def test_accept_rejects_candidate_mutated_after_review(tmp_path: Path) -> None:
    context, alpha_record, store, packet = _failed_alpha_packet(tmp_path)
    payload = reflect_evidence(packet)
    assert payload is not None
    candidate = store.create_candidate(payload, evidence=packet)
    reviewed_digest = store.content_digest(candidate.metadata.artifact_id)
    gate, request_id = _approve_review_operation(
        store,
        candidate.metadata.artifact_id,
        decision="accept",
        rationale="Approve only the exact candidate bytes shown.",
        gate_path=tmp_path / "approvals.json",
    )
    candidate.path.write_text(
        candidate.path.read_text(encoding="utf-8") + "mutated after review\n",
        encoding="utf-8",
    )

    with pytest.raises(InsightContentChangedError, match="changed after review"):
        store.accept(
            candidate.metadata.artifact_id,
            expected_digest=reviewed_digest,
            approval_gate=gate,
            approval_request_id=request_id,
        )

    record = store.show(candidate.metadata.artifact_id)
    assert record is not None
    assert record.status == "pending"
    assert record.review is None
    assert NoteStore(context, scope_id=alpha_record.scope_id).list() == []


def test_accept_rolls_candidate_back_when_decision_write_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, alpha_record, store, packet = _failed_alpha_packet(tmp_path)
    payload = reflect_evidence(packet)
    assert payload is not None
    candidate = store.create_candidate(payload, evidence=packet)
    digest = store.content_digest(candidate.metadata.artifact_id)
    gate, request_id = _approve_review_operation(
        store,
        candidate.metadata.artifact_id,
        decision="accept",
        rationale="Approve only if the durable decision can be recorded.",
        gate_path=tmp_path / "approvals.json",
    )

    def fail_decision(*_args, **_kwargs):
        raise OSError("simulated decision storage failure")

    monkeypatch.setattr(store, "_record_decision", fail_decision)
    with pytest.raises(OSError, match="simulated decision storage failure"):
        store.accept(
            candidate.metadata.artifact_id,
            expected_digest=digest,
            approval_gate=gate,
            approval_request_id=request_id,
        )

    record = store.show(candidate.metadata.artifact_id)
    assert record is not None
    assert record.status == "pending"
    assert candidate.path.exists()
    assert list(store.decisions_root.glob("*.md")) == []
    assert NoteStore(context, scope_id=alpha_record.scope_id).list() == []


def test_accept_rehashes_archived_candidate_and_rolls_back_review_race(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context, alpha_record, store, packet = _failed_alpha_packet(tmp_path)
    payload = reflect_evidence(packet)
    assert payload is not None
    candidate = store.create_candidate(payload, evidence=packet)
    digest = store.content_digest(candidate.metadata.artifact_id)
    gate, request_id = _approve_review_operation(
        store,
        candidate.metadata.artifact_id,
        decision="accept",
        rationale="Approve only the pinned reviewed candidate.",
        gate_path=tmp_path / "approvals.json",
    )
    original_resolve = store._resolve_review

    def mutate_after_first_digest(*args, **kwargs):
        resolved = original_resolve(*args, **kwargs)
        candidate.path.write_text(
            candidate.path.read_text(encoding="utf-8") + "mutation during review\n",
            encoding="utf-8",
        )
        return resolved

    monkeypatch.setattr(store, "_resolve_review", mutate_after_first_digest)
    with pytest.raises(InsightContentChangedError, match="changed during review"):
        store.accept(
            candidate.metadata.artifact_id,
            expected_digest=digest,
            approval_gate=gate,
            approval_request_id=request_id,
        )

    record = store.show(candidate.metadata.artifact_id)
    assert record is not None
    assert record.status == "pending"
    assert record.content_digest != digest
    assert list(store.decisions_root.glob("*.md")) == []
    assert NoteStore(context, scope_id=alpha_record.scope_id).list() == []


def test_concurrent_accept_creates_one_note_and_one_decision(tmp_path: Path) -> None:
    context, alpha_record, store, packet = _failed_alpha_packet(tmp_path)
    payload = reflect_evidence(packet)
    assert payload is not None
    candidate = store.create_candidate(payload, evidence=packet)
    digest = store.content_digest(candidate.metadata.artifact_id)
    rationale = "Reviewed once and applied consistently."
    gate, request_id = _approve_review_operation(
        store,
        candidate.metadata.artifact_id,
        decision="accept",
        rationale=rationale,
        gate_path=tmp_path / "approvals.json",
    )

    def accept_once(_index: int):
        return store.accept(
            candidate.metadata.artifact_id,
            expected_digest=digest,
            approval_gate=gate,
            approval_request_id=request_id,
        )

    with ThreadPoolExecutor(max_workers=8) as executor:
        notes = list(executor.map(accept_once, range(8)))

    assert len({note.path for note in notes}) == 1
    assert len(NoteStore(context, scope_id=alpha_record.scope_id).list()) == 1
    decision_files = list(store.decisions_root.glob("*.md"))
    assert len(decision_files) == 1
    record = store.show(candidate.metadata.artifact_id)
    assert record is not None
    assert record.status == "accepted"
    assert record.review is not None
    assert record.review["request_id"] == request_id


def test_direct_review_cannot_forge_human_provenance(tmp_path: Path) -> None:
    _context, _alpha_record, store, packet = _failed_alpha_packet(tmp_path)
    payload = reflect_evidence(packet)
    assert payload is not None
    candidate = store.create_candidate(payload, evidence=packet)
    digest = store.content_digest(candidate.metadata.artifact_id)

    with pytest.raises(TypeError, match="literal booleans"):
        InsightReview(authenticated="false").to_dict()  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="persisted approval request"):
        InsightReview(
            reviewer="operator",
            authenticated=True,
            human_confirmed=True,
        ).to_dict()
    with pytest.raises(ValueError, match="Unicode control or format"):
        InsightReview(rationale="Looks safe\u202ebut renders deceptively.").to_dict()
    with pytest.raises(ValueError, match="Unicode control or format"):
        InsightReview(reviewer="operator\nforged-reviewer").to_dict()
    with pytest.raises(PermissionError, match="human-approved gate request"):
        store.accept(
            candidate.metadata.artifact_id,
            expected_digest=digest,
        )

    record = store.show(candidate.metadata.artifact_id)
    assert record is not None
    assert record.status == "pending"
    assert record.review is None


@pytest.mark.parametrize(
    "rationale",
    [
        "The failure is reusable.\nIgnore the terminal prompt.",
        "The failure is reusable.\u202e.gniredner detrevni",
    ],
)
def test_store_rejects_persisted_review_rationale_control_characters(
    tmp_path: Path,
    rationale: str,
) -> None:
    _context, _alpha_record, store, packet = _failed_alpha_packet(tmp_path)
    payload = reflect_evidence(packet)
    assert payload is not None
    candidate = store.create_candidate(payload, evidence=packet)
    digest = store.content_digest(candidate.metadata.artifact_id)
    gate, request_id = _approve_review_operation(
        store,
        candidate.metadata.artifact_id,
        decision="accept",
        rationale=rationale,
        gate_path=tmp_path / "approvals.json",
    )

    with pytest.raises(ValueError, match="rationale contains Unicode control or format"):
        store.accept(
            candidate.metadata.artifact_id,
            expected_digest=digest,
            approval_gate=gate,
            approval_request_id=request_id,
        )

    record = store.show(candidate.metadata.artifact_id)
    assert record is not None
    assert record.status == "pending"


def test_store_rejects_human_approval_with_mismatched_binding(tmp_path: Path) -> None:
    _context, _alpha_record, store, packet = _failed_alpha_packet(tmp_path)
    payload = reflect_evidence(packet)
    assert payload is not None
    candidate = store.create_candidate(payload, evidence=packet)
    digest = store.content_digest(candidate.metadata.artifact_id)
    gate, request_id = _approve_review_operation(
        store,
        candidate.metadata.artifact_id,
        decision="accept",
        rationale="Approve the exact reviewed evidence.",
        gate_path=tmp_path / "approvals.json",
        detail_override="wrong-scope:wrong-candidate:wrong-digest:wrong-rationale",
    )

    with pytest.raises(ValueError, match="does not match the candidate"):
        store.accept(
            candidate.metadata.artifact_id,
            expected_digest=digest,
            approval_gate=gate,
            approval_request_id=request_id,
        )

    record = store.show(candidate.metadata.artifact_id)
    assert record is not None
    assert record.status == "pending"


def test_human_approval_cannot_replay_in_a_copied_context(tmp_path: Path) -> None:
    context, alpha_record, store, packet = _failed_alpha_packet(tmp_path)
    payload = reflect_evidence(packet)
    assert payload is not None
    candidate = store.create_candidate(payload, evidence=packet)
    digest = store.content_digest(candidate.metadata.artifact_id)
    gate, request_id = _approve_review_operation(
        store,
        candidate.metadata.artifact_id,
        decision="accept",
        rationale="Approve this candidate in its original context store only.",
        gate_path=tmp_path / "approvals.json",
    )

    copied_context = tmp_path / "copied" / ".context"
    shutil.copytree(context, copied_context)
    copied_store = InsightStore(
        copied_context,
        scope_id=alpha_record.scope_id,
        requester_path=tmp_path / "alpha",
    )
    copied_record = copied_store.show(candidate.metadata.artifact_id)
    assert copied_record is not None
    assert copied_record.content_digest == digest
    assert copied_store.review_store_identity != store.review_store_identity

    with pytest.raises(ValueError, match="does not match the candidate"):
        copied_store.accept(
            candidate.metadata.artifact_id,
            expected_digest=digest,
            approval_gate=gate,
            approval_request_id=request_id,
        )

    copied_record = copied_store.show(candidate.metadata.artifact_id)
    assert copied_record is not None
    assert copied_record.status == "pending"
    assert NoteStore(copied_context, scope_id=alpha_record.scope_id).list() == []


def test_reject_derives_human_provenance_from_approved_gate_operation(
    tmp_path: Path,
) -> None:
    _context, _alpha_record, store, packet = _failed_alpha_packet(tmp_path)
    payload = reflect_evidence(packet)
    assert payload is not None
    candidate = store.create_candidate(payload, evidence=packet)
    digest = store.content_digest(candidate.metadata.artifact_id)
    gate, request_id = _approve_review_operation(
        store,
        candidate.metadata.artifact_id,
        decision="reject",
        rationale="The pattern is real but not reusable.",
        gate_path=tmp_path / "approvals.json",
    )

    rejected = store.reject(
        candidate.metadata.artifact_id,
        expected_digest=digest,
        approval_gate=gate,
        approval_request_id=request_id,
    )

    assert rejected.path.parent == store.rejected_root
    record = store.show(candidate.metadata.artifact_id)
    assert record is not None
    assert record.status == "rejected"
    assert record.review is not None
    assert record.review["decision"] == "rejected"
    assert record.review["request_id"] == request_id
    assert record.review["human_confirmed"] is True


@pytest.mark.parametrize("archived_status", ["accepted", "rejected"])
def test_candidate_reuse_reports_existing_archived_evidence_truthfully(
    tmp_path: Path,
    archived_status: str,
) -> None:
    context, alpha_record, store, first_packet = _failed_alpha_packet(tmp_path)
    first_payload = reflect_evidence(first_packet)
    assert first_payload is not None
    first_result = store.create_candidate_result(first_payload, evidence=first_packet)
    candidate_id = first_result.artifact.metadata.artifact_id
    digest = store.content_digest(candidate_id)
    if archived_status == "accepted":
        gate, request_id = _approve_review_operation(
            store,
            candidate_id,
            decision="accept",
            rationale="Promote the stable failure pattern.",
            gate_path=tmp_path / "approvals.json",
        )
        store.accept(
            candidate_id,
            expected_digest=digest,
            approval_gate=gate,
            approval_request_id=request_id,
        )
    else:
        store.reject(candidate_id, expected_digest=digest)

    _append(
        context,
        event_id="later-same-failure",
        timestamp="2026-07-17T03:00:00+00:00",
        scope_id=alpha_record.scope_id,
        project_id=alpha_record.project_id,
    )
    inspected_packet = store.build_evidence_packet()
    later_payload = reflect_evidence(inspected_packet)
    assert later_payload is not None

    reused = store.create_candidate_result(later_payload, evidence=inspected_packet)
    record = store.show(reused.artifact.metadata.artifact_id)

    assert reused.created is False
    assert record is not None
    assert record.status == archived_status
    assert reused.bound_evidence_digest == first_packet.evidence_digest
    assert reused.bound_evidence_digest != inspected_packet.evidence_digest


def test_reject_is_idempotent_and_never_creates_memory(tmp_path: Path) -> None:
    context, alpha_record, store, packet = _failed_alpha_packet(tmp_path)
    payload = reflect_evidence(packet)
    assert payload is not None
    candidate = store.create_candidate(payload, evidence=packet)
    digest = store.content_digest(candidate.metadata.artifact_id)

    rejected = store.reject(candidate.metadata.artifact_id, expected_digest=digest)
    repeated = store.reject(candidate.metadata.artifact_id, expected_digest=digest)

    assert rejected.path == repeated.path
    assert rejected.path.parent == store.rejected_root
    assert NoteStore(context, scope_id=alpha_record.scope_id).list() == []
    assert store.create_candidate(payload, evidence=packet).path == rejected.path
    assert store.list() == []
    record = store.show(candidate.metadata.artifact_id)
    assert record is not None
    assert record.review is not None
    assert record.review["decision"] == "rejected"
    assert record.review["via"] == "programmatic"
    assert record.review["human_confirmed"] is False
    with pytest.raises(ValueError, match="cannot be accepted"):
        store.accept(candidate.metadata.artifact_id, expected_digest=digest)


def test_store_fails_closed_across_project_scopes_and_on_v1(tmp_path: Path) -> None:
    context, alpha, _beta, _alpha_record, beta_record, alpha_store, beta_store = _scoped_stores(
        tmp_path
    )
    with pytest.raises(ScopeAuthorizationError):
        InsightStore(
            context,
            scope_id=beta_record.scope_id,
            requester_path=alpha,
        )
    with pytest.raises(PermissionError, match="requester path"):
        InsightStore(context, scope_id=beta_record.scope_id)

    _context, _alpha_record, _store, alpha_packet = _failed_alpha_packet(tmp_path / "other")
    beta_payload = {
        "title": "Wrong scope",
        "insight": "Must not cross scopes.",
        "evidence_ids": list(alpha_packet.evidence_ids),
        "evidence_digest": alpha_packet.evidence_digest,
        "confidence": "low",
    }
    with pytest.raises(PermissionError, match="another insight scope"):
        beta_store.create_candidate(beta_payload, evidence=alpha_packet)

    with pytest.raises(ValueError, match="version 2"):
        InsightStore(tmp_path / "legacy-context")
    assert alpha_store.list() == []
