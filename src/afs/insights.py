"""Scoped, evidence-bound insight candidates with explicit review outcomes.

This module deliberately stops before scheduling, models, or network access.
It builds deterministic packets from already-attributed local history, stores
immutable candidates in the v2 scratchpad, and verifies human provenance only
against an exact persisted approval-gate request.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import threading
import unicodedata
from collections import defaultdict
from collections.abc import Iterable, Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, cast

from .artifacts import MarkdownArtifact, MarkdownArtifactCodec, NoteStore, validate_scope_id
from .context_layout import LAYOUT_VERSION, detect_layout_version
from .history import iter_history_events, read_recent_history_events, resolve_history_root
from .models import ContextCategory
from .project_registry import COMMON_SCOPE_ID, ProjectRegistry
from .response_schemas import validate_structured_response
from .scratchpad import archive_markdown_artifact

if TYPE_CHECKING:
    from .agents.guardrails import ApprovalGate, ApprovalRequest

try:  # POSIX
    import fcntl  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - Windows
    fcntl = None  # type: ignore[assignment]

try:  # Windows
    import msvcrt  # type: ignore[import-not-found]
except ImportError:  # pragma: no cover - POSIX
    msvcrt = None  # type: ignore[assignment]

INSIGHT_EVIDENCE_SCHEMA_VERSION = "1"
INSIGHT_EVENT_SOURCE = "afs.insights"
INSIGHT_REVIEW_AGENT = "insights"
MAX_EVIDENCE_EVENTS = 200
MAX_EVIDENCE_METADATA_CHARS = 256
MAX_INSIGHT_CANDIDATE_BYTES = 64 * 1024
MAX_INSIGHT_BODY_BYTES = 64 * 1024
MAX_INSIGHT_AGENT_NAME_CHARS = 128
MAX_INSIGHT_REVIEW_RATIONALE_CHARS = 4096
MAX_INSIGHT_REVIEW_IDENTITY_CHARS = 512
DEFAULT_SCHEDULED_INSIGHT_HISTORY_WINDOW = 1000
MAX_INSIGHT_HISTORY_WINDOW = 5000

InsightStatus = Literal["pending", "accepted", "rejected"]

_LIFECYCLE_LOCKS_GUARD = threading.Lock()
_LIFECYCLE_LOCKS: dict[str, threading.RLock] = {}

_PAYLOAD_KEYS = frozenset({"payload", "payload_preview", "payload_ref", "payload_sha256"})
_EVIDENCE_METADATA_KEYS = frozenset(
    {
        "agent_name",
        "check_name",
        "check_status",
        "exit_code",
        "ok",
        "operation",
        "outcome",
        "phase",
        "project_id",
        "scope_attribution",
        "scope_id",
        "status",
        "step",
        "tool_name",
        "verification_status",
        "workflow",
    }
)
_FAILURE_WORDS = frozenset(
    {"blocked", "error", "fail", "failed", "failure", "reject", "rejected", "timeout"}
)
_COMPLETION_WORDS = frozenset(
    {
        "complete",
        "completed",
        "done",
        "finish",
        "finished",
        "pass",
        "passed",
        "succeed",
        "succeeded",
        "success",
    }
)


def _contains_unicode_control_or_format(value: str) -> bool:
    """Return whether ``value`` contains terminal-unsafe Unicode characters."""

    return any(unicodedata.category(character) in {"Cc", "Cf"} for character in value)


def _assert_candidate_text_safe(name: str, value: str) -> None:
    """Reject text that cannot be rendered byte-for-byte for human review."""

    if _contains_unicode_control_or_format(value):
        raise ValueError(
            f"candidate {name} cannot contain Unicode control or format characters"
        )


def assert_insight_artifact_reviewable(artifact: MarkdownArtifact) -> None:
    """Require one persisted candidate to be safe and complete for terminal review.

    Candidate payload fields may not contain control or format characters.  The
    renderer itself adds LF separators, so LF is the only control character
    permitted in the final Markdown artifact.  This also protects promotion of
    legacy or externally modified candidates that bypassed creation validation.
    """

    _assert_candidate_text_safe("title", artifact.metadata.title)
    body_size = len(artifact.body.encode("utf-8"))
    if body_size > MAX_INSIGHT_BODY_BYTES:
        raise ValueError(
            f"candidate body exceeds the {MAX_INSIGHT_BODY_BYTES}-byte review limit"
        )
    if any(
        character != "\n" and unicodedata.category(character) in {"Cc", "Cf"}
        for character in artifact.body
    ):
        raise ValueError(
            "candidate body contains Unicode control or format characters and "
            "cannot be rendered exactly for review"
        )


class InsightContentChangedError(ValueError):
    """Raised when a reviewed candidate no longer matches its approved snapshot."""


@dataclass(frozen=True)
class InsightReview:
    """Non-authoritative provenance for a programmatic candidate rejection.

    Human provenance is never accepted through caller-claimable fields.  It is
    derived from an exact persisted :class:`ApprovalRequest` by
    :meth:`InsightStore.accept` or :meth:`InsightStore.reject`.
    """

    rationale: str = ""
    request_id: str = ""
    reviewer: str = ""
    via: str = "programmatic"
    authenticated: bool = False
    human_confirmed: bool = False

    def to_dict(self) -> dict[str, str | bool]:
        raw_strings = {
            "rationale": self.rationale,
            "request_id": self.request_id,
            "reviewer": self.reviewer,
            "via": self.via,
        }
        for name, value in raw_strings.items():
            if not isinstance(value, str):
                raise TypeError(f"review {name} must be a string")
            if _contains_unicode_control_or_format(value):
                raise ValueError(
                    f"review {name} cannot contain Unicode control or format characters"
                )
        if type(self.authenticated) is not bool or type(self.human_confirmed) is not bool:
            raise TypeError("review authenticated and human_confirmed must be literal booleans")
        fields: dict[str, str | bool] = {
            "rationale": self.rationale.strip(),
            "request_id": self.request_id.strip(),
            "reviewer": self.reviewer.strip(),
            "via": self.via.strip(),
            "authenticated": self.authenticated,
            "human_confirmed": self.human_confirmed,
        }
        if len(self.rationale) > MAX_INSIGHT_REVIEW_RATIONALE_CHARS:
            raise ValueError(
                f"review rationale exceeds the {MAX_INSIGHT_REVIEW_RATIONALE_CHARS}-character limit"
            )
        if any(
            len(value) > MAX_INSIGHT_REVIEW_IDENTITY_CHARS
            for value in (self.request_id, self.reviewer, self.via)
        ):
            raise ValueError(
                "review identity field exceeds the "
                f"{MAX_INSIGHT_REVIEW_IDENTITY_CHARS}-character limit"
            )
        if fields["via"] != "programmatic":
            raise ValueError("direct insight reviews must use via='programmatic'")
        if fields["request_id"]:
            raise ValueError(
                "approval request IDs must be verified through approval_gate and "
                "approval_request_id"
            )
        if self.authenticated or self.human_confirmed:
            raise ValueError("human provenance must be derived from a persisted approval request")
        return fields


def insight_review_gate_binding(
    store: InsightStore,
    record: InsightRecord,
    *,
    decision: Literal["accept", "reject"],
    rationale: str,
) -> tuple[str, str]:
    """Return the exact approval-gate action and detail for one review."""

    if decision not in {"accept", "reject"}:  # pragma: no cover - Literal callers
        raise ValueError("insight decision must be accept or reject")
    if not isinstance(rationale, str) or not rationale.strip():
        raise ValueError("a rationale is required for an insight review")
    normalized_rationale = rationale.strip()
    artifact = record.artifact
    if artifact.metadata.scope_id != store.scope_id:
        raise PermissionError("insight review record belongs to another store scope")
    store_identity = store.review_store_identity
    content_digest = record.content_digest
    rationale_digest = hashlib.sha256(normalized_rationale.encode("utf-8")).hexdigest()
    action = (
        f"insight_{decision}_{artifact.metadata.artifact_id}_"
        f"{content_digest[:16]}_{rationale_digest[:12]}_{store_identity[:12]}"
    )
    detail = ":".join(
        [
            store_identity,
            artifact.metadata.scope_id,
            artifact.metadata.artifact_id,
            content_digest,
            rationale_digest,
        ]
    )
    return action, detail


def _assert_verified_approval_request(
    store: InsightStore,
    record: InsightRecord,
    *,
    decision: Literal["accept", "reject"],
    request: ApprovalRequest,
) -> None:
    string_fields = {
        "agent": request.agent,
        "action": request.action,
        "detail": request.detail,
        "status": request.status,
        "reviewed_by": request.reviewed_by,
        "rationale": request.rationale,
        "request_id": request.request_id,
        "reviewed_via": request.reviewed_via,
        "reviewer_subject": request.reviewer_subject,
    }
    if any(not isinstance(value, str) for value in string_fields.values()):
        raise ValueError("persisted insight approval contains non-string fields")
    if (
        type(request.identity_authenticated) is not bool
        or type(request.human_confirmed) is not bool
    ):
        raise ValueError("persisted insight approval contains non-boolean provenance")
    if request.agent != INSIGHT_REVIEW_AGENT:
        raise ValueError("persisted insight approval belongs to another agent")
    # The gate authorizes performing the review operation.  Candidate accept
    # versus reject remains the decision recorded by the insight lifecycle.
    if request.status != "approved":
        raise ValueError("persisted insight review operation was not approved")
    if request.identity_authenticated is not True or request.human_confirmed is not True:
        raise ValueError("persisted insight approval is not human-confirmed")
    if request.reviewed_via != "controlling_terminal":
        raise ValueError("persisted insight approval was not confirmed at a terminal")
    if not request.reviewed_by.strip() or not request.reviewer_subject.strip():
        raise ValueError("persisted insight approval has no authenticated reviewer identity")
    if not request.rationale or request.rationale != request.rationale.strip():
        raise ValueError("persisted insight approval has no canonical rationale")
    persisted_review_fields = {
        "rationale": request.rationale,
        "request_id": request.request_id,
        "reviewer": request.reviewed_by,
        "reviewed_via": request.reviewed_via,
        "reviewer_subject": request.reviewer_subject,
    }
    for name, value in persisted_review_fields.items():
        if _contains_unicode_control_or_format(value):
            raise ValueError(
                "persisted insight approval "
                f"{name} contains Unicode control or format characters"
            )
    if len(request.rationale) > MAX_INSIGHT_REVIEW_RATIONALE_CHARS:
        raise ValueError("persisted insight approval rationale exceeds the review limit")
    if any(
        len(value) > MAX_INSIGHT_REVIEW_IDENTITY_CHARS
        for value in (
            request.request_id,
            request.reviewed_by,
            request.reviewed_via,
            request.reviewer_subject,
        )
    ):
        raise ValueError("persisted insight approval identity exceeds the review limit")
    expected_action, expected_detail = insight_review_gate_binding(
        store,
        record,
        decision=decision,
        rationale=request.rationale,
    )
    if request.action != expected_action or request.detail != expected_detail:
        raise ValueError(
            "persisted insight approval does not match the candidate, digest, and rationale"
        )


def _thread_lifecycle_lock(path: Path) -> threading.RLock:
    key = str(path)
    with _LIFECYCLE_LOCKS_GUARD:
        return _LIFECYCLE_LOCKS.setdefault(key, threading.RLock())


def _lock_fd(fd: int) -> None:
    if fcntl is not None:
        fcntl.flock(fd, fcntl.LOCK_EX)
        return
    if msvcrt is not None:  # pragma: no cover - Windows
        if os.fstat(fd).st_size == 0:
            os.write(fd, b"0")
            os.fsync(fd)
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_LOCK, 1)
        return
    raise RuntimeError("no supported lifecycle lock backend is available")


def _unlock_fd(fd: int) -> None:
    if fcntl is not None:
        fcntl.flock(fd, fcntl.LOCK_UN)
    elif msvcrt is not None:  # pragma: no cover - Windows
        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)


def _canonical_json(value: Any) -> str:
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )


def _packet_digest(scope_id: str, events: Iterable[Mapping[str, Any]]) -> str:
    payload = {
        "schema_version": INSIGHT_EVIDENCE_SCHEMA_VERSION,
        "scope_id": scope_id,
        "events": list(events),
    }
    return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()


def _candidate_content_digest(artifact: MarkdownArtifact) -> str:
    canonical = {
        "metadata": artifact.metadata.to_dict(),
        "body": artifact.body,
    }
    return hashlib.sha256(_canonical_json(canonical).encode("utf-8")).hexdigest()


def _without_payloads(value: Any) -> Any:
    """Return a detached JSON value with history payload fields removed."""

    if isinstance(value, Mapping):
        return {
            str(key): _without_payloads(item)
            for key, item in value.items()
            if isinstance(key, str) and key not in _PAYLOAD_KEYS
        }
    if isinstance(value, list):
        return [_without_payloads(item) for item in value]
    return value


def _bounded_metadata_value(value: Any) -> str | bool | int | float | None:
    """Keep only compact scalar evidence metadata."""

    if isinstance(value, str):
        return value[:MAX_EVIDENCE_METADATA_CHARS]
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return value
    raise TypeError("evidence metadata must be a JSON scalar")


def _event_scope_id(event: Mapping[str, Any]) -> str | None:
    """Return one unambiguous explicit event scope, otherwise fail closed."""

    metadata = event.get("metadata")
    if not isinstance(metadata, Mapping):
        return None
    metadata_scope = metadata.get("scope_id")
    if (
        not isinstance(metadata_scope, str)
        or not metadata_scope
        or metadata_scope != metadata_scope.strip()
    ):
        return None
    try:
        scope_id = validate_scope_id(metadata_scope)
    except ValueError:
        return None
    event_scope = event.get("scope_id")
    if event_scope not in (None, ""):
        if not isinstance(event_scope, str):
            return None
        try:
            if event_scope != event_scope.strip() or validate_scope_id(event_scope) != scope_id:
                return None
        except ValueError:
            return None

    raw_project_id = metadata.get("project_id")
    if scope_id.startswith("project:"):
        if not isinstance(raw_project_id, str):
            return None
        if raw_project_id != scope_id.removeprefix("project:"):
            return None
        if metadata.get("scope_attribution") != "registry":
            return None
    else:
        if raw_project_id not in (None, ""):
            return None
        if metadata.get("scope_attribution") != "common":
            return None
    return scope_id


def _is_insight_event(event: Mapping[str, Any]) -> bool:
    source = event.get("source")
    if not isinstance(source, str):
        return False
    return any(source.startswith(prefix) for prefix in (INSIGHT_EVENT_SOURCE, "agent.insights"))


def _evidence_event(
    event: Mapping[str, Any],
    *,
    visible_scopes: frozenset[str],
) -> dict[str, Any] | None:
    if _is_insight_event(event):
        return None
    scope_id = _event_scope_id(event)
    if scope_id is None or scope_id not in visible_scopes:
        return None

    event_id = event.get("id")
    timestamp = event.get("timestamp")
    event_type = event.get("type")
    source = event.get("source")
    metadata = event.get("metadata")
    if not isinstance(event_id, str) or not event_id.strip():
        return None
    if not isinstance(timestamp, str) or not timestamp.strip():
        return None
    if not isinstance(event_type, str) or not event_type.strip():
        return None
    if not isinstance(source, str) or not source.strip():
        return None
    if not isinstance(metadata, Mapping):
        return None
    op = event.get("op")
    if op is not None and not isinstance(op, str):
        return None

    sanitized_metadata: dict[str, Any] = {}
    for key, value in metadata.items():
        if not isinstance(key, str) or key not in _EVIDENCE_METADATA_KEYS:
            continue
        try:
            sanitized_metadata[key] = _bounded_metadata_value(_without_payloads(value))
        except TypeError:
            continue
    try:
        # A canonical round trip detaches caller-owned objects and rejects
        # non-JSON values instead of producing a non-reproducible packet.
        canonical_metadata = json.loads(_canonical_json(sanitized_metadata))
    except (TypeError, ValueError):
        return None
    if not isinstance(canonical_metadata, dict):
        return None
    return {
        "id": event_id.strip(),
        "timestamp": timestamp.strip(),
        "type": event_type.strip(),
        "source": source.strip(),
        "op": op.strip() if isinstance(op, str) and op.strip() else None,
        "scope_id": scope_id,
        "metadata": canonical_metadata,
    }


@dataclass(frozen=True)
class InsightEvidencePacket:
    """Deterministic, payload-free local history evidence."""

    schema_version: str
    scope_id: str
    evidence_ids: tuple[str, ...]
    evidence_digest: str
    events: tuple[dict[str, Any], ...]

    def assert_valid(self) -> None:
        """Reject a packet mutated after construction or bound to another scope."""

        if self.schema_version != INSIGHT_EVIDENCE_SCHEMA_VERSION:
            raise ValueError(f"unsupported insight evidence schema_version: {self.schema_version}")
        canonical_scope = validate_scope_id(self.scope_id)
        if len(self.events) > MAX_EVIDENCE_EVENTS:
            raise ValueError(f"insight evidence packet exceeds {MAX_EVIDENCE_EVENTS} events")
        canonical_events: list[dict[str, Any]] = []
        for event in self.events:
            normalized = _evidence_event(
                event,
                visible_scopes=frozenset({canonical_scope}),
            )
            if normalized is None or normalized != event:
                raise ValueError(
                    "insight evidence event is unscoped, self-authored, or non-canonical"
                )
            canonical_events.append(normalized)
        ordered = sorted(
            canonical_events,
            key=lambda event: (
                str(event["timestamp"]),
                str(event["id"]),
                _canonical_json(event),
            ),
        )
        if ordered != canonical_events:
            raise ValueError("insight evidence events are not in canonical order")
        event_ids = tuple(str(event.get("id", "")).strip() for event in self.events)
        if any(not event_id for event_id in event_ids):
            raise ValueError("insight evidence packet must contain identified events")
        if len(event_ids) != len(set(event_ids)):
            raise ValueError("insight evidence packet contains duplicate event IDs")
        if event_ids != self.evidence_ids:
            raise ValueError("insight evidence IDs do not match the packet events")
        digest = _packet_digest(canonical_scope, self.events)
        if digest != self.evidence_digest:
            raise ValueError("insight evidence digest does not match the packet events")

    def to_dict(self) -> dict[str, Any]:
        self.assert_valid()
        return {
            "schema_version": self.schema_version,
            "scope_id": self.scope_id,
            "evidence_ids": list(self.evidence_ids),
            "evidence_digest": self.evidence_digest,
            "events": json.loads(_canonical_json(self.events)),
        }


@dataclass(frozen=True)
class InsightRecord:
    """One candidate plus the lifecycle directory that currently owns it."""

    status: InsightStatus
    artifact: MarkdownArtifact
    review: dict[str, Any] | None = None

    @property
    def content_digest(self) -> str:
        """Canonical metadata/body digest for binding a review decision."""

        return _candidate_content_digest(self.artifact)

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "content_digest": self.content_digest,
            "review": dict(self.review) if self.review is not None else None,
            **self.artifact.to_dict(),
        }


@dataclass(frozen=True)
class InsightCandidateResult:
    """Truthful outcome of one idempotent candidate creation attempt."""

    artifact: MarkdownArtifact
    created: bool

    @property
    def bound_evidence_digest(self) -> str:
        """Evidence digest permanently bound to the returned artifact."""

        provenance = self.artifact.metadata.provenance or {}
        digest = provenance.get("evidence_digest")
        if (
            not isinstance(digest, str)
            or len(digest) != 64
            or any(character not in "0123456789abcdef" for character in digest)
        ):
            raise ValueError("insight candidate has invalid bound evidence provenance")
        return digest


def _outcome(event: Mapping[str, Any]) -> Literal["failure", "completion", "activity"]:
    metadata = event.get("metadata")
    if not isinstance(metadata, Mapping):
        metadata = {}
    op = str(event.get("op") or "").strip().casefold()
    status = str(metadata.get("status") or "").strip().casefold()
    outcome = str(metadata.get("outcome") or "").strip().casefold()
    check_status = str(metadata.get("check_status") or "").strip().casefold()
    verification_status = str(metadata.get("verification_status") or "").strip().casefold()
    ok = metadata.get("ok")
    exit_code = metadata.get("exit_code")
    words = {op, status, outcome, check_status, verification_status}
    if ok is False or (
        isinstance(exit_code, int) and not isinstance(exit_code, bool) and exit_code != 0
    ):
        return "failure"
    if words.intersection(_FAILURE_WORDS):
        return "failure"
    if ok is True or (
        isinstance(exit_code, int) and not isinstance(exit_code, bool) and exit_code == 0
    ):
        return "completion"
    if words.intersection(_COMPLETION_WORDS):
        return "completion"
    return "activity"


def reflect_evidence(packet: InsightEvidencePacket) -> dict[str, Any] | None:
    """Derive one deterministic frequency insight without a model or network.

    The initial policy deliberately emits only repeated failures.  Normal
    completions and generic recurring activity are too common for a
    frequency-only detector: a rolling history window would otherwise create
    a new candidate whenever another healthy event arrives.  Ties between
    failure patterns are resolved lexically after frequency so the same packet
    always yields the same candidate payload.
    """

    packet.assert_valid()
    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for event in packet.events:
        outcome = _outcome(event)
        group_key = (
            outcome,
            str(event.get("type") or "unknown"),
            str(event.get("source") or "unknown"),
            str(event.get("op") or "event"),
        )
        grouped[group_key].append(event)

    eligible = [
        (key, events) for key, events in grouped.items() if key[0] == "failure" and len(events) >= 2
    ]
    if not eligible:
        return None
    selected_key, events = min(
        eligible,
        key=lambda item: (
            -len(item[1]),
            item[0][1],
            item[0][2],
            item[0][3],
        ),
    )
    selected_outcome, event_type, source, op = selected_key
    # Cite the first threshold-establishing pair rather than the full rolling
    # series.  The candidate stays stable when another event in the same
    # failure pattern arrives, while its packet provenance still records the
    # complete evidence snapshot used during the first creation.
    evidence_ids = [str(event["id"]) for event in events[:2]]
    label = "Repeated failure"
    insight = (
        f"{source} repeatedly recorded {selected_outcome} events for "
        f"{event_type}:{op} in attributed local history."
    )
    next_step = "Review the repeated failure before changing automation."
    return {
        "title": f"{label}: {source} {op}",
        "insight": insight,
        "evidence_ids": evidence_ids,
        "evidence_digest": packet.evidence_digest,
        "confidence": "medium",
        "limitations": ["Frequency alone does not establish causation."],
        "next_step": next_step,
    }


class InsightStore:
    """Store and review insight candidates within one authorized v2 scope."""

    def __init__(
        self,
        context_path: Path,
        *,
        scope_id: str = COMMON_SCOPE_ID,
        requester_path: Path | None = None,
        config: Any = None,
    ) -> None:
        self.context_path = context_path.expanduser().resolve()
        if detect_layout_version(self.context_path) != LAYOUT_VERSION:
            raise ValueError("insight artifacts require a version 2 context")
        self.scope_id = validate_scope_id(scope_id)
        if self.scope_id != COMMON_SCOPE_ID and requester_path is None:
            raise PermissionError("project-scoped insights require a requester path")
        self.requester_path = (
            requester_path.expanduser().resolve()
            if requester_path is not None
            else self.context_path
        )
        registry = ProjectRegistry(self.context_path)
        resolved_scope, scratchpad_scope_root = registry.resolve_scope_root(
            ContextCategory.SCRATCHPAD,
            requester_path=self.requester_path,
            scope_id=self.scope_id,
        )
        if resolved_scope != self.scope_id:  # pragma: no cover - registry contract
            raise PermissionError("resolved insight scope does not match the requested scope")

        self.root = scratchpad_scope_root / "insights"
        self.pending_root = self.root / "candidates"
        self.accepted_root = self.root / "archive" / "accepted"
        self.rejected_root = self.root / "archive" / "rejected"
        self.decisions_root = self.root / "decisions"
        self._pending = MarkdownArtifactCodec(self.pending_root)
        self._accepted = MarkdownArtifactCodec(self.accepted_root)
        self._rejected = MarkdownArtifactCodec(self.rejected_root)
        self._decisions = MarkdownArtifactCodec(self.decisions_root)
        self._config = config

    @property
    def project_id(self) -> str:
        return (
            self.scope_id.removeprefix("project:") if self.scope_id.startswith("project:") else ""
        )

    @property
    def review_store_identity(self) -> str:
        """Stable digest binding approvals to this canonical context store."""

        payload = {
            "schema": "afs.insights.review-store.v1",
            "context_path": str(self.context_path),
            "insights_root": str(self.root.resolve()),
            "scope_id": self.scope_id,
        }
        return hashlib.sha256(_canonical_json(payload).encode("utf-8")).hexdigest()

    @property
    def visible_scope_ids(self) -> frozenset[str]:
        return frozenset({self.scope_id})

    @contextmanager
    def _lifecycle_transaction(self) -> Iterator[None]:
        """Serialize lifecycle mutations within this exact scope."""

        thread_lock = _thread_lifecycle_lock(self.pending_root / ".lifecycle.lock")
        with thread_lock:
            fd = self._pending._open_control_file(
                ".lifecycle.lock",
                os.O_RDWR | os.O_CREAT,
            )
            locked = False
            try:
                _lock_fd(fd)
                locked = True
                yield
            finally:
                if locked:
                    _unlock_fd(fd)
                os.close(fd)

    def build_evidence_packet(
        self,
        *,
        limit: int = 100,
        event_types: Iterable[str] | None = None,
        recent_history_limit: int | None = None,
    ) -> InsightEvidencePacket:
        """Build a bounded deterministic packet from explicitly scoped history.

        ``recent_history_limit`` opts into a bounded append-order read before
        scope filtering.  Interactive callers retain the complete-history
        behavior by default; scheduled reflection uses the bounded window.
        """

        if limit <= 0 or limit > MAX_EVIDENCE_EVENTS:
            raise ValueError(f"evidence limit must be between 1 and {MAX_EVIDENCE_EVENTS}")
        if recent_history_limit is not None and (
            isinstance(recent_history_limit, bool)
            or not isinstance(recent_history_limit, int)
            or recent_history_limit <= 0
            or recent_history_limit > MAX_INSIGHT_HISTORY_WINDOW
        ):
            raise ValueError(
                "recent history limit must be between 1 and "
                f"{MAX_INSIGHT_HISTORY_WINDOW}"
            )
        normalized_types = {
            event_type.strip()
            for event_type in (event_types or ())
            if isinstance(event_type, str) and event_type.strip()
        }
        history_root = resolve_history_root(self.context_path, config=self._config)
        by_id: dict[str, dict[str, Any]] = {}
        if recent_history_limit is None:
            history_events = iter_history_events(
                history_root,
                event_types=normalized_types or None,
                include_payloads=False,
            )
        else:
            history_events = read_recent_history_events(
                history_root,
                limit=recent_history_limit,
                event_types=normalized_types or None,
            )
        for event in history_events:
            if not isinstance(event, Mapping):
                continue
            evidence = _evidence_event(event, visible_scopes=self.visible_scope_ids)
            if evidence is None:
                continue
            event_id = str(evidence["id"])
            previous = by_id.get(event_id)
            if previous is not None and previous != evidence:
                raise ValueError(f"conflicting history events share evidence ID {event_id!r}")
            by_id[event_id] = evidence

        events = sorted(
            by_id.values(),
            key=lambda event: (
                str(event["timestamp"]),
                str(event["id"]),
                _canonical_json(event),
            ),
        )[-limit:]
        evidence_ids = tuple(str(event["id"]) for event in events)
        digest = _packet_digest(self.scope_id, events)
        return InsightEvidencePacket(
            schema_version=INSIGHT_EVIDENCE_SCHEMA_VERSION,
            scope_id=self.scope_id,
            evidence_ids=evidence_ids,
            evidence_digest=digest,
            events=tuple(events),
        )

    def create_candidate(
        self,
        payload: Mapping[str, Any],
        *,
        evidence: InsightEvidencePacket,
        agent_name: str = "",
        sensitivity: str = "internal",
        recent_history_limit: int | None = None,
    ) -> MarkdownArtifact:
        """Compatibility wrapper returning the created or existing artifact."""

        return self.create_candidate_result(
            payload,
            evidence=evidence,
            agent_name=agent_name,
            sensitivity=sensitivity,
            recent_history_limit=recent_history_limit,
        ).artifact

    def create_candidate_result(
        self,
        payload: Mapping[str, Any],
        *,
        evidence: InsightEvidencePacket,
        agent_name: str = "",
        sensitivity: str = "internal",
        recent_history_limit: int | None = None,
    ) -> InsightCandidateResult:
        """Validate a candidate and report whether this call created it."""

        evidence.assert_valid()
        if evidence.scope_id != self.scope_id:
            raise PermissionError("evidence packet belongs to another insight scope")
        if recent_history_limit is not None and (
            isinstance(recent_history_limit, bool)
            or not isinstance(recent_history_limit, int)
            or recent_history_limit <= 0
            or recent_history_limit > MAX_INSIGHT_HISTORY_WINDOW
        ):
            raise ValueError(
                "recent history limit must be between 1 and "
                f"{MAX_INSIGHT_HISTORY_WINDOW}"
            )
        self._verify_local_evidence(
            evidence,
            recent_history_limit=recent_history_limit,
        )
        if not isinstance(agent_name, str):
            raise TypeError("insight agent_name must be a string")
        normalized_agent_name = agent_name.strip()
        if len(normalized_agent_name) > MAX_INSIGHT_AGENT_NAME_CHARS:
            raise ValueError(
                f"insight agent_name exceeds {MAX_INSIGHT_AGENT_NAME_CHARS} characters"
            )
        result = validate_structured_response("insight-candidate", dict(payload))
        if not result.valid or not isinstance(result.parsed, dict):
            details = result.parse_error or "; ".join(result.errors[:5])
            raise ValueError(f"invalid insight candidate: {details}")
        canonical_candidate = _canonical_json(result.parsed)
        candidate_size = len(canonical_candidate.encode("utf-8"))
        if candidate_size > MAX_INSIGHT_CANDIDATE_BYTES:
            raise ValueError(
                "insight candidate exceeds the "
                f"{MAX_INSIGHT_CANDIDATE_BYTES}-byte canonical payload limit"
            )
        detached = json.loads(canonical_candidate)
        if not isinstance(detached, dict):  # pragma: no cover - validator contract
            raise ValueError("insight candidate must be a JSON object")
        candidate = cast(dict[str, Any], detached)
        evidence_digest = candidate.get("evidence_digest")
        if evidence_digest != evidence.evidence_digest:
            raise ValueError("candidate evidence_digest does not match the evidence packet")
        raw_ids = candidate.get("evidence_ids")
        if not isinstance(raw_ids, list) or any(
            not isinstance(event_id, str) for event_id in raw_ids
        ):
            raise ValueError("candidate evidence_ids must be strings")
        evidence_ids = cast(list[str], raw_ids)
        unknown_ids = sorted(set(evidence_ids).difference(evidence.evidence_ids))
        if unknown_ids:
            raise ValueError(
                "candidate cites evidence outside the packet: " + ", ".join(unknown_ids)
            )

        title = candidate.get("title")
        insight = candidate.get("insight")
        confidence = candidate.get("confidence")
        limitations = candidate.get("limitations", [])
        next_step = candidate.get("next_step", "")
        if not isinstance(title, str):  # narrowed separately for mypy and defense in depth
            raise ValueError("candidate title must be a string")
        if not isinstance(insight, str):
            raise ValueError("candidate insight must be a string")
        if not isinstance(confidence, str):
            raise ValueError("candidate confidence must be a string")
        if not isinstance(limitations, list) or any(
            not isinstance(item, str) for item in limitations
        ):
            raise ValueError("candidate limitations must be strings")
        if not isinstance(next_step, str):
            raise ValueError("candidate next_step must be a string")

        _assert_candidate_text_safe("title", title)
        _assert_candidate_text_safe("insight", insight)
        for index, limitation in enumerate(cast(list[str], limitations)):
            _assert_candidate_text_safe(f"limitations[{index}]", limitation)
        _assert_candidate_text_safe("next_step", next_step)
        for index, evidence_id in enumerate(evidence_ids):
            _assert_candidate_text_safe(f"evidence_ids[{index}]", evidence_id)
        _assert_candidate_text_safe("agent_name", normalized_agent_name)

        candidate_digest = hashlib.sha256(canonical_candidate.encode("utf-8")).hexdigest()
        identity_payload = dict(candidate)
        identity_payload.pop("evidence_digest", None)
        candidate_identity = hashlib.sha256(
            _canonical_json(identity_payload).encode("utf-8")
        ).hexdigest()
        body = _render_candidate_body(
            insight=insight,
            evidence_ids=evidence_ids,
            evidence_digest=evidence.evidence_digest,
            confidence=confidence,
            limitations=cast(list[str], limitations),
            next_step=next_step,
        )
        body_size = len(body.encode("utf-8"))
        if body_size > MAX_INSIGHT_BODY_BYTES:
            raise ValueError(
                f"insight candidate exceeds the {MAX_INSIGHT_BODY_BYTES}-byte rendered body limit"
            )
        with self._lifecycle_transaction():
            existing = self._candidate_by_identity(candidate_identity)
            if existing is not None:
                return InsightCandidateResult(artifact=existing, created=False)
            artifact_id = hashlib.sha256(
                f"insight-candidate\0{self.scope_id}\0{candidate_identity}".encode()
            ).hexdigest()[:32]
            try:
                artifact = self._pending.create(
                    kind="insight-candidate",
                    title=title,
                    body=body,
                    scope_id=self.scope_id,
                    project_id=self.project_id,
                    agent_name=normalized_agent_name,
                    author_kind="agent",
                    sensitivity=sensitivity,
                    provenance={
                        "source": "afs.insights.reflect",
                        "schema": "insight-candidate",
                        "candidate_sha256": candidate_digest,
                        "candidate_identity_sha256": candidate_identity,
                        "evidence_digest": evidence.evidence_digest,
                        "evidence_ids": evidence_ids,
                    },
                    artifact_id=artifact_id,
                )
                return InsightCandidateResult(artifact=artifact, created=True)
            except FileExistsError:
                existing = self._candidate_by_identity(candidate_identity)
                if existing is not None:
                    return InsightCandidateResult(artifact=existing, created=False)
                raise

    def list(
        self,
        *,
        status: InsightStatus | None = "pending",
        limit: int = 100,
    ) -> list[InsightRecord]:
        """List newest candidates in one lifecycle state, or all states."""

        if limit <= 0:
            return []
        states: tuple[InsightStatus, ...] = (
            ("pending", "accepted", "rejected") if status is None else (status,)
        )
        reviews = self._review_index()
        records: list[InsightRecord] = []
        for current in states:
            codec = self._codec(current)
            for artifact in codec.iter_artifacts(kind="insight-candidate"):
                if artifact.metadata.scope_id == self.scope_id:
                    records.append(
                        InsightRecord(
                            status=current,
                            artifact=artifact,
                            review=reviews.get(artifact.metadata.artifact_id),
                        )
                    )
        records.sort(
            key=lambda record: (
                record.artifact.metadata.created_at,
                record.artifact.metadata.artifact_id,
            ),
            reverse=True,
        )
        return records[:limit]

    def show(self, identifier: str | Path) -> InsightRecord | None:
        """Read a candidate by artifact ID, filename, or exact current path."""

        raw = str(identifier).strip()
        if not raw:
            return None
        matches: list[InsightRecord] = []
        for status in cast(tuple[InsightStatus, ...], ("pending", "accepted", "rejected")):
            for record in self.list(status=status, limit=1_000_000):
                artifact = record.artifact
                if (
                    artifact.metadata.artifact_id == raw
                    or artifact.path.name == raw
                    or str(artifact.path) == raw
                ):
                    matches.append(record)
        if len(matches) > 1:
            raise ValueError("insight candidate exists in multiple lifecycle states")
        return matches[0] if matches else None

    def content_digest(self, identifier: str | Path) -> str:
        """Read and hash one candidate through its pinned artifact codec."""

        record = self.show(identifier)
        if record is None:
            raise FileNotFoundError(f"insight candidate not found: {identifier}")
        return record.content_digest

    def accept(
        self,
        identifier: str | Path,
        *,
        expected_digest: str,
        approval_gate: ApprovalGate | None = None,
        approval_request_id: str = "",
    ) -> MarkdownArtifact:
        """Human-promote one candidate and archive it idempotently.

        The first and every retrying call must present an exact persisted,
        human-confirmed approval request.  There is intentionally no
        programmatic promotion path: enforcement lives at the memory-writing
        resource rather than only at CLI callers or scheduled agents.
        """

        with self._lifecycle_transaction():
            record = self.show(identifier)
            if record is None:
                raise FileNotFoundError(f"insight candidate not found: {identifier}")
            assert_insight_artifact_reviewable(record.artifact)
            approved_digest = self._assert_expected_digest(record, expected_digest)
            if record.status == "rejected":
                raise ValueError("rejected insight candidate cannot be accepted")
            if approval_gate is None and (
                not isinstance(approval_request_id, str) or not approval_request_id.strip()
            ):
                raise PermissionError(
                    "accepting an insight requires an exact human-approved gate request"
                )
            review_payload = self._resolve_review(
                record,
                decision="accept",
                review=None,
                approval_gate=approval_gate,
                approval_request_id=approval_request_id,
            )
            candidate = record.artifact
            if record.status == "pending":
                candidate = self._archive(
                    candidate,
                    destination="accepted",
                    expected_digest=approved_digest,
                )
            try:
                decision = self._record_decision(
                    candidate,
                    decision="accepted",
                    content_digest=approved_digest,
                    review=review_payload,
                )
            except BaseException:
                if record.status == "pending":
                    archive_markdown_artifact(
                        self._accepted,
                        self._pending,
                        candidate.path.name,
                    )
                raise
            decision_review = self._review_from_decision(decision)
            note = self._accepted_note(candidate, review=decision_review)
            if note is None:
                note = self._create_accepted_note(candidate, review=decision_review)
            return note

    def reject(
        self,
        identifier: str | Path,
        *,
        expected_digest: str,
        review: InsightReview | None = None,
        approval_gate: ApprovalGate | None = None,
        approval_request_id: str = "",
    ) -> MarkdownArtifact:
        """Archive one rejected candidate without creating durable memory."""

        with self._lifecycle_transaction():
            record = self.show(identifier)
            if record is None:
                raise FileNotFoundError(f"insight candidate not found: {identifier}")
            approved_digest = self._assert_expected_digest(record, expected_digest)
            if record.status == "accepted":
                raise ValueError("accepted insight candidate cannot be rejected")
            review_payload = self._resolve_review(
                record,
                decision="reject",
                review=review,
                approval_gate=approval_gate,
                approval_request_id=approval_request_id,
            )
            candidate = record.artifact
            if record.status == "pending":
                candidate = self._archive(
                    candidate,
                    destination="rejected",
                    expected_digest=approved_digest,
                )
            try:
                self._record_decision(
                    candidate,
                    decision="rejected",
                    content_digest=approved_digest,
                    review=review_payload,
                )
            except BaseException:
                if record.status == "pending":
                    archive_markdown_artifact(
                        self._rejected,
                        self._pending,
                        candidate.path.name,
                    )
                raise
            return candidate

    def _codec(self, status: InsightStatus) -> MarkdownArtifactCodec:
        if status == "pending":
            return self._pending
        if status == "accepted":
            return self._accepted
        if status == "rejected":
            return self._rejected
        raise ValueError(f"unknown insight status: {status}")

    def _archive(
        self,
        artifact: MarkdownArtifact,
        *,
        destination: Literal["accepted", "rejected"],
        expected_digest: str,
    ) -> MarkdownArtifact:
        codec = self._accepted if destination == "accepted" else self._rejected
        archive_markdown_artifact(self._pending, codec, artifact.path.name)
        try:
            archived = codec.read(codec.root / artifact.path.name)
            archived_digest = _candidate_content_digest(archived)
            if not hmac.compare_digest(archived_digest, expected_digest):
                raise InsightContentChangedError(
                    "insight candidate changed during review; inspect it and approve a new digest"
                )
            return archived
        except BaseException:
            try:
                archive_markdown_artifact(codec, self._pending, artifact.path.name)
            except BaseException as rollback_error:
                raise RuntimeError(
                    "insight candidate archive verification failed and rollback "
                    "requires manual repair"
                ) from rollback_error
            raise

    @staticmethod
    def _assert_expected_digest(record: InsightRecord, expected_digest: str) -> str:
        if (
            not isinstance(expected_digest, str)
            or len(expected_digest) != 64
            or any(char not in "0123456789abcdef" for char in expected_digest)
        ):
            raise ValueError("expected candidate digest must be 64 lowercase hex characters")
        if not hmac.compare_digest(record.content_digest, expected_digest):
            raise InsightContentChangedError(
                "insight candidate changed after review; inspect it and approve a new digest"
            )
        return expected_digest

    def _decision_for(self, candidate_id: str) -> MarkdownArtifact | None:
        matches: list[MarkdownArtifact] = []
        for artifact in self._decisions.iter_artifacts(kind="insight-decision"):
            if artifact.metadata.scope_id != self.scope_id:
                continue
            provenance = artifact.metadata.provenance or {}
            if provenance.get("source_artifact_id") == candidate_id:
                matches.append(artifact)
        if len(matches) > 1:
            raise ValueError("insight candidate has multiple decision artifacts")
        return matches[0] if matches else None

    def _resolve_review(
        self,
        record: InsightRecord,
        *,
        decision: Literal["accept", "reject"],
        review: InsightReview | None,
        approval_gate: ApprovalGate | None,
        approval_request_id: str,
    ) -> dict[str, str | bool]:
        if not isinstance(approval_request_id, str):
            raise TypeError("approval_request_id must be a string")
        has_gate = approval_gate is not None
        has_request = bool(approval_request_id.strip())
        if has_gate != has_request:
            raise ValueError("approval_gate and approval_request_id must be supplied together")
        if not has_gate:
            return (review or InsightReview()).to_dict()
        if review is not None:
            raise ValueError(
                "human review provenance is derived from the persisted approval request"
            )
        return self._verified_approval_review(
            record,
            decision=decision,
            approval_gate=approval_gate,
            approval_request_id=approval_request_id.strip(),
        )

    def _verified_approval_review(
        self,
        record: InsightRecord,
        *,
        decision: Literal["accept", "reject"],
        approval_gate: ApprovalGate | None,
        approval_request_id: str,
    ) -> dict[str, str | bool]:
        from .agents.guardrails import ApprovalGate

        if not isinstance(approval_gate, ApprovalGate):
            raise TypeError("approval_gate must be an ApprovalGate")
        matches = [
            request
            for request in approval_gate.all_requests()
            if request.request_id == approval_request_id
        ]
        if len(matches) != 1:
            raise ValueError("exact persisted insight approval request was not found")
        request = matches[0]
        _assert_verified_approval_request(
            self,
            record,
            decision=decision,
            request=request,
        )
        return {
            "rationale": request.rationale,
            "request_id": request.request_id,
            "reviewer": request.reviewed_by,
            "via": request.reviewed_via,
            "authenticated": True,
            "human_confirmed": True,
        }

    @staticmethod
    def _review_from_decision(decision: MarkdownArtifact) -> dict[str, Any]:
        provenance = decision.metadata.provenance or {}
        if provenance.get("source") != "afs.insights.review":
            raise ValueError("insight decision has invalid source provenance")
        required = {
            "decision",
            "candidate_content_digest",
            "rationale",
            "request_id",
            "reviewer",
            "via",
            "authenticated",
            "human_confirmed",
        }
        if not required.issubset(provenance):
            raise ValueError("insight decision provenance is incomplete")
        for key in (
            "decision",
            "candidate_content_digest",
            "rationale",
            "request_id",
            "reviewer",
            "via",
        ):
            if not isinstance(provenance[key], str):
                raise ValueError(f"insight decision {key} must be a string")
        if provenance["decision"] not in {"accepted", "rejected"}:
            raise ValueError("insight decision has an invalid outcome")
        digest = provenance["candidate_content_digest"]
        if len(digest) != 64 or any(char not in "0123456789abcdef" for char in digest):
            raise ValueError("insight decision has an invalid candidate digest")
        if (
            type(provenance["authenticated"]) is not bool
            or type(provenance["human_confirmed"]) is not bool
        ):
            raise ValueError("insight decision review flags must be literal booleans")
        if provenance["human_confirmed"] is True and provenance["authenticated"] is not True:
            raise ValueError("human-confirmed insight decision must be authenticated")
        payload = {key: provenance[key] for key in sorted(required)}
        payload.update(
            {
                "decision_artifact_id": decision.metadata.artifact_id,
                "decision_path": str(decision.path),
            }
        )
        return payload

    def _review_index(self) -> dict[str, dict[str, Any]]:
        reviews: dict[str, dict[str, Any]] = {}
        for artifact in self._decisions.iter_artifacts(kind="insight-decision"):
            if artifact.metadata.scope_id != self.scope_id:
                continue
            provenance = artifact.metadata.provenance or {}
            candidate_id = provenance.get("source_artifact_id")
            if not isinstance(candidate_id, str) or not candidate_id:
                continue
            if candidate_id in reviews:
                raise ValueError("insight candidate has multiple decision artifacts")
            reviews[candidate_id] = self._review_from_decision(artifact)
        return reviews

    def _record_decision(
        self,
        candidate: MarkdownArtifact,
        *,
        decision: Literal["accepted", "rejected"],
        content_digest: str,
        review: Mapping[str, str | bool],
    ) -> MarkdownArtifact:
        expected = {
            "decision": decision,
            "candidate_content_digest": content_digest,
            **dict(review),
        }
        existing = self._decision_for(candidate.metadata.artifact_id)
        if existing is not None:
            actual = self._review_from_decision(existing)
            if any(actual.get(key) != value for key, value in expected.items()):
                raise ValueError("insight candidate has conflicting review provenance")
            return existing

        body = "\n".join(
            [
                f"Decision: {decision}",
                f"Candidate: {candidate.metadata.artifact_id}",
                f"Candidate digest: {content_digest}",
                f"Reviewer: {review['reviewer'] or '(unspecified)'}",
                f"Via: {review['via']}",
                f"Authenticated: {str(review['authenticated']).lower()}",
                f"Human confirmed: {str(review['human_confirmed']).lower()}",
                "",
                "## Rationale",
                str(review["rationale"]) or "(none supplied)",
                "",
            ]
        )
        artifact_id = hashlib.sha256(
            (f"insight-decision\0{self.scope_id}\0{candidate.metadata.artifact_id}").encode()
        ).hexdigest()[:32]
        try:
            decision_title = f"{decision.title()}: {candidate.metadata.title}"[:240].rstrip()
            return self._decisions.create(
                kind="insight-decision",
                title=decision_title,
                body=body,
                scope_id=self.scope_id,
                project_id=self.project_id,
                agent_name=candidate.metadata.agent_name,
                author_kind="system",
                sensitivity=candidate.metadata.sensitivity,
                provenance={
                    "source": "afs.insights.review",
                    "source_artifact_id": candidate.metadata.artifact_id,
                    "source_path": str(candidate.path),
                    **expected,
                },
                artifact_id=artifact_id,
            )
        except FileExistsError:
            existing = self._decision_for(candidate.metadata.artifact_id)
            if existing is not None:
                actual = self._review_from_decision(existing)
                if all(actual.get(key) == value for key, value in expected.items()):
                    return existing
            raise

    def _accepted_note(
        self,
        candidate: MarkdownArtifact,
        *,
        review: Mapping[str, Any],
    ) -> MarkdownArtifact | None:
        notes = NoteStore(
            self.context_path,
            scope_id=self.scope_id,
            config=self._config,
        )
        for note in notes.list(limit=1_000_000):
            provenance = note.metadata.provenance or {}
            if provenance.get("source_artifact_id") != candidate.metadata.artifact_id:
                continue
            candidate_provenance = candidate.metadata.provenance or {}
            if provenance.get("source") != "afs.insights.accept":
                raise ValueError("accepted insight note has invalid source provenance")
            if (
                note.metadata.scope_id != self.scope_id
                or note.metadata.project_id != candidate.metadata.project_id
                or note.metadata.title != candidate.metadata.title
                or note.body != candidate.body
            ):
                raise ValueError("accepted insight note conflicts with the reviewed candidate")
            if provenance.get("evidence_digest") != candidate_provenance.get("evidence_digest"):
                raise ValueError("accepted insight note has conflicting evidence provenance")
            if any(provenance.get(key) != value for key, value in review.items()):
                raise ValueError("accepted insight note has conflicting review provenance")
            return note
        return None

    def _verify_local_evidence(
        self,
        packet: InsightEvidencePacket,
        *,
        recent_history_limit: int | None = None,
    ) -> None:
        """Require every packet event to still match exact scoped local history."""

        wanted_ids = set(packet.evidence_ids)
        local: dict[str, dict[str, Any]] = {}
        history_root = resolve_history_root(self.context_path, config=self._config)
        history_events = (
            iter_history_events(history_root, include_payloads=False)
            if recent_history_limit is None
            else read_recent_history_events(
                history_root,
                limit=recent_history_limit,
            )
        )
        for event in history_events:
            if not isinstance(event, Mapping):
                continue
            event_id = event.get("id")
            if not isinstance(event_id, str) or event_id.strip() not in wanted_ids:
                continue
            canonical = _evidence_event(
                event,
                visible_scopes=self.visible_scope_ids,
            )
            if canonical is None:
                continue
            canonical_id = str(canonical["id"])
            previous = local.get(canonical_id)
            if previous is not None and previous != canonical:
                raise ValueError(
                    f"conflicting local history events share evidence ID {canonical_id!r}"
                )
            local[canonical_id] = canonical

        for expected in packet.events:
            event_id = str(expected["id"])
            actual = local.get(event_id)
            if actual is None:
                raise ValueError(
                    f"evidence event is not present in exact-scope local history: {event_id}"
                )
            if actual != expected:
                raise ValueError(
                    f"evidence event conflicts with exact-scope local history: {event_id}"
                )

    def _candidate_by_identity(self, candidate_identity: str) -> MarkdownArtifact | None:
        for record in self.list(status=None, limit=1_000_000):
            provenance = record.artifact.metadata.provenance or {}
            if provenance.get("candidate_identity_sha256") == candidate_identity:
                return record.artifact
        return None

    def _create_accepted_note(
        self,
        candidate: MarkdownArtifact,
        *,
        review: Mapping[str, Any],
    ) -> MarkdownArtifact:
        provenance = dict(candidate.metadata.provenance or {})
        provenance.update(
            {
                "source": "afs.insights.accept",
                "source_artifact_id": candidate.metadata.artifact_id,
                "source_path": str(candidate.path),
                **dict(review),
            }
        )
        return NoteStore(
            self.context_path,
            scope_id=self.scope_id,
            config=self._config,
        ).create(
            title=candidate.metadata.title,
            body=candidate.body,
            project_id=candidate.metadata.project_id,
            task_id=candidate.metadata.task_id,
            agent_name=candidate.metadata.agent_name,
            author_kind="system",
            sensitivity=candidate.metadata.sensitivity,
            provenance=provenance,
        )


def _render_candidate_body(
    *,
    insight: str,
    evidence_ids: list[str],
    evidence_digest: str,
    confidence: str,
    limitations: list[str],
    next_step: str,
) -> str:
    lines = [insight.strip(), "", "## Evidence"]
    lines.extend(f"- `{event_id}`" for event_id in evidence_ids)
    lines.extend([f"- Digest: `{evidence_digest}`", "", "## Confidence", confidence])
    if limitations:
        lines.extend(["", "## Limitations"])
        lines.extend(f"- {item}" for item in limitations)
    if next_step.strip():
        lines.extend(["", "## Next step", next_step.strip()])
    return "\n".join(lines).rstrip() + "\n"


__all__ = [
    "INSIGHT_EVENT_SOURCE",
    "INSIGHT_EVIDENCE_SCHEMA_VERSION",
    "MAX_EVIDENCE_EVENTS",
    "DEFAULT_SCHEDULED_INSIGHT_HISTORY_WINDOW",
    "MAX_INSIGHT_HISTORY_WINDOW",
    "MAX_EVIDENCE_METADATA_CHARS",
    "MAX_INSIGHT_BODY_BYTES",
    "MAX_INSIGHT_CANDIDATE_BYTES",
    "MAX_INSIGHT_AGENT_NAME_CHARS",
    "MAX_INSIGHT_REVIEW_IDENTITY_CHARS",
    "MAX_INSIGHT_REVIEW_RATIONALE_CHARS",
    "InsightContentChangedError",
    "InsightCandidateResult",
    "InsightEvidencePacket",
    "InsightRecord",
    "InsightReview",
    "InsightStatus",
    "InsightStore",
    "assert_insight_artifact_reviewable",
    "insight_review_gate_binding",
    "reflect_evidence",
]
