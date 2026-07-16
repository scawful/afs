"""Calibration trail: resurface past decisions and score their outcomes.

Approving, rejecting, and closing missions are judgment calls. This module
keeps the loop honest: rationales and predictions are recorded when decisions
happen, then a periodic (typically weekly) pass resurfaces them next to what
actually happened so the human can score their own calibration instead of
offloading the judgment entirely to agents.

Storage is append-only JSONL under ``scratchpad/calibration/``:

- ``predictions.jsonl`` — predict-before-reveal entries (e.g. session
  bootstrap ``--engage``), each with the prediction and the revealed actual.
- ``outcomes.jsonl`` — human outcome scores (``hit``/``miss``/``unclear``)
  keyed by the decision ref (approval id, mission id, or prediction id).

Agent-gate decisions are the one exception: the approval gate store is
global, not per-context, so their outcome scores live in a global
``approval_outcomes.jsonl`` next to the gate store. Scoring a global
decision per context would resurface it as unscored in every other
context and let it be scored once per context.
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from .context_paths import resolve_mount_root
from .models import MountType

CALIBRATION_DIR_NAME = "calibration"
PREDICTIONS_FILE_NAME = "predictions.jsonl"
OUTCOMES_FILE_NAME = "outcomes.jsonl"
GATE_OUTCOMES_FILE_NAME = "approval_outcomes.jsonl"
VALID_OUTCOMES = ("hit", "miss", "unclear")


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _parse_timestamp(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value.strip():
        return None
    try:
        parsed = datetime.fromisoformat(value)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def calibration_root(context_path: Path, *, config: Any = None) -> Path:
    return (
        resolve_mount_root(context_path, MountType.SCRATCHPAD, config=config)
        / CALIBRATION_DIR_NAME
    )


def _append_jsonl(path: Path, entry: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry, ensure_ascii=False) + "\n")


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    entries: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError:
        return []
    for line in lines:
        line = line.strip()
        if not line:
            continue
        try:
            parsed = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            entries.append(parsed)
    return entries


# -- write paths --------------------------------------------------------------


def record_prediction(
    context_path: Path,
    *,
    kind: str,
    predicted: str,
    actual: str,
    match: bool | None = None,
    session_id: str = "",
    config: Any = None,
) -> dict[str, Any]:
    """Append a non-authoritative programmatic prediction annotation."""
    return _record_prediction(
        context_path,
        kind=kind,
        predicted=predicted,
        actual=actual,
        match=match,
        session_id=session_id,
        predicted_by="unauthenticated",
        predicted_via="programmatic",
        reviewer_subject="",
        identity_authenticated=False,
        human_confirmed=False,
        config=config,
    )


def human_prediction_scope(
    context_path: Path,
    *,
    kind: str,
    predicted: str,
    actual: str,
    match: bool | None,
    session_id: str = "",
    config: Any = None,
) -> str:
    """Return the broker scope for a prediction in its exact trail."""
    from .human_provenance import decision_scope_parts

    store_path = (
        calibration_root(context_path, config=config) / PREDICTIONS_FILE_NAME
    ).expanduser().resolve()
    return decision_scope_parts(
        "calibration-prediction",
        "record",
        str(store_path),
        kind.strip(),
        predicted.strip(),
        actual.strip(),
        json.dumps(match),
        session_id.strip(),
    )


def record_human_prediction(
    context_path: Path,
    *,
    kind: str,
    predicted: str,
    actual: str,
    authorization: Any,
    match: bool | None = None,
    session_id: str = "",
    config: Any = None,
) -> dict[str, Any]:
    """Append a broker-confirmed human prediction."""
    from .human_provenance import consume_human_authorization

    scope = human_prediction_scope(
        context_path,
        kind=kind,
        predicted=predicted,
        actual=actual,
        match=match,
        session_id=session_id,
        config=config,
    )
    if not consume_human_authorization(authorization, scope=scope):
        raise ValueError("a HumanDecisionBroker authorization is required")
    identity = authorization.identity
    return _record_prediction(
        context_path,
        kind=kind,
        predicted=predicted,
        actual=actual,
        match=match,
        session_id=session_id,
        predicted_by=identity.reviewer,
        predicted_via=authorization.confirmed_via,
        reviewer_subject=identity.subject,
        identity_authenticated=identity.authenticated,
        human_confirmed=True,
        config=config,
    )


def _record_prediction(
    context_path: Path,
    *,
    kind: str,
    predicted: str,
    actual: str,
    match: bool | None,
    session_id: str,
    predicted_by: str,
    predicted_via: str,
    reviewer_subject: str,
    identity_authenticated: bool,
    human_confirmed: bool,
    config: Any,
) -> dict[str, Any]:
    entry = {
        "id": f"pred_{uuid.uuid4().hex[:12]}",
        "timestamp": _now().isoformat(),
        "kind": kind,
        "predicted": predicted.strip(),
        "actual": actual.strip(),
        "match": match,
        "session_id": session_id,
        "predicted_by": predicted_by,
        "predicted_via": predicted_via,
        "reviewer_subject": reviewer_subject,
        "identity_authenticated": identity_authenticated,
        "human_confirmed": human_confirmed,
    }
    _append_jsonl(
        calibration_root(context_path, config=config) / PREDICTIONS_FILE_NAME, entry
    )
    return entry


class UnknownDecisionRefError(ValueError):
    """The ref does not name any known decision in this context."""


def record_outcome(
    context_path: Path,
    *,
    ref: str,
    outcome: str,
    note: str = "",
    scored_by: str = "",
    scored_via: str = "",
    config: Any = None,
    force: bool = False,
) -> dict[str, Any]:
    """Record a non-authoritative programmatic outcome annotation.

    ``ref`` is an approval/mission/prediction id. This compatibility API does
    not accept caller-supplied ``scored_by``/``scored_via`` as proof of human
    provenance; the entry is explicitly marked programmatic and never counts
    as a human calibration score. Use :func:`record_human_outcome` with a
    :class:`~afs.human_provenance.HumanAuthorization` for an authoritative
    score.

    The ref must name a real decision — a typo'd or fabricated ref would
    silently poison the trail. Raises :class:`UnknownDecisionRefError` for
    refs that no known store contains; ``force=True`` is the explicit
    escape hatch for scoring a decision whose store is unavailable.

    """
    return _record_outcome(
        context_path,
        ref=ref,
        outcome=outcome,
        note=note,
        scored_by="unauthenticated",
        scored_via="programmatic",
        reviewer_subject="",
        identity_authenticated=False,
        human_confirmed=False,
        config=config,
        force=force,
    )


def record_human_outcome(
    context_path: Path,
    *,
    ref: str,
    outcome: str,
    authorization: Any,
    note: str = "",
    config: Any = None,
    force: bool = False,
) -> dict[str, Any]:
    """Record an authoritative outcome using a broker-minted capability."""
    from .human_provenance import consume_human_authorization

    normalized_ref = ref.strip()
    if outcome not in VALID_OUTCOMES:
        raise ValueError(
            f"invalid outcome {outcome!r}; valid: " + ", ".join(VALID_OUTCOMES)
        )
    if not normalized_ref:
        raise ValueError("a decision ref is required")
    if not force and not ref_is_known(context_path, normalized_ref, config=config):
        raise UnknownDecisionRefError(
            f"unknown decision ref {normalized_ref!r}; run `afs calibration review` "
            "to see valid refs (approval request ids, mission ids, prediction ids)"
        )

    scope = human_outcome_scope(
        context_path,
        ref=normalized_ref,
        outcome=outcome,
        note=note,
        config=config,
    )
    if not consume_human_authorization(authorization, scope=scope):
        raise ValueError("a HumanDecisionBroker authorization is required")
    identity = authorization.identity
    return _record_outcome(
        context_path,
        ref=normalized_ref,
        outcome=outcome,
        note=note,
        scored_by=identity.reviewer,
        scored_via=authorization.confirmed_via,
        reviewer_subject=identity.subject,
        identity_authenticated=identity.authenticated,
        human_confirmed=True,
        config=config,
        force=force,
    )


def _record_outcome(
    context_path: Path,
    *,
    ref: str,
    outcome: str,
    note: str,
    scored_by: str,
    scored_via: str,
    reviewer_subject: str,
    identity_authenticated: bool,
    human_confirmed: bool,
    config: Any,
    force: bool,
) -> dict[str, Any]:
    if outcome not in VALID_OUTCOMES:
        raise ValueError(
            f"invalid outcome {outcome!r}; valid: " + ", ".join(VALID_OUTCOMES)
        )
    ref = ref.strip()
    if not ref:
        raise ValueError("a decision ref is required")
    kind = _ref_kind(ref)
    if not force and not ref_is_known(context_path, ref, config=config):
        raise UnknownDecisionRefError(
            f"unknown decision ref {ref!r}; run `afs calibration review` to see "
            "valid refs (approval request ids, mission ids, prediction ids)"
        )
    entry = {
        "ref": ref,
        "kind": kind,
        "outcome": outcome,
        "note": note.strip(),
        "scored_by": scored_by.strip(),
        "scored_via": scored_via.strip(),
        "reviewer_subject": reviewer_subject,
        "identity_authenticated": identity_authenticated,
        "human_confirmed": human_confirmed,
        "timestamp": _now().isoformat(),
    }
    _append_jsonl(_outcomes_path_for_kind(context_path, kind, config=config), entry)
    return entry


def _ref_kind(ref: str) -> str:
    if ref.startswith("mission_"):
        return "mission"
    if ref.startswith("pred_"):
        return "prediction"
    if ref.startswith("gate_"):
        return "gate"
    return "approval"


def _gate_outcomes_path() -> Path:
    """Global trail for agent-gate outcomes, co-located with the gate store."""
    from .agents.guardrails import _prefer_writable_state_path

    return _prefer_writable_state_path(
        GATE_OUTCOMES_FILE_NAME, env_var="AFS_AGENT_APPROVAL_OUTCOMES_PATH"
    )


def _outcomes_path_for_kind(
    context_path: Path, kind: str, *, config: Any = None
) -> Path:
    if kind == "gate":
        return _gate_outcomes_path()
    return calibration_root(context_path, config=config) / OUTCOMES_FILE_NAME


def human_outcome_scope(
    context_path: Path,
    *,
    ref: str,
    outcome: str,
    note: str = "",
    config: Any = None,
) -> str:
    """Return the broker scope for a score in its exact outcome store."""
    from .human_provenance import decision_scope_parts

    normalized_ref = ref.strip()
    kind = _ref_kind(normalized_ref)
    store_path = _outcomes_path_for_kind(
        context_path, kind, config=config
    ).expanduser().resolve()
    return decision_scope_parts(
        "calibration",
        outcome,
        str(store_path),
        normalized_ref,
        note.strip(),
    )


def ref_is_known(context_path: Path, ref: str, *, config: Any = None) -> bool:
    """True when ``ref`` names a decision some known store actually contains."""
    kind = _ref_kind(ref)
    if kind == "prediction":
        return any(
            entry.get("id") == ref
            for entry in load_predictions(context_path, config=config)
        )
    if kind == "mission":
        try:
            from .missions import MissionStore

            return MissionStore(context_path, config=config).get(ref) is not None
        except Exception:
            return False
    if kind == "gate":
        try:
            from .agents.guardrails import ApprovalGate

            return any(
                request.request_id == ref for request in ApprovalGate().all_requests()
            )
        except Exception:
            return False
    try:
        from .work_assistant import WorkAssistantStore

        store = WorkAssistantStore(context_path, config=config)
        return store.get_approval(ref) is not None
    except Exception:
        return False


# -- read paths ----------------------------------------------------------------


def load_outcomes(context_path: Path, *, config: Any = None) -> list[dict[str, Any]]:
    """Context outcomes plus the global gate-outcome trail.

    Merging the global trail means a gate decision scored while reviewing one
    context shows as scored in every context, instead of resurfacing as
    unscored elsewhere.
    """
    entries = _load_jsonl(
        calibration_root(context_path, config=config) / OUTCOMES_FILE_NAME
    )
    try:
        entries += _load_jsonl(_gate_outcomes_path())
    except Exception:
        pass
    return entries


def load_predictions(
    context_path: Path,
    *,
    since: datetime | None = None,
    config: Any = None,
) -> list[dict[str, Any]]:
    entries = _load_jsonl(
        calibration_root(context_path, config=config) / PREDICTIONS_FILE_NAME
    )
    if since is None:
        return entries
    kept: list[dict[str, Any]] = []
    for entry in entries:
        stamp = _parse_timestamp(entry.get("timestamp"))
        if stamp is not None and stamp >= since:
            kept.append(entry)
    return kept


# Work approvals that represent a human "yes": applied/failed are approved
# decisions whose execution already ran (successfully or not) — dropping them
# would hide exactly the decisions most worth scoring.
_WORK_DECIDED_STATUSES = ("approved", "rejected", "applied", "failed")


def _gate_decisions(since: datetime) -> list[dict[str, Any]]:
    """Approved/rejected entries from the agent approval gate, in window.

    The gate store is global (not per-context), so the store path is included
    for provenance and refs use the per-request id — never the agent:action
    pair, which repeats across requests and contexts.
    """
    try:
        from .agents.guardrails import ApprovalGate

        gate = ApprovalGate()
    except Exception:
        return []
    decisions: list[dict[str, Any]] = []
    for request in gate.all_requests():
        if request.status not in ("approved", "rejected"):
            continue
        if not request.human_confirmed:
            continue
        decided_at = _parse_timestamp(request.reviewed_at)
        if decided_at is None or decided_at < since:
            continue
        decisions.append(
            {
                "ref": request.request_id or f"{request.agent}:{request.action}",
                "source": "gate",
                "store": str(getattr(gate, "_path", "")),
                "agent": request.agent,
                "action": request.action,
                "detail": request.detail,
                "status": request.status,
                "decided_at": request.reviewed_at,
                "decided_by": request.reviewed_by,
                "decided_via": request.reviewed_via,
                "reviewer_subject": request.reviewer_subject,
                "identity_authenticated": request.identity_authenticated,
                "human_confirmed": request.human_confirmed,
                "rationale": request.rationale,
            }
        )
    return decisions


def _work_decisions(
    context_path: Path, since: datetime, *, config: Any = None
) -> list[dict[str, Any]]:
    """Decided work approvals in window (approved/rejected/applied/failed)."""
    try:
        from .work_assistant import WorkAssistantStore

        store = WorkAssistantStore(context_path, config=config)
        approvals = store.list_approvals(status=None, limit=500)
    except Exception:
        return []
    decisions: list[dict[str, Any]] = []
    for approval in approvals:
        if approval.get("status") not in _WORK_DECIDED_STATUSES:
            continue
        if not approval.get("human_confirmed"):
            continue
        decided_at = _parse_timestamp(approval.get("updated_at"))
        if decided_at is None or decided_at < since:
            continue
        decisions.append(
            {
                "ref": str(approval.get("approval_id", "")),
                "source": "work",
                "action": str(approval.get("action", "")),
                "detail": str(approval.get("summary", "")),
                "status": str(approval.get("status", "")),
                "decided_at": str(approval.get("updated_at", "")),
                "decided_by": str(approval.get("approved_by", "")),
                "decided_via": str(approval.get("decision_via", "")),
                "reviewer_subject": str(approval.get("reviewer_subject", "")),
                "identity_authenticated": bool(
                    approval.get("identity_authenticated", False)
                ),
                "human_confirmed": True,
                "rationale": str(approval.get("rationale", "")),
            }
        )
    return decisions


def _closed_missions(
    context_path: Path, since: datetime, *, config: Any = None
) -> list[dict[str, Any]]:
    """Missions closed (done/abandoned) in window, with their acceptance."""
    try:
        from .missions import MissionStore

        missions = MissionStore(context_path, config=config).list(limit=1000)
    except Exception:
        return []
    closed: list[dict[str, Any]] = []
    for mission in missions:
        if mission.status not in ("done", "abandoned"):
            continue
        closed_at = _parse_timestamp(mission.updated_at)
        if closed_at is None or closed_at < since:
            continue
        closed.append(
            {
                "ref": mission.mission_id,
                "title": mission.title,
                "status": mission.status,
                "acceptance": (
                    mission.acceptance if mission.acceptance_human_confirmed else ""
                ),
                "acceptance_human_confirmed": mission.acceptance_human_confirmed,
                "summary": mission.summary,
                "closed_at": mission.updated_at,
            }
        )
    return closed


def collect_decisions(
    context_path: Path,
    *,
    days: int = 7,
    config: Any = None,
    now: datetime | None = None,
) -> dict[str, Any]:
    """Gather decisions from the last ``days`` days for calibration review.

    Every source is optional; a failing or absent source contributes an empty
    list so the review never blocks on one broken store.
    """
    current = now or _now()
    since = current - timedelta(days=max(1, days))
    scored = {
        entry.get("ref"): entry
        for entry in load_outcomes(context_path, config=config)
        if entry.get("human_confirmed") is True
    }
    return {
        "window_days": max(1, days),
        "since": since.isoformat(),
        "approvals": _gate_decisions(since)
        + _work_decisions(context_path, since, config=config),
        "missions": _closed_missions(context_path, since, config=config),
        "predictions": [
            entry
            for entry in load_predictions(
                context_path, since=since, config=config
            )
            if entry.get("human_confirmed") is True
        ],
        "scored": scored,
    }
