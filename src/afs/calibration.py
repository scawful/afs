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
    """Append one predict-before-reveal entry to the calibration trail."""
    entry = {
        "id": f"pred_{uuid.uuid4().hex[:12]}",
        "timestamp": _now().isoformat(),
        "kind": kind,
        "predicted": predicted.strip(),
        "actual": actual.strip(),
        "match": match,
        "session_id": session_id,
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
    config: Any = None,
    force: bool = False,
) -> dict[str, Any]:
    """Score a past decision. ``ref`` is an approval/mission/prediction id.

    The ref must name a real decision — a typo'd or fabricated ref would
    silently poison the trail. Raises :class:`UnknownDecisionRefError` for
    refs that no known store contains; ``force=True`` is the explicit
    escape hatch for scoring a decision whose store is unavailable.
    """
    if outcome not in VALID_OUTCOMES:
        raise ValueError(
            f"invalid outcome {outcome!r}; valid: " + ", ".join(VALID_OUTCOMES)
        )
    ref = ref.strip()
    if not ref:
        raise ValueError("a decision ref is required")
    if not force and not ref_is_known(context_path, ref, config=config):
        raise UnknownDecisionRefError(
            f"unknown decision ref {ref!r}; run `afs calibration review` to see "
            "valid refs (approval request ids, mission ids, prediction ids)"
        )
    entry = {
        "ref": ref,
        "kind": _ref_kind(ref),
        "outcome": outcome,
        "note": note.strip(),
        "timestamp": _now().isoformat(),
    }
    _append_jsonl(
        calibration_root(context_path, config=config) / OUTCOMES_FILE_NAME, entry
    )
    return entry


def _ref_kind(ref: str) -> str:
    if ref.startswith("mission_"):
        return "mission"
    if ref.startswith("pred_"):
        return "prediction"
    return "approval"


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
    try:
        from .agents.guardrails import ApprovalGate

        if any(request.request_id == ref for request in ApprovalGate()._pending):
            return True
    except Exception:
        pass
    try:
        from .work_assistant import WorkAssistantStore

        store = WorkAssistantStore(context_path, config=config)
        return store.get_approval(ref) is not None
    except Exception:
        return False


# -- read paths ----------------------------------------------------------------


def load_outcomes(context_path: Path, *, config: Any = None) -> list[dict[str, Any]]:
    return _load_jsonl(
        calibration_root(context_path, config=config) / OUTCOMES_FILE_NAME
    )


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
    for request in gate._pending:
        if request.status not in ("approved", "rejected"):
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
                "acceptance": mission.acceptance,
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
    scored = {entry.get("ref"): entry for entry in load_outcomes(context_path, config=config)}
    return {
        "window_days": max(1, days),
        "since": since.isoformat(),
        "approvals": _gate_decisions(since)
        + _work_decisions(context_path, since, config=config),
        "missions": _closed_missions(context_path, since, config=config),
        "predictions": load_predictions(context_path, since=since, config=config),
        "scored": scored,
    }
