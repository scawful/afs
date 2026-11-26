"""Agentic evaluation harness for AFS retrieval, context packing, safety, and trajectories."""

from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
import time
import uuid
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .context_layout import LAYOUT_VERSION, detect_layout_version
from .context_paths import resolve_mount_root
from .hybrid_search import HybridSearchEngine
from .manager import AFSManager
from .mcp.catalog import FROZEN_SLIM_MCP_TOOLS
from .models import MountType
from .path_safety import assert_no_linklike_components

VALID_EVAL_KINDS = ("retrieval", "context_pack", "safety", "trajectory")

_LOWER_IS_BETTER_METRICS = frozenset(
    {
        "avg_latency_ms",
        "p95_latency_ms",
        "avg_steps",
        "avg_observe_act_ratio",
        "max_observe_act_ratio",
        "failed",
    }
)

_OBSERVE_TOOLS = frozenset(
    {
        "context.read",
        "context.list",
        "context.query",
        "context.status",
        "messages.read",
        "note.read",
        "note.list",
        "skill.match",
        "skill.read",
        "handoff.read",
        "handoff.list",
        "read",
        "list",
        "query",
        "search",
        "observe",
        "status",
    }
)

_ACT_TOOLS = frozenset(
    {
        "context.write",
        "messages.send",
        "messages.ack",
        "note.create",
        "handoff.create",
        "write",
        "edit",
        "apply",
        "act",
        "send",
        "ack",
    }
)


@dataclass
class EvalCaseResult:
    """Outcome of a single evaluation case."""

    case_id: str
    suite_kind: str
    passed: bool
    score: float
    metrics: dict[str, float | int | bool | str] = field(default_factory=dict)
    details: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_id": self.case_id,
            "suite_kind": self.suite_kind,
            "passed": self.passed,
            "score": round(self.score, 6),
            "metrics": dict(self.metrics),
            "details": dict(self.details),
            "latency_ms": round(self.latency_ms, 3),
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EvalCaseResult:
        return cls(
            case_id=str(payload.get("case_id", "")),
            suite_kind=str(payload.get("suite_kind", "")),
            passed=bool(payload.get("passed", False)),
            score=float(payload.get("score", 0.0) or 0.0),
            metrics=dict(payload.get("metrics") or {}),
            details=dict(payload.get("details") or {}),
            latency_ms=float(payload.get("latency_ms", 0.0) or 0.0),
        )


@dataclass
class EvalSuiteSummary:
    """Aggregated evaluation report for one suite kind and configuration."""

    suite_name: str
    suite_kind: str
    run_id: str
    created_at: str
    git_sha: str
    config: dict[str, Any]
    total: int
    passed: int
    failed: int
    pass_rate: float
    metrics: dict[str, float]
    cases: list[EvalCaseResult] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "suite_name": self.suite_name,
            "suite_kind": self.suite_kind,
            "run_id": self.run_id,
            "created_at": self.created_at,
            "git_sha": self.git_sha,
            "config": dict(self.config),
            "total": self.total,
            "passed": self.passed,
            "failed": self.failed,
            "pass_rate": round(self.pass_rate, 6),
            "metrics": {k: round(float(v), 6) for k, v in self.metrics.items()},
            "cases": [case.to_dict() for case in self.cases],
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> EvalSuiteSummary:
        raw_cases = payload.get("cases") or []
        return cls(
            suite_name=str(payload.get("suite_name", "")),
            suite_kind=str(payload.get("suite_kind", "")),
            run_id=str(payload.get("run_id", "")),
            created_at=str(payload.get("created_at", "")),
            git_sha=str(payload.get("git_sha", "")),
            config=dict(payload.get("config") or {}),
            total=int(payload.get("total", 0) or 0),
            passed=int(payload.get("passed", 0) or 0),
            failed=int(payload.get("failed", 0) or 0),
            pass_rate=float(payload.get("pass_rate", 0.0) or 0.0),
            metrics={
                str(k): float(v)
                for k, v in (payload.get("metrics") or {}).items()
                if isinstance(v, (int, float)) and not isinstance(v, bool)
            },
            cases=[
                EvalCaseResult.from_dict(item)
                for item in raw_cases
                if isinstance(item, dict)
            ],
        )


@dataclass
class EvalComparison:
    """Metric diff and regression check between two persisted eval runs."""

    baseline_run_id: str
    candidate_run_id: str
    suite_kind: str
    metric_deltas: dict[str, float]
    regressions: list[dict[str, Any]]
    improvements: list[dict[str, Any]]
    passed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "baseline_run_id": self.baseline_run_id,
            "candidate_run_id": self.candidate_run_id,
            "suite_kind": self.suite_kind,
            "metric_deltas": {k: round(float(v), 6) for k, v in self.metric_deltas.items()},
            "regressions": list(self.regressions),
            "improvements": list(self.improvements),
            "passed": self.passed,
        }


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_run_id(prefix: str = "eval") -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


def _current_git_sha(start_dir: Path | None = None) -> str:
    cwd = str(start_dir or Path.cwd())
    try:
        completed = subprocess.run(
            ["git", "-C", cwd, "rev-parse", "--short", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            timeout=2.0,
        )
    except (OSError, subprocess.SubprocessError):
        return ""
    if completed.returncode != 0:
        return ""
    return completed.stdout.strip()


def _percentile(values: Sequence[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, math.ceil(q * len(ordered)) - 1))
    return float(ordered[idx])


def _normalize_target(value: str) -> str:
    return value.strip().replace("\\", "/").lower()


def _as_str_list(raw: Any) -> list[str]:
    if isinstance(raw, str):
        return [raw.strip()] if raw.strip() else []
    if isinstance(raw, Iterable) and not isinstance(raw, (dict, bytes)):
        return [str(item).strip() for item in raw if str(item).strip()]
    return []


def _extract_retrieval_targets(case: dict[str, Any]) -> tuple[list[str], list[str]]:
    expected_paths: list[str] = []
    expected_doc_ids: list[str] = []
    expected_paths.extend(_as_str_list(case.get("expected_paths")))
    expected_paths.extend(_as_str_list(case.get("expected_path")))
    expected_paths.extend(_as_str_list(case.get("expected_path_contains")))
    expected_doc_ids.extend(_as_str_list(case.get("expected_doc_ids")))
    expected_doc_ids.extend(_as_str_list(case.get("expected_doc_id")))
    raw_expected = case.get("expected")
    if isinstance(raw_expected, str) and raw_expected.strip():
        expected_paths.append(raw_expected.strip())
    elif isinstance(raw_expected, list):
        for item in raw_expected:
            if isinstance(item, str) and item.strip():
                expected_paths.append(item.strip())
            elif isinstance(item, dict):
                doc_id = item.get("doc_id")
                path_val = item.get("path_contains") or item.get("path")
                if isinstance(doc_id, str) and doc_id.strip():
                    expected_doc_ids.append(doc_id.strip())
                if isinstance(path_val, str) and path_val.strip():
                    expected_paths.append(path_val.strip())
    deduped_paths = list(dict.fromkeys(expected_paths))
    deduped_ids = list(dict.fromkeys(expected_doc_ids))
    return deduped_paths, deduped_ids


def _matching_target_key(
    relative_path: str,
    source_path: str,
    doc_id: str,
    expected_paths: list[str],
    expected_doc_ids: list[str],
) -> str | None:
    if doc_id in expected_doc_ids:
        return f"id:{doc_id}"
    rel_norm = _normalize_target(relative_path)
    src_norm = _normalize_target(source_path)
    for target in expected_paths:
        target_norm = _normalize_target(target)
        if target_norm and (target_norm in rel_norm or target_norm in src_norm):
            return f"path:{target_norm}"
    return None


def evaluate_retrieval_suite(
    cases: Sequence[dict[str, Any]],
    *,
    engine: HybridSearchEngine,
    modes: Sequence[str] = ("text", "hybrid"),
    top_k: int = 5,
    scope_ids: Sequence[str] = (),
    include_common: bool = True,
    all_projects: bool = True,
    query_embed_fn: Callable[[str], list[float]] | None = None,
    recreate_query_embedder: bool = True,
    suite_name: str = "retrieval",
    git_sha: str = "",
) -> dict[str, EvalSuiteSummary]:
    """Evaluate retrieval accuracy and latency across one or more search modes."""
    normalized_modes = [mode.strip().lower() for mode in modes if mode.strip()]
    if not normalized_modes:
        normalized_modes = ["text"]
    sha = git_sha or _current_git_sha()
    summaries: dict[str, EvalSuiteSummary] = {}

    for mode in normalized_modes:
        case_results: list[EvalCaseResult] = []
        latencies: list[float] = []
        hit1_total = 0
        hitk_total = 0
        mrr_sum = 0.0
        ndcg_sum = 0.0
        semantic_ready_count = 0

        for idx, case in enumerate(cases, start=1):
            query = str(case.get("query", "")).strip()
            if not query:
                continue
            case_id = str(case.get("id") or case.get("case_id") or f"retrieval_{idx}")
            case_top_k = int(case.get("top_k") or top_k)
            expected_paths, expected_doc_ids = _extract_retrieval_targets(case)
            target_count = max(1, len(expected_paths) + len(expected_doc_ids))

            t0 = time.perf_counter()
            response = engine.search(
                query,
                scope_ids=scope_ids,
                include_common=include_common,
                all_projects=all_projects,
                mode=mode,
                top_k=case_top_k,
                query_embed_fn=query_embed_fn if mode in {"hybrid", "semantic"} else None,
                recreate_query_embedder=(
                    recreate_query_embedder if mode in {"hybrid", "semantic"} else False
                ),
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(elapsed_ms)

            if response.semantic_status == "ready":
                semantic_ready_count += 1

            matched_keys: set[str] = set()
            first_rank: int | None = None
            dcg = 0.0
            top_hits: list[dict[str, Any]] = []
            for rank, hit in enumerate(response.results, start=1):
                top_hits.append(
                    {
                        "rank": rank,
                        "relative_path": hit.relative_path,
                        "score": round(hit.score, 6),
                        "signals": hit.signals,
                    }
                )
                key = _matching_target_key(
                    hit.relative_path,
                    hit.source_path,
                    hit.doc_id,
                    expected_paths,
                    expected_doc_ids,
                )
                if key is not None and key not in matched_keys:
                    matched_keys.add(key)
                    if first_rank is None:
                        first_rank = rank
                    dcg += 1.0 / math.log2(rank + 1)

            ideal_hits = min(target_count, case_top_k)
            idcg = sum(1.0 / math.log2(r + 1) for r in range(1, ideal_hits + 1))
            ndcg = dcg / idcg if idcg > 0 else 0.0
            reciprocal_rank = (1.0 / first_rank) if first_rank is not None else 0.0
            hit1 = first_rank == 1
            hitk = first_rank is not None

            if hit1:
                hit1_total += 1
            if hitk:
                hitk_total += 1
            mrr_sum += reciprocal_rank
            ndcg_sum += ndcg

            case_results.append(
                EvalCaseResult(
                    case_id=case_id,
                    suite_kind="retrieval",
                    passed=hitk,
                    score=reciprocal_rank,
                    metrics={
                        "mode": mode,
                        "hit_at_1": hit1,
                        "hit_at_k": hitk,
                        "reciprocal_rank": round(reciprocal_rank, 6),
                        "ndcg_at_k": round(ndcg, 6),
                        "first_rank": first_rank or 0,
                        "semantic_status": response.semantic_status,
                    },
                    details={
                        "query": query,
                        "expected_paths": expected_paths,
                        "expected_doc_ids": expected_doc_ids,
                        "semantic_reason": response.semantic_reason,
                        "top_hits": top_hits[:3],
                    },
                    latency_ms=elapsed_ms,
                )
            )

        total = len(case_results)
        passed = sum(1 for item in case_results if item.passed)
        metrics: dict[str, float] = {
            "hit_at_1": (hit1_total / total) if total else 0.0,
            "hit_at_k": (hitk_total / total) if total else 0.0,
            "mrr": (mrr_sum / total) if total else 0.0,
            "ndcg_at_k": (ndcg_sum / total) if total else 0.0,
            "avg_latency_ms": (sum(latencies) / total) if total else 0.0,
            "p95_latency_ms": _percentile(latencies, 0.95),
            "semantic_ready_rate": (semantic_ready_count / total) if total else 0.0,
        }
        summaries[mode] = EvalSuiteSummary(
            suite_name=f"{suite_name}:{mode}",
            suite_kind="retrieval",
            run_id=_new_run_id(f"ret_{mode}"),
            created_at=_now_iso(),
            git_sha=sha,
            config={
                "mode": mode,
                "top_k": top_k,
                "index_root": str(engine.index_root),
            },
            total=total,
            passed=passed,
            failed=total - passed,
            pass_rate=(passed / total) if total else 0.0,
            metrics=metrics,
            cases=case_results,
        )
    return summaries


def compare_retrieval_modes(
    text_summary: EvalSuiteSummary,
    hybrid_summary: EvalSuiteSummary,
) -> dict[str, float]:
    """Compute metric lift from text retrieval to hybrid semantic retrieval."""
    keys = ("hit_at_1", "hit_at_k", "mrr", "ndcg_at_k")
    deltas = {
        f"{key}_delta": round(
            hybrid_summary.metrics.get(key, 0.0) - text_summary.metrics.get(key, 0.0),
            6,
        )
        for key in keys
    }
    deltas["latency_delta_ms"] = round(
        hybrid_summary.metrics.get("avg_latency_ms", 0.0)
        - text_summary.metrics.get("avg_latency_ms", 0.0),
        3,
    )
    return deltas


def evaluate_context_pack_suite(
    cases: Sequence[dict[str, Any]],
    *,
    manager: AFSManager,
    context_path: Path,
    project_path: Path,
    suite_name: str = "context_pack",
    git_sha: str = "",
) -> EvalSuiteSummary:
    """Evaluate token-budgeted context packing recall, utilization, and leak prevention."""
    from .context_pack import build_context_pack

    sha = git_sha or _current_git_sha(project_path)
    case_results: list[EvalCaseResult] = []
    latencies: list[float] = []
    recall_scores: list[float] = []
    utilizations: list[float] = []
    budget_ok_count = 0
    zero_leak_count = 0

    for idx, case in enumerate(cases, start=1):
        case_id = str(case.get("id") or case.get("case_id") or f"pack_{idx}")
        budget = int(case.get("token_budget") or 4000)
        t0 = time.perf_counter()
        pack = build_context_pack(
            manager,
            context_path,
            project_path=project_path,
            query=str(case.get("query", "")),
            task=str(case.get("task", "")),
            model=str(case.get("model", "generic")),
            workflow=str(case.get("workflow", "general")),
            pack_mode=str(case.get("pack_mode", "focused")),
            token_budget=budget,
            include_content=bool(case.get("include_content", False)),
            semantic=bool(case.get("semantic", False)),
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(elapsed_ms)

        estimated_tokens = int(pack.get("estimated_tokens", 0) or 0)
        resolved_budget = int(pack.get("token_budget", budget) or budget)
        sources = [_normalize_target(str(s)) for s in (pack.get("sources") or [])]
        section_text = "\n".join(
            f"{sec.get('title', '')}\n{sec.get('body', '')}"
            for sec in (pack.get("sections") or [])
            if isinstance(sec, dict)
        ).lower()
        haystack = "\n".join([*sources, section_text])

        expected_items = [
            _normalize_target(item)
            for item in (
                _as_str_list(case.get("expected_paths"))
                + _as_str_list(case.get("expected_terms"))
            )
        ]
        matched_expected = [item for item in expected_items if item in haystack]
        recall = (len(matched_expected) / len(expected_items)) if expected_items else 1.0

        forbidden_items = [
            _normalize_target(item)
            for item in (
                _as_str_list(case.get("forbidden_paths"))
                + _as_str_list(case.get("forbidden_terms"))
            )
        ]
        leaked = [item for item in forbidden_items if item in haystack]
        zero_secret_leaks = len(leaked) == 0
        budget_respected = estimated_tokens <= resolved_budget if resolved_budget > 0 else True
        utilization = (
            min(1.0, estimated_tokens / resolved_budget) if resolved_budget > 0 else 0.0
        )

        if budget_respected:
            budget_ok_count += 1
        if zero_secret_leaks:
            zero_leak_count += 1
        recall_scores.append(recall)
        utilizations.append(utilization)

        passed = budget_respected and zero_secret_leaks and recall >= float(
            case.get("min_recall", 1.0)
        )
        case_results.append(
            EvalCaseResult(
                case_id=case_id,
                suite_kind="context_pack",
                passed=passed,
                score=recall if (budget_respected and zero_secret_leaks) else 0.0,
                metrics={
                    "recall_in_budget": round(recall, 6),
                    "budget_respected": budget_respected,
                    "zero_secret_leaks": zero_secret_leaks,
                    "estimated_tokens": estimated_tokens,
                    "token_budget": resolved_budget,
                    "token_utilization": round(utilization, 6),
                },
                details={
                    "expected": expected_items,
                    "matched": matched_expected,
                    "leaked": leaked,
                    "omitted_sections": list(pack.get("omitted_sections") or []),
                },
                latency_ms=elapsed_ms,
            )
        )

    total = len(case_results)
    passed_count = sum(1 for item in case_results if item.passed)
    return EvalSuiteSummary(
        suite_name=suite_name,
        suite_kind="context_pack",
        run_id=_new_run_id("pack"),
        created_at=_now_iso(),
        git_sha=sha,
        config={"context_path": str(context_path), "project_path": str(project_path)},
        total=total,
        passed=passed_count,
        failed=total - passed_count,
        pass_rate=(passed_count / total) if total else 0.0,
        metrics={
            "recall_in_budget": (sum(recall_scores) / total) if total else 0.0,
            "budget_respected_rate": (budget_ok_count / total) if total else 0.0,
            "zero_secret_leak_rate": (zero_leak_count / total) if total else 0.0,
            "avg_token_utilization": (sum(utilizations) / total) if total else 0.0,
            "avg_latency_ms": (sum(latencies) / total) if total else 0.0,
            "p95_latency_ms": _percentile(latencies, 0.95),
        },
        cases=case_results,
    )


def _mint_test_human_auth(gate: Any, request_id: str, agent: str, action: str, rationale: str):
    from .human_provenance import _broker_for_reader

    expected_token = f"{agent}:{action}"
    scope = gate.human_authorization_scope("approve", request_id, rationale)
    broker = _broker_for_reader(lambda _prompt: expected_token)
    auth = broker.confirm_token(expected_token, "eval confirmation", scope=scope)
    if auth is None:
        raise RuntimeError("Failed to mint broker capability for isolated safety eval")
    return auth


def evaluate_safety_suite(
    cases: Sequence[dict[str, Any]],
    *,
    suite_name: str = "safety",
    git_sha: str = "",
) -> EvalSuiteSummary:
    """Evaluate ApprovalGate enforcement, payload binding, and audit ledger in temp DBs."""
    from .agents.approvals import ApprovalGate

    sha = git_sha or _current_git_sha()
    case_results: list[EvalCaseResult] = []
    latencies: list[float] = []
    enforcement_hits = 0
    tamper_scenarios = 0
    tamper_blocked = 0
    audit_complete_hits = 0

    with tempfile.TemporaryDirectory(prefix="afs_safety_eval_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        for idx, case in enumerate(cases, start=1):
            case_id = str(case.get("id") or case.get("case_id") or f"safety_{idx}")
            scenario = str(case.get("scenario", "human_gate")).strip().lower()
            agent = str(case.get("agent", "eval_agent"))
            action = str(case.get("action", "deploy"))
            detail = str(case.get("detail", "staging"))
            payload = case.get("payload", {"target": "staging"})
            gate = ApprovalGate(path=tmp_root / f"{case_id}_{idx}.sqlite3")

            t0 = time.perf_counter()
            gate_ok = False
            audit_ok = False
            details: dict[str, Any] = {"scenario": scenario}

            if scenario == "auto_approve":
                expected_allowed = bool(case.get("expected_allowed", True))
                allowed = gate.check(agent, action, detail, payload=payload)
                entries = gate.audit_log(action=action, decision="auto_approved")
                gate_ok = allowed is expected_allowed
                audit_ok = len(entries) == 1 if expected_allowed else True
                details.update({"allowed": allowed, "audit_count": len(entries)})

            elif scenario == "human_gate":
                decision_after_queue = str(case.get("decision", "approve")).strip().lower()
                initial = gate.check(agent, action, detail, run_id="run_1", payload=payload)
                pending = gate.pending_requests()
                if not initial and len(pending) == 1:
                    req = pending[0]
                    if decision_after_queue == "reject":
                        gate.reject(agent, action, rationale="blocked in eval")
                        recheck = gate.check(agent, action, detail, run_id="run_1", payload=payload)
                        gate_ok = recheck is False
                        audit_ok = True
                        details["recheck"] = recheck
                    else:
                        rationale = "verified in safety eval"
                        auth = _mint_test_human_auth(gate, req.request_id, agent, action, rationale)
                        gate.approve_human(
                            rationale=rationale,
                            authorization=auth,
                            request_id=req.request_id,
                        )
                        recheck = gate.check(agent, action, detail, run_id="run_1", payload=payload)
                        entries = gate.audit_log(decision="human_authorized")
                        gate_ok = recheck is True
                        audit_ok = len(entries) == 1
                        details.update({"recheck": recheck, "audit_count": len(entries)})
                else:
                    details["error"] = f"unexpected initial={initial} pending={len(pending)}"

            elif scenario in {"tamper_reject", "loopjacking"}:
                tamper_scenarios += 1
                tampered_payload = case.get("tampered_payload", {"target": "production"})
                initial = gate.check(agent, action, detail, payload=payload)
                pending = gate.pending_requests()
                if not initial and len(pending) == 1:
                    req = pending[0]
                    rationale = "approved original payload"
                    auth = _mint_test_human_auth(gate, req.request_id, agent, action, rationale)
                    gate.approve_human(
                        rationale=rationale,
                        authorization=auth,
                        request_id=req.request_id,
                    )
                    original_ok = gate.check(agent, action, detail, payload=payload)
                    tampered_ok = gate.check(agent, action, detail, payload=tampered_payload)
                    mismatches = gate.audit_log(decision="payload_mismatch")
                    gate_ok = original_ok is True and tampered_ok is False
                    audit_ok = len(mismatches) == 1
                    if gate_ok and audit_ok:
                        tamper_blocked += 1
                    details.update(
                        {
                            "original_ok": original_ok,
                            "tampered_ok": tampered_ok,
                            "mismatch_logged": len(mismatches),
                        }
                    )
                else:
                    details["error"] = f"unexpected initial={initial} pending={len(pending)}"

            elif scenario == "once_single_use":
                initial = gate.check(
                    agent, action, detail, run_id="run_1", scope="once", payload=payload
                )
                pending = gate.pending_requests()
                if not initial and len(pending) == 1:
                    req = pending[0]
                    rationale = "single-use authorization"
                    auth = _mint_test_human_auth(gate, req.request_id, agent, action, rationale)
                    gate.approve_human(
                        rationale=rationale,
                        authorization=auth,
                        request_id=req.request_id,
                    )
                    first_run = gate.check(
                        agent, action, detail, run_id="run_1", scope="once", payload=payload
                    )
                    second_run = gate.check(
                        agent, action, detail, run_id="run_2", scope="once", payload=payload
                    )
                    entries = gate.audit_log(decision="human_authorized")
                    gate_ok = first_run is True and second_run is False
                    audit_ok = len(entries) == 1
                    details.update({"first_run": first_run, "second_run": second_run})
                else:
                    details["error"] = f"unexpected initial={initial} pending={len(pending)}"

            else:
                details["error"] = f"unknown safety scenario: {scenario}"

            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            latencies.append(elapsed_ms)

            if gate_ok:
                enforcement_hits += 1
            if audit_ok:
                audit_complete_hits += 1
            passed = gate_ok and audit_ok
            case_results.append(
                EvalCaseResult(
                    case_id=case_id,
                    suite_kind="safety",
                    passed=passed,
                    score=1.0 if passed else 0.0,
                    metrics={
                        "gate_enforced": gate_ok,
                        "audit_complete": audit_ok,
                    },
                    details=details,
                    latency_ms=elapsed_ms,
                )
            )

    total = len(case_results)
    passed_count = sum(1 for item in case_results if item.passed)
    return EvalSuiteSummary(
        suite_name=suite_name,
        suite_kind="safety",
        run_id=_new_run_id("safety"),
        created_at=_now_iso(),
        git_sha=sha,
        config={"isolated": True},
        total=total,
        passed=passed_count,
        failed=total - passed_count,
        pass_rate=(passed_count / total) if total else 0.0,
        metrics={
            "gate_enforcement_rate": (enforcement_hits / total) if total else 0.0,
            "tamper_block_rate": (tamper_blocked / tamper_scenarios) if tamper_scenarios else 1.0,
            "audit_completeness_rate": (audit_complete_hits / total) if total else 0.0,
            "avg_latency_ms": (sum(latencies) / total) if total else 0.0,
        },
        cases=case_results,
    )


def _load_trajectory_events(case: dict[str, Any], *, base_dir: Path | None) -> list[dict[str, Any]]:
    raw_events = case.get("events")
    if isinstance(raw_events, list):
        return [item for item in raw_events if isinstance(item, dict)]
    trace_path_raw = case.get("trace_path")
    if not isinstance(trace_path_raw, str) or not trace_path_raw.strip():
        return []
    path = Path(trace_path_raw).expanduser()
    if not path.is_absolute() and base_dir is not None:
        path = (base_dir / path).resolve()
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
        if isinstance(parsed, dict) and isinstance(parsed.get("events"), list):
            return [item for item in parsed["events"] if isinstance(item, dict)]
        return []
    events: list[dict[str, Any]] = []
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parsed_line = json.loads(line)
        if isinstance(parsed_line, dict):
            events.append(parsed_line)
    return events


def evaluate_trajectory_suite(
    cases: Sequence[dict[str, Any]],
    *,
    base_dir: Path | None = None,
    suite_name: str = "trajectory",
    git_sha: str = "",
) -> EvalSuiteSummary:
    """Evaluate recorded agent trajectories for efficiency, observe:act ratio, and slim compliance."""
    sha = git_sha or _current_git_sha(base_dir)
    case_results: list[EvalCaseResult] = []
    latencies: list[float] = []
    step_counts: list[float] = []
    observe_act_ratios: list[float] = []
    slim_ok_count = 0

    for idx, case in enumerate(cases, start=1):
        case_id = str(case.get("id") or case.get("case_id") or f"traj_{idx}")
        t0 = time.perf_counter()
        events = _load_trajectory_events(case, base_dir=base_dir)

        tool_calls: list[str] = []
        mcp_calls: list[str] = []
        observe_count = 0
        act_count = 0
        error_count = 0

        for ev in events:
            ev_type = str(ev.get("type", "")).strip()
            raw_meta = ev.get("metadata")
            meta: dict[str, Any] = raw_meta if isinstance(raw_meta, dict) else {}
            tool_name = str(
                ev.get("tool")
                or ev.get("tool_name")
                or meta.get("tool_name")
                or ev.get("op")
                or ""
            ).strip()
            if tool_name:
                tool_calls.append(tool_name)
                if ev_type == "mcp_tool" or "." in tool_name:
                    mcp_calls.append(tool_name)

            if ev.get("ok") is False or meta.get("ok") is False:
                error_count += 1

            phase = str(ev.get("phase") or meta.get("phase") or "").strip().lower()
            lowered_tool = tool_name.lower()
            if phase == "observe" or ev_type == "phase_observe" or lowered_tool in _OBSERVE_TOOLS:
                observe_count += 1
            elif phase == "act" or ev_type == "phase_act" or lowered_tool in _ACT_TOOLS:
                act_count += 1

        step_count = len(tool_calls) if tool_calls else len(events)
        if act_count > 0:
            observe_act_ratio = observe_count / act_count
        elif observe_count > 0:
            observe_act_ratio = float(observe_count)
        else:
            observe_act_ratio = 0.0

        non_slim_tools = [
            tool for tool in mcp_calls if tool not in FROZEN_SLIM_MCP_TOOLS
        ]
        slim_compliant = len(non_slim_tools) == 0
        if slim_compliant:
            slim_ok_count += 1

        failures: list[str] = []
        max_steps = case.get("max_steps")
        if max_steps is not None and step_count > int(max_steps):
            failures.append(f"steps {step_count} > max_steps {int(max_steps)}")

        max_ratio = case.get("max_observe_act_ratio")
        if max_ratio is not None and observe_act_ratio > float(max_ratio):
            failures.append(
                f"observe_act_ratio {observe_act_ratio:.2f} > {float(max_ratio):.2f}"
            )

        must_call = _as_str_list(case.get("must_call"))
        missing_calls = [req for req in must_call if req not in tool_calls]
        if missing_calls:
            failures.append(f"missing required tools: {missing_calls}")

        must_not_call = _as_str_list(case.get("must_not_call"))
        forbidden_calls = [bad for bad in must_not_call if bad in tool_calls]
        if forbidden_calls:
            failures.append(f"called forbidden tools: {forbidden_calls}")

        if bool(case.get("slim_mcp_only", False)) and not slim_compliant:
            failures.append(f"non-slim MCP tools used: {non_slim_tools}")

        expected_status = case.get("expected_status")
        actual_status = str(case.get("status", "")).strip()
        if expected_status and actual_status != str(expected_status).strip():
            failures.append(f"status {actual_status!r} != {expected_status!r}")

        required_outputs = _as_str_list(case.get("required_outputs"))
        output_blob = str(case.get("final_output") or "")
        missing_outputs = [out for out in required_outputs if out not in output_blob]
        if missing_outputs:
            failures.append(f"missing required outputs: {missing_outputs}")

        elapsed_ms = (time.perf_counter() - t0) * 1000.0
        latencies.append(elapsed_ms)
        step_counts.append(float(step_count))
        observe_act_ratios.append(observe_act_ratio)

        passed = len(failures) == 0
        case_results.append(
            EvalCaseResult(
                case_id=case_id,
                suite_kind="trajectory",
                passed=passed,
                score=1.0 if passed else 0.0,
                metrics={
                    "steps": step_count,
                    "observe_count": observe_count,
                    "act_count": act_count,
                    "observe_act_ratio": round(observe_act_ratio, 4),
                    "slim_mcp_compliant": slim_compliant,
                    "error_count": error_count,
                },
                details={
                    "tool_calls": tool_calls,
                    "non_slim_tools": non_slim_tools,
                    "failures": failures,
                },
                latency_ms=elapsed_ms,
            )
        )

    total = len(case_results)
    passed_count = sum(1 for item in case_results if item.passed)
    return EvalSuiteSummary(
        suite_name=suite_name,
        suite_kind="trajectory",
        run_id=_new_run_id("traj"),
        created_at=_now_iso(),
        git_sha=sha,
        config={},
        total=total,
        passed=passed_count,
        failed=total - passed_count,
        pass_rate=(passed_count / total) if total else 0.0,
        metrics={
            "slim_compliance_rate": (slim_ok_count / total) if total else 0.0,
            "avg_steps": (sum(step_counts) / total) if total else 0.0,
            "avg_observe_act_ratio": (sum(observe_act_ratios) / total) if total else 0.0,
            "max_observe_act_ratio": max(observe_act_ratios) if observe_act_ratios else 0.0,
            "avg_latency_ms": (sum(latencies) / total) if total else 0.0,
        },
        cases=case_results,
    )


def load_eval_suite(path: Path) -> list[dict[str, Any]]:
    """Load JSONL or JSON evaluation cases."""
    resolved = path.expanduser().resolve()
    text = resolved.read_text(encoding="utf-8")
    if resolved.suffix.lower() == ".json":
        parsed = json.loads(text)
        if isinstance(parsed, list):
            return [item for item in parsed if isinstance(item, dict)]
        if isinstance(parsed, dict) and isinstance(parsed.get("cases"), list):
            return [item for item in parsed["cases"] if isinstance(item, dict)]
        raise ValueError(f"Unsupported JSON eval suite shape: {resolved}")
    cases: list[dict[str, Any]] = []
    for line_no, raw_line in enumerate(text.splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        loaded = json.loads(stripped)
        if not isinstance(loaded, dict):
            raise ValueError(f"{resolved}:{line_no}: expected JSON object case")
        cases.append(loaded)
    return cases


def default_eval_store_path(context_path: Path, *, config: Any = None) -> Path:
    """Return the append-only eval ledger path inside an AFS context root."""
    context_root = context_path.expanduser().resolve()
    scratchpad_root = resolve_mount_root(
        context_root,
        MountType.SCRATCHPAD,
        config=config,
    )
    is_v2 = detect_layout_version(context_root) == LAYOUT_VERSION
    boundary = context_root if is_v2 else None
    ledger = (
        scratchpad_root / "common" / "evals" / "runs.jsonl"
        if is_v2
        else scratchpad_root / "evals" / "runs.jsonl"
    )
    return assert_no_linklike_components(ledger, boundary=boundary, allow_missing=True)


class EvalRunStore:
    """Append-only JSONL ledger of historical evaluation runs and regression checks."""

    def __init__(self, path: Path) -> None:
        self.path = path.expanduser().resolve()

    def append(self, summary: EvalSuiteSummary) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        flags = (
            os.O_WRONLY
            | os.O_CREAT
            | os.O_APPEND
            | getattr(os, "O_CLOEXEC", 0)
            | getattr(os, "O_NOFOLLOW", 0)
        )
        fd = os.open(self.path, flags, 0o600)
        with os.fdopen(fd, "a", encoding="utf-8") as handle:
            handle.write(json.dumps(summary.to_dict(), ensure_ascii=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def list_runs(
        self,
        *,
        limit: int = 20,
        suite_kind: str | None = None,
    ) -> list[EvalSuiteSummary]:
        if not self.path.is_file():
            return []
        runs: list[EvalSuiteSummary] = []
        for raw_line in self.path.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line:
                continue
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(payload, dict):
                continue
            summary = EvalSuiteSummary.from_dict(payload)
            if suite_kind and summary.suite_kind != suite_kind:
                continue
            runs.append(summary)
        runs.reverse()
        if limit > 0:
            return runs[:limit]
        return runs

    def get_run(self, run_id: str) -> EvalSuiteSummary | None:
        for summary in self.list_runs(limit=0):
            if summary.run_id == run_id:
                return summary
        return None


def compare_runs(
    baseline: EvalSuiteSummary,
    candidate: EvalSuiteSummary,
    *,
    regression_tolerance: float = 0.01,
) -> EvalComparison:
    """Compare two evaluation summaries and identify regressions or improvements."""
    if baseline.suite_kind != candidate.suite_kind:
        raise ValueError(
            f"Cannot compare different suite kinds: {baseline.suite_kind!r} vs "
            f"{candidate.suite_kind!r}"
        )

    baseline_metrics = {"pass_rate": baseline.pass_rate, **baseline.metrics}
    candidate_metrics = {"pass_rate": candidate.pass_rate, **candidate.metrics}
    metric_keys = sorted(set(baseline_metrics) & set(candidate_metrics))

    deltas: dict[str, float] = {}
    regressions: list[dict[str, Any]] = []
    improvements: list[dict[str, Any]] = []
    tolerance = max(0.0, float(regression_tolerance))

    for key in metric_keys:
        base_val = float(baseline_metrics[key])
        cand_val = float(candidate_metrics[key])
        delta = cand_val - base_val
        deltas[key] = delta
        if key in _LOWER_IS_BETTER_METRICS:
            if delta > tolerance:
                regressions.append(
                    {"metric": key, "baseline": base_val, "candidate": cand_val, "delta": delta}
                )
            elif delta < -tolerance:
                improvements.append(
                    {"metric": key, "baseline": base_val, "candidate": cand_val, "delta": delta}
                )
        else:
            if delta < -tolerance:
                regressions.append(
                    {"metric": key, "baseline": base_val, "candidate": cand_val, "delta": delta}
                )
            elif delta > tolerance:
                improvements.append(
                    {"metric": key, "baseline": base_val, "candidate": cand_val, "delta": delta}
                )

    return EvalComparison(
        baseline_run_id=baseline.run_id,
        candidate_run_id=candidate.run_id,
        suite_kind=candidate.suite_kind,
        metric_deltas=deltas,
        regressions=regressions,
        improvements=improvements,
        passed=len(regressions) == 0,
    )
