from __future__ import annotations

import json
from pathlib import Path

import pytest

from afs.cli import build_parser
from afs.context_layout import scaffold_v2
from afs.evals import (
    EvalRunStore,
    EvalSuiteSummary,
    compare_retrieval_modes,
    compare_runs,
    evaluate_context_pack_suite,
    evaluate_retrieval_suite,
    evaluate_safety_suite,
    evaluate_trajectory_suite,
    load_eval_suite,
)
from afs.hybrid_search import HybridSearchEngine, HybridSource
from afs.manager import AFSManager
from afs.project_registry import ProjectRegistry
from afs.schema import AFSConfig, SensitivityConfig


def test_evaluate_retrieval_suite_shows_semantic_lift(tmp_path: Path) -> None:
    source = tmp_path / "repo"
    source.mkdir()
    # Lexical mismatch: distractor matches all lexical terms while the true target
    # expresses the same concept with disjoint vocabulary ("payload_sha256 hmac").
    (source / "noise.py").write_text(
        "def prevent_argument():\n    return 'prevent unverified argument handler'\n",
        encoding="utf-8",
    )
    (source / "approvals.py").write_text(
        "def verify_binding():\n    return 'payload_sha256 cryptographic hmac seal'\n",
        encoding="utf-8",
    )

    def embed_fn(text: str) -> list[float]:
        lowered = text.lower()
        if any(tok in lowered for tok in ("payload_sha256", "cryptographic", "seal")):
            return [1.0, 0.0]
        if "prevent" in lowered and "consent" in lowered:
            return [1.0, 0.0]
        return [0.0, 1.0]

    engine = HybridSearchEngine(tmp_path / "search")
    engine.build(
        [HybridSource(source, scope_id="project:afs", embed_allowed=True)],
        embed_fn=embed_fn,
    )

    cases = [
        {
            "id": "ret_semantic",
            "query": "prevent unverified argument consent",
            "expected_paths": ["approvals.py"],
        }
    ]
    summaries = evaluate_retrieval_suite(
        cases,
        engine=engine,
        modes=("text", "hybrid"),
        top_k=5,
        query_embed_fn=embed_fn,
        recreate_query_embedder=False,
    )
    lift = compare_retrieval_modes(summaries["text"], summaries["hybrid"])

    assert summaries["hybrid"].metrics["hit_at_k"] == 1.0
    assert summaries["hybrid"].metrics["mrr"] >= 0.5
    assert summaries["hybrid"].metrics["semantic_ready_rate"] == 1.0
    assert summaries["text"].metrics["mrr"] == 0.0
    assert lift["hit_at_k_delta"] == 1.0
    assert lift["mrr_delta"] >= 0.5


def test_evaluate_context_pack_suite_respects_budget_and_zero_leaks(tmp_path: Path) -> None:
    context = tmp_path / ".context"
    project = tmp_path / "project"
    project.mkdir()
    scaffold_v2(context)
    record = ProjectRegistry(context).register(project)
    knowledge_dir = context / "knowledge" / "projects" / record.project_id
    knowledge_dir.mkdir(parents=True, exist_ok=True)
    (knowledge_dir / "architecture.md").write_text(
        "# Architecture\nHybrid retrieval fuses FTS5 and normalized vectors.\n",
        encoding="utf-8",
    )
    (knowledge_dir / "secret.md").write_text(
        "# Secret\ntop_secret_token_never_export_marker\n",
        encoding="utf-8",
    )

    config = AFSConfig(sensitivity=SensitivityConfig(never_export=["**/secret.md", "secret.md"]))
    manager = AFSManager(config=config)
    summary = evaluate_context_pack_suite(
        [
            {
                "id": "pack_arch",
                "query": "hybrid retrieval",
                "task": "Review architecture",
                "token_budget": 4000,
                "include_content": True,
                "expected_terms": ["hybrid retrieval", "architecture"],
                "forbidden_terms": ["top_secret_token_never_export_marker"],
                "forbidden_paths": ["secret.md"],
            }
        ],
        manager=manager,
        context_path=context,
        project_path=project,
    )

    assert summary.total == 1
    assert summary.passed == 1
    assert summary.metrics["budget_respected_rate"] == 1.0
    assert summary.metrics["zero_secret_leak_rate"] == 1.0
    assert summary.metrics["recall_in_budget"] == 1.0


def test_evaluate_safety_suite_enforces_tamper_and_once_tokens() -> None:
    summary = evaluate_safety_suite(
        [
            {
                "id": "s_auto",
                "scenario": "auto_approve",
                "agent": "agy",
                "action": "file_read",
                "detail": "README.md",
                "expected_allowed": True,
            },
            {
                "id": "s_approve",
                "scenario": "human_gate",
                "agent": "deployer",
                "action": "git_push",
                "decision": "approve",
            },
            {
                "id": "s_reject",
                "scenario": "human_gate",
                "agent": "deployer",
                "action": "drop_db",
                "decision": "reject",
            },
            {
                "id": "s_tamper",
                "scenario": "tamper_reject",
                "agent": "deployer",
                "action": "deploy",
                "payload": {"cmd": "staging"},
                "tampered_payload": {"cmd": "prod"},
            },
            {
                "id": "s_once",
                "scenario": "once_single_use",
                "agent": "migrator",
                "action": "migrate",
            },
        ]
    )

    assert summary.total == 5
    assert summary.passed == 5
    assert summary.pass_rate == 1.0
    assert summary.metrics["gate_enforcement_rate"] == 1.0
    assert summary.metrics["tamper_block_rate"] == 1.0
    assert summary.metrics["audit_completeness_rate"] == 1.0


def test_evaluate_trajectory_suite_slim_compliance_and_trace_file(tmp_path: Path) -> None:
    trace_file = tmp_path / "trace.jsonl"
    trace_file.write_text(
        "\n".join(
            [
                json.dumps({"type": "mcp_tool", "metadata": {"tool_name": "context.read", "ok": True}}),
                json.dumps({"type": "mcp_tool", "metadata": {"tool_name": "context.write", "ok": True}}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    summary = evaluate_trajectory_suite(
        [
            {
                "id": "traj_good_trace",
                "trace_path": "trace.jsonl",
                "max_steps": 4,
                "max_observe_act_ratio": 1.5,
                "slim_mcp_only": True,
                "must_call": ["context.read", "context.write"],
            },
            {
                "id": "traj_bad_non_slim",
                "slim_mcp_only": True,
                "max_observe_act_ratio": 1.5,
                "events": [
                    {"type": "mcp_tool", "metadata": {"tool_name": "events.read", "ok": True}},
                    {"type": "mcp_tool", "metadata": {"tool_name": "context.read", "ok": True}},
                    {"type": "mcp_tool", "metadata": {"tool_name": "context.read", "ok": True}},
                    {"type": "mcp_tool", "metadata": {"tool_name": "context.write", "ok": True}},
                ],
            },
        ],
        base_dir=tmp_path,
    )

    assert summary.total == 2
    assert summary.passed == 1
    assert summary.failed == 1
    assert summary.metrics["slim_compliance_rate"] == 0.5
    assert summary.cases[0].passed is True
    assert summary.cases[1].passed is False
    assert "non-slim MCP tools" in " ".join(summary.cases[1].details["failures"])


def test_eval_run_store_persistence_and_regression_detection(tmp_path: Path) -> None:
    store = EvalRunStore(tmp_path / "evals" / "runs.jsonl")
    baseline = EvalSuiteSummary(
        suite_name="core",
        suite_kind="retrieval",
        run_id="run_base",
        created_at="2026-11-26T00:00:00+00:00",
        git_sha="abc123",
        config={},
        total=4,
        passed=4,
        failed=0,
        pass_rate=1.0,
        metrics={"mrr": 0.90, "avg_latency_ms": 12.0},
        cases=[],
    )
    candidate = EvalSuiteSummary(
        suite_name="core",
        suite_kind="retrieval",
        run_id="run_cand",
        created_at="2026-11-26T00:05:00+00:00",
        git_sha="def456",
        config={},
        total=4,
        passed=3,
        failed=1,
        pass_rate=0.75,
        metrics={"mrr": 0.70, "avg_latency_ms": 9.0},
        cases=[],
    )
    store.append(baseline)
    store.append(candidate)

    runs = store.list_runs(limit=10)
    assert [r.run_id for r in runs] == ["run_cand", "run_base"]
    assert store.get_run("run_base") is not None

    comparison = compare_runs(baseline, candidate, regression_tolerance=0.01)
    assert comparison.passed is False
    reg_metrics = {r["metric"] for r in comparison.regressions}
    imp_metrics = {i["metric"] for i in comparison.improvements}
    assert {"pass_rate", "mrr"}.issubset(reg_metrics)
    assert "avg_latency_ms" in imp_metrics


def test_cli_eval_seed_run_list_and_compare_fail_on_regression(
    tmp_path: Path, capsys
) -> None:
    parser = build_parser([])
    seed_path = tmp_path / "seeded.jsonl"
    store_path = tmp_path / "runs.jsonl"

    seed_args = parser.parse_args(["eval", "seed", "--output", str(seed_path)])
    assert seed_args.func(seed_args) == 0
    capsys.readouterr()
    assert len(load_eval_suite(seed_path)) == 4

    run_args = parser.parse_args(
        [
            "eval",
            "run",
            "--suite",
            str(seed_path),
            "--kind",
            "safety",
            "--save",
            "--store",
            str(store_path),
            "--json",
        ]
    )
    assert run_args.func(run_args) == 0
    run_payload = json.loads(capsys.readouterr().out)
    run_id = run_payload["summaries"][0]["run_id"]

    list_args = parser.parse_args(["eval", "list", "--store", str(store_path), "--json"])
    assert list_args.func(list_args) == 0
    listed = json.loads(capsys.readouterr().out)
    assert listed[0]["run_id"] == run_id

    store = EvalRunStore(store_path)
    degraded = EvalSuiteSummary.from_dict(run_payload["summaries"][0])
    degraded.run_id = "safety_degraded"
    degraded.pass_rate = 0.0
    degraded.metrics["tamper_block_rate"] = 0.0
    store.append(degraded)

    cmp_args = parser.parse_args(
        [
            "eval",
            "compare",
            run_id,
            "safety_degraded",
            "--store",
            str(store_path),
            "--fail-on-regression",
            "--json",
        ]
    )
    rc = cmp_args.func(cmp_args)
    assert rc == 2
    cmp_payload = json.loads(capsys.readouterr().out)
    assert cmp_payload["passed"] is False


def test_core_eval_suite_safety_and_trajectory_pass_locally() -> None:
    suite_path = Path(__file__).resolve().parents[1] / "configs" / "evals" / "afs_core_eval.jsonl"
    cases = load_eval_suite(suite_path)
    safety_cases = [c for c in cases if c.get("kind") == "safety"]
    traj_cases = [c for c in cases if c.get("kind") == "trajectory"]

    safety_summary = evaluate_safety_suite(safety_cases)
    traj_summary = evaluate_trajectory_suite(traj_cases)

    assert safety_summary.pass_rate == 1.0
    assert traj_summary.pass_rate == 1.0
    assert traj_summary.metrics["slim_compliance_rate"] == 1.0
    assert pytest.approx(safety_summary.metrics["tamper_block_rate"]) == 1.0
