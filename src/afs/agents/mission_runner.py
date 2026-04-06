"""Mission runner agent — reads TOML mission definitions and executes OODA phases."""

from __future__ import annotations

import json
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .base import (
    AgentResult,
    build_base_parser,
    configure_logging,
    emit_progress,
    emit_result,
    now_iso,
)
from .guardrails import GuardrailConfig, GuardrailedAgent

# Late imports to avoid circular deps — resolved at call sites
# from ..config import load_config_model
# from ..context_index import ContextSQLiteIndex
# from ..discovery import discover_contexts
# from ..history import query_events
# from ..manager import AFSManager
# from ..models import MountType

logger = logging.getLogger(__name__)

# Default paths for Zelda projects (can be overridden via mission config)
DEFAULT_OOS_PATH = Path.home() / "src" / "hobby" / "oracle-of-secrets"
DEFAULT_YAZE_PATH = Path.home() / "src" / "hobby" / "yaze"
DEFAULT_MESEN2_PATH = Path.home() / "src" / "third_party" / "forks" / "mesen2"
DEFAULT_DISASM_PATH = Path.home() / "src" / "hobby" / "USDASM"

AGENT_NAME = "mission-runner"
AGENT_DESCRIPTION = (
    "Read mission definitions from scratchpad/missions/, execute OODA phases "
    "(observe → orient → decide → act), write results, and respect guardrails."
)

AGENT_CAPABILITIES = {
    "tools": [
        "context.read", "context.write", "context.query", "context.list",
        "context.diff", "embedding_search", "embedding_update",
    ],
    "topics": ["missions", "analysis", "automation"],
    "mount_types": ["scratchpad", "knowledge", "memory", "history"],
    "description": "Autonomous mission executor with OODA loop and guardrails",
}

MISSIONS_DIR = "scratchpad/missions"


# ---------------------------------------------------------------------------
# Mission model
# ---------------------------------------------------------------------------

@dataclass
class MissionPhase:
    name: str
    description: str = ""
    tools: list[str] = field(default_factory=list)
    outputs: list[str] = field(default_factory=list)
    requires_approval: bool = False


@dataclass
class MissionGuardrails:
    require_worktree: bool = False
    require_approval_for: list[str] = field(default_factory=list)
    max_iterations: int = 50
    dry_run: bool = False


@dataclass
class Mission:
    name: str
    description: str = ""
    tier: str = "background"
    owner: str = ""
    status: str = "pending"
    guardrails: MissionGuardrails = field(default_factory=MissionGuardrails)
    phases: list[MissionPhase] = field(default_factory=list)


def _load_mission(path: Path) -> Mission | None:
    """Load a mission definition from a TOML file."""
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ImportError:
            logger.error("No TOML parser available (need Python 3.11+ or tomli)")
            return None

    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to parse mission %s: %s", path, exc)
        return None

    m = data.get("mission", {})
    guardrails_data = m.get("guardrails", {})
    phases_data = m.get("phases", [])

    guardrails = MissionGuardrails(
        require_worktree=guardrails_data.get("require_worktree", False),
        require_approval_for=guardrails_data.get("require_approval_for", []),
        max_iterations=guardrails_data.get("max_iterations", 50),
        dry_run=guardrails_data.get("dry_run", False),
    )

    phases = []
    for phase_data in phases_data:
        phases.append(MissionPhase(
            name=phase_data.get("name", ""),
            description=phase_data.get("description", ""),
            tools=phase_data.get("tools", []),
            outputs=phase_data.get("outputs", []),
            requires_approval=phase_data.get("requires_approval", False),
        ))

    return Mission(
        name=m.get("name", path.stem),
        description=m.get("description", ""),
        tier=m.get("tier", "background"),
        owner=m.get("owner", ""),
        status=m.get("status", "pending"),
        guardrails=guardrails,
        phases=phases,
    )


def load_mission(path: Path) -> dict[str, Any]:
    """Load a mission TOML file and return its raw contents (dict)."""
    try:
        import tomllib
    except ImportError:
        try:
            import tomli as tomllib  # type: ignore[no-redef]
        except ImportError as exc:
            raise RuntimeError("No TOML parser available (need Python 3.11+ or tomli)") from exc
    with path.open("rb") as f:
        return tomllib.load(f)


def _is_zelda_mission(mission_data: dict[str, Any]) -> bool:
    """Detect whether a mission is zelda-related by name, owner, or description."""
    m = mission_data.get("mission", {})
    name = str(m.get("name", "")).lower()
    owner = str(m.get("owner", "")).lower()
    description = str(m.get("description", "")).lower()
    return (
        "zelda" in name
        or "zelda" in owner
        or "oos" in name
        or "oracle-of-secrets" in name
        or "oracle of secrets" in description
        or "mesen" in name
        or "yaze" in name
    )


def _discover_missions(context_root: Path) -> list[tuple[Path, Mission]]:
    """Find all pending mission files."""
    missions_dir = context_root / MISSIONS_DIR
    if not missions_dir.is_dir():
        return []

    results = []
    for path in sorted(missions_dir.glob("*.toml")):
        mission = _load_mission(path)
        if mission and mission.status == "pending":
            results.append((path, mission))
    return results


# ---------------------------------------------------------------------------
# Phase tool execution — real AFS operations
# ---------------------------------------------------------------------------

def _run_phase_tools(
    phase: MissionPhase,
    mission: Mission,
    guard: GuardrailedAgent,
    output_dir: Path,
    context_root: Path | None = None,
) -> dict[str, Any]:
    """Execute real AFS tools for a phase. Returns collected data and notes."""
    notes: list[str] = []
    data: dict[str, Any] = {}

    if context_root is None:
        notes.append(f"Phase {phase.name}: no context_root, skipped tool calls")
        return {"notes": notes, "data": data}

    context_path = context_root
    if not (context_path / "metadata.json").exists():
        # Try .context subdirectory
        if (context_path / ".context").is_dir():
            context_path = context_path / ".context"

    if phase.name == "observe":
        data = _phase_observe(phase, context_path, guard, notes)
    elif phase.name == "orient":
        data = _phase_orient(phase, context_path, guard, notes, output_dir)
    elif phase.name == "decide":
        data = _phase_decide(phase, context_path, guard, notes, output_dir)
    elif phase.name == "act":
        data = _phase_act(phase, context_path, guard, notes, output_dir)
    else:
        notes.append(f"Unknown phase type '{phase.name}', no tools executed")

    return {"notes": notes, "data": data}


def _phase_observe(
    phase: MissionPhase,
    context_path: Path,
    guard: GuardrailedAgent,
    notes: list[str],
) -> dict[str, Any]:
    """Observe phase: read-only data gathering from context, git, and index."""
    data: dict[str, Any] = {"context_health": {}, "index_summary": {}, "recent_events": []}

    # 1. Context health check
    try:
        from ..config import load_config_model
        from ..manager import AFSManager
        config = load_config_model(merge_user=True)
        manager = AFSManager(config=config)
        health = manager.context_health(context_path=context_path)
        data["context_health"] = health
        issues = health.get("missing_directories", []) + health.get("broken_mounts", [])
        if issues:
            notes.append(f"Context health: {len(issues)} issues found")
        else:
            notes.append("Context health: OK")
    except Exception as exc:
        notes.append(f"Context health check failed: {exc}")

    # 2. Index query — search for recently modified content
    try:
        from ..context_index import ContextSQLiteIndex
        from ..manager import AFSManager
        from ..models import MountType
        config = load_config_model(merge_user=True)
        manager = AFSManager(config=config)
        index = ContextSQLiteIndex(manager, context_path)
        if index.has_entries():
            summary = index.summary()
            data["index_summary"] = {
                "total_entries": summary.rows_written,
                "mount_types": summary.by_mount_type,
            }
            # Check for stale content
            diff = index.diff()
            data["index_diff"] = {
                "added": len(diff.get("added", [])),
                "modified": len(diff.get("modified", [])),
                "deleted": len(diff.get("deleted", [])),
            }
            notes.append(
                f"Index: {summary.rows_written} entries, "
                f"{len(diff.get('modified', []))} modified since last index"
            )
        else:
            notes.append("Index: empty, needs rebuild")
    except Exception as exc:
        notes.append(f"Index query failed: {exc}")

    # 3. Recent history events
    try:
        from ..context_paths import resolve_mount_root
        from ..history import query_events
        from ..models import MountType
        history_root = resolve_mount_root(context_path, MountType.HISTORY)
        if history_root.exists():
            events = query_events(history_root, limit=20)
            data["recent_events"] = [
                {"type": e.get("type", ""), "source": e.get("source", ""),
                 "op": e.get("op", ""), "timestamp": e.get("timestamp", "")}
                for e in events
            ]
            notes.append(f"History: {len(events)} recent events")
    except Exception as exc:
        notes.append(f"History query failed: {exc}")

    return data


def _phase_orient(
    phase: MissionPhase,
    context_path: Path,
    guard: GuardrailedAgent,
    notes: list[str],
    output_dir: Path,
) -> dict[str, Any]:
    """Orient phase: analyze observations, run index queries, cross-reference."""
    data: dict[str, Any] = {"findings": [], "query_results": []}

    # Read observe phase output if available
    obs_path = output_dir / "observations.json"
    observations = {}
    if obs_path.exists():
        try:
            observations = json.loads(obs_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    # Run targeted queries based on mission description
    try:
        from ..config import load_config_model
        from ..context_index import ContextSQLiteIndex
        from ..manager import AFSManager
        from ..models import MountType
        config = load_config_model(merge_user=True)
        manager = AFSManager(config=config)
        index = ContextSQLiteIndex(manager, context_path)

        # Query knowledge for mission-relevant content
        if index.has_entries():
            results = index.query(
                query=phase.description or "recent changes",
                mount_types=[MountType.KNOWLEDGE, MountType.MEMORY],
                limit=15,
                include_content=False,
            )
            data["query_results"] = [
                {"path": r.get("relative_path", ""), "mount": r.get("mount_type", ""),
                 "excerpt": r.get("content_excerpt", "")[:200]}
                for r in results
            ]
            notes.append(f"Orient: found {len(results)} relevant context entries")
        else:
            notes.append("Orient: index empty, no queries possible")
    except Exception as exc:
        notes.append(f"Orient query failed: {exc}")

    # Analyze index diff for patterns
    obs_data = observations.get("data", {})
    index_diff = obs_data.get("index_diff", {})
    if index_diff:
        total_changes = index_diff.get("added", 0) + index_diff.get("modified", 0)
        if total_changes > 0:
            data["findings"].append({
                "type": "index_drift",
                "severity": "info" if total_changes < 10 else "warning",
                "detail": f"{total_changes} files changed since last index",
            })

    health = obs_data.get("context_health", {})
    if health.get("missing_directories"):
        data["findings"].append({
            "type": "missing_dirs",
            "severity": "warning",
            "detail": f"Missing: {', '.join(health['missing_directories'])}",
        })

    notes.append(f"Orient: {len(data['findings'])} findings")

    # LLM enrichment — optional, never gates the phase
    try:
        from .llm_bridge import query_llm
        model = guard.resolve_model(task_tier="background")
        findings_json = json.dumps({
            "query_results": data.get("query_results", []),
            "findings": data.get("findings", []),
        }, indent=2, default=str)
        llm_response = query_llm(
            prompt=(
                "Analyze these findings from the observe phase and identify "
                "patterns, risks, and priorities: " + findings_json
            ),
            context=data,
            model_route=model,
        )
        if not llm_response.startswith("ERROR:"):
            data["llm_analysis"] = llm_response
            guard.record_call(model.provider)
            notes.append("Orient: LLM analysis added")
        else:
            notes.append(f"Orient: LLM skipped — {llm_response}")
    except Exception as exc:
        notes.append(f"Orient: LLM enrichment failed (non-fatal): {exc}")

    return data


def _phase_decide(
    phase: MissionPhase,
    context_path: Path,
    guard: GuardrailedAgent,
    notes: list[str],
    output_dir: Path,
) -> dict[str, Any]:
    """Decide phase: produce action plan from analysis."""
    data: dict[str, Any] = {"actions": [], "deferred": []}

    # Read orient phase output
    orient_path = output_dir / "analysis.json"
    analysis = {}
    if orient_path.exists():
        try:
            analysis = json.loads(orient_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    findings = analysis.get("data", {}).get("findings", [])

    for finding in findings:
        severity = finding.get("severity", "info")
        finding_type = finding.get("type", "unknown")

        if finding_type == "index_drift":
            data["actions"].append({
                "action": "embedding_update",
                "reason": finding["detail"],
                "auto_approve": True,
            })
        elif finding_type == "missing_dirs":
            data["deferred"].append({
                "action": "context_repair",
                "reason": finding["detail"],
                "requires_approval": True,
            })
        elif severity == "warning":
            data["deferred"].append({
                "action": "investigate",
                "reason": finding.get("detail", ""),
                "requires_approval": True,
            })

    notes.append(
        f"Decide: {len(data['actions'])} auto-actions, "
        f"{len(data['deferred'])} deferred for review"
    )

    # LLM enrichment — optional, never gates the phase
    try:
        from .llm_bridge import query_llm
        model = guard.resolve_model(task_tier="background")
        analysis_json = json.dumps({
            "findings": findings,
            "current_actions": data.get("actions", []),
            "current_deferred": data.get("deferred", []),
            "llm_analysis": analysis.get("data", {}).get("llm_analysis", ""),
        }, indent=2, default=str)
        llm_response = query_llm(
            prompt=(
                "Based on this analysis, produce a JSON action plan with "
                "'actions' (auto-approvable) and 'deferred' (needs review): "
                + analysis_json
            ),
            context={"findings": findings},
            model_route=model,
        )
        if not llm_response.startswith("ERROR:"):
            # Try to parse the LLM response as JSON and merge actions
            try:
                llm_plan = json.loads(llm_response)
                if isinstance(llm_plan, dict):
                    for action in llm_plan.get("actions", []):
                        if isinstance(action, dict) and action not in data["actions"]:
                            action["source"] = "llm"
                            data["actions"].append(action)
                    for item in llm_plan.get("deferred", []):
                        if isinstance(item, dict) and item not in data["deferred"]:
                            item["source"] = "llm"
                            data["deferred"].append(item)
                    notes.append("Decide: LLM action plan merged")
            except (json.JSONDecodeError, TypeError):
                # LLM didn't return valid JSON — store as raw analysis
                data["llm_action_plan_raw"] = llm_response
                notes.append("Decide: LLM response stored (non-JSON)")
            guard.record_call(model.provider)
        else:
            notes.append(f"Decide: LLM skipped — {llm_response}")
    except Exception as exc:
        notes.append(f"Decide: LLM enrichment failed (non-fatal): {exc}")

    return data


def _phase_act(
    phase: MissionPhase,
    context_path: Path,
    guard: GuardrailedAgent,
    notes: list[str],
    output_dir: Path,
) -> dict[str, Any]:
    """Act phase: execute approved actions within guardrails."""
    data: dict[str, Any] = {"executed": [], "skipped": [], "errors": []}

    # Read decide phase output
    decide_path = output_dir / "decisions.json"
    decisions = {}
    if decide_path.exists():
        try:
            decisions = json.loads(decide_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    actions = decisions.get("data", {}).get("actions", [])

    for action_spec in actions:
        action = action_spec.get("action", "")
        reason = action_spec.get("reason", "")

        if not guard.can_do(action, reason):
            data["skipped"].append({"action": action, "reason": "approval required"})
            notes.append(f"Skipped {action}: needs approval")
            continue

        try:
            if action == "embedding_update":
                # Rebuild stale index entries
                from ..config import load_config_model
                from ..context_index import ContextSQLiteIndex
                from ..manager import AFSManager
                config = load_config_model(merge_user=True)
                manager = AFSManager(config=config)
                index = ContextSQLiteIndex(manager, context_path)
                summary = index.rebuild()
                data["executed"].append({
                    "action": action,
                    "result": {
                        "total_entries": summary.rows_written,
                        "mount_types": summary.by_mount_type,
                    },
                })
                notes.append(f"Rebuilt index: {summary.rows_written} entries")

            elif action == "context_write":
                # Write findings to scratchpad
                scratchpad = context_path / "scratchpad"
                if scratchpad.exists() and guard.can_do("file_write_scratchpad", "write mission results"):
                    report_path = scratchpad / "afs_agents" / "mission_report.json"
                    report_path.parent.mkdir(parents=True, exist_ok=True)
                    report_path.write_text(json.dumps({
                        "mission": output_dir.name,
                        "timestamp": now_iso(),
                        "actions_executed": len(data["executed"]),
                    }, indent=2), encoding="utf-8")
                    data["executed"].append({"action": action, "result": str(report_path)})
                    notes.append(f"Wrote report to {report_path.name}")

            else:
                data["skipped"].append({"action": action, "reason": f"no handler for {action}"})

        except Exception as exc:
            data["errors"].append({"action": action, "error": str(exc)})
            notes.append(f"Error executing {action}: {exc}")

    notes.append(
        f"Act: {len(data['executed'])} executed, "
        f"{len(data['skipped'])} skipped, {len(data['errors'])} errors"
    )
    return data


# ---------------------------------------------------------------------------
# Phase dispatch
# ---------------------------------------------------------------------------

def _execute_phase(
    phase: MissionPhase,
    mission: Mission,
    guard: GuardrailedAgent,
    output_dir: Path,
    context_root: Path | None = None,
) -> dict[str, Any]:
    """Execute a single OODA phase. Returns phase result dict."""
    started = now_iso()
    emit_progress(AGENT_NAME, f"phase_{phase.name}", mission.name)

    result: dict[str, Any] = {
        "phase": phase.name,
        "started_at": started,
        "status": "ok",
        "notes": [],
        "outputs": [],
    }

    # Check approval gate for phases that require it
    if phase.requires_approval:
        if not guard.can_do(
            f"mission_phase_{phase.name}",
            f"Mission '{mission.name}' phase '{phase.name}': {phase.description}",
        ):
            result["status"] = "awaiting_approval"
            result["notes"].append(f"Phase {phase.name} requires human approval")
            result["finished_at"] = now_iso()
            return result

    # Resolve model for this phase
    try:
        model = guard.resolve_model(
            task_tier="background" if phase.name == "observe" else mission.tier,
        )
        result["model"] = {"provider": model.provider, "model_id": model.model_id}
        guard.record_call(model.provider)
    except RuntimeError as exc:
        result["status"] = "error"
        result["notes"].append(f"Model resolution failed: {exc}")
        result["finished_at"] = now_iso()
        return result

    # Execute phase with real AFS tool calls
    phase_data = _run_phase_tools(phase, mission, guard, output_dir, context_root)
    result["notes"].extend(phase_data.get("notes", []))
    result["data"] = phase_data.get("data", {})

    # Write phase outputs
    for output_file in phase.outputs:
        out_path = output_dir / output_file
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({
            "phase": phase.name,
            "mission": mission.name,
            "data": result.get("data", {}),
            "notes": result["notes"],
            "model": result.get("model", {}),
        }, indent=2), encoding="utf-8")
    result["outputs"] = [str(output_dir / out) for out in phase.outputs]

    result["finished_at"] = now_iso()
    return result


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def _run_mission(
    mission_path: Path,
    mission: Mission,
    context_root: Path,
) -> dict[str, Any]:
    """Run a single mission through all its phases."""
    output_dir = context_root / MISSIONS_DIR / mission.name
    output_dir.mkdir(parents=True, exist_ok=True)

    guard = GuardrailedAgent(
        AGENT_NAME,
        config=GuardrailConfig(
            task_tier=mission.tier,
            max_iterations=mission.guardrails.max_iterations,
            dry_run=mission.guardrails.dry_run,
        ),
    )

    mission_result: dict[str, Any] = {
        "mission": mission.name,
        "description": mission.description,
        "started_at": now_iso(),
        "phases": [],
        "status": "ok",
    }

    for phase in mission.phases:
        if not guard.should_continue():
            mission_result["status"] = "iteration_cap"
            mission_result["phases"].append({
                "phase": phase.name, "status": "skipped", "reason": "iteration cap",
            })
            break

        phase_result = _execute_phase(phase, mission, guard, output_dir, context_root)
        mission_result["phases"].append(phase_result)

        if phase_result["status"] == "awaiting_approval":
            mission_result["status"] = "awaiting_approval"
            break
        if phase_result["status"] == "error":
            mission_result["status"] = "error"
            break

        if guard.should_checkpoint():
            checkpoint_path = output_dir / "checkpoint.json"
            checkpoint_path.write_text(json.dumps(mission_result, indent=2), encoding="utf-8")

    mission_result["finished_at"] = now_iso()
    mission_result["quota_usage"] = guard.usage_summary()
    mission_result["pending_approvals"] = [a.to_dict() for a in guard.pending_approvals()]

    # Write final result
    result_path = output_dir / "result.json"
    result_path.write_text(json.dumps(mission_result, indent=2), encoding="utf-8")

    return mission_result


def build_parser():
    parser = build_base_parser("Execute pending missions from scratchpad/missions/.")
    parser.add_argument(
        "--context-root",
        default=str(Path.home() / "src" / "lab" / ".context"),
        help="Context root directory.",
    )
    parser.add_argument(
        "--mission",
        default="",
        help="Run a specific mission by name (default: run all pending).",
    )
    return parser


def run(args) -> int:
    configure_logging(args.quiet)
    started_at = now_iso()
    start = time.time()
    context_root = Path(args.context_root).expanduser()

    # Load context snapshot for index/memory awareness during mission execution
    try:
        from ..agent_context import load_agent_context_snapshot
        ctx = load_agent_context_snapshot()
        if ctx:
            logger.info(
                "Mission runner context: %d indexed, %d memory topics",
                ctx.index_total, len(ctx.memory_topics),
            )
    except Exception:
        pass

    missions = _discover_missions(context_root)
    if args.mission:
        missions = [(p, m) for p, m in missions if m.name == args.mission]

    if not missions:
        logger.info("No pending missions found")
        result = AgentResult(
            name=AGENT_NAME,
            status="ok",
            started_at=started_at,
            finished_at=now_iso(),
            duration_seconds=time.time() - start,
            notes=["no pending missions"],
        )
        emit_result(result, output_path=None, force_stdout=bool(args.stdout), pretty=bool(args.pretty))
        return 0

    all_results = []
    for mission_path, mission in missions:
        logger.info("Running mission: %s", mission.name)
        emit_progress(AGENT_NAME, "mission_start", mission.name)
        mission_result = _run_mission(mission_path, mission, context_root)
        all_results.append(mission_result)

    result = AgentResult(
        name=AGENT_NAME,
        status="ok",
        started_at=started_at,
        finished_at=now_iso(),
        duration_seconds=time.time() - start,
        metrics={
            "missions_found": len(missions),
            "missions_completed": sum(1 for r in all_results if r["status"] == "ok"),
            "missions_blocked": sum(1 for r in all_results if r["status"] == "awaiting_approval"),
        },
        payload={"missions": all_results},
    )
    emit_result(
        result,
        output_path=Path(args.output).expanduser() if args.output else None,
        force_stdout=bool(args.stdout),
        pretty=bool(args.pretty),
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
