"""Model-aware session workflow profiles used by context packs and adapters."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

WORKFLOW_CHOICES = (
    "general",
    "scan_fast",
    "edit_fast",
    "review_deep",
    "root_cause_deep",
)
TOOL_PROFILE_CHOICES = (
    "default",
    "context_readonly",
    "context_repair",
    "edit_and_verify",
    "handoff_only",
)


@dataclass(frozen=True)
class SessionWorkflowDefinition:
    name: str
    summary: str
    intent: str
    default_tool_profile: str
    prompt_contract: tuple[str, ...]
    verification_contract: tuple[str, ...]
    model_hints: dict[str, str]
    retry_contract: tuple[str, ...]
    retry_hints: dict[str, str]


@dataclass(frozen=True)
class SessionToolProfileDefinition:
    name: str
    summary: str
    preferred_surfaces: tuple[str, ...]
    notes: tuple[str, ...]


PROMPT_ONLY_LOOP_POLICY = (
    "Prompt-only rail. Let the host CLI manage turn state; AFS supplies context, "
    "schemas, tool narrowing, and digest hints without owning execution state."
)


WORKFLOW_DEFINITIONS = {
    "general": SessionWorkflowDefinition(
        name="general",
        summary="Balanced workflow when the task is not classified yet.",
        intent="Read the cited context, keep the plan flat, and move to action without over-scaffolding.",
        default_tool_profile="default",
        prompt_contract=(
            "Use cited context before assuming missing facts.",
            "Keep the plan flat and short.",
            "Prefer the smallest working change or recommendation.",
        ),
        verification_contract=(
            "Run the fastest relevant verification before handoff.",
            "If checks cannot run, state the exact blocker and residual risk.",
        ),
        model_hints={
            "generic": "Use the model's default reasoning level and escalate only when the evidence stays ambiguous.",
            "gemini": "Start with Flash for narrow retrieval or extraction, move to Pro for multi-file synthesis, and use Deep Think only for hard root-cause work.",
            "claude": "Prefer a short execution plan before editing.",
            "codex": "Bias toward concrete files, diffs, and checks over general discussion.",
        },
        retry_contract=(
            "If the answer gets diffuse, retry with a narrower query or retrieval pack before changing workflows.",
            "Use a schema-bound prompt only when the response format matters more than free-form explanation.",
        ),
        retry_hints={
            "generic": "Retry with a narrower query or smaller pack before escalating the workflow.",
            "gemini": "If Flash diffuses, retry with a narrower query or retrieval pack; escalate to Pro only after the context is clean.",
            "claude": "Retry with a shorter execution contract before widening the context.",
            "codex": "Retry with fewer files and a concrete verification target before broadening the task.",
        },
    ),
    "scan_fast": SessionWorkflowDefinition(
        name="scan_fast",
        summary="Fast triage, search, classification, or short read-only investigation.",
        intent="Optimize for narrowing and evidence collection rather than full implementation.",
        default_tool_profile="context_readonly",
        prompt_contract=(
            "Prefer search, status, and diff before reading large files.",
            "Return a shortlist or decision, not a long essay.",
            "Attach file paths or commands to key conclusions.",
        ),
        verification_contract=(
            "Show the evidence that drove the shortlist or conclusion.",
            "Escalate to a deeper workflow only when the evidence conflicts.",
        ),
        model_hints={
            "generic": "Use a fast/default reasoning mode focused on extraction.",
            "gemini": "Flash is the default fit here; escalate to Pro only if the evidence needs synthesis.",
            "claude": "Keep the scan bounded and path-oriented.",
            "codex": "Prefer concrete hits and next files to inspect.",
        },
        retry_contract=(
            "Retry with a tighter query or smaller shortlist before widening the search.",
            "Escalate to a deeper workflow only when the evidence conflicts or stays incomplete.",
        ),
        retry_hints={
            "generic": "Retry with a narrower retrieval query before switching workflows.",
            "gemini": "Stay on Flash for the retry; move to Pro only if the extracted evidence still needs synthesis.",
            "claude": "Retry with a shorter prompt and clearer decision target.",
            "codex": "Retry with concrete file filters instead of reading more files by default.",
        },
    ),
    "edit_fast": SessionWorkflowDefinition(
        name="edit_fast",
        summary="Bounded code or config edit with immediate verification.",
        intent="Make the smallest working change, then verify immediately.",
        default_tool_profile="edit_and_verify",
        prompt_contract=(
            "Plan in three steps or fewer.",
            "Edit the smallest surface that can satisfy the task.",
            "Keep cited files attached to claims and edits.",
        ),
        verification_contract=(
            "Run the fastest relevant check immediately after editing.",
            "Report only failing or missing verification unless asked for a full log.",
        ),
        model_hints={
            "generic": "Use a balanced reasoning mode with a short edit loop.",
            "gemini": "Flash works for narrow edits; switch to Pro when the change spans multiple modules or constraints.",
            "claude": "Translate the pack into a concise edit checklist before touching files.",
            "codex": "Focus on implementation order and verification commands.",
        },
        retry_contract=(
            "If the edit loop drifts, rerun with `afs.workflow.structured` and the `edit-intent` schema.",
            "Retry with fewer files or a retrieval pack before widening the patch.",
        ),
        retry_hints={
            "generic": "Retry with a smaller patch scope and explicit verification target.",
            "gemini": "Retry on Flash for a narrower edit; escalate to Pro only when the change crosses modules or constraints.",
            "claude": "Retry with a three-step checklist and concrete files.",
            "codex": "Retry with a reduced file set and immediate verification command.",
        },
    ),
    "review_deep": SessionWorkflowDefinition(
        name="review_deep",
        summary="Deep review or design evaluation focused on bugs, risks, and gaps.",
        intent="Surface concrete findings with evidence instead of rewriting the implementation.",
        default_tool_profile="context_readonly",
        prompt_contract=(
            "Order findings by severity.",
            "Use concrete file references and behavior-focused reasoning.",
            "Separate findings from summaries or change narration.",
        ),
        verification_contract=(
            "Flag missing tests and unverified paths explicitly.",
            "Call out assumptions when they affect severity or confidence.",
        ),
        model_hints={
            "generic": "Use a deeper reasoning mode than edit workflows.",
            "gemini": "Prefer Pro for most reviews; use Deep Think only when cross-cutting reasoning is still unresolved.",
            "claude": "Keep the report structured and evidence-backed.",
            "codex": "Bias toward bugs, regressions, and missing tests over summaries.",
        },
        retry_contract=(
            "If findings become essay-like, rerun with `afs.workflow.structured` and the `review-findings` schema.",
            "Prefer smaller cited context before escalating to a deeper model or workflow.",
        ),
        retry_hints={
            "generic": "Retry with a smaller cited context and clearer severity target.",
            "gemini": "Use Pro first; reserve Deep Think for unresolved cross-cutting reasoning after the evidence is narrowed.",
            "claude": "Retry with stricter finding ordering and fewer files in scope.",
            "codex": "Retry with concrete risk categories and missing-test focus.",
        },
    ),
    "root_cause_deep": SessionWorkflowDefinition(
        name="root_cause_deep",
        summary="Multi-step debugging with explicit hypotheses and evidence.",
        intent="Form a small hypothesis set, test it against context, and converge before editing.",
        default_tool_profile="edit_and_verify",
        prompt_contract=(
            "State the leading hypotheses before changing code.",
            "Prefer evidence that narrows the search space over speculative rewrites.",
            "Keep the debugging loop explicit: inspect, test, conclude.",
        ),
        verification_contract=(
            "Record what disproved each failed hypothesis.",
            "Confirm the final fix or stopping point with the fastest relevant check.",
        ),
        model_hints={
            "generic": "Use the deepest reasoning mode available only while the root cause is unresolved.",
            "gemini": "Deep Think or Pro fits this workflow; keep the hypothesis list short so the model does not wander.",
            "claude": "Translate each hypothesis into a concrete file or command check.",
            "codex": "Stay close to stack traces, diffs, and runnable checks.",
        },
        retry_contract=(
            "If hypotheses multiply, retry with a retrieval pack and operator digests for the failing output.",
            "Keep the hypothesis list short; only widen scope after disproving the current top candidate.",
        ),
        retry_hints={
            "generic": "Retry with compressed evidence before adding more hypotheses.",
            "gemini": "Retry on Pro with compressed traces and digests first; use Deep Think only after the evidence is already clean.",
            "claude": "Retry with one hypothesis per check and explicit stop conditions.",
            "codex": "Retry with stack traces, diffs, and one runnable check per hypothesis.",
        },
    ),
}

TOOL_PROFILE_DEFINITIONS = {
    "default": SessionToolProfileDefinition(
        name="default",
        summary="Balanced AFS surface for normal repo work.",
        preferred_surfaces=(
            "session.bootstrap",
            "session.pack",
            "operator.digest",
            "context.status",
            "context.diff",
            "context.query",
            "context.read",
            "context.list",
            "handoff.create",
        ),
        notes=(
            "Escalate to narrower tool profiles when the task is clearly read-only or repair-oriented.",
        ),
    ),
    "context_readonly": SessionToolProfileDefinition(
        name="context_readonly",
        summary="Read-only context discovery and evidence gathering.",
        preferred_surfaces=(
            "session.bootstrap",
            "session.pack",
            "operator.digest",
            "context.status",
            "context.diff",
            "context.query",
            "context.read",
            "context.list",
        ),
        notes=(
            "Prefer this when the task is review, triage, or design analysis.",
        ),
    ),
    "context_repair": SessionToolProfileDefinition(
        name="context_repair",
        summary="AFS context repair and freshness recovery.",
        preferred_surfaces=(
            "session.bootstrap",
            "operator.digest",
            "context.status",
            "context.diff",
            "context.repair",
            "context.index.rebuild",
            "handoff.create",
        ),
        notes=(
            "Use this when the problem is stale mounts, stale indexes, or broken context state.",
        ),
    ),
    "edit_and_verify": SessionToolProfileDefinition(
        name="edit_and_verify",
        summary="Context-guided edit loop with immediate verification.",
        preferred_surfaces=(
            "session.bootstrap",
            "session.pack",
            "operator.digest",
            "context.status",
            "context.diff",
            "context.query",
            "context.read",
            "handoff.create",
        ),
        notes=(
            "Use normal editor/shell tooling for repo edits; keep AFS focused on context, evidence, and handoff.",
        ),
    ),
    "handoff_only": SessionToolProfileDefinition(
        name="handoff_only",
        summary="Minimal surfaces for summarizing status and handing work off cleanly.",
        preferred_surfaces=(
            "session.bootstrap",
            "context.status",
            "handoff.read",
            "handoff.create",
            "hivemind.read",
        ),
        notes=(
            "Prefer this when the main need is coordination rather than implementation.",
        ),
    ),
}


def normalize_workflow(name: str | None) -> str:
    normalized = (name or "general").strip().lower()
    if normalized not in WORKFLOW_DEFINITIONS:
        return "general"
    return normalized


def normalize_tool_profile(name: str | None, *, workflow: str | None = None) -> str:
    normalized = (name or "default").strip().lower()
    if normalized == "default":
        workflow_name = normalize_workflow(workflow)
        return WORKFLOW_DEFINITIONS[workflow_name].default_tool_profile
    if normalized not in TOOL_PROFILE_DEFINITIONS:
        workflow_name = normalize_workflow(workflow)
        return WORKFLOW_DEFINITIONS[workflow_name].default_tool_profile
    return normalized


def get_tool_profile_surfaces(name: str | None) -> frozenset[str]:
    """Return the set of allowed MCP tool names for a tool profile.

    Returns an empty frozenset for unknown profiles (callers should treat
    empty as "no restriction" or "profile not found" depending on context).
    """
    normalized = (name or "").strip().lower()
    definition = TOOL_PROFILE_DEFINITIONS.get(normalized)
    if definition is None:
        return frozenset()
    return frozenset(definition.preferred_surfaces)


def is_tool_in_profile(tool_name: str, profile_name: str) -> bool:
    """Check whether *tool_name* is allowed by *profile_name*.

    Returns True when the profile is unknown (fail-open for unrecognised
    profiles keeps backward compatibility).
    """
    surfaces = get_tool_profile_surfaces(profile_name)
    if not surfaces:
        return True  # unknown profile → no restriction
    return tool_name in surfaces


def build_session_execution_profile(
    *,
    model: str,
    workflow: str | None = None,
    tool_profile: str | None = None,
) -> dict[str, Any]:
    workflow_name = normalize_workflow(workflow)
    tool_profile_name = normalize_tool_profile(tool_profile, workflow=workflow_name)
    workflow_definition = WORKFLOW_DEFINITIONS[workflow_name]
    tool_definition = TOOL_PROFILE_DEFINITIONS[tool_profile_name]
    return {
        "workflow": workflow_name,
        "summary": workflow_definition.summary,
        "intent": workflow_definition.intent,
        "loop_policy": PROMPT_ONLY_LOOP_POLICY,
        "model_hint": workflow_definition.model_hints.get(
            model,
            workflow_definition.model_hints["generic"],
        ),
        "retry_hint": workflow_definition.retry_hints.get(
            model,
            workflow_definition.retry_hints["generic"],
        ),
        "prompt_contract": list(workflow_definition.prompt_contract),
        "verification_contract": list(workflow_definition.verification_contract),
        "retry_contract": list(workflow_definition.retry_contract),
        "tool_profile": {
            "name": tool_profile_name,
            "summary": tool_definition.summary,
            "preferred_surfaces": list(tool_definition.preferred_surfaces),
            "notes": list(tool_definition.notes),
        },
    }
