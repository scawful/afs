"""Dynamic system prompt construction for AFS model families.

Composes system prompts from static base sections (cacheable) and
dynamic session-specific sections, mirroring Claude Code's split
architecture.  The static prefix stays stable across calls for
prompt-cache efficiency; the dynamic suffix incorporates live
session state, workflow hints, and memory manifests.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class PromptSection:
    """One block of a system prompt."""

    content: str
    cacheable: bool = True
    priority: int = 0  # higher = more important (survives truncation)
    label: str = ""


def build_model_system_prompt(
    *,
    base_prompt: str,
    model_family: str = "generic",
    role: str = "",
    context_path: Path | None = None,
    session_state: dict[str, Any] | None = None,
    pack_state: dict[str, Any] | None = None,
    skills_state: dict[str, Any] | None = None,
    workflow: str | None = None,
    tool_profile: str | None = None,
    token_budget: int = 0,
) -> str:
    """Compose a system prompt from static base + dynamic session context.

    Args:
        base_prompt: The model's base system prompt (from registry or Modelfile).
        model_family: One of "oracle", "avatar", "persona", "cloud", "generic".
        role: Specific role within the family (e.g., "echo", "din", "scribe").
        context_path: AFS .context root for session bootstrap injection.
        session_state: Pre-built bootstrap summary (if already available).
        pack_state: Prepared AFS session-pack metadata.
        skills_state: Prepared AFS skill-match metadata.
        workflow: AFS workflow name for model hints.
        tool_profile: AFS tool profile for capability hints.
        token_budget: If > 0, truncate dynamic sections to fit.

    Returns:
        Assembled system prompt string.
    """
    sections: list[PromptSection] = []

    # --- Static section: base prompt (highest priority, cacheable) ---
    if base_prompt.strip():
        sections.append(PromptSection(
            content=base_prompt.strip(),
            cacheable=True,
            priority=100,
            label="base",
        ))

    # --- Static section: family-specific constraints ---
    family_constraints = _family_constraints(model_family, role)
    if family_constraints:
        sections.append(PromptSection(
            content=family_constraints,
            cacheable=True,
            priority=90,
            label="family_constraints",
        ))

    # --- Dynamic section: workflow and tool profile hints ---
    workflow_hints = _workflow_hints(workflow, tool_profile, model_family)
    if workflow_hints:
        sections.append(PromptSection(
            content=workflow_hints,
            cacheable=False,
            priority=70,
            label="workflow_hints",
        ))

    # --- Dynamic section: session-pack context (query/task/pack mode) ---
    pack_block = _pack_context_block(pack_state)
    if pack_block:
        sections.append(PromptSection(
            content=pack_block,
            cacheable=False,
            priority=60,
            label="pack_context",
        ))

    # --- Dynamic section: skill matches for the active task/session ---
    skills_block = _skills_context_block(skills_state)
    if skills_block:
        sections.append(PromptSection(
            content=skills_block,
            cacheable=False,
            priority=55,
            label="skills_context",
        ))

    # --- Dynamic section: session context (scratchpad, diff, memory) ---
    context_block = _session_context_block(context_path, session_state)
    if context_block:
        sections.append(PromptSection(
            content=context_block,
            cacheable=False,
            priority=50,
            label="session_context",
        ))

    # --- Assemble: static first (cacheable prefix), then dynamic ---
    static_parts = sorted(
        [s for s in sections if s.cacheable],
        key=lambda s: -s.priority,
    )
    dynamic_parts = sorted(
        [s for s in sections if not s.cacheable],
        key=lambda s: -s.priority,
    )

    parts = static_parts + dynamic_parts

    # Apply token budget by dropping lowest-priority dynamic sections
    if token_budget > 0:
        parts = _apply_budget(parts, token_budget)

    return "\n\n".join(s.content for s in parts if s.content.strip())


def _family_constraints(model_family: str, role: str) -> str:
    """Return family-specific behavioral constraints."""
    family = model_family.lower()

    if family == "oracle":
        return (
            "## Oracle Constraints\n"
            "You are an Oracle-family model specializing in 65816 ASM, "
            "SNES ROM hacking, and Zelda game engine analysis.\n"
            "- Use exact opcode mnemonics (LDA, STA, JSL, etc.)\n"
            "- Reference ROM addresses in $XX:XXXX bank:offset format\n"
            "- Prefer disassembly labels when available\n"
            "- Validate memory access patterns against SNES memory map"
        )

    if family == "avatar":
        constraints = (
            "## Avatar Constraints\n"
            "You are an Avatar-family model trained on personal voice and style."
        )
        if role.lower() == "echo":
            constraints += (
                "\n- Write in lowercase stream-of-consciousness style\n"
                "- Match the user's cadence and energy\n"
                "- Brevity over explanation"
            )
        elif role.lower() == "memory":
            constraints += (
                "\n- Factual recall mode — be precise and cite sources\n"
                "- Temperature 0.1 behavior expected"
            )
        elif role.lower() == "muse":
            constraints += (
                "\n- Creative divergence mode\n"
                "- Explore tangents and unexpected connections"
            )
        return constraints

    if family == "persona":
        return (
            "## Persona Constraints\n"
            "You are a specialized Persona-family model.\n"
            "- Stay within your defined role boundaries\n"
            "- Refer to capabilities you don't have rather than hallucinating"
        )

    return ""


def _workflow_hints(
    workflow: str | None,
    tool_profile: str | None,
    model_family: str,
) -> str:
    """Extract workflow and tool profile hints from session_workflows.py."""
    if not workflow:
        return ""

    try:
        from .session_workflows import build_session_execution_profile
    except ImportError:
        return ""

    try:
        profile = build_session_execution_profile(
            model=model_family,
            workflow=workflow,
            tool_profile=tool_profile,
        )
    except Exception:
        return ""

    lines = ["## Execution Profile"]
    hint = profile.get("model_hint", "")
    if hint:
        lines.append(f"Model hint: {hint}")

    contracts = profile.get("prompt_contract", [])
    if contracts:
        lines.append("Prompt contract:")
        for item in contracts:
            lines.append(f"- {item}")

    tool_info = profile.get("tool_profile", {})
    if tool_info.get("summary"):
        lines.append(f"Tool profile: {tool_info['summary']}")
    notes = tool_info.get("notes", [])
    for note in notes:
        lines.append(f"- {note}")

    return "\n".join(lines)


def _session_context_block(
    context_path: Path | None,
    session_state: dict[str, Any] | None,
) -> str:
    """Build a compact session context block from AFS bootstrap state."""
    if session_state is None and context_path is None:
        return ""

    # Use provided state or build fresh
    state = session_state
    if state is None and context_path is not None:
        try:
            from .config import load_config_model
            from .manager import AFSManager
            from .session_bootstrap import build_session_bootstrap

            config = load_config_model()
            manager = AFSManager(config=config)
            state = build_session_bootstrap(
                manager,
                context_path,
                token_budget=2000,
                record_event=False,
            )
        except Exception:
            return ""

    if not state:
        return ""

    lines = ["## Session Context"]

    # Project and profile
    project = state.get("project", "")
    profile = state.get("profile", "")
    if project:
        lines.append(f"Project: {project} (profile: {profile})")

    # Scratchpad state (most important dynamic context)
    scratchpad = state.get("scratchpad", {})
    scratchpad_text = scratchpad.get("state_text", "")
    if scratchpad_text:
        lines.append(f"Scratchpad state: {scratchpad_text[:500]}")

    deferred = scratchpad.get("deferred_text", "")
    if deferred:
        lines.append(f"Deferred: {deferred[:300]}")

    # Recent drift summary
    diff = state.get("diff", {})
    if diff.get("available") and diff.get("total_changes", 0) > 0:
        lines.append(f"Recent changes: {diff['total_changes']} files changed")

    # Memory manifest topics (from C2 implementation)
    memory = state.get("memory", {})
    manifest = memory.get("memory_manifest", [])
    if manifest:
        topic_names = [t["topic"] for t in manifest[:8]]
        lines.append(f"Memory topics: {', '.join(topic_names)}")

    # Tasks summary
    tasks = state.get("tasks", {})
    if tasks.get("total", 0) > 0:
        lines.append(f"Tasks: {tasks['total']} ({', '.join(f'{k}={v}' for k, v in sorted(tasks.get('counts', {}).items()))})")

    # Handoff from last session
    handoff = state.get("handoff", {})
    if handoff.get("available"):
        next_steps = handoff.get("next_steps", [])
        if next_steps:
            lines.append("Last session next steps:")
            for step in next_steps[:3]:
                lines.append(f"- {step}")

    return "\n".join(lines)


def _pack_context_block(pack_state: dict[str, Any] | None) -> str:
    if not isinstance(pack_state, dict) or not pack_state.get("available"):
        return ""

    lines = ["## Session Pack"]
    query = str(pack_state.get("query", "") or "").strip()
    task = str(pack_state.get("task", "") or "").strip()
    model = str(pack_state.get("model", "") or "").strip()
    workflow = str(pack_state.get("workflow", "") or "").strip()
    tool_profile = str(pack_state.get("tool_profile", "") or "").strip()
    pack_mode = str(pack_state.get("pack_mode", "") or "").strip()
    estimated_tokens = pack_state.get("estimated_tokens")

    if query:
        lines.append(f"Query focus: {query[:400]}")
    if task:
        lines.append(f"Task focus: {task[:400]}")
    if model or workflow or tool_profile or pack_mode:
        summary_bits = []
        if model:
            summary_bits.append(f"model={model}")
        if workflow:
            summary_bits.append(f"workflow={workflow}")
        if tool_profile:
            summary_bits.append(f"tool_profile={tool_profile}")
        if pack_mode:
            summary_bits.append(f"pack_mode={pack_mode}")
        lines.append(f"Pack settings: {', '.join(summary_bits)}")
    if isinstance(estimated_tokens, int) and estimated_tokens > 0:
        lines.append(f"Pack tokens: {estimated_tokens}")

    return "\n".join(lines) if len(lines) > 1 else ""


def _skills_context_block(skills_state: dict[str, Any] | None) -> str:
    if not isinstance(skills_state, dict) or not skills_state.get("available"):
        return ""

    matches = skills_state.get("matches", [])
    if not isinstance(matches, list) or not matches:
        return ""

    lines = ["## Relevant Skills"]
    for match in matches[:5]:
        if not isinstance(match, dict):
            continue
        name = str(match.get("name", "") or "").strip()
        if not name:
            continue
        score = match.get("score")
        triggers = match.get("triggers", [])
        line = name
        if isinstance(score, int):
            line += f" (score={score})"
        if isinstance(triggers, list) and triggers:
            trigger_values = [
                str(trigger).strip()
                for trigger in triggers
                if isinstance(trigger, str) and str(trigger).strip()
            ]
            if trigger_values:
                line += f" triggers={', '.join(trigger_values[:4])}"
        lines.append(f"- {line}")

    return "\n".join(lines) if len(lines) > 1 else ""


def _apply_budget(sections: list[PromptSection], budget: int) -> list[PromptSection]:
    """Drop lowest-priority sections until total fits within token budget."""
    def _estimate(text: str) -> int:
        return max(1, (len(text) + 3) // 4)

    total = sum(_estimate(s.content) for s in sections)
    if total <= budget:
        return sections

    # Sort by priority ascending to drop lowest first
    by_priority = sorted(enumerate(sections), key=lambda t: t[1].priority)
    dropped: set[int] = set()

    for idx, section in by_priority:
        if total <= budget:
            break
        if section.cacheable:
            continue  # never drop cacheable/static sections
        total -= _estimate(section.content)
        dropped.add(idx)

    return [s for i, s in enumerate(sections) if i not in dropped]
