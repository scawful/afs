"""Model and client profile hints for modern agent harnesses.

These profiles are descriptive metadata. They let AFS prepare better packs and
client payloads without requiring core AFS to call every provider API directly.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModelClientProfile:
    name: str
    family: str
    aliases: tuple[str, ...] = ()
    context_window_hint: str = ""
    reasoning_effort: str = ""
    cache_strategy: str = ""
    tool_surface_strategy: str = ""
    structured_output_preference: str = ""
    prompt_update_strategy: str = ""
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "family": self.family,
            "aliases": list(self.aliases),
            "context_window_hint": self.context_window_hint,
            "reasoning_effort": self.reasoning_effort,
            "cache_strategy": self.cache_strategy,
            "tool_surface_strategy": self.tool_surface_strategy,
            "structured_output_preference": self.structured_output_preference,
            "prompt_update_strategy": self.prompt_update_strategy,
            "notes": list(self.notes),
        }


_MODEL_PROFILES = (
    ModelClientProfile(
        name="codex:gpt-5.5",
        family="codex",
        aliases=("codex", "gpt-5.5", "codex-5.5"),
        context_window_hint="frontier long-context coding model; keep stable prefix and task delta separate",
        reasoning_effort="medium default; raise only for ambiguous root-cause or design work",
        cache_strategy="stable prefix with static AFS contract first, dynamic workspace/task context last",
        tool_surface_strategy="small default MCP catalog plus explicit CLI/tool-search routes",
        structured_output_preference="prefer schema-bound outputs for plans, reviews, and verification summaries",
        prompt_update_strategy="outcome-first task suffix with concrete files and verification target",
        notes=("Use Responses-style tool semantics when the host supports them.",),
    ),
    ModelClientProfile(
        name="claude:opus-4.8",
        family="claude",
        aliases=("claude", "opus-4.8", "claude-opus-4-8"),
        context_window_hint="very large context; still prefer source-ranked slices over dumping everything",
        reasoning_effort="adaptive thinking only; avoid manual budget assumptions",
        cache_strategy="keep at least a stable 1024-token prefix for prompt caching",
        tool_surface_strategy="explicit tool descriptions and short plans before edits",
        structured_output_preference="findings-first reviews and concise execution checklists",
        prompt_update_strategy="mid-conversation system updates can carry changing AFS session hints",
        notes=("Avoid non-default temperature/top_p/top_k where unsupported.",),
    ),
    ModelClientProfile(
        name="gemini:3.5-flash",
        family="gemini",
        aliases=("gemini", "gemini-3.5-flash", "Gemini 3.5 Flash"),
        context_window_hint="stable fast model for sustained agentic/coding work",
        reasoning_effort="use Low/Medium/High labels in Antigravity CLI when available",
        cache_strategy="prefer retrieval/focused packs; keep thought/signature handling extension-owned",
        tool_surface_strategy="route interactive terminal work through Antigravity CLI; keep API provider separate",
        structured_output_preference="JSON summaries for sync/status operations",
        prompt_update_strategy="query-first context, then escalate to broader pack only when evidence is thin",
        notes=("Good default for hcode/Antigravity fast scans and bounded edits.",),
    ),
    ModelClientProfile(
        name="gemini:3.1-pro",
        family="gemini",
        aliases=("gemini-3.1-pro", "Gemini 3.1 Pro"),
        context_window_hint="preview advanced reasoning model; use for multi-file synthesis and hard debugging",
        reasoning_effort="prefer High only for unresolved root-cause or design analysis",
        cache_strategy="stable AFS prefix with compact task/retrieval suffix",
        tool_surface_strategy="Antigravity/Interactions-capable profile with explicit permission boundaries",
        structured_output_preference="schema-bound design briefs, triage, and review results",
        prompt_update_strategy="preserve the retrieval trail so preview behavior is auditable",
        notes=("Treat preview availability and behavior as more drift-prone than stable Flash.",),
    ),
    ModelClientProfile(
        name="hcode:opencode",
        family="hcode",
        aliases=("hcode", "opencode"),
        context_window_hint="host-model dependent; AFS should provide slim, path-oriented context",
        reasoning_effort="delegate reasoning level to selected hcode provider/model",
        cache_strategy="plugin/slash-command guidance stays stable; session payload carries dynamic state",
        tool_surface_strategy="slim AFS MCP by default, slash commands for heavier flows",
        structured_output_preference="prefer command-oriented markdown and JSON payload artifacts",
        prompt_update_strategy="inject AFS context path and payload hints via hcode plugin/commands",
        notes=("Keep hcode integration provider-neutral so forks can choose their own model backend.",),
    ),
    ModelClientProfile(
        name="antigravity:agy",
        family="gemini",
        aliases=("antigravity", "agy", "jetski"),
        context_window_hint="terminal agent harness with model selection via `agy models`",
        reasoning_effort="map to Antigravity model labels rather than hardcoded Gemini CLI flags",
        cache_strategy="stable settings/MCP contract plus dynamic session artifacts",
        tool_surface_strategy="permission-safe default; do not add dangerous skip-permission flags automatically",
        structured_output_preference="JSON for status/setup; markdown for user-facing command guidance",
        prompt_update_strategy="use AFS MCP/settings and session payloads instead of Gemini CLI-specific env-only setup",
        notes=("Public binary is `agy`; keep Jetski as an alias only for local/user shorthand.",),
    ),
)


ALIASES: dict[str, ModelClientProfile] = {}
for profile in _MODEL_PROFILES:
    ALIASES[profile.name.lower()] = profile
    for alias in profile.aliases:
        ALIASES[alias.lower()] = profile


def resolve_model_client_profile(name: str | None) -> ModelClientProfile:
    key = (name or "generic").strip().lower()
    profile = ALIASES.get(key)
    if profile is not None:
        return profile
    return ModelClientProfile(
        name=key or "generic",
        family="generic",
        cache_strategy="stable AFS contract first, dynamic context last",
        tool_surface_strategy="small default MCP catalog plus explicit CLI routes",
    )


def profile_for_client_model(client: str, model: str) -> ModelClientProfile:
    for candidate in (model, client):
        profile = resolve_model_client_profile(candidate)
        if profile.family != "generic" or profile.name in ALIASES:
            return profile
    return resolve_model_client_profile("generic")
