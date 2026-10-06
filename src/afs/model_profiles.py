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
        name="claude:context-efficient",
        family="claude",
        aliases=(
            "claude",
            "claude:opus-thinking",
            "opus-thinking",
            "claude-opus-thinking",
            "sonnet-5",
            "claude-sonnet-5",
            "opus-5",
            "claude-opus-5",
            "haiku-4.5",
            "claude-haiku-4-5",
            "opus-4.6",
            "claude-opus-4-6",
            "Claude Opus 4.6",
            "Claude Opus 4.6 (Thinking)",
            # Keep the user's older shorthand resolving, but prefer runtime
            # model discovery via `agy models` over this compatibility alias.
            "opus-4.8",
            "claude-opus-4-8",
        ),
        context_window_hint="long context still degrades; load one bounded session summary and retrieve details on demand",
        reasoning_effort="use provider-advertised effort controls; raise effort only for unresolved design or root-cause work",
        cache_strategy="keep tool definitions and system guidance byte-identical at the front; append changing AFS state last",
        tool_surface_strategy="prefer a small default tool catalog and tool search over sending every definition up front",
        structured_output_preference="findings-first reviews and concise execution checklists",
        prompt_update_strategy="bootstrap once, then use focused context.query/context.read calls instead of reinjecting full state",
        notes=(
            "Prompt caching reduces repeated input cost and latency; it does not reduce context-window use.",
            "Use native compaction for long sessions and create an AFS handoff only when work must cross sessions.",
        ),
    ),
    ModelClientProfile(
        name="gemini:flash",
        family="gemini",
        aliases=(
            "gemini",
            "gemini-flash",
            "gemini-3.8-flash",
            "gemini-3.7-flash",
            "gemini-3.6-flash",
            "gemini-3.5-flash",
            "Gemini 3.8 Flash",
            "Gemini 3.7 Flash",
            "Gemini 3.6 Flash",
            "Gemini 3.5 Flash",
        ),
        context_window_hint="stable Flash model for sustained agentic work and bounded coding subtasks",
        reasoning_effort="use model-advertised Minimal/Low/Medium/High labels when available",
        cache_strategy="prefer retrieval-focused packs; preserve provider-managed tool-turn state",
        tool_surface_strategy="route interactive terminal work through Antigravity CLI; keep API provider separate",
        structured_output_preference="JSON summaries for sync/status operations",
        prompt_update_strategy="query-first context, then escalate to broader pack only when evidence is thin",
        notes=("Discover Antigravity model labels with `agy models`; do not pin a project model.",),
    ),
    ModelClientProfile(
        name="gemini:pro",
        family="gemini",
        aliases=("gemini-pro", "gemini-3.1-pro", "gemini-3.1-pro-preview", "Gemini 3.1 Pro"),
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
        notes=(
            "Keep hcode integration provider-neutral so forks can choose their own model backend.",
        ),
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
