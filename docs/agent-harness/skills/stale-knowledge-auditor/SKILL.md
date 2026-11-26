---
name: stale-knowledge-auditor
description: Audit docs, skills, configs, model aliases, MCP names, scripts, and handoffs for stale claims or outdated workflow references. Use when modernizing instructions, checking accuracy, or cleaning agent knowledge.
---

# Stale Knowledge Auditor

## Goal
Find and fix outdated operational knowledge before agents act on it.

## Workflow
1. Define the truth source: current code, official docs, live command output, current config, or user correction.
2. Search for stale terms and duplicates across the relevant instruction surfaces.
3. Classify each finding:
   - broken path or command
   - deprecated tool or server (`hafs`, `fs.*`, legacy full-catalog assumptions)
   - stale model ID or endpoint
   - old behavior contradicted by current code
   - vague guidance that should become a concrete workflow
4. Patch only the surfaces in scope. Do not rewrite unrelated docs.
5. Verify with parsers, focused grep, or the smallest live check.

## Common Searches
- MCP/tools: `mcp`, server names, deprecated `fs.*` aliases.
- Models: explicit IDs, local aliases, ports, host names.
- Harnesses: `AGENTS.md`, `CLAUDE.md`, `GEMINI.md`, `SKILL.md`, config files.
- Runtime workflows: launch scripts, deploy scripts, socket paths, service names.

## Reporting
Lead with stale items fixed, then remaining stale items intentionally left alone.
