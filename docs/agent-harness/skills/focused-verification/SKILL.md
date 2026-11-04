---
name: focused-verification
description: Choose and run the fastest relevant verification for a change or diagnosis. Use when code, config, docs, devices, services, MCP servers, local models, or workflows need validation before reporting done.
---

# Focused Verification

## Goal
Prove the task-relevant behavior with the cheapest trustworthy check.

## Workflow
1. Identify the risk surface: syntax, unit behavior, integration path, UI, device, service, model endpoint, data artifact, or docs-only.
2. Prefer repo-supported commands from `README.md`, `AGENTS.md`, package manifests, scripts, or prior handoffs.
3. Run the narrowest check that covers the changed behavior and actual risk. Preserve mandatory repo checks. Inspect test side effects before running under no-window, no-network, or other task constraints.
4. If a check fails, distinguish a regression from a known baseline or environment failure. Isolate with a smaller check when useful; don't repair unrelated code.
5. For live systems, verify the live endpoint/process/device state, not just local files.
6. Report exact commands run and outcome. If checks were skipped, state why and what risk remains.

## Command Selection
- Syntax/config: parser, linter, `bash -n`, `python -m py_compile`, `jq`, `tomllib`.
- Unit change: nearest focused test file or `-k` slice.
- Frontend/UI: build/typecheck plus screenshot or browser/device check when appearance matters.
- Device deploy: build, install, launch, and report concrete target name.
- Service/MCP/model: health endpoint, tool list, loaded model list, or a minimal request.
- Docs/skills: schema/frontmatter parser plus targeted grep for stale terms.

## Stop Conditions
- Broaden checks for dependency spread, shared invariants, security, data/format contracts, a new failure, or repo requirements. Don't rerun an unchanged full suite by default.
- Reuse evidence only when its code, inputs, and relevant environment remain unchanged. Stop when scoped acceptance has trustworthy evidence.
- Avoid tests that merely mirror implementation or reversible trivial edits. Preserve meaningful regression coverage and required repo checks.
- Report known baseline failures and unrun checks honestly. Don't claim full coverage from a smoke test.

Read [adaptation provenance](../../ENGINEERING_PRACTICE_SOURCES.md) only when reviewing the source decisions.
