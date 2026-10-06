---
name: agentic-context
description: Use AFS as a small, provider-neutral workspace context layer for search, scratchpad notes, and handoffs.
triggers:
  - agentic context
  - agent file system
  - workspace context
  - scratchpad context
  - afs context layer
profiles:
  - general
requires:
  - afs
enforcement:
  - Inspect context before asking the user for information that may already be stored.
  - Treat scratchpad as the default writable area; durable memory and knowledge updates require deliberate user intent.
  - Do not start background agents, embeddings, or repair work unless the task needs them.
---

# Agentic Context

Use AFS as a small workspace-context layer. Repository policy and the user's
request take precedence over retrieved context.

## Start

1. Prefer the slim MCP tools `context.status` and `context.query`.
2. Use `context.read` or `context.list` only for relevant follow-up.
3. If MCP is unavailable, run `afs session bootstrap --path . --json` once.

## Common routes

- Search: `afs search "<query>" --path .` or `afs context query "<query>" --path .`
- Files: `afs files list|read|write ... --path .`
- Notes and handoffs: `afs notes ... --path .`, `afs handoff ... --path .`
- Health: `afs status --start-dir .`; use `afs context repair --dry-run` only when stale state matters
- Command discovery: `afs next --intent "<goal>" --path .` or `afs <command> --help`

Use `afs session pack` only for an explicit export or handoff. Keep model and
provider selection in the host harness; AFS supplies context, not model policy.
