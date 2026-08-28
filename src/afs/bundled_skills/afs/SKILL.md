---
name: afs
description: Route a workspace request through the smallest useful AFS context, verification, repair, or handoff operation.
triggers:
  - use afs
  - afs overview
  - workspace context
  - afs context layer
profiles:
  - general
requires:
  - afs
enforcement:
  - Prefer a cheap scoped lookup before broad retrieval or repair.
  - Keep model selection and filesystem layout in the host harness or local configuration.
  - Treat scratchpad as the default writable area; durable memory and knowledge updates require deliberate user intent.
---

# AFS

Handle the request with the smallest useful AFS operation.

1. Prefer the slim MCP tools `context.status` and `context.query`, followed by
   `context.read` or `context.list` only when useful.
2. If MCP is unavailable, run `afs session bootstrap --path . --json` once.
3. For command discovery, run `afs next --intent "<goal>" --path . --json` or
   `afs <command> --help`.
4. Use `afs context repair --path . --dry-run --json` before applying repair.
5. Use `afs verify plan --cwd . --json` to choose repository verification when
   a policy exists.
6. Use `afs handoff` only for an explicit cross-session handoff.

Do not infer a central context root, source checkout location, model, or
provider. Do not start background agents, embeddings, or session packs merely
because they are available.
