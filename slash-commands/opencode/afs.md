---
description: use AFS for workspace context and the next useful action
---

Handle this request with the smallest useful AFS operation:

`$ARGUMENTS`

Rules:

1. Prefer the available AFS MCP equivalents of `context.status` and
   `context.query`; follow with `context.read` or `context.list` only when the
   result points to a relevant file.
2. If MCP is unavailable, use the `afs` executable on `PATH`. Start with
   `afs session bootstrap --path . --json` or
   `afs next --intent "$ARGUMENTS" --path . --json`.
3. Treat scratchpad as the default writable area. Update memory or knowledge
   only when the user deliberately asks for durable state.
4. Do not start background agents, embeddings, repair, or session-pack work
   merely because those features exist.
5. Keep the result concise and include the exact context path or command that
   matters next.

Related commands: `/afs/status`, `/afs/verify`, `/afs/handoff`, `/afs/repair`.
