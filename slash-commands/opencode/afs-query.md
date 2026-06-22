---
description: query AFS context, memory, scratchpad, and knowledge
---

Answer this workspace-context question:

`$ARGUMENTS`

Rules:

- Start with MCP `context.query` if available.
- If MCP is unavailable, run `~/src/lab/afs/scripts/afs context query "$ARGUMENTS" --path .`.
- Use `context.read` or `context.list` only for directly relevant follow-up.
- Do not call session pack unless the user explicitly asks for a pack/export.
- Include source paths or mount names when they materially help.
