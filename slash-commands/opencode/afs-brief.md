---
description: produce a concise AFS workspace brief
---

Create a short workspace brief for this hcode session.

Rules:

- Start with MCP `context.status` and `context.query` when available.
- If MCP is unavailable, use `~/src/lab/afs/scripts/afs session bootstrap --json` once.
- Summarize health, recent scratchpad/deferred notes, active tasks/jobs, and one next action.
- Keep it under ten bullets unless the user asks for a longer handoff.

$ARGUMENTS
