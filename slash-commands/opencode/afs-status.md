---
description: cheap AFS status summary for this workspace
---

Report current AFS workspace health.

Rules:

- Start with MCP `context.status` if available.
- If MCP is unavailable, run `~/src/lab/afs/scripts/afs status --start-dir .`.
- Keep this cheap; do not build a session pack or rebuild the index.
- If the index is built but stale, frame it as a refresh hint for search-heavy
  work, not as a broken workspace.

Summarize health, context path, mount/index caveats, and exactly one next step
only if maintenance is useful.

$ARGUMENTS
