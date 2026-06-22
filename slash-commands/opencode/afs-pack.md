---
description: build an explicit AFS session pack for handoff or export
---

Build an explicit AFS session pack only because the user asked for one or a
handoff/export artifact is actually needed.

Rules:

- Prefer `~/src/lab/afs/scripts/afs session pack --path . --json` or the MCP
  prompt `afs.session.pack` if the client exposes prompts.
- Run it once; do not retry in a loop.
- Mention cache/artifact reuse if the command reports it.
- Report pack path, freshness caveats, and the most relevant sections.

$ARGUMENTS
