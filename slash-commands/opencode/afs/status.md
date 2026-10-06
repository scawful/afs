---
description: report cheap AFS workspace health
---

Report AFS health for the current workspace.

Prefer the AFS MCP equivalent of `context.status`. If MCP is unavailable, run
`afs status --start-dir .`. Do not build a session pack or rebuild an index.

Summarize the resolved context path, mount/index caveats, and one maintenance
action only when maintenance is actually useful.

`$ARGUMENTS`
