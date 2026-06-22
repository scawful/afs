---
description: inspect or update AFS context files
---

Use the AFS file surface for this request:

`$ARGUMENTS`

Rules:

- Prefer MCP `context.list`, `context.read`, and `context.write`.
- Treat `scratchpad` as the default writable working area.
- Treat `memory` and `knowledge` as deliberate durable updates; do not write
  them unless the user explicitly asks for durable memory/knowledge.
- Do not delete or move files unless the user explicitly requested it.
- Keep writes small and report the exact path changed.
