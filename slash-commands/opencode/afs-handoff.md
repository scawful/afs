---
description: create or inspect an AFS scratchpad handoff
---

Handle AFS handoff state for this workspace:

`$ARGUMENTS`

Rules:

- If the user asks to inspect, list/read relevant files under
  `scratchpad/handoffs` with `context.list` and `context.read`.
- If the user asks to create or update a handoff, write a concise markdown file
  under `scratchpad/handoffs/` with current state, changed files, verification,
  gaps, and next step.
- Prefer scratchpad handoff files over ad hoc chat summaries.
- Do not build a full session pack unless the user explicitly asks.
