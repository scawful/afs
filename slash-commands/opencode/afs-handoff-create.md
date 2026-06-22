---
description: create a durable AFS handoff note
---

Create a durable handoff for this workspace:

`$ARGUMENTS`

Rules:

- Write a concise markdown handoff under `.context/scratchpad/handoffs/` or use `context.write` into `scratchpad/handoffs/`.
- Include: goal, changed files, verification run, known gaps, and exact next command.
- Prefer this command when work must pause or transfer between hcode/Codex/Claude/Antigravity.
- Do not write durable memory/knowledge unless the user explicitly asks.
