---
description: show the AFS command menu for this workspace
---

Explain the available AFS slash commands and when to use each one.

Default posture:

- Use the slim MCP surface first: `context.status`, `context.query`,
  `context.read`, `context.write`, and `context.list`.
- Use CLI/framework commands for heavier flows: work preflight, approvals,
  verification, refresh/repair, handoffs, and session packs.
- Do not start background agents, embeddings, training, or domain MCP servers
  unless the user explicitly asks.

Commands to summarize:

- `/afs-status` cheap workspace health
- `/afs-next <intent>` one recommended AFS-guided next action
- `/afs-brief` concise workspace brief
- `/afs-query <question>` context/memory/scratchpad search
- `/afs-files <path or intent>` read/list/write context files
- `/afs-tasks` inspect AFS tasks and background job inbox
- `/afs-review-context <topic>` review relevant context before edits/review
- `/afs-work-preflight <purpose>` writing style and external-action approval guardrail
- `/afs-verify <change>` fastest relevant verification
- `/afs-handoff <summary>` create or inspect a scratchpad handoff
- `/afs-handoff-create <summary>` durable handoff note for another harness
- `/afs-refresh` repair/rebuild context when freshness matters
- `/afs-pack <goal>` explicit session pack/export
- `/afs-update-work` preview or apply the AFS local harness update path

Keep the answer short and command-oriented.

$ARGUMENTS
