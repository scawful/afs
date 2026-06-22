---
description: inspect AFS tasks and agent jobs
---

Inspect AFS tasks/jobs for this workspace:

`$ARGUMENTS`

Rules:

- Run `~/src/lab/afs/scripts/afs tasks list --path .` for task state.
- Run `~/src/lab/afs/scripts/afs agent-jobs inbox --path .` when background job output may matter.
- Do not create or archive jobs unless the user explicitly asks.
- Report actionable items only, with IDs and exact follow-up commands.
