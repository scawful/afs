---
description: choose the next AFS-guided action for this workspace
---

Pick the next concrete AFS-guided action for this request:

`$ARGUMENTS`

Rules:

- Prefer `~/src/lab/afs/scripts/afs next --intent "$ARGUMENTS" --path . --json`.
- If the intent is empty, infer one short intent from the current user request.
- Report one recommended next step, the reason, and the exact command/file to use.
- Do not start broad background work unless the user explicitly asks.
