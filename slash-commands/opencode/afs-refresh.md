---
description: repair/rebuild AFS context when freshness matters
---

Refresh AFS context freshness for this workspace.

Rules:

- First run a dry check: `~/src/lab/afs/scripts/afs context repair --path . --dry-run --json`.
- If stale search/index freshness matters for the user's task, run
  `~/src/lab/afs/scripts/afs context repair --path . --rebuild-index --json`.
- Do not refresh just because the command exists; explain why it is needed.
- Report rows/index status and any repair errors.

$ARGUMENTS
