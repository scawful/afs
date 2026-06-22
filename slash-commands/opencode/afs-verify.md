---
description: choose and run the fastest relevant verification
---

Verify the current change or request:

`$ARGUMENTS`

Rules:

- Prefer the narrowest relevant check for touched files.
- If an AFS session payload exists, inspect the verify plan with
  `~/src/lab/afs/scripts/afs verify plan --payload-file .context/scratchpad/afs_agents/session_client_hcode.json --json`
  before falling back to repo-native tests.
- Run one verification command at a time.
- If a check cannot run, report the exact blocker and residual risk.
