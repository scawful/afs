---
description: choose and run focused verification for the current change
---

Verify this request or change:

`$ARGUMENTS`

Inspect the changed files first. Prefer the repository's narrowest relevant
check. Use `afs verify plan --cwd . --json` when a repository verification
policy or prepared AFS session payload exists. Run one bounded check at a time
and report the command, result, and residual risk.
