---
description: inspect and repair stale AFS context deliberately
---

Investigate this AFS freshness or context issue:

`$ARGUMENTS`

First run `afs context repair --path . --dry-run --json`. Apply repair or add
`--rebuild-index` only when the dry run and current task show that stale state
matters. Do not change profiles, mounts, or external storage conventions as a
side effect. Report planned/applied actions and any remaining error exactly.
