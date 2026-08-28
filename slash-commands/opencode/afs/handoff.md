---
description: inspect or create a durable AFS handoff
---

Handle this handoff request:

`$ARGUMENTS`

For inspection, use the AFS handoff/list/read surface. For creation, prefer
`afs handoff create --path . --title "<title>" --accomplished "<result>" --next "<next step>" --json`.
Include changed files, verification, blockers, and one exact next action. Keep
handoffs in scratchpad; do not promote them to durable memory automatically.
