---
description: run AFS communication preflight before external writing
---

Prepare for external-facing writing or posting:

`$ARGUMENTS`

Rules:

- Run `~/src/lab/afs/scripts/afs work communication preflight --path . --json`.
- If a purpose is provided, pass it with `--purpose` when supported.
- Report style evidence, pending approvals, `ready_to_post`, and
  `requires_explicit_approval`.
- Draft locally only. Do not send, post, submit, approve, or execute an
  external action without explicit user approval.
