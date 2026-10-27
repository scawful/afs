---
name: approvals-and-gates
triggers:
  - approval
  - approve
  - pending request
  - decision gate
  - optimization gate
  - optimize decide
profiles:
  - general
requires:
  - afs
enforcement:
  - Never approve, reject, or clear an approval request without explicit user direction.
  - Never promote an optimization candidate; `afs optimize decide` only recommends.
  - Surface pending approvals to the user; do not resolve them to unblock yourself.
---

# Approvals & Decision Gates

Human-gated flows: agents may create and inspect requests, only the user
resolves them.

## Supervisor Approvals

```bash
afs approvals list [--json]                           # pending requests (expired ones are reaped)
afs approvals show --request-id <gate_id>             # one request with its payload
afs approvals history                                 # every state, with rationales
afs approvals approve --request-id <gate_id> --because "…"  # USER-DIRECTED ONLY
afs approvals reject --request-id <gate_id> --because "…"   # USER-DIRECTED ONLY
afs approvals reject --stale --because "…"            # USER-DIRECTED ONLY; one confirmation
afs approvals revoke --request-id <gate_id> --because "…"   # USER-DIRECTED ONLY
afs approvals clear                                   # prune old system-closed rows
```

Agents read the same store through the MCP `approvals.list` tool; it never
decides anything.

`--because` is required on approve/reject/revoke (empty rationales are
refused), and approve/reject prompt for an interactive human confirmation —
headless approval fails closed. The rationale is resurfaced by
`afs calibration review`. `--approvals-file <path>` overrides this global
supervisor-approval store (SQLite).

Requests expire after 72 hours, a newer request from the same source
supersedes the older one, and a requester can withdraw its own request. These
system states are not human judgments. `clear` never removes approved or
rejected rows. The terminal broker is a cooperative same-user guard, not a
hostile-process sandbox; public programmatic decisions are labeled
unauthenticated and never authorize guarded execution.

## Context-Local Work Approvals

```bash
afs work approvals list --path <ws>
afs work approvals show <approval_id> --path <ws>
afs work approvals approve <approval_id> --path <ws> --because "…"  # USER-DIRECTED
afs work approvals reject <approval_id> --path <ws> --because "…"   # USER-DIRECTED
```

These ID-based requests gate work-assistant external writes; the same
rationale and interactive-confirmation rules apply. Their pending count,
not the global supervisor store, surfaces in `afs session bootstrap`.

## Optimization Gate

```bash
afs optimize decide --baseline base.json --candidate cand.json --policy policy.json
```

Returns a deterministic review recommendation for one candidate step. It never
executes candidates; treat its output as input to a human decision, not as
permission to proceed.

## Related Gates

- `afs skills promote` (skill candidates) and `afs agent-jobs promote`
  (job review packets) are also promote-style gates: run them only when the
  user asks
- When blocked on a gate, say so plainly and stop; do not route around it
