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

Agents may create and inspect requests; only the user resolves them.

## Supervisor Approvals

```bash
afs approvals list [--json]                 # pending; expired ones are reaped
afs approvals show --request-id <gate_id>   # one request with its payload
afs approvals history                       # every state, with rationales
afs approvals approve --request-id <gate_id> --because "…"  # USER-DIRECTED ONLY
afs approvals reject --request-id <gate_id> --because "…"   # USER-DIRECTED ONLY
afs approvals reject --stale --because "…"  # USER-DIRECTED ONLY
afs approvals revoke --request-id <gate_id> --because "…"   # USER-DIRECTED ONLY
afs approvals clear                         # prune old system-closed rows
```

`--because` is required, and approve/reject need an interactive human
confirmation, so headless approval fails closed. MCP `approvals.list` is
read-only. Requests expire after 72 hours, a newer request from the same
source supersedes the older one, and a requester may withdraw its own;
`clear` never removes approved or rejected rows. Programmatic decisions are
unauthenticated and never authorize guarded execution.

## Context-Local Work Approvals

```bash
afs work approvals list --path <ws>
afs work approvals show <approval_id> --path <ws>
afs work approvals approve <approval_id> --path <ws> --because "…"  # USER-DIRECTED
afs work approvals reject <approval_id> --path <ws> --because "…"   # USER-DIRECTED
```

Same rationale and confirmation rules. Their pending count,
not the global supervisor store, surfaces in `afs session bootstrap`.

## Optimization Gate

`afs optimize decide --baseline … --candidate … --policy …` only recommends
one candidate step; treat its output as input to a human decision.

## Related Gates

`afs skills promote` and `afs agent-jobs promote` are promote-style gates:
run them only when the user asks. When blocked on a gate, say so and stop.
