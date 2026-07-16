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
afs approvals list                                    # pending agent/action pairs
afs approvals history                                 # including resolved, with rationales
afs approvals approve <agent> <action> --because "…"  # USER-DIRECTED ONLY
afs approvals reject <agent> <action> --because "…"   # USER-DIRECTED ONLY
afs approvals clear                                   # archive completed; compact active
```

`--because` is required on approve/reject (empty rationales are refused), and
both prompt for an interactive human confirmation — headless approval fails
closed. The rationale is stored in history and resurfaced by
`afs calibration review`. `--approvals-file <path>` overrides this global
supervisor-approval store.
Clearing preserves completed decision records in append-only history. The
terminal broker is a cooperative same-user guard, not a hostile-process
sandbox; public programmatic decisions are labeled unauthenticated and never
authorize guarded execution.

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
