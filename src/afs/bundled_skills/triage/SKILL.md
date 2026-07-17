---
name: triage
triggers:
  - triage
  - severity
  - incoming bug
  - failure report
  - ci failure
  - flaky test
  - dedupe findings
  - classify defect
profiles:
  - general
requires:
  - afs
enforcement:
  - Reproduce or read the primary failure log before classifying; never triage from a summary alone.
  - A canceled CI job is not a failure — find the matrix sibling that actually failed first.
  - Every dismissed or deferred finding gets a recorded reason; silence is not a disposition.
verification:
  - Confirm each triaged item has a severity, an owner or route, and a recorded disposition.
---

# Triage

Turn incoming failures, bug reports, and review findings into classified,
routed, recorded work — without guessing.

## Order of Operations

1. **Get primary evidence.** Read the actual failing log/stack, not the
   rollup. For CI: `gh run view <id> --log-failed`; matrix fail-fast marks
   siblings "canceled" — only the first real failure matters.
2. **Reproduce or bound.** Reproduce cheaply if possible; otherwise state
   exactly what was observed and what could not be checked.
3. **Dedupe.** Search first: `afs search`/`context.query` for prior
   findings, `gh issue list --search`, existing missions. Link duplicates
   instead of filing twins.
4. **Classify severity by impact:**
   - `blocker` — data loss, security, broken main/CI, blocks the team
   - `major` — wrong behavior with real consequence; workaround exists
   - `minor` — edge case, debt, papercut
   - `info` — observation, no action required
5. **Route.** Assign an owner, file a ticket with repro steps, or fix
   on the spot only if smaller than filing it.
6. **Record the disposition** where the team will see it (ticket,
   mission note, handoff): fixed / filed / duplicate-of / won't-fix + why.

## Red Flags

- Severity assigned from the report title instead of the evidence.
- "Flaky" used as a disposition without a failure-mode hypothesis.
- The same failure triaged twice because nobody searched.
- Environment differences (platform, resolver versions) dismissed instead
  of investigated — they are frequently the root cause.
