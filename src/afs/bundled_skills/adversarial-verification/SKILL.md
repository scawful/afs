---
name: adversarial-verification
triggers:
  - adversarial verification
  - refute
  - reproduce
  - reproduction
  - false positive
  - confirm finding
  - bite test
  - does it actually fail
profiles:
  - general
requires:
  - afs
enforcement:
  - Try to refute the claim, not confirm it; a finding you could not break is worth reporting, one you did not try to break is not.
  - A fix is validated only when a previously failing reproduction flips to passing; a testbed that never reproduced the failure validates nothing.
  - Verdicts are CONFIRMED (reproduced/observed) or PLAUSIBLE (code-read only) — never blend them.
  - Never claim a verification that was not actually run.
verification:
  - For each verdict, state the exact command or probe that produced it and its environment (platform, tool versions).
---

# Adversarial Verification

Verify findings, fixes, and claims by trying to break them. Applies to
review findings, bug reports, "this fix works", and guard/ratchet checks.

## Verifying a Finding

1. Restate the claim as a concrete failure scenario: inputs/state → wrong
   outcome. If it cannot be stated, it cannot be confirmed.
2. Build the cheapest honest reproduction — isolated venv/scratch dir,
   exact versions from the failing environment. Match what actually
   failed: resolver picks, platform stubs, and wheel contents differ
   across Python versions and OSes.
3. Run it. Reproduces → CONFIRMED with the probe recorded. Doesn't →
   either the claim is wrong or the testbed differs; find which before
   downgrading the finding.

## Verifying a Fix

1. Reproduce the failure FIRST on the unfixed code. No repro, no signal.
2. Apply the fix; the same probe must flip to passing.
3. Check the fix didn't narrow the guard: run the surrounding tests.

## Verifying a Guard or Ratchet (bite test)

A check that never fails protects nothing. Prove new lint rules, type
gates, and CI guards bite: introduce a violating sample, watch it fail,
remove the sample.

## Red Flags

- "Verified locally" where local differs from the failing environment
  (macOS vs Linux CI, different dependency resolution).
- Confirming evidence collected while refuting evidence was never sought.
- A passing suite treated as proof for a claim no test actually covers.
