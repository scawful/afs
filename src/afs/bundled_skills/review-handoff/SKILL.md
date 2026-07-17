---
name: review-handoff
triggers:
  - review package
  - review handoff
  - pre-PR review
  - review verdict
  - reviewer
  - review findings
  - constraint checklist
  - review someone
profiles:
  - general
requires:
  - afs
enforcement:
  - Never modify, commit, stage, or clean the worktree under review; it belongs to its author.
  - Every constraint marked FAIL or PASS carries one line of evidence; every finding carries file:line, severity, and a suggested fix.
  - Declare what was NOT reviewed as explicitly as what was.
verification:
  - Confirm the package states verdicts, validation commands with pass/fail counts, and reviewed-at SHAs before handing off.
---

# Review Handoff

Produce review results another engineer or agent can act on without
re-deriving your work: verdicts, evidence, and exact validation.
Complements `code-review` (how to find defects); this is how to deliver
them across a team.

## Package Shape

1. **Header**: what was reviewed (branch@SHA, diff base), by whom, when.
   Pin SHAs — authors keep working while you review; note if HEAD moved.
2. **Verdict per unit**: READY / NEEDS_WORK / BLOCKED, one line of why.
3. **Findings**, most severe first:
   `severity — file:line — defect (one sentence) — failure scenario —
   suggested fix`. Separate must-fix-before-merge from should-fix from
   note-in-PR-description.
4. **Constraint checklist** when a plan/spec exists: each stated
   constraint → PASS / FAIL / NOT_IMPLEMENTED + one-line evidence.
   NOT_IMPLEMENTED with a stated deferral is a scoping decision for the
   owner, not a defect.
5. **Validation**: exact commands run, pass/fail counts, environment.
   Distinguish CONFIRMED (reproduced) from PLAUSIBLE (code-read) findings.
6. **Not reviewed**: skipped files, unpushed commits, angles not taken.

## Reviewing Others' Live Work

- Read-only in their worktree; run tests from a disposable scratch
  worktree if execution is needed.
- Findings may already be fixed in unpushed work — review pinned SHAs and
  say so, rather than guessing at intent.
- Deliver where the team coordinates (ticket, handoff, message bus), not
  only in the transcript.

## Red Flags

- A verdict with no validation section behind it.
- Style notes ranked above correctness findings.
- "Looks good" covering files the reviewer never opened.
