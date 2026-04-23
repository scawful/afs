---
name: code-review
triggers:
  - review
  - code review
  - bug
  - regression
  - risk
  - anti-pattern
  - code smell
  - bad practice
profiles:
  - general
requires:
  - afs
enforcement:
  - Lead with correctness, safety, regressions, and missing tests before style.
  - Report concrete findings with file and line references when available.
  - Do not conclude the change is safe until verification is run or the risk is stated.
verification:
  - Run the fastest relevant lint, type-check, or test command for touched code.
  - If checks cannot run, state the exact blocker and residual risk.
---

# Code Review

Prioritize correctness, safety, regressions, and missing tests before style.

## Review Order

1. Confirm the change still matches the stated behavior and done criteria.
2. Look for state corruption, invalid assumptions, and edge-case breakage.
3. Check ownership, lifetime, nullability, error handling, and concurrency hazards.
4. Flag maintainability risks only after behavior and safety are covered.
5. Verify the fastest relevant tests or explain the remaining risk.

## Expected Output

- Lead with concrete findings, not a summary.
- Cite the file and line when possible.
- Explain impact in one sentence.
- Suggest the smallest corrective change that removes the risk.

## Red Flags

- Silent fallbacks that hide errors.
- Broad exception or error suppression.
- Hidden side effects at import or construction time.
- Multiple sources of truth for the same state.
- Boolean or mode flags that create unrelated code paths in one function.
- Commented-out code, dead branches, and TODOs in touched paths.
