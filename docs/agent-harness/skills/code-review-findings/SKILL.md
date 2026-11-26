---
name: code-review-findings
description: Perform a strict code review with findings first. Use when the user asks for review, re-review, audit, regression check, or wants bugs, risks, and missing tests identified before summaries.
---

# Code Review Findings

## Goal
Review like a blocking reviewer: prioritize correctness, regressions, safety, and test gaps.

## Workflow
1. Identify review base, current HEAD, and user-provided fix commits or files.
2. Inspect the actual diff and relevant surrounding code.
3. Focus on user-visible bugs, data loss, security boundaries, broken workflows, missing verification, and maintainability risks that affect behavior.
4. Reproduce or reason from exact code paths; avoid style-only comments unless style hides a bug.
5. If a previous finding was claimed fixed, re-review current code against that fix instead of repeating stale findings.

## Output Contract
1. Findings first, ordered by severity.
2. Each finding includes file/line, impact, and why it happens.
3. Then open questions or assumptions.
4. Then brief change summary only if useful.
5. If no issues are found, say so and list residual test gaps.

## Guardrails
- Do not bury findings after a prose summary.
- Do not claim a regression without pointing to the code path or verification evidence.
