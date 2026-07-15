---
name: implementation-planning
triggers:
  - plan
  - planning
  - refactor
  - migration
  - rollout
  - breakdown
  - milestone
  - roadmap
  - implementation plan
profiles:
  - general
requires:
  - afs
enforcement:
  - Define the problem, constraints, done criteria, and rollback before major edits.
  - Break work into reversible, testable slices with one verification point each.
  - Prefer vertical slices over broad cross-repo churn.
verification:
  - Record the plan in scratchpad or the task queue when the change spans multiple steps.
  - Run the fastest relevant verification after each slice, not only at the end.
---

# Implementation Planning

Break work into reversible slices with clear verification points.

## Planning Rules

1. Define the problem, constraints, and done criteria before editing.
2. Split large changes into behavior-preserving steps.
3. Keep each step independently reviewable and testable.
4. Identify the fastest verification command for every step.
5. Note rollback strategy for risky migrations or refactors.

## Strong Defaults

- Land scaffolding before behavior changes.
- Prefer vertical slices over broad cross-repo churn.
- Preserve old and new paths briefly when measuring a migration.
- Remove temporary compatibility code once the new path is verified.

## AFS Integration

- Keep the active plan in scratchpad:

```bash
afs fs write scratchpad plans/<topic>.md --content "goal, constraints, slices, checks, rollback"
```

- Check existing queue state with `afs tasks list`.
- Under MCP clients, use `task.create` for sub-work and `hivemind.send` for handoffs.
