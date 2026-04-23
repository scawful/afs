---
name: software-design
triggers:
  - design
  - architecture
  - abstraction
  - interface
  - boundary
  - coupling
  - cohesion
  - api
  - domain model
profiles:
  - general
requires:
  - afs
enforcement:
  - State invariants, ownership, and single sources of truth before adding abstractions.
  - Prefer clear module boundaries over wrappers that only rename complexity.
  - Keep failure modes and rollback paths explicit for risky design changes.
verification:
  - Write the active design note to scratchpad before large structural edits.
  - Validate the chosen design with the smallest end-to-end slice before expanding scope.
---

# Software Design

Clarify invariants and module boundaries before introducing new abstractions.

## Design Checklist

1. State the problem, constraints, and done criteria in one short note.
2. Identify the owner of each important piece of state.
3. Keep one source of truth per concept.
4. Prefer modules shaped around behavior and data flow, not placeholder layers.
5. Make failure modes and rollback paths explicit for risky changes.

## Avoid

- Premature abstractions created before two real callers exist.
- Bidirectional dependencies between modules.
- God objects that mix orchestration, persistence, and policy.
- Wrappers that only rename APIs while leaking the same complexity.
- Optional-parameter growth that should become separate types or commands.

## AFS Integration

- Capture working notes in scratchpad first:

```bash
afs fs write scratchpad design/<topic>.md --content "problem, constraints, invariants, options"
```

- Promote stable conclusions into knowledge only after the design survives implementation.
