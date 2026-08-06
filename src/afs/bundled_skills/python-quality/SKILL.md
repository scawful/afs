---
name: python-quality
triggers:
  - python
  - pyproject
  - ruff
  - mypy
  - pyright
  - pytest
  - pydantic
  - typing
  - type hints
  - asyncio
  - dataclass
profiles:
  - general
requires:
  - afs
enforcement:
  - Keep I/O, side effects, and async boundaries explicit; avoid import-time work.
  - Validate untrusted data and use narrow exceptions; never hide failures or cancellation.
  - Add or preserve precise types on public interfaces and shared models; avoid implicit Any and Optional results.
verification:
  - Run repository-standard Ruff, focused tests, and the configured type checker for touched Python.
  - Report the exact blocker and residual risk for every skipped quality gate.
---

# Python Quality

Prefer explicit data flow, typed boundaries, deterministic cleanup, and small units with visible side effects.

## Core Rules

- Type public functions and shared models. Use dataclasses, TypedDicts, protocols, or repository-standard models where they clarify contracts.
- Validate untrusted input at the boundary, then pass typed values through the core.
- Use context managers for resources and dependency injection for clocks, clients, storage, and other I/O.
- Keep blocking I/O out of async paths. Preserve timeouts, cancellation, and cleanup behavior.
- Raise narrow exceptions with useful context; preserve the original cause when translating failures.

## Avoid

- Bare `except`, broad exception swallowing, hidden fallbacks, or partially handled branches that return `None`.
- Mutable default arguments, implicit `Any`, unnecessary casts, and unchecked `# type: ignore` comments.
- Import-time I/O, mutable module globals, and side effects hidden in constructors.
- Boolean mode flags when separate functions, types, or strategies make behavior clearer.
- Patch-heavy tests when a pure helper or injected boundary is simpler.

## Quality Gates

Use project scripts and the narrowest touched paths first. Typical checks:

```bash
ruff check <paths>
ruff format --check <paths>
pytest -q <tests>
mypy <package-or-paths>
pyright <paths>
```

Run only the configured type checker when the repository standardizes on one.
