---
name: python-quality
triggers:
  - python
  - ruff
  - mypy
  - pyright
  - pytest
  - pydantic
  - typing
  - type hints
  - asyncio
profiles:
  - general
requires:
  - afs
enforcement:
  - Keep I/O boundaries explicit and avoid import-time side effects.
  - Avoid bare except, mutable defaults, and hidden None returns.
  - Add or preserve type hints on public interfaces and shared models.
verification:
  - Run repo-standard Ruff, tests, and type checks for touched Python.
  - Report the exact blocker and residual risk if verification is skipped.
---

# Python Quality

Prefer explicit data flow, typed boundaries, and small units with obvious side effects.

## Prefer

- Type hints on public functions and shared models.
- `pathlib`, context managers, and dependency injection over implicit globals.
- Pure helpers around I/O boundaries so behavior is testable without patch-heavy fixtures.
- Structured return types such as dataclasses, TypedDicts, or repo-standard models.
- Narrow exceptions with clear error messages.

## Avoid

- Bare `except` or broad exception swallowing.
- Mutable default arguments.
- Import-time side effects.
- Hidden `None` returns from partially handled branches.
- Boolean flags that should be separate functions or strategy objects.

## Quality Gates

Use the repo's existing toolchain. Typical checks:

```bash
ruff check .
ruff format --check .
pytest -q
mypy .
pyright
```
