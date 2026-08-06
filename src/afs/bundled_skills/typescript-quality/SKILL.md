---
name: typescript-quality
triggers:
  - typescript
  - tsx
  - tsc
  - typecheck
  - tsconfig
  - tsconfig.json
  - typescript-eslint
  - type guard
  - discriminated union
  - vitest
  - zod
profiles:
  - general
requires:
  - afs
enforcement:
  - Keep strict types at module boundaries and validate every untrusted runtime input.
  - Avoid any, unchecked type assertions, non-null assertions, and floating promises.
  - Keep browser, server, module-format, and async-cleanup boundaries explicit.
verification:
  - Use the lockfile-selected package manager for focused lint, typecheck, build, and tests.
  - Report runtime-validation gaps and every skipped quality gate with residual risk.
---

# TypeScript Quality

Use static types for trusted program state and runtime validation for data that crosses a trust boundary.

## Core Rules

- Keep `strict` compiler settings unless the repository documents a narrower migration step.
- Accept untrusted values as `unknown`, validate once, and return a typed result.
- Use discriminated unions, exhaustive checks, and branded or opaque types when states must not mix.
- Keep transport, validation, state transitions, rendering, and side effects in distinct units.
- Handle promise rejection, cancellation, effect cleanup, and resource disposal explicitly.
- Preserve the repository's package manager, module format, browser/server boundary, and generated-code policy.

## Avoid

- `any`, double assertions, broad `as` casts, and non-null assertions as design tools.
- Floating promises, swallowed rejections, and async callbacks in APIs that do not await them.
- Boolean prop or option explosions and invalid states represented by unrelated optional fields.
- Import-time side effects, mutable ambient state, and accidental browser-only or server-only imports.
- Editing generated files when the source schema or generator owns the change.

## Quality Gates

Use the lockfile-selected package manager and repository scripts. Typical checks:

```bash
<package-manager> run lint
<package-manager> run typecheck
<package-manager> run build
<package-manager> test -- <focused-tests>
```

Run `tsc --noEmit` directly only when the repository has no equivalent script.
