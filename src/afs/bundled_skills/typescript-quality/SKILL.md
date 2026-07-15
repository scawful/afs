---
name: typescript-quality
triggers:
  - typescript
  - tsx
  - eslint
  - tsc
  - typecheck
  - tsconfig
  - react
  - node
  - npm
profiles:
  - general
requires:
  - afs
enforcement:
  - Keep strict types at boundaries and validate untrusted input instead of using any.
  - Avoid non-null assertions, assertion chains, and boolean option explosions.
  - Separate rendering or transport concerns from stateful side effects.
verification:
  - Run repo-standard lint, typecheck, and tests for touched TypeScript.
  - Call out runtime validation gaps at API or user-input boundaries.
---

# TypeScript Quality

Bias toward explicit types at module boundaries and runtime validation at untrusted inputs.

## Prefer

- `strict` compiler settings and `tsc --noEmit` in CI or local verification.
- `unknown` plus validation instead of `any`.
- Discriminated unions, exhaustive `switch` statements, and typed result objects.
- Small modules with explicit imports and minimal ambient state.
- UI code that separates rendering from data loading and side effects.

## Avoid

- `any`, assertion chains, and non-null assertions as design tools.
- Boolean prop or option explosions.
- Default exports in utility-heavy modules where named imports read better.
- Mixing transport, validation, and presentation concerns in one function.
- Silent promise rejection handling.

## Quality Gates

Use the repo package manager and scripts. Common checks:

```bash
npm run lint
npm run typecheck
npm test
tsc --noEmit
```
