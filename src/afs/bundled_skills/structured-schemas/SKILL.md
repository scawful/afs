---
name: structured-schemas
triggers:
  - afs schema
  - structured response
  - structured output
  - response schema
  - schema validate
profiles:
  - general
requires:
  - afs
enforcement:
  - Never write, fill, or edit the human_intent section of an implementation
    plan; reproduce it exactly as the human provided it, or leave it absent.
---

# Structured Response Schemas

Named response schemas that AFS workflows use to validate structured agent
output.

## Commands

```bash
afs schema list                          # available schema names
afs schema show <name>                   # print schema as JSON
afs schema validate --schema <name> --file response.json
afs schema validate --workflow review_deep --text '{"..."}'
afs schema validate --schema implementation-plan --file plan.json \
  --skeleton human_plan.json   # fails if human_intent was edited/authored
```

## Validation

- `--schema <name>` or `--workflow <workflow>` (resolves the schema bound to
  that workflow, e.g. `edit_fast`, `review_deep`) — exactly one is required
- Input via `--text` inline, `--file <path>`, or stdin (`--file -` or omitted)
- Exit code 1 on mismatch; `--json` emits the full validation result including
  a suggested correction

## Usage Pattern

1. Before producing structured output, `afs schema show <name>` to see the
   exact contract
2. After producing it, pipe through `afs schema validate` instead of eyeballing
3. Session packs and `prepare-client` payloads may name a recommended schema
   (e.g. `implementation-plan`, `handoff-summary`) — honor it when present
