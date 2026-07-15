---
name: verification-plans
triggers:
  - verify
  - verification
  - verification plan
  - quality gate
  - test plan
profiles:
  - general
requires:
  - afs
verification:
  - Prefer `afs verify run` output over hand-picked test commands when a plan exists.
  - Report pass/fail per check; never summarize a failing plan as passing.
---

# Verification Plans

Repo-aware verification: AFS computes which checks fit the current repo state
and changed paths, then runs them.

## Commands

```bash
afs verify plan --cwd <repo>             # show the computed plan
afs verify run --cwd <repo>              # execute runnable checks
afs verify run --check <name>            # run a single named check
afs verify run --continue-on-fail        # do not stop at first failure
```

## Plan Inputs

- Changed paths auto-detected from git; override with `--changed-path` (repeatable)
- `--payload-file <json>` reuses a `afs session prepare-client` payload
- `--workflow`, `--tool-profile`, `--model` select structured guidance
- `--verification-profile <name>` picks a profile from `afs.toml`;
  `--repo-policy-file` points at an explicit `.afs/policy.toml`
- `--skill <name>` feeds a matched skill into planning

## Run Behavior

- `--require-checks` fails if the plan has no runnable checks (good for CI)
- `--max-digest-items` bounds digested failure output
- `--json` for machine-readable results

## Discipline

Run `plan` first when unsure what will execute. Keep the failing surface
narrow: one check at a time when debugging, full run before handoff.
