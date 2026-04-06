# Development Guide

## Scope

This repository documents and develops core AFS platform features.

Domain-specific and model-training development flows live in afs-ext.
See docs/SCAWFUL_MIGRATION.md.

## Local Setup

```bash
cd <afs-root>
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Fast Validation

```bash
pytest -q tests/test_config.py tests/test_manager.py tests/test_profiles.py
python3 -m py_compile src/afs/*.py
```

## Contributor Workflow

1. Keep changes scoped to the requested subsystem.
2. Prefer small, testable increments.
3. Add or update targeted tests for behavior changes.
4. Run the fastest relevant test subset before commit.
5. Document user-facing changes in docs/ when CLI or config behavior changes.

## Branching Workflow

Use the shared integration model for this repo:

1. Create topic branches from `features`.
2. Merge topic branches into `features` via PRs.
3. Promote tested work from `features` into `development`.
4. Promote release-ready work from `development` into `main`.
5. When needed, sync `main` back into `features` and `development` to keep history aligned.

Preferred PR targets:

- Day-to-day feature work: `features`
- Integration/stabilization: `development`
- Release promotion: `main` (from `development`)

Use the repo PR template checklist to confirm base-branch targeting on every PR.

## Recommended Branch Protection

Configure branch protection in GitHub for:

- `features`: require at least 1 review before merge.
- `development`: require at least 1 review and required status checks.
- `main`: require at least 1 review, required status checks, and restrict direct pushes.

Recommended required checks: the `AFS CI/CD Pipeline` workflow jobs that validate tests/lint/type checks.
Also require `Branch Policy / Enforce staged branch flow` to block invalid PR base/head combinations.

## Areas

- Context manager and filesystem helpers: `src/afs/manager.py`, `src/afs/context_fs.py`
- Config/profile/extension system: `src/afs/config.py`, `src/afs/schema.py`, `src/afs/profiles.py`, `src/afs/extensions.py`
- CLI surface: `src/afs/cli/`
- MCP server: `src/afs/mcp_server.py`
- Health diagnostics: `src/afs/health/`

## Documentation Updates

If you add or modify:

- profile behavior
- extension manifest schema
- MCP tools
- health/status output

update the corresponding docs page and command examples in the same change.
