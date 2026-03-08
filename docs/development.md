# Development Guide

## Scope

This repository documents and develops core AFS platform features.

Domain-specific and model-training development flows live in afs-scawful.
See docs/SCAWFUL_MIGRATION.md.

## Local Setup

```bash
cd ~/src/lab/afs
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
