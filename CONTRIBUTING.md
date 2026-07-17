# Contributing to AFS

Thanks for helping improve AFS. This repo is the **core platform**: context filesystem primitives, CLI/MCP surfaces, extension loading, memory/session flows, and generic workflow infrastructure.

Domain-specific agents, corpora, model families, private connectors, and personal workstation workflows should live in companion extension repos.

## Quick setup

```bash
git clone https://github.com/scawful/afs.git
cd afs
make setup
make check
```

If you prefer explicit commands:

```bash
python3 -m venv .venv
. .venv/bin/activate
python -m pip install --upgrade pip
pip install -e ".[dev,test,docs]"
```

## Branching flow

AFS uses staged integration:

1. Create topic branches from `features`.
2. Open feature PRs into `features`.
3. Promote `features -> development` after checks pass.
4. Promote `development -> main` for releases.

The branch-policy workflow enforces this flow for protected branches.

## What belongs in core

Core AFS may include:

- `.context` filesystem primitives
- context indexing and context packs
- CLI and MCP surface contracts
- extension/plugin discovery
- generic agent-job, work-assistant, memory, health, and session workflows
- provider-neutral schemas and interfaces

Use an extension repo for:

- domain-specific datasets or models
- private company/personal connectors
- private/domain-specific assistant workflows
- machine-specific paths or local model names
- provider-specific tools that not every AFS user needs

## Development checklist

Code standards live in [docs/ENGINEERING_PRACTICES.md](docs/ENGINEERING_PRACTICES.md)
(exception policy, atomic filesystem writes, typing, test expectations).
Ruff and mypy enforce parts of it through ratcheted baselines in
`pyproject.toml`: never add a baseline entry; when you touch a listed
file, fix its violations and delete its entry.

Before opening a PR:

```bash
make lint
make type-check
make test
make package-check
```

For docs or onboarding changes:

```bash
make docs
```

For release-prep changes:

```bash
make release-check
```

Also update docs when you change user-facing CLI behavior, MCP tool names, extension manifests, configuration, or setup flow.

## Compatibility expectations

Pre-1.0 versions may still refine APIs, but changes should be intentional:

- note breaking changes in `CHANGELOG.md`
- keep compatibility aliases when inexpensive
- prefer extension shims over hard failures when moving functionality out of core
- add tests for new CLI/MCP/setup behavior

## Security and external writes

AFS treats external writes cautiously. Do not add code that posts, sends, publishes, force-pushes, deletes, or mutates external systems without explicit user approval paths and dry-run behavior.

See `SECURITY.md` for reporting guidance.
