# AFS (Agentic File System)

Research-only. Not a product.

Scope: core AFS primitives + internal workflow tooling.

Provenance: avoid employer/internal sources; skip unclear origins.

Docs:
- `docs/STATUS.md`
- `docs/ROADMAP.md`
- `docs/REPO_FACTS.json`
- `docs/NARRATIVE.md`

Quickstart:
- `python -m afs init --context-root ~/src/context --workspace-name trunk`
- `python -m afs status`
- `python -m afs workspace add --path ~/src/trunk --name trunk`
- `python -m afs context init --path ~/src/trunk`
- `python -m afs context validate --path ~/src/trunk`
- `python -m afs context discover --path ~/src/trunk`
- `python -m afs context ensure-all --path ~/src/trunk`
- `python -m afs graph export --path ~/src/trunk`

Discovery skips directory names in `general.discovery_ignore` (default: legacy, archive, archives).
