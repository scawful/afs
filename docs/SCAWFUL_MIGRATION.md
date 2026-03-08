# AFS Scawful Migration

This repository (`lab/afs`) now documents and ships **core AFS platform capabilities** only.

Domain-specific content has moved to `lab/afs-scawful`, including:

- Persona/model lineages
- Fine-tuning and dataset orchestration workflows
- Domain adapters and specialist content
- Local model deployment playbooks tied to personal environments

## Migration Rule

If a workflow is specific to:

- a model family or persona strategy
- game/domain corpora
- personal workstation/laptop deployment paths

it belongs in `afs-scawful`.

## What Stays in Core AFS

- Context filesystem primitives
- Profiles, extensions, hooks
- MCP server integration
- Skills metadata and discovery
- Embeddings indexing/search interfaces
- Health and diagnostics primitives

## Recommended Layout

- Core docs: `lab/afs/docs/`
- Domain/training docs: `lab/afs-scawful/docs/`
- Domain skills: `lab/afs-scawful/skills/`
- Domain registries/policies: `lab/afs-scawful/config/`
