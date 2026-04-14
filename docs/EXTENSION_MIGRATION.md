# AFS Extension Migration

This repository (`lab/afs`) now documents and ships **core AFS platform capabilities** only.

Domain-specific content has moved to `lab/afs-ext`, including:

- Persona/model lineages
- Fine-tuning and dataset orchestration workflows
- Domain adapters and specialist content
- Local model deployment playbooks tied to personal environments

## Migration Rule

If a workflow is specific to:

- a model family or persona strategy
- game/domain corpora
- personal workstation/laptop deployment paths

it belongs in `afs-ext`.

## What Stays in Core AFS

- Context filesystem primitives
- Profiles, extensions, hooks
- MCP server integration
- Skills metadata and discovery
- Embeddings indexing/search interfaces
- Health and diagnostics primitives
- Generic dataset/run/eval/feedback orchestration primitives
- Shared schemas, metrics, and status artifacts for training workflows

## Recommended Layout

- Core docs: `lab/afs/docs/`
- Domain/training docs: `lab/afs-ext/docs/`
- Domain skills: `lab/afs-ext/skills/`
- Domain registries/policies: `lab/afs-ext/config/`
