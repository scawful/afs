# AFS Extension Migration

This repository (`lab/afs`) ships **core AFS platform capabilities** only. It
should not assume one user's games, corpora, model lineages, external tools, or
machine layout.

Domain-specific content belongs in companion extension repos named like
`afs_<name>` or `afs-<name>`. In this workspace, Zelda/Oracle/persona/training
content lives in the `lab/afs-scawful` repo and imports as the `afs_scawful`
Python package.

## Migration Rule

If a workflow is specific to:

- a model family or persona strategy
- game/domain corpora
- personal workstation/laptop deployment paths
- external connector implementations such as Google Workspace public API adapters
- MCP/domain servers that are not useful for every AFS user

it belongs in a companion extension repo, not in core AFS.

## What Stays in Core AFS

- Context filesystem primitives
- Profiles, extensions, hooks
- MCP server integration
- Skills metadata and discovery
- Embeddings indexing/search interfaces
- Health and diagnostics primitives
- Generic dataset/run/eval/feedback orchestration primitives
- Shared schemas, metrics, and status artifacts for training workflows

## Companion Repo Layout

A companion repo can be a sibling of `afs`:

```text
~/src/lab/afs_example/
  extension.toml
  src/afs_example/
    __init__.py
    cli.py
    agents.py
    mcp_surface.py
```

Core AFS discovers it with:

```toml
[extensions]
enabled_extensions = ["afs_example"]
extension_repo_roots = ["~/src/lab"]
```

`src/` is added to the import path automatically when present, so src-layout
packages do not need to be copied into `lab/afs/extensions/`.

Extensions can also expose manager-visible actions without becoming part of
core AFS:

```toml
[manager]
actions = [
  "afs gws status",
  "afs work communication preflight --path .",
]
```

The Python `afs manager` GUI lists those actions next to the extension and
keeps execution explicit.

## Current Scawful/Zelda Boundary

- Core docs/code: `lab/afs/`
- Personal/domain extension repo: `lab/afs-scawful/`
- Python package: `afs_scawful`
- Zelda/Oracle modules: `afs_scawful.oracle.*`, `afs_scawful.agents.zelda_tools`
- Scawful model/tool shims: `afs_scawful.agent_model_presets`,
  `afs_scawful.agent_tools`

Core AFS may keep compatibility shims that explain where a moved module lives,
but new domain implementation should be added to the companion repo.
