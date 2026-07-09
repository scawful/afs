# AFS Extension Migration

This repository ships **core AFS platform capabilities** only. It
should not assume one user's private corpora, model lineages, external tools, or
machine layout.

Domain-specific content belongs in companion extension repos named like
`afs_<name>` or `afs-<name>`. Those repos can mount knowledge, skills, CLI
commands, agents, context sources, and MCP surfaces without hardcoding them into
core AFS.

## Migration Rule

If a workflow is specific to:

- a model family or assistant strategy
- private or domain-specific corpora
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

## Compatibility Shims

Core AFS may keep small compatibility shims that explain where a moved module
lives, but new domain implementation should be added to a companion extension
repo. Shims should fail with clear install/enable instructions rather than
reintroducing private implementation details into core.
