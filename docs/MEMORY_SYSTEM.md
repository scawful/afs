# AFS Memory and Context Layout

## Overview

AFS stores long-lived agent context in a filesystem tree rooted at `.context`.

Each mount role has a dedicated directory with policy-aware access.

## Default Layout

```text
.context/
  memory/
  knowledge/
  tools/
  scratchpad/
  history/
  hivemind/
  global/
  items/
  monorepo/
  metadata.json
```

## Directory Roles

- `memory`: durable notes and snapshots
- `knowledge`: reference corpora and indexes
- `tools`: executable utilities and skill packs
- `scratchpad`: writable working area
- `history`: immutable event logs
- `hivemind`: shared multi-agent coordination files
- `global`: cross-project shared state
- `items`: structured artifacts
- `monorepo`: workspace bridge metadata (`active_workspace.toml`)

## Metadata

`metadata.json` stores project-level metadata, including mount directory mapping.
That mapping is authoritative for built-in subsystems too: task queues, hivemind
messages, history logs, MCP prompts/tools, and background-agent reports resolve
their mount roots by role, not by hardcoded directory name.

## Access Patterns

Use the context filesystem CLI for scoped operations:

```bash
afs fs list scratchpad --path ~/src/project-a
afs fs read scratchpad state.md --path ~/src/project-a
afs fs write scratchpad notes.md --path ~/src/project-a --content "..."
```

## Profiles and Mount Injection

Profiles can mount external knowledge/skill/registry roots into context.

```bash
./scripts/afs context profile-show --profile work
./scripts/afs context profile-apply --profile work
```

## Embedding Indexes

Embedding indexes are stored alongside mounted knowledge roots (for example,
`embedding_index.json` under the selected knowledge directory).

Use:

```bash
./scripts/afs embeddings index --knowledge-dir <path> --source <path>
./scripts/afs embeddings search "<query>" --knowledge-dir <path>
```

## Monorepo Bridge

Workspace switch tooling should update:

- `.context/monorepo/active_workspace.toml`

AFS warns when this file is stale and surfaces status via `./scripts/afs health`.
