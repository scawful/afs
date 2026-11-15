# AFS Lite Boundary

AFS core (`afs-lite`) is a small context OS for agents: files, mail, status,
handoffs, and skill roots. Everything else is optional, gated, or scheduled for
extraction.

## Core (always on)

| Surface | Notes |
|---|---|
| `context.read` / `write` / `list` / `query` / `status` | Preferred file + index tools |
| `messages.*` | SQLite mail (`hivemind/messages.sqlite3`) |
| `note.*` / `handoff.*` / `skill.*` | Slim MCP defaults |
| `afs doctor` / `afs history reap` | Hygiene and retention |
| Skill roots | Host loader; AFS supplies roots + task context |

Default MCP `tools/list` is frozen as `afs.mcp.catalog.FROZEN_SLIM_MCP_TOOLS`.

## Optional extras

| Surface | Enable with | Notes |
|---|---|---|
| `afs training *` | `pip install 'afs[training]'` or `AFS_ENABLE_TRAINING=1` | Generators, dataset export, run control |
| `afs embeddings *` | provider SDK / `afs[embeddings]` or `AFS_ENABLE_EMBEDDINGS=1` | Index build/search/eval; full MCP only |
| Gemini / Claude adapters | `afs[gemini]` / `afs[claude]` | Client-specific packs |

The repo `./scripts/afs` wrapper sets both `AFS_ENABLE_*` flags by default so a
checkout keeps working. A plain `python -m afs` install stays fail-closed.

## Personal / operator (in-tree, out of slim MCP)

These remain in the tree for now but are not part of the agent slim catalog.
Candidates for an `afs-scawful` / work extension later:

- `work_assistant`, `storage` / `storage_doctor`, `gws`, `briefing`
- Full-catalog MCP tools (`agent.*`, `events.*`, `work.*`, embeddings index)

## Deprecation schedule

| Item | Status |
|---|---|
| `fs.*` MCP aliases | Soft-deprecated; removal in **0.3** |
| `afs.search` / `afs.codebase.*` | Legacy full-catalog only; prefer `context.query` / scoped `context.search` |
| Physical move of `training/` / embeddings stacks | Follow-on PR after this gate lands |

## Retention defaults

- History: `history.retention_days = 30`, gzip older daily jsonl (`gzip_older`)
- Scratchpad: `scratchpad.retention_days = 14`; protect `state.md`, `deferred.md`, `handoffs/`, `common/`, `plans/`
- Operator command: `afs history reap --dry-run` / `--apply` (also suggested by `afs doctor`)
- Indexers already exclude `embeddings/`, `*.bak*`, `embedding_index.json`, `.afs/`; `afs index rebuild` after VACUUM remains operator-owned

## Explicitly deferred

- Single `.afs/afs.sqlite3` mega-db
- Deleting live empty embedding JSON blobs (needs owner approval)
- Hard-removing `fs.*` or layout v1/v2 dual paths
