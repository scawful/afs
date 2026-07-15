---
name: context-search
triggers:
  - afs query
  - context query
  - context index
  - stale context
  - context freshness
  - index rebuild
  - embedding search
profiles:
  - general
requires:
  - afs
enforcement:
  - Start the indefinite watch loop or execute an --on-change shell command only with explicit user direction.
---

# Context Search

Indexed retrieval over AFS context mounts, plus index freshness management.

## Commands

| Command | Description |
|---------|-------------|
| `afs query "<text>" --path <ws>` | Shortcut for `afs context query` |
| `afs context query "<text>" --mount <m> --prefix <p> --limit N` | Indexed path/content search |
| `afs index rebuild --path <ws>` | Rebuild the SQLite context index |
| `afs context freshness` | Per-file freshness scores |
| `afs watch --path <ws> --debounce 30` | Watch for changes, auto-rebuild |
| `afs embeddings index/search/eval` | Embedding index build, search, retrieval eval |

## Query Flags

- `--mount` (repeatable) restricts to mount types (scratchpad, memory, knowledge, ...)
- `--include-content` returns full indexed content instead of excerpts
- `--no-auto-index` / `--no-auto-refresh` skip implicit index maintenance
- `--json` for machine-readable results

## Staleness Policy

A built-but-stale index is a freshness advisory, not a hard failure. If mounts
are healthy and the index exists, keep using cheap reads; rebuild only before
search-heavy work. Missing or empty index: rebuild first.

`afs watch` runs until interrupted and rebuilds after each change batch.
`--on-change` executes a shell command; treat it as code execution, not as a
notification-only option.

## MCP Tools

Slim catalog: `context.query`, `context.status`. Full catalog adds
`context.diff` and `context.freshness`.
