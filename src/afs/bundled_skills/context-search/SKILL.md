---
name: context-search
triggers:
  - afs search
  - afs query
  - context query
  - context index
  - stale context
  - context freshness
  - index rebuild
  - embedding search
  - insights research
profiles:
  - general
requires:
  - afs
enforcement:
  - Start the indefinite watch loop or execute an --on-change shell command only with explicit user direction.
---

# Context Search

Scoped local-first retrieval over project files and AFS context.

## Commands

| Command | Description |
|---------|-------------|
| `afs search "<text>" --path <ws>` | Search the current project plus common context |
| `afs search "<text>" --path <ws> --semantic` | Explicitly enable semantic retrieval (Gemini by default) |
| `afs insights research "<text>" --path <ws>` | Refresh and research the current project plus common |
| `afs query "<text>" --path <ws>` | Version 1 SQLite compatibility search |
| `afs context query "<text>" --mount <m> --prefix <p> --limit N` | Indexed path/content search |
| `afs index rebuild --path <ws>` | Rebuild the SQLite context index |
| `afs context freshness` | Per-file freshness scores |
| `afs watch --path <ws> --debounce 30` | Watch for changes, auto-rebuild |
| `afs embeddings index/search/eval` | Advanced legacy embedding workflows |

## Search Flags

- `--rebuild` publishes a new immutable local index generation
- `--semantic` explicitly allows embeddings; Gemini defaults to stable
  `gemini-embedding-2` at 768 dimensions
- `--all-projects` is required for cross-project results
- `--json` for machine-readable results

## Staleness Policy

The first `afs search` builds a local keyword/symbol index automatically.
Semantic opt-in upgrades a keyword-only index automatically. Source scope is
filtered before text, symbol, association, or vector ranking.

`afs insights research` refreshes its local index by default; use
`--reuse-index` only when freshness is already known. Its semantic mode is
explicit: Ollama keeps embedding input local, while Gemini transmits indexed
content and the query to Gemini. Research is always limited to the current
project plus `common`; it has no all-projects mode. Internet research is a
separate extension opt-in with an explicit domain allowlist.

`afs watch` runs until interrupted and rebuilds after each change batch.
`--on-change` executes a shell command; treat it as code execution, not as a
notification-only option.

## MCP Tools

Slim catalog: `context.search`, `context.query` (version 1 compatibility), and
`context.status`. `context.search` uses the existing v2 index and never grants
another project merely because the central context path is known.
