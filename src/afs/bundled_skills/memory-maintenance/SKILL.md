---
name: memory-maintenance
triggers:
  - afs memory
  - memory consolidate
  - memory search
  - durable memory
  - session history
profiles:
  - general
requires:
  - afs
---

# Memory Maintenance

Durable memory distilled from session history, stored under the `memory`
mount. Search it before asking the user for already-known context.

## Commands

| Command | Description |
|---------|-------------|
| `afs memory search "<query>" --path <ws>` | Search memory entries |
| `afs memory status` | Pipeline status (pending history, entry counts) |
| `afs memory consolidate` | Distill recent history into durable entries |

## Consolidation Flags

`--max-events`, `--max-events-per-entry`, `--event-type <t>` (limit to
specific history event types), `--no-markdown` (skip markdown summaries),
`--json`.

## Discipline

- Read path: `memory search` first, then `context.query` over the memory
  mount for exact files
- Write path: prefer consolidation over hand-written entries so provenance
  links back to history events
- Run `consolidate` after significant work slices or when `status` shows a
  backlog; it is idempotent over already-consolidated events

## MCP Tools

`memory.status` and `memory.search` exist in the full catalog
(`AFS_MCP_TOOL_CATALOG=full`); slim-catalog clients use the CLI.
