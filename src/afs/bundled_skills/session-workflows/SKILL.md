---
name: session-workflows
triggers:
  - afs session
  - session handoff
  - session bootstrap
  - context pack
  - session continuity
  - session replay
  - prepare-client
profiles:
  - general
requires:
  - afs
---

# Session Workflows

Start, pack, hand off, and replay AFS-aware agent sessions.

## Commands

| Command | Description |
|---------|-------------|
| `afs session bootstrap --path <ws>` | Startup packet: context health, scratchpad, tasks, hivemind, memory, missions, skills |
| `afs session pack --path <ws> [query]` | Token-budgeted context pack for a client |
| `afs session prepare-client` | Full bootstrap/pack/skill payload for a client harness |
| `afs session handoff create/list/read` | Handoff packets between sessions |
| `afs session replay-list` / `replay` | List and replay recorded session timelines |
| `afs cache status` / `clear` | Inspect or clear the session pack cache |

## Key Flags

- `bootstrap`: `--task-limit`, `--message-limit`, `--agent-name`, `--json`; writes `scratchpad/afs_agents/session_bootstrap.{json,md}` unless `--no-write-artifacts`
- `pack`: `--model generic|gemini|claude|codex`, `--workflow general|scan_fast|edit_fast|review_deep|root_cause_deep`, `--tool-profile`, `--pack-mode focused|retrieval|full_slice`, `--token-budget`
- Optional positional `query` on `pack` adds retrieval results

## Handoff Discipline

1. End a work slice with `afs session handoff create` (what shipped, next steps, risks)
2. Next session starts with `afs session bootstrap`, which surfaces the latest handoff
3. Use `afs next --intent handoff` when unsure whether a handoff is due

## MCP Tools

`handoff.create`, `handoff.read`, `handoff.list` (full catalog only —
`AFS_MCP_TOOL_CATALOG=full`). Slim-catalog clients should shell out to the CLI.
