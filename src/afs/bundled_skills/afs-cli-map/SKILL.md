---
name: afs-cli-map
triggers:
  - which afs command
  - which afs subcommand
  - afs command map
  - route afs intent
profiles:
  - general
requires:
  - afs
enforcement:
  - Route to read-only inspection first; state-changing, destructive, or persistent commands require explicit user direction.
  - A command recommendation is not permission to bypass a domain skill's approval or safety rules.
---

# AFS CLI Map

Routing table for the `afs` CLI. Use this skill only for command-discovery
questions; domain-specific prompts should load the corresponding skill instead.
When unsure, start with
`afs next --intent <continue|context|review|ship|verify|handoff|setup|refresh|pack> --path <ws>`.

## Core

| Area | Commands |
|------|----------|
| Setup | `init`, `setup`, `status`, `profile`, `bundle`, `workspace` |
| Search | `query`, `context query`, `index rebuild`, `embeddings`, `watch` |
| Files | `fs read/write/list/move/delete`, `context mount/unmount/list` |
| Health | `health check/status/trend`, `doctor`, `context repair --dry-run`, `context validate` |
| Sessions | `session bootstrap/pack/prepare-client/handoff/replay`, `cache` |
| Coordination | `tasks`, `hivemind`, `mission`, `approvals`, `work`, `calibration` |
| Agents | `agents list/ps/run/watch`, `agent-jobs`, `agent-runs`, `agent-hooks`, `agent-manifest` |
| Quality | `verify plan/run`, `schema list/show/validate`, `review`, `optimize decide`, `execution inspect` |
| Knowledge | `memory consolidate/status/search`, `skills list/match/mine/promote`, `events`, `graph`, `sources` |
| Serving | `mcp serve`, `plugins`, `services`, `orchestrator` |
| Integrations | `personal`, `briefing`, `claude`, `gemini`, `antigravity`, `gws`, `studio`, `training` |

## Conventions

- Most commands take `--path <workspace>` (resolves the context), `--json`
  (machine-readable), and `--config <afs.toml>`
- `afs help <command>` or `afs <command> --help` for full flags
- `afs guide` prints friendly workflow guides; `afs manager` opens the GUI
- Deeper usage lives in dedicated skills: session-workflows, context-search,
  health-repair, mission-tracking, skill-authoring, extension-authoring
