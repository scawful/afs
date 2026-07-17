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
| Setup | `start`, `projects`, `init`, `profile`, `bundle`, `workspace` |
| Search | `search`, `query` (v1), `index rebuild`, `embeddings`, `watch` |
| Files | `files read/write/list/move/delete`, `notes`, `context mount/unmount/list` |
| Health | `check`, `repair`, `context validate` |
| Sessions | `start`, `session pack/prepare-client/replay`, `handoff`, `cache` |
| Coordination | `messages`, `tasks`, `missions`, `approvals`, `work`, `calibration` |
| Agents | `agents list/ps/run/watch`, `jobs`, `agent-runs`, `agent-hooks`, `agent-manifest` |
| Quality | `verify plan/run`, `schema list/show/validate`, `review`, `optimize decide`, `execution inspect` |
| Knowledge | `memory consolidate/status/search`, `skills list/match/mine/promote`, `insights research/reflect/list/show/accept/reject`, `events`, `graph`, `sources` |
| Serving | `mcp serve`, `plugins`, `services`, `orchestrator` |
| Integrations | `personal`, `briefing`, `claude`, `gemini`, `antigravity`, `gws`, `studio`, `training` |

## Conventions

- Most commands take `--path <workspace>` (resolves the context), `--json`
  (machine-readable), and `--config <afs.toml>`
- `afs help <command>` or `afs <command> --help` for full flags
- `afs guide` prints friendly workflow guides; `afs manager` opens the GUI
- Deeper usage lives in dedicated skills: session-workflows, context-search,
  health-repair, mission-tracking, skill-authoring, extension-authoring
