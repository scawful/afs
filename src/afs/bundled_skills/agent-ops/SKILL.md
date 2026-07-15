---
name: agent-ops
triggers:
  - agent
  - spawn
  - supervisor
  - background
profiles:
  - general
requires:
  - afs
enforcement:
  - Inspect agent and job state before changing it.
  - Never start or stop background agents, enqueue or move jobs, or run a worker without explicit user direction.
  - Never install or load shell or LaunchAgent hooks without explicit user direction.
  - Never promote an agent-job review packet without explicit user direction.
  - Never enable --allow-destructive unless the user explicitly authorizes that worker mode.
---

# Agent Operations

Manage background agent processes via the AFS supervisor.

## Default Set

An empty profile agent list gets `context-warm` (network-free daily audit),
`index-rebuild` (knowledge/memory changes), `skills-mine` (weekly), and
`morning-briefing` (daily interval). Custom lists are unchanged. Disable with
`[agents] default_set = false` or `AFS_DEFAULT_AGENTS=off`. Starting the
supervisor is always an explicit operator action.

## Commands

| Command | Description |
|---------|-------------|
| `afs agents list` | Show available built-in agents |
| `afs agents ps` | Show running background agents |
| `afs agents ps --all` | Include stopped/failed agents |
| `afs agents run <name>` | Run an agent in foreground |
| `afs agents watch <name>` | Tail progress events for an agent |

Full-catalog MCP equivalents are `agent.spawn`, `agent.ps`, and `agent.stop`;
the same explicit-user gates apply.

Lifecycle is `stopped -> running -> stopped|failed`; review-gated work pauses
at `awaiting_review` until the user approves or rejects it.

## Jobs, Runs, and Hooks

| Command | Description |
|---------|-------------|
| `afs agent-jobs list/show/status/inbox/review` | Inspect durable jobs |
| `afs agent-jobs create/seed/claim/move/archive` | Change job state (user-directed) |
| `afs agent-jobs work [--dry-run]` | Preview or invoke a worker |
| `afs agent-jobs promote` | Promote a review packet (user-gated) |
| `afs agent-runs start/list/show/event/finish` | Record agent runs |
| `afs agent-hooks show/status/install-*` | Inspect or explicitly install hooks |
| `afs agent-manifest show/validate/export/sync` | Manage the harness manifest |

## Tips

- State defaults to `<context_root>/scratchpad/afs_agents/supervisor/`.
- Set `allowed_mounts` and `allowed_tools` per agent.
- `--allow-destructive` requires exceptional user authorization.
