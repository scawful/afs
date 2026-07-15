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
| `afs agent-jobs list/show/status/inbox/review` | Inspect queued and reviewable markdown jobs |
| `afs agent-jobs create/seed/claim/move/archive` | Change durable job state (user-directed) |
| `afs agent-jobs work --dry-run` | Preview runnable work before claiming anything |
| `afs agent-jobs work` | Claim jobs and invoke a local worker (user-directed) |
| `afs agent-jobs promote` | Promote a review packet to scratchpad/handoffs (user-gated) |
| `afs agent-runs start/list/show/event/finish` | Record and inspect agent runs |
| `afs agent-hooks show/status` | Preview hooks or inspect installation state |
| `afs agent-hooks install-shell/install-worker` | Install persistent hooks only with `--apply`; `--load` loads the LaunchAgent |
| `afs agent-manifest show/validate/export/sync` | Inspect or export the harness manifest; `sync` applies only with `--apply` |

## Tips

- Agent state dir defaults to `<context_root>/scratchpad/afs_agents/supervisor/`
- Per-agent sandbox: set `allowed_mounts` and `allowed_tools` in config
- Treat `--allow-destructive` as an exceptional user-authorized worker mode,
  never as a way to bypass a blocked job
