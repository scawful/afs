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

## MCP Tools

When running under MCP (`afs mcp serve`), agents can be managed via:

- `agent.spawn` — start a background agent
- `agent.ps` — list running agents
- `agent.stop` — stop an agent by name

## Agent Lifecycle

```
stopped -> running -> stopped
              |
              v
           failed
              |
              v
       awaiting_review -> stopped (approved)
                       -> failed  (rejected)
```

## Tips

- Agent state dir defaults to `<context_root>/scratchpad/afs_agents/supervisor/`
- Use `afs agents watch <name>` to see progress events from history
- Review gates pause agent work until human approval
- Per-agent sandbox: set `allowed_mounts` and `allowed_tools` in config
