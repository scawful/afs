---
name: mcp-server
triggers:
  - mcp
  - serve
  - tool
  - protocol
profiles:
  - general
requires:
  - afs
---

# MCP Server

AFS exposes all context operations as MCP tools over stdio JSON-RPC.

## Starting

```bash
afs mcp serve                          # start MCP server
python -m afs.mcp_server --demo       # demo mode with sample data
python -m afs.mcp_server --verbose    # debug logging to stderr
```

## Built-in Tools

### Context
- `context.discover` — find .context directories
- `context.init` — create .context for a path
- `context.mount` — mount a directory into context
- `context.unmount` — remove a mount
- `context.index.rebuild` — rebuild the context index
- `context.query` — search indexed context
- `context.diff` — diff context state
- `context.status` — show context health
- `context.repair` — fix broken context state

### Filesystem
- `fs.list` — list files in a mount
- `fs.read` — read file contents
- `fs.write` — write file contents
- `fs.delete` — delete a file from a mount
- `fs.move` — move/rename a file within mounts

### Agents
- `agent.spawn` — start a background agent
- `agent.ps` — list running agents
- `agent.stop` — stop an agent
- `agent.logs` — read agent event history

### Communication
- `hivemind.send` — send an inter-agent message
- `hivemind.read` — read hivemind messages

### Tasks
- `task.create` — create a task
- `task.list` — list tasks
- `task.claim` — claim a task
- `task.complete` — complete a task

### Review
- `review.list` — list agents awaiting review
- `review.approve` — approve an agent's work
- `review.reject` — reject an agent's work

## Claude Code Integration

Add to `~/.claude/claude_desktop_config.json`:

```json
{
  "mcpServers": {
    "afs": {
      "command": "/path/to/afs",
      "args": ["mcp", "serve"]
    }
  }
}
```
