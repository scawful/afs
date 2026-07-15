---
name: event-log
triggers:
  - afs events
  - afs event log
  - afs event analytics
  - mcp tool errors
  - afs session timeline
profiles:
  - general
requires:
  - afs
---

# Event Log

Query the append-only AFS event log for telemetry, auditing, and session
timelines.

## Commands

| Command | Description |
|---------|-------------|
| `afs events tail --path <ws>` | Most recent events |
| `afs events list --type <t> --since <iso> --limit N` | Filtered event listing |
| `afs events analytics` | Event volume, MCP tool usage, error rates |
| `afs events replay --session-id <id>` | Replay one recorded session timeline |
| `afs session event` | Record prompt/turn/task lifecycle activity |

## Filters

`list` supports `--type`, `--since` (ISO 8601), `--limit`, `--source`,
`--session-id`, `--json`. Prefer `--json` plus a narrow `--since` window over
dumping the full log.

## Uses

- Debugging: `analytics` shows which MCP tools error most. For raw MCP events,
  use `events list --type mcp_tool --json` and filter their error metadata;
  `--type` is an exact event-type match, not a severity filter
- Auditing: `replay` reconstructs what an agent session actually did
- Skill mining reads this log — a healthy event stream feeds
  `afs skills mine` (see the skill-authoring skill)
- The supervisor's event reactor consumes it: `AgentConfig.on_event` patterns
  (e.g. `error`, `hivemind:context:repair`) start agents or enqueue jobs when
  matching entries appear (see the agent-ops skill)

Events are recorded automatically by MCP tools, session hooks, and CLI flows;
`afs session event` is for client wrappers that need to record activity
explicitly.
