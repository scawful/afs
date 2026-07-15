---
name: hivemind-comms
triggers:
  - hivemind
  - message
  - communicate
  - bus
profiles:
  - general
requires:
  - afs
---

# Hivemind Communication

Inter-agent message passing via the hivemind mount.

## CLI

```bash
afs hivemind list              # show recent messages
afs hivemind list --limit 50   # show more
```

## MCP Tools

- `hivemind.send` — post a message (finding, request, status)
- `hivemind.read` — read messages with optional filters

## Message Types

| Type | Purpose |
|------|---------|
| `finding` | Agent discovered something worth sharing |
| `request` | Agent needs help or input from another |
| `status` | Progress or state update |

## Python API

```python
from afs.hivemind import HivemindBus
from pathlib import Path

bus = HivemindBus(Path(".context"))
bus.send("my-agent", "finding", {"key": "value"})
messages = bus.read(agent_name="my-agent", limit=10)
```

## Storage

Messages are JSON files in `.context/hivemind/<agent>/<timestamp>-<uuid>.json`.
Old messages auto-cleanup after 24h via `bus.cleanup()`.
