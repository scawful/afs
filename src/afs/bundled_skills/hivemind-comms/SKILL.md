---
name: hivemind-comms
triggers:
  - afs messages
  - scoped messages
  - inter-agent message
  - hivemind
profiles:
  - general
requires:
  - afs
---

# Scoped Messages

Project-scoped inter-agent messages. The skill's historical name and the
`afs hivemind` command remain compatibility aliases for one cycle.

## CLI

```bash
afs messages list --path .
afs messages send --path . --from my-agent --type status --payload '{"ok":true}'
```

Normal reads include the current project and `common`; use `--all-projects`
only for an intentional cross-project operation. Queue-wide cleanup also
requires that explicit flag and is a dry run until `--apply` is passed.

## MCP tools

- `messages.send` posts a project/common scoped message.
- `messages.read` reads current-project plus common messages.
- Legacy `hivemind.*` tools are full-catalog compatibility aliases.

## Python API

```python
from pathlib import Path
from afs.messages import MessageBus

bus = MessageBus(Path.home() / ".context", scope_id="project:prj_example")
bus.send("my-agent", "finding", {"key": "value"})
messages = bus.read(agent_name="my-agent", limit=10)
```

Messages are stored below `.context/.afs/queue/messages/` and carry a
`scope_id`; knowing the central context path is not project authorization.
