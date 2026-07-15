---
name: task-queue
triggers:
  - task
  - queue
  - items
  - todo
profiles:
  - general
requires:
  - afs
---

# Task Queue

File-backed task queue for agent coordination via the items mount.

## CLI

```bash
afs tasks list                 # show all tasks
afs tasks list --status pending  # filter by status
afs tasks list --json          # machine-readable
```

## MCP Tools

- `task.create` — create a new task
- `task.list` — list tasks with optional status filter
- `task.claim` — claim a pending task for an agent
- `task.complete` — mark a task done with result

## Task Lifecycle

```
pending -> claimed -> done
                   -> failed
```

## Python API

```python
from afs.tasks import TaskQueue
from pathlib import Path

queue = TaskQueue(Path(".context"))
task = queue.create("Fix lint errors", created_by="planner", priority=1)
claimed = queue.claim(task.id, "worker-agent")
queue.complete(task.id, result={"files_fixed": 3})
```

## Storage

Tasks are JSON files in `.context/items/task-<uuid>.json`.
Priority 0 = highest, default = 5.
