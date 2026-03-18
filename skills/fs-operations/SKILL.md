---
name: fs-operations
triggers:
  - read
  - write
  - list
  - file
  - mount
profiles:
  - general
requires:
  - afs
---

# Filesystem Operations

Read, write, and list files across AFS context mounts.

## Commands

| Command | Description |
|---------|-------------|
| `afs fs list <mount>` | List files in a mount |
| `afs fs read <mount> <path>` | Read file from mount |
| `afs fs write <mount> <path> --content ...` | Write to mount |

## Mount Types

| Mount | Purpose | Policy |
|-------|---------|--------|
| `memory` | Long-term agent memory | read-only |
| `knowledge` | Reference material | read-only |
| `scratchpad` | Working scratch space | writable |
| `tools` | Executable tools/scripts | executable |
| `history` | CLI invocation logs | read-only |
| `hivemind` | Inter-agent messages | writable |
| `items` | Task queue items | writable |
| `global` | Shared state, index DB | writable |
| `monorepo` | Multi-project links | read-only |

## MCP Tools

- `fs.list` — list files: `{"mount": "knowledge"}`
- `fs.read` — read file: `{"mount": "memory", "path": "notes.md"}`
- `fs.write` — write file: `{"mount": "scratchpad", "path": "draft.md", "content": "..."}`
- `fs.delete` — delete file: `{"mount": "scratchpad", "path": "old.md"}`
- `fs.move` — move file: `{"mount": "scratchpad", "from": "a.md", "to": "b.md"}`

## Tips

- `--relative` flag on `fs list` scopes to a subdirectory
- All mounts are readable; only scratchpad/hivemind/items/global are writable
- Sandbox agents can only write to their `allowed_mounts`
