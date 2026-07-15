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

For MCP clients, prefer `context.read`, `context.write`, `context.list`,
`context.move`, and `context.delete`. The older `fs.*` tool names remain
compatibility aliases. The CLI still uses `afs fs ...`.

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

- `context.list` — list files: `{"path": "/path/to/project/.context/knowledge", "max_depth": 2}`
- `context.read` — read file: `{"path": "/path/to/project/.context/memory/notes.md"}`
- `context.write` — write file: `{"path": "/path/to/project/.context/scratchpad/draft.md", "content": "...", "mkdirs": true}`
- `context.delete` — delete file: `{"path": "/path/to/project/.context/scratchpad/old.md"}`
- `context.move` — move file: `{"source": "/path/to/project/.context/scratchpad/a.md", "destination": "/path/to/project/.context/scratchpad/b.md"}`

## Tips

- `--relative` flag on `fs list` scopes to a subdirectory
- All mounts are readable; only scratchpad/hivemind/items/global are writable
- Sandbox agents can only write to their `allowed_mounts`
- For MCP file tools, prefer absolute paths under the repo-local `.context`
  root instead of mount-relative JSON arguments
