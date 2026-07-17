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

Read, write, and list files in the current project scope.

For MCP clients, prefer `context.read`, `context.write`, `context.list`,
`context.move`, and `context.delete`. The older `fs.*` tool names remain
compatibility aliases. Prefer the plain `afs files ...` CLI spelling.

## Commands

| Command | Description |
|---------|-------------|
| `afs files list <category>` | List files in the current project scope |
| `afs files read <category> <path>` | Read a scoped file |
| `afs files write <category> <path> --content ...` | Write a scoped file |

## Version 2 categories

| Mount | Purpose | Policy |
|-------|---------|--------|
| `memory` | Long-term agent memory | read-only |
| `knowledge` | Reference material | read-only |
| `scratchpad` | Working scratch space | writable |
| `tools` | Executable tools/scripts | executable |
| `history` | CLI invocation logs | read-only |
| `human` | Human intent and decisions | policy-controlled |

Internal queues, indexes, and compatibility mounts live below `.afs`; use
`afs messages`, `afs jobs`, and other domain commands instead of raw file access.

## MCP Tools

- Pass both the central `context_path` and registered `project_path`.
- Keep category-relative paths such as `scratchpad/draft.md`; AFS resolves the
  project scope and rejects absolute paths belonging to another project.

## Tips

- `--relative` on `files list` scopes to a subdirectory
- Use `--common` only for intentionally shared category files
- Sandbox agents can only write to their `allowed_mounts`
