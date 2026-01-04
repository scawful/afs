# Agent Surfaces (CLI + MCP)

AFS is CLI-first. MCP is optional for clients that prefer a tool surface.

## CLI (preferred)

Use the installed `afs` entrypoint when available. If it is not on `PATH`, use
the repo wrapper:

```bash
~/src/lab/afs/scripts/afs status
```

The wrapper sets `AFS_ROOT` and `PYTHONPATH` automatically.

## Shell setup (bash/zsh)

Add this to `~/.bashrc` or `~/.zshrc`:

```bash
source ~/src/lab/afs/scripts/afs-shell-init.sh
```

This exports:
- `AFS_ROOT` (repo root)
- `AFS_CLI` (full path to the wrapper)
- `PATH` updated to include `~/src/lab/afs/scripts`

## Agent runtime (non-interactive)

Use one of:
- `AFS_CLI=~/src/lab/afs/scripts/afs`
- `~/src/lab/afs/scripts/afs <command>`
- `PYTHONPATH=~/src/lab/afs/src python -m afs <command>`

Warm context/cache:
- `~/src/lab/afs/scripts/afs-warm`

Agent contract:
- `~/.context/AFS_SPEC.md`

## MCP (optional)

If an MCP client needs direct filesystem access to context state, run a
filesystem MCP server scoped to `~/.context` (and any project `.context`).
Requires an MCP filesystem server installation.

```json
{
  "mcpServers": {
    "afs-context": {
      "command": "mcp-server-filesystem",
      "args": ["~/.context"]
    }
  }
}
```

This exposes context files only; higher-level operations should still use the
AFS CLI.
