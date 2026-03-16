# Workspace Integration Notes

These notes are for operating AFS inside the Scawful workspace. Keep details
generic here; refer to workspace docs for specifics.

## Source of Truth

- Workspace infrastructure: `~/src/docs/NERV_INFRASTRUCTURE.md`
- Source universe sync: `~/src/docs/SRC_UNIVERSE_NETWORK.md`
- Windows workflow: `~/src/lab/afs-scawful/docs/WINDOWS_WORKFLOW.md`

## Codenames (no IPs)

- **ORACLE**: macOS primary workstation (`mac`)
- **MECHANICA**: Windows GPU node (`medical-mechanica`)
- **NEXUS**: Linux server (`halext-nj`)

Use SSH host aliases rather than hardcoded IPs.

## Mounts + Contexts

- Use mount points (`~/Mounts/...`) to browse remote filesystems.
- For Windows, prefer `/mnt/d/src` when working in WSL.
- Keep `.context/` local to each machine.
- For Gemini CLI workspaces under `/google`, add `/google` to
  `general.workspace_directories` and `general.mcp_allowed_roots` so MCP path
  validation matches your real work root.

Example:

```toml
[general]
mcp_allowed_roots = ["/google"]

[[general.workspace_directories]]
path = "/google"
description = "Mercurial cloud workspaces"
```

Temporary shell override:

```bash
export AFS_MCP_ALLOWED_ROOTS=/google
```

## Tooling

- Use `ws` for workspace navigation (`ws list`, `ws go`, `ws status`).
- Use `afs` CLI for context operations and mounts.
- Use `afs workspace sync --root ~/src` to mirror `WORKSPACE.toml` paths into AFS discovery.

## Monorepo Bridge

AFS reserves `.context/monorepo/` for workspace bridge state.

Expected file:

- `.context/monorepo/active_workspace.toml`

Recommended pattern:

- let your workspace switch tool update that file on each switch
- use `afs health` to catch stale bridge state
- keep the bridge machine-local instead of committing it into project repos

Template hook:

- `extensions/afs_google/hooks/context-sync-active-workspace.sh`
