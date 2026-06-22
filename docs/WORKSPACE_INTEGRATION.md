# Workspace Integration Notes

These notes are for operating AFS inside the Scawful workspace. Keep details
generic here; refer to workspace docs for specifics.

## Source of Truth

- Workspace infrastructure: `<workspace-docs-root>/NERV_INFRASTRUCTURE.md`
- Source universe sync: `<workspace-docs-root>/SRC_UNIVERSE_NETWORK.md`
- Windows workflow: companion repo docs, for example
  `<afs_scawful-root>/docs/WINDOWS_WORKFLOW.md` in this workspace

## Codenames (no IPs)

- **ORACLE**: macOS primary workstation (`mac`)
- **MECHANICA**: Windows GPU node (`remote-gpu`)
- **NEXUS**: Linux server (`remote-server`)

Use SSH host aliases rather than hardcoded IPs.

## Mounts + Contexts

- Use mount points (`~/Mounts/...`) to browse remote filesystems.
- For Windows, prefer `/mnt/d/src` when working in WSL.
- Keep `.context/` local to each machine.
- For Antigravity or Gemini compatibility workspaces under a managed root, add that root to
  `general.workspace_directories` and `general.mcp_allowed_roots` so MCP path
  validation matches your real workspace root.

Example:

```toml
[general]
mcp_allowed_roots = ["~/workspaces/company"]

[[general.workspace_directories]]
path = "~/workspaces/company"
description = "Managed workspace root"
```

Temporary shell override:

```bash
export AFS_MCP_ALLOWED_ROOTS=~/workspaces/company
```

If you want local harness bundles or extensions to stay repo-local instead of
landing in a shared user directory, set `extensions.extension_dirs` to a path
inside the workspace/context or set `extensions.extension_repo_roots` to a
parent that contains companion repos such as `afs_example` or `afs_scawful`. AFS
prefers earlier extension roots over later defaults, so a work-local install can
safely override an older `~/.config/afs/extensions/<name>` copy with the same
extension name.

When a workspace path under one of those roots moves, `afs context repair` and the
background `context-warm` / `context-watch` services will try a conservative
remap against registered workspace roots before leaving the mount broken. This
works best when the real workspace roots are listed in
`general.workspace_directories`, not just broad allowed roots.

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

- `extensions/workspace_adapter/hooks/context-sync-active-workspace.sh`
