# Workspace Integration Notes

These notes describe generic workspace integration patterns. Keep private
infrastructure details in workspace-specific docs or extensions.

## Source of Truth

- Workspace infrastructure: your workspace documentation repo
- Source universe sync: your workspace inventory or project registry
- Windows/remote workflow: companion extension docs or local runbooks

## Machine Names

Machine names, SSH aliases, and infrastructure roles belong in user or
organization configuration, not core AFS. Use whatever naming convention the
current environment supplies and avoid committing hostnames or addresses to a
portable profile.

## Mounts + Contexts

- Configure remote mount points where the operating environment expects them.
- Do not assume a drive letter, WSL path, home-directory layout, or shared
  network namespace.
- Use repo-local `.context/` or a configured central context root. Do not infer
  one computer's placement from another computer's checkout path.
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
parent that contains companion repos such as `afs_example` or `afs_company`. AFS
prefers earlier extension roots over later defaults, so a work-local install can
safely override an older `~/.config/afs/extensions/<name>` copy with the same
extension name.

When a workspace path under one of those roots moves, `afs context repair` and the
background `context-warm` / `context-watch` services will try a conservative
remap against registered workspace roots before leaving the mount broken. This
works best when the real workspace roots are listed in
`general.workspace_directories`, not just broad allowed roots.

## Tooling

- Use `afs` for context operations and mounts.
- Workspace navigation tools are optional; AFS does not require `ws` or a
  particular source root.
- Use `afs workspace sync --root /path/to/workspace` only when that root has a
  `WORKSPACE.toml` inventory you want AFS to mirror.

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
