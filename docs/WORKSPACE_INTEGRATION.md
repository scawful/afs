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

## Tooling

- Use `ws` for workspace navigation (`ws list`, `ws go`, `ws status`).
- Use `afs` CLI for context operations and mounts.
- Use `afs workspace sync --root ~/src` to mirror `WORKSPACE.toml` paths into AFS discovery.
