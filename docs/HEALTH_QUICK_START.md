# Health Quick Start

Use AFS health commands to diagnose profile/context/extension issues quickly.

## One-Command Snapshot

```bash
./scripts/afs health
```

This reports:

- active profile
- context path and mount counts
- mount health: broken symlinks, duplicate sources, missing profile-managed mounts
- provenance health: untracked mounts, stale provenance records, remapped profile mounts
- monorepo bridge freshness
- embedding index age summary
- extension/hook status
- MCP status
- maintenance status for `context-warm`, `context-watch`, `agent-supervisor`, and `gemini-workspace-brief`
- supervisor state: running, failed, manually stopped, and stale agent state files

## JSON Output

```bash
./scripts/afs health --json
```

## Extended Checks

```bash
./scripts/afs health check --level basic
./scripts/afs health check --level standard
./scripts/afs health check --level comprehensive
```

## Monitor Mode

```bash
./scripts/afs health monitor --interval 60
```

## History and Trend

```bash
./scripts/afs health history --limit 10
./scripts/afs health trend --hours 24
```

## Typical Workflow

1. Run `./scripts/afs health`.
2. If profile or mounts look wrong, run `./scripts/afs profile current` and `./scripts/afs context profile-show`.
3. If `mount_health` or provenance looks wrong, run `./scripts/afs context repair --dry-run`.
4. Apply the repair with `./scripts/afs context repair --rebuild-index`, or let `./scripts/afs services start context-warm` handle it in the background.
5. For continuous repo-local maintenance, start `./scripts/afs services start context-watch`.
6. If you rely on profile-defined background agents, start `./scripts/afs services start agent-supervisor`.
7. If SQLite index health is stale, rebuild with `./scripts/afs index rebuild --path "$PWD" --json`, or have an MCP client call `context.index.rebuild`.
8. If monorepo bridge is stale, refresh `monorepo/active_workspace.toml` via workspace switch hook.
9. If MCP tools look wrong, check `./scripts/afs mcp serve` and extension `[mcp_tools]` / `[mcp_server]` config.
