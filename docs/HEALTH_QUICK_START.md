# Health Quick Start

Use AFS health commands to diagnose profile/context/extension issues quickly.

## One-Command Snapshot

```bash
afs health
```

This reports:

- active profile
- context path and mount counts
- monorepo bridge freshness
- embedding index age summary
- extension/hook status
- MCP status

## JSON Output

```bash
afs health --json
```

## Extended Checks

```bash
afs health check --level basic
afs health check --level standard
afs health check --level comprehensive
```

## Monitor Mode

```bash
afs health monitor --interval 60
```

## History and Trend

```bash
afs health history --limit 10
afs health trend --hours 24
```

## Typical Workflow

1. Run `afs health`.
2. If profile or mounts look wrong, run `afs profile current` and `afs context profile-show`.
3. If monorepo bridge is stale, refresh `monorepo/active_workspace.toml` via workspace switch hook.
4. If MCP tools look wrong, check `afs mcp serve` and extension `[mcp_tools]` config.
