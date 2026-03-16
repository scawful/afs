# AFS CLI Reference

## Invocation

Preferred during local development:

- `./scripts/afs <command>`

Also supported once installed into the active environment:

- `afs <command>`

## Quickstart

- `./scripts/afs`
- `./scripts/afs help context`
- `./scripts/afs init --context-root ~/.context --workspace-name src`
- `./scripts/afs status`
- `./scripts/afs status --json`
- `./scripts/afs context init --path ~/src`
- `./scripts/afs context discover --path ~/src`
- `./scripts/afs context ensure-all --path ~/src`
- `./scripts/afs graph export --path ~/src`

## Profiles

```bash
./scripts/afs profile current
./scripts/afs profile list
./scripts/afs profile switch work
```

## Context

```bash
./scripts/afs context init
./scripts/afs context ensure
./scripts/afs context list
./scripts/afs context validate
./scripts/afs context repair --dry-run
./scripts/afs context mount knowledge ~/src/docs --alias docs
./scripts/afs context unmount knowledge docs
```

## Workspace

```bash
./scripts/afs workspace list
./scripts/afs workspace add ~/src/project-a --description "project-a"
./scripts/afs workspace remove ~/src/project-a
./scripts/afs workspace sync --root ~/src
```

## Plugins and Extensions

```bash
./scripts/afs plugins --details
./scripts/afs plugins --json
```

## Skills

```bash
./scripts/afs skills list --profile work
./scripts/afs skills match "mcp context mount" --profile work
```

## Embeddings

```bash
./scripts/afs embeddings index --knowledge-dir ~/.context/knowledge/work --source ~/.context/knowledge/work
./scripts/afs embeddings search "workspace policy" --knowledge-dir ~/.context/knowledge/work
```

## MCP

```bash
./scripts/afs mcp serve
```

Useful Gemini-oriented MCP operations:

- `context.query` for indexed path/content search
- `context.diff` for “what changed since the last index build”
- `context.status` for mount counts, mount health, profile, and index health
- `context.repair` for provenance seeding, conservative source remapping, and stale index repair

Gemini work-root override:

```bash
export AFS_MCP_ALLOWED_ROOTS=/google
```

Gemini brief agent:

```bash
./scripts/afs agents run gemini-workspace-brief --stdout
./scripts/afs services start gemini-workspace-brief
./scripts/afs agents run claude-orchestrator --prompt "Summarize this repo"
./scripts/afs services start context-warm
```

`context-warm` now audits each discovered context for broken symlink mounts,
duplicate mount targets, missing profile-managed mounts, untracked/stale mount
provenance, and stale SQLite indexes. The built-in service now runs with
`--repair-mounts --rebuild-stale-indexes` by default.

For continuous maintenance, start the watcher:

```bash
./scripts/afs services start context-watch
```

`context-watch` uses `context-warm --watch` and reacts to changes under the
context root and mounted source paths. If the optional `watchfiles` package is
not installed, it falls back to polling.

Codex MCP config:

```toml
[mcp_servers.afs]
command = "/Users/scawful/src/lab/afs/scripts/afs"
args = ["mcp", "serve"]
```

## Health

```bash
./scripts/afs health
./scripts/afs health --json
./scripts/afs health check --level standard
```

`afs health` reports AFS MCP registration for Gemini, Claude, and Codex, and it
detects both `python -m afs.mcp_server` and `afs mcp serve` processes. It also
reports broken mounts, duplicate mount targets, provenance drift, repair/remap
activity, and maintenance service state so you can see context drift without
opening the directory manually.

Repair a context directly when you want an explicit fix instead of waiting for a
background service:

```bash
./scripts/afs context repair --dry-run
./scripts/afs context repair --rebuild-index
```
