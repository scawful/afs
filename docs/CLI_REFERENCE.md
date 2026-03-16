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
- `context.status` for mount counts, profile, and index health

Gemini work-root override:

```bash
export AFS_MCP_ALLOWED_ROOTS=/google
```

Gemini brief agent:

```bash
./scripts/afs agents run gemini-workspace-brief --stdout
./scripts/afs services start gemini-workspace-brief
./scripts/afs agents run claude-orchestrator --prompt "Summarize this repo"
```

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
detects both `python -m afs.mcp_server` and `afs mcp serve` processes.
