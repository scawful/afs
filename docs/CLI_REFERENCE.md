# AFS CLI Reference

## Quickstart

- `afs`
- `afs help context`
- `afs init --context-root ~/.context --workspace-name src`
- `afs status`
- `afs context init --path ~/src`
- `afs context discover --path ~/src`
- `afs context ensure-all --path ~/src`
- `afs graph export --path ~/src`

## Profiles

```bash
afs profile current
afs profile list
afs profile switch work
```

## Context

```bash
afs context init
afs context ensure
afs context list
afs context validate
afs context mount knowledge ~/src/docs --alias docs
afs context unmount knowledge docs
```

## Workspace

```bash
afs workspace list
afs workspace add ~/src/project-a --description "project-a"
afs workspace remove ~/src/project-a
afs workspace sync --root ~/src
```

## Plugins and Extensions

```bash
afs plugins --details
afs plugins --json
```

## Skills

```bash
afs skills list --profile work
afs skills match "mcp context mount" --profile work
```

## Embeddings

```bash
afs embeddings index --knowledge-dir ~/.context/knowledge/work
afs embeddings search "workspace policy" --knowledge-dir ~/.context/knowledge/work
```

## MCP

```bash
afs mcp serve
```

## Health

```bash
afs health
afs health --json
afs health check --level standard
```
