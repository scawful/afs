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

## Bundles

```bash
./scripts/afs bundle pack work --output ./dist
./scripts/afs bundle inspect ./dist/work
./scripts/afs bundle install ./dist/work --install-dir ./.afs/extensions
./scripts/afs bundle list
```

`bundle install` now writes:

- an installable extension under the chosen extension root
- generated MCP/agent shims when the bundled profile defines `mcp_tools` or `agent_configs`
- `profile-snippet.toml` you can merge into `afs.toml` to re-enable the bundled profile semantics safely

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

## Review

`review` now operates on the active context instead of a separate
`~/.context/projects/...` tree. Pending drafts live under the context
scratchpad at `review/<category>/`.

```bash
./scripts/afs review list --path ~/src/project-a
./scripts/afs review approve --path ~/src/project-a draft.md
./scripts/afs review reject --path ~/src/project-a draft.md --reason "needs revision"
```

Approved plans move into `memory/reviewed/plans/`. Other approved review
documents move into `history/reviewed/<category>/`. Rejections are archived
under `history/rejected/<category>/`.

For compatibility, `./scripts/afs review approve project-a draft.md` still
works when `project-a` can be resolved from configured
`general.workspace_directories`.

## Memory

```bash
./scripts/afs memory consolidate --path ~/src/project-a
./scripts/afs memory consolidate --path ~/src/project-a --json
./scripts/afs agents run history-memory --stdout
./scripts/afs services start history-memory
```

`memory consolidate` is the canonical history-to-memory step. It reads new
metadata-first history events, writes durable summaries into
`memory/entries.jsonl`, writes markdown summaries into
`memory/history_consolidation/`, and checkpoints incremental progress under
`.context/scratchpad/afs_agents/history_memory_checkpoint.json`.

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
./scripts/afs agents ps --all
./scripts/afs agents ps --all --json
./scripts/afs services start gemini-workspace-brief
./scripts/afs services start agent-supervisor
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

If you have profile-driven background agents with `auto_start`, `schedule`, or
`watch_paths`, start the supervisor too:

```bash
./scripts/afs services start agent-supervisor
./scripts/afs services start history-memory
```

The supervisor stores state under
`.context/scratchpad/afs_agents/supervisor/` by default, so repo- or
context-scoped configs do not get shadowed by a single global PID cache.

If you want background services to stay pinned to a repo-local config and
`.context`, start them with `--config`:

```bash
./scripts/afs services start --config /path/to/afs.toml context-warm
./scripts/afs services start --config /path/to/afs.toml agent-supervisor
```

`afs services render|start|stop|status|restart` preserve that explicit
`AFS_CONFIG_PATH` for the spawned service process. `afs status` and
`afs health` also surface the `history-memory` maintenance report alongside
`context-warm`, `context-watch`, and `agent-supervisor`.

`context-watch` uses `context-warm --watch` and reacts to changes under the
context root and mounted source paths. If the optional `watchfiles` package is
not installed, it falls back to polling.

If you need the built-in maintenance daemons scoped to a subset of contexts
without overriding the full command, use `context_filters` in your service
config:

```toml
[services.services.context-watch]
context_filters = ["/Users/scawful/src/lab"]
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
detects both `python -m afs.mcp_server` and `afs mcp serve` processes. It also
reports broken mounts, duplicate mount targets, provenance drift, repair/remap
activity, maintenance service state, and supervisor agent state so you can see
context drift or failed background agents without opening the directory
manually.

Repair a context directly when you want an explicit fix instead of waiting for a
background service:

```bash
./scripts/afs context repair --dry-run
./scripts/afs context repair --rebuild-index
```
