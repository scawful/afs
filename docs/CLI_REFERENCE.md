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
- `./scripts/afs doctor`
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

## Session

```bash
./scripts/afs session bootstrap
./scripts/afs session bootstrap --json
./scripts/afs session pack
./scripts/afs session pack "sqlite indexing" --model gemini
./scripts/afs session pack "runtime bug" --model codex --token-budget 12000 --json
```

`session bootstrap` is the preferred start-of-session surface. It combines:

- `context.status`
- `context.diff`
- scratchpad state and deferred notes
- queued tasks from `items/`
- recent `hivemind/` messages
- latest durable memory summary

It also refreshes:

- `.context/scratchpad/afs_agents/session_bootstrap.json`
- `.context/scratchpad/afs_agents/session_bootstrap.md`

`session pack` is the compact follow-on surface when an agent needs a bounded
working set for Gemini, Claude, Codex, or a generic client. It builds a
token-budgeted packet from bootstrap state, scratchpad, queued tasks, hivemind,
durable memory, and indexed retrieval hits, then writes:

- `.context/scratchpad/afs_agents/session_pack_<model>.json`
- `.context/scratchpad/afs_agents/session_pack_<model>.md`

`never_export` sensitivity rules are applied to indexed content included in the
pack, so blocked paths do not leak into session exports.

## Events

```bash
./scripts/afs events tail --json
./scripts/afs events list --type mcp_tool --limit 25
./scripts/afs events list --path ~/src/project-a --source afs.mcp
```

`events` reads the active context history log with the same config/context
resolution as the rest of the CLI.

## Claude

```bash
./scripts/afs claude setup --path ~/src/project-a
./scripts/afs claude context --path ~/src/project-a
./scripts/afs claude session-report --session <uuid> --write-scratchpad
```

`claude setup` writes `project/.claude/settings.json` and `project/CLAUDE.md`
for the resolved project path, not just the current shell directory. When an
`afs.toml` is present, the generated Claude MCP entry pins `AFS_CONFIG_PATH`
and `AFS_PREFER_REPO_CONFIG=1` so Claude uses the repo-local AFS config.

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
# Index with keyword-only (no embedding provider needed)
./scripts/afs embeddings index --knowledge-path ~/.context/knowledge --provider none --include "*.md"

# Index with Gemini vectors (requires GEMINI_API_KEY)
./scripts/afs embeddings index --knowledge-path ~/.context/knowledge --provider gemini --include "*.md"

# Index with other providers
./scripts/afs embeddings index --knowledge-path ~/.context/knowledge --provider ollama
./scripts/afs embeddings index --knowledge-path ~/.context/knowledge --provider openai --model text-embedding-3-small
./scripts/afs embeddings index --knowledge-path ~/.context/knowledge --provider hf --model nomic-embed-text

# Semantic search (auto-uses RETRIEVAL_QUERY for Gemini asymmetric retrieval)
./scripts/afs embeddings search --knowledge-path ~/.context/knowledge --provider gemini "how to debug a sprite"

# Keyword search (no provider needed if index exists)
./scripts/afs embeddings search --knowledge-path ~/.context/knowledge --provider none "sprite RAM tables"

# Evaluate retrieval quality
./scripts/afs embeddings eval --knowledge-path ~/.context/knowledge --provider gemini --query-file eval_cases.jsonl
```

Embedding providers: `none` (keyword-only), `ollama`, `hf` (HuggingFace), `openai`, `gemini`.

For Gemini, the system auto-selects the correct task type: `RETRIEVAL_DOCUMENT` for
indexing, `RETRIEVAL_QUERY` for search queries (asymmetric retrieval). Override with
`--gemini-task-type`.

## Gemini

```bash
# Set up Gemini settings.json with AFS MCP server
./scripts/afs gemini setup
./scripts/afs gemini setup --force                    # overwrite existing entry
./scripts/afs gemini setup --settings-path ~/custom/settings.json

# Check integration health
./scripts/afs gemini status
./scripts/afs gemini status --json
./scripts/afs gemini status --skip-ping               # skip live embedding test
./scripts/afs gemini status --project afs             # inspect one project subtree
./scripts/afs gemini status --context-root ~/src/lab/.context

# Generate context from knowledge base
./scripts/afs gemini context                           # dump full INDEX.md
./scripts/afs gemini context "sprite development"      # search for relevant docs
./scripts/afs gemini context "debugging" --top-k 3     # limit results
./scripts/afs gemini context "hooks" --include-content  # include full doc text
./scripts/afs gemini context "training" --json          # machine-readable output
./scripts/afs gemini context --project afs "sqlite"     # search one project subtree
./scripts/afs gemini context --knowledge-path ~/.context/knowledge/afs "hooks"
```

`afs gemini setup` writes the AFS MCP server entry into `~/.gemini/settings.json`
so Gemini CLI can discover AFS tools automatically.

`afs gemini status` checks: API key, google-genai SDK, settings.json, MCP registration,
embedding index, and live embedding ping.

`afs gemini context` generates context from the knowledge base using semantic search
(when embeddings are indexed) or dumps the full knowledge INDEX.md. When no
`--project` or `--knowledge-path` is given, it searches across every indexed
project subtree under the active context knowledge root.

## Briefing

```bash
./scripts/afs briefing
./scripts/afs briefing --short
./scripts/afs briefing --json
./scripts/afs briefing --org
./scripts/afs briefing --no-gws                        # skip Google Workspace lookups
```

## GWS

```bash
./scripts/afs gws status                               # gws auth status
./scripts/afs gws agenda                               # today's calendar agenda
./scripts/afs gws unread                               # unread primary inbox
./scripts/afs gws raw gmail +triage --output-format json
```

## MCP

```bash
./scripts/afs mcp serve
```

Useful Gemini-oriented MCP operations:

- `afs.session.bootstrap` for the full session-start packet
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
`--repair-mounts --rebuild-stale-indexes --doctor-snapshot` by default.

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

OS-managed service lifecycle is also available:

```bash
./scripts/afs services install context-warm --enable
./scripts/afs services status --system
./scripts/afs services logs context-warm
./scripts/afs services disable context-warm
./scripts/afs services uninstall context-warm
```

`afs services render|install|enable|disable|start|stop|status|restart|logs` preserve that explicit
`AFS_CONFIG_PATH` for the spawned service process. `afs status` and
`afs health` also surface the `history-memory` maintenance report alongside
`context-warm`, `context-watch`, `agent-supervisor`, and the periodic
`doctor_snapshot`.

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

Client launch wrappers:

```bash
./scripts/afs-gemini
./scripts/afs-claude
./scripts/afs-codex
```

These wrappers prefer repo-local config, refresh the session bootstrap packet,
and export the bootstrap artifact paths before launching the client.

## Doctor

```bash
./scripts/afs doctor                               # diagnose all common issues
./scripts/afs doctor --fix                          # auto-apply available fixes
./scripts/afs doctor --json                         # machine-readable output
```

`afs doctor` runs diagnostic checks across the full AFS stack and reports
actionable results. Checks include: Python environment, config loading,
context root integrity, context mount/provenance health, optional dependencies,
MCP registration, embedding indexes, extension loading, context index freshness,
configured auto-start service state, and MCP server build.

When `--fix` is passed, the doctor auto-applies fixes for issues that have
automated remediation (e.g., creating missing context directories, rebuilding
mount structures, seeding/pruning mount provenance, and rebuilding stale
indexes). Issues without auto-fix include suggested manual commands.

The CLI also catches common runtime errors (missing dependencies, file not
found, permission denied) and suggests running `afs doctor` instead of showing
raw tracebacks.

The MCP server runs startup diagnostics on launch and logs warnings/errors to
stderr, so Gemini and other MCP clients can surface context/index/runtime
problems without the server crashing. The startup subset is lighter than the
full `doctor` run and skips operator-only checks like service registration
state.

`context-warm` and `context-watch` now write
`.context/scratchpad/afs_agents/doctor_snapshot.json` so `afs health` can
surface the latest maintenance-time diagnosis even when you have not run the
doctor manually.

## Health

```bash
./scripts/afs health
./scripts/afs health --json
./scripts/afs health check --level standard
```

`afs health` reports AFS MCP registration for Gemini, Claude, and Codex, and it
detects both `python -m afs.mcp_server` and `afs mcp serve` processes. It also
reports broken mounts, duplicate mount targets, provenance drift, repair/remap
activity, recent MCP workflow usage (`afs.session.bootstrap`, `context.status`,
`context.diff`, `context.query`, `session.pack`), maintenance service state,
and supervisor agent state so you can see context drift or failed background
agents without opening the directory manually.

Repair a context directly when you want an explicit fix instead of waiting for a
background service:

```bash
./scripts/afs context repair --dry-run
./scripts/afs context repair --rebuild-index
```
