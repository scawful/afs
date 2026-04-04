# Agent Surfaces (CLI + MCP)

AFS is CLI-first. The built-in MCP server is the preferred structured tool
surface for Gemini, Antigravity, and other MCP-aware clients.

## Preferred Entry Point

Use the repo wrapper during local development:

```bash
~/src/lab/afs/scripts/afs status
```

Why:

- it sets `AFS_ROOT`
- it adds repo `src/` to `PYTHONPATH`
- it avoids relying on whichever `python` happens to be first on `PATH`

Use the installed `afs` entrypoint only after `pip install -e .` into the
environment the agent actually runs in.

Help:

- `~/src/lab/afs/scripts/afs`
- `~/src/lab/afs/scripts/afs help <command>`
- `~/src/lab/afs/scripts/afs <command> --help`

## Shell Setup

For interactive shells:

```bash
source ~/src/lab/afs/scripts/afs-shell-init.sh
```

This exports:

- `AFS_ROOT`
- `AFS_CLI`
- `PATH` including `~/src/lab/afs/scripts`

## Venv Setup

Bootstrap a repo-local venv:

```bash
~/src/lab/afs/scripts/afs-venv
```

Optional extras:

```bash
AFS_VENV_EXTRAS=test ~/src/lab/afs/scripts/afs-venv
```

For non-interactive agents:

```bash
export AFS_CLI=~/src/lab/afs/scripts/afs
export AFS_VENV=~/src/lab/afs/.venv
```

## Useful Agent Commands

```bash
~/src/lab/afs/scripts/afs context discover --path ~/src
~/src/lab/afs/scripts/afs context ensure-all --path ~/src
~/src/lab/afs/scripts/afs session bootstrap --json
~/src/lab/afs/scripts/afs session prepare-client --client codex --json
~/src/lab/afs/scripts/afs session hook session_start --client codex --session-id "$AFS_SESSION_ID"
~/src/lab/afs/scripts/afs session event user_prompt_submit --client codex --session-id "$AFS_SESSION_ID" --prompt "current task"
~/src/lab/afs/scripts/afs-session-notify task_created --task-id bg-1 --task-title "Index context"
~/src/lab/afs/scripts/afs events tail --json
~/src/lab/afs/scripts/afs claude setup --path ~/src/project-a
~/src/lab/afs/scripts/afs claude setup --scope user
~/src/lab/afs/scripts/afs claude doctor
~/src/lab/afs/scripts/afs claude reap --limit 20
~/src/lab/afs/scripts/afs doctor
~/src/lab/afs/scripts/afs profile current
~/src/lab/afs/scripts/afs skills list --profile work
~/src/lab/afs/scripts/afs health
```

Warm context/cache:

```bash
~/src/lab/afs/scripts/afs-warm
```

Operational repair entrypoint:

```bash
~/src/lab/afs/scripts/afs doctor
~/src/lab/afs/scripts/afs doctor --fix
```

`afs doctor` is the operator-facing diagnostic surface. It checks config,
active context health, mount provenance, index freshness, extension loading,
MCP registration, and configured auto-start maintenance services. The built-in
MCP server reuses a lighter startup subset of these checks so clients get
warnings without paying for the full operator sweep on every launch.

Agent contract:

- `~/.context/AFS_SPEC.md`
- `./AGENTS.md`
- `./docs/PROFILES.md`

## MCP

Run the built-in stdio MCP server:

```bash
~/src/lab/afs/scripts/afs mcp serve
```

Built-in tools:

Preferred agent-facing file operations:

- `context.read`
- `context.write`
- `context.delete`
- `context.move`
- `context.list`

Legacy compatibility aliases:

- `fs.read`
- `fs.write`
- `fs.delete`
- `fs.move`
- `fs.list`
- `context.discover`
- `context.init`
- `context.mount`
- `context.unmount`
- `context.index.rebuild`
- `context.query`
- `context.diff`
- `context.status`
- `context.repair`
- `session.pack`
- `events.query`
- `events.tail`
- `events.analytics`
- `events.replay`
- `hivemind.reap`
- `handoff.create`
- `handoff.read`
- `handoff.list`

Paths are scoped to:

- `~/.context`
- configured `general.context_root`
- configured `general.workspace_directories`
- configured `general.mcp_allowed_roots`
- `AFS_MCP_ALLOWED_ROOTS`
- local project `.context`

`context.init` is intended for Gemini-style project bootstrap:

- local project init when the target project is under the current working directory
- init under configured workspace roots
- explicit `context_root` under an allowed root for centralized/shared contexts

Gemini-friendly prompts/resources are also exposed over MCP:

- prompts: `afs.session.bootstrap`, `afs.session.pack`, `afs.workflow.structured`, `afs.context.overview`, `afs.query.search`, `afs.scratchpad.review`
- resources: `afs://contexts`, `afs://claude/bootstrap`, `afs://context/<path>/bootstrap`, `.../metadata`, `.../mounts`, `.../index`
- schema resources: `afs://schemas/plan`, `afs://schemas/file-shortlist`, `afs://schemas/review-findings`, `afs://schemas/edit-intent`, `afs://schemas/verification-summary`, `afs://schemas/handoff-summary`

`afs.session.bootstrap` is the preferred start-of-session surface. It combines:

- context health and index freshness
- recent filesystem drift from `context.diff`
- scratchpad state and deferred notes
- queued `items` tasks
- recent `hivemind` messages
- latest durable memory summary

The CLI equivalent is:

```bash
~/src/lab/afs/scripts/afs session bootstrap
~/src/lab/afs/scripts/afs session bootstrap --json
~/src/lab/afs/scripts/afs session pack
~/src/lab/afs/scripts/afs session prepare-client --client codex --json
~/src/lab/afs/scripts/afs session event task_created --client codex --session-id "$AFS_SESSION_ID" --task-id bg-1 --task-title "Index context"
~/src/lab/afs/scripts/afs session pack "sqlite" --model gemini --workflow scan_fast --task "Shortlist the relevant SQLite files"
~/src/lab/afs/scripts/afs session pack "sqlite" --model gemini --pack-mode retrieval
~/src/lab/afs/scripts/afs session pack --model gemini --pack-mode full_slice
~/src/lab/afs/scripts/afs session pack "sqlite" --model codex --json
~/src/lab/afs/scripts/afs events analytics --hours 24 --json
~/src/lab/afs/scripts/afs events replay --session-id "$AFS_SESSION_ID"
```

The CLI also refreshes:

- `.context/scratchpad/afs_agents/session_bootstrap.json`
- `.context/scratchpad/afs_agents/session_bootstrap.md`
- `.context/scratchpad/afs_agents/session_pack_<model>.json`
- `.context/scratchpad/afs_agents/session_pack_<model>.md`
- `.context/scratchpad/afs_agents/session_client_<client>.json`
- `.context/scratchpad/afs_agents/session_skills_<client>.json`

`session pack` is an explicit follow-on step, not the default startup path.
When the bootstrap snapshot and pack inputs have not changed, repeated calls
reuse the stored pack artifact instead of rebuilding all sections. Packs now
also carry an `execution_profile` block, a task-at-end suffix via `--task`, and
a stable `cache.prefix_hash` for adapter-side cache reuse work. `--pack-mode`
lets callers choose between the normal focused pack, a query-first retrieval
pack, and a broader full-slice pack for long-context models. The
`execution_profile` now also spells out that `afs.workflow.structured` is a
prompt-only rail and includes retry guidance so the host loop stays in Gemini
CLI or Claude Code instead of moving into core AFS.

`session prepare-client` packages the same bootstrap, pack, skill, and prompt
surfaces into a single JSON artifact for wrappers and IDE adapters.
`afs-client-session` exports the resulting `AFS_SESSION_BOOTSTRAP_*`,
`AFS_SESSION_PACK_*`, `AFS_SESSION_SKILLS_JSON`,
`AFS_SESSION_SYSTEM_PROMPT_*`, `AFS_SESSION_CLIENT_PAYLOAD_JSON`, and
`AFS_SESSION_EVENT_BIN` variables, then fires `session_start` / `session_end`
hooks around the client run. By default it also hands the prompt artifact to
the native client surface when available: Codex via
`-c model_instructions_file=...`, Claude via `--append-system-prompt-file`,
and Gemini via `GEMINI_SYSTEM_MD`. Set `AFS_CLIENT_NATIVE_PROMPT=0` or the
client-specific `AFS_<CLIENT>_NATIVE_PROMPT=0` to disable that native handoff.
When the wrapper is launched with `--prompt`, `--prompt-file`, or `--turn-id`,
it also exports `AFS_SESSION_DEFAULT_TURN_ID` and emits
`user_prompt_submit`, `turn_started`, and `turn_completed` / `turn_failed`
around the client invocation.

`session event` is the harness-side write path for prompt, turn, and task
lifecycle. It appends durable `session` history events and updates the live
`session_client_<client>.json` payload with `activity.current_prompt`,
`activity.current_turn`, `activity.active_tasks`, counters, and a rolling
`recent_events` list. `afs-session-notify` is the shell-facing helper that
reuses the wrapper-exported client, session, payload, and default turn
metadata so child scripts only need to provide the event-specific fields.

For noisy command output, use MCP tool `operator.digest` before pasting raw
logs back into a model turn. It can auto-detect and compress `pytest`,
`traceback`, `grep`, and `git diff --stat` style output into a compact summary
plus structured fields, and it is included in the default, readonly, repair,
and edit-oriented tool bundles.

For schema-bound plan or verification work, use prompt `afs.workflow.structured`.
It inlines one built-in response schema together with a normal `session.pack`
payload so the model can stay inside a small JSON contract without the caller
manually stitching the prompt together.

Extensions can add their own MCP tools, prompts, and resources with
`[mcp_server]` in `extension.toml`. Legacy tool-only factories under
`[mcp_tools]` still work. Core `afs://contexts`, `afs://context/...`, and the
built-in `afs.*` prompts remain reserved.

Gemini work setup example:

```toml
[general]
mcp_allowed_roots = ["~/workspaces/company"]

[[general.workspace_directories]]
path = "~/workspaces/company"
description = "Managed workspace root"
```

Session-only override:

```bash
export AFS_MCP_ALLOWED_ROOTS=~/workspaces/company
```

Gemini background agent surfaces:

```bash
~/src/lab/afs/scripts/afs agents run gemini-workspace-brief --stdout
~/src/lab/afs/scripts/afs agents run history-memory --stdout
~/src/lab/afs/scripts/afs agents ps --all
~/src/lab/afs/scripts/afs services start gemini-workspace-brief
~/src/lab/afs/scripts/afs services start history-memory
~/src/lab/afs/scripts/afs services start agent-supervisor
~/src/lab/afs/scripts/afs agents run claude-orchestrator --prompt "Summarize this workspace"
```

The brief agent writes JSON and Markdown summaries under
`.context/scratchpad/afs_agents/` and requires `GEMINI_API_KEY` or
`GOOGLE_API_KEY`. `claude-orchestrator` is now a built-in agent surface and can
be listed with `afs agents list`.

`context-warm` is the background maintenance surface for contexts:

```bash
~/src/lab/afs/scripts/afs agents run context-warm --stdout
~/src/lab/afs/scripts/afs services start context-warm
~/src/lab/afs/scripts/afs services start context-watch
```

Each run now audits discovered contexts for:

- broken symlink mounts
- duplicate aliases that point at the same source
- missing or mismatched profile-managed mounts
- untracked or stale mount provenance
- empty or stale SQLite indexes

The built-in `context-warm` service now runs with `--repair-mounts
--rebuild-stale-indexes --doctor-snapshot` by default, so it can seed provenance, remap missing
sources conservatively, and reapply profile-managed mounts when the configured
source still exists.

`context-watch` is the event-driven companion surface. It uses `context-warm
--watch` to watch the context root and mounted source paths, then reruns repair
and index maintenance only for affected contexts. If the optional `watchfiles`
package is unavailable, it degrades to polling.

If you need service processes to use a repo-local or workspace-local config
instead of `~/.config/afs/config.toml`, start them with an explicit config:

```bash
~/src/lab/afs/scripts/afs services start --config /path/to/afs.toml context-warm
~/src/lab/afs/scripts/afs services start --config /path/to/afs.toml agent-supervisor
```

Managed units can also be installed through the OS service adapter:

```bash
~/src/lab/afs/scripts/afs services install context-warm --enable
~/src/lab/afs/scripts/afs services status --system
~/src/lab/afs/scripts/afs services logs context-warm
```

`afs services render|start|stop|status|restart` now preserve that explicit
`AFS_CONFIG_PATH` for the spawned service process, so background maintenance can
stay pinned to a repo-local `.context` such as `~/src/lab/.context`.

For the built-in `context-warm` and `context-watch` services, you can scope the
watched/audited contexts declaratively without replacing the whole command:

```toml
[services.services.context-watch]
context_filters = ["~/src/lab"]
```

`agent-supervisor` is the process reconciler for profile-defined background
agents. It applies:

- `auto_start`
- interval-style `schedule` values such as `5m`, `1h`, `daily`, or `weekly`
- `watch_paths` change detection

By default it stores state under
`.context/scratchpad/afs_agents/supervisor/`, which makes repo-local and
context-local configs safer than a single user-global state cache.

`history-memory` is the canonical durable-memory surface. It incrementally
consolidates new `history/` events into compact summaries in `memory/` and
tracks progress with a context-scoped checkpoint under
`.context/scratchpad/afs_agents/`. By default it summarizes low-sensitivity
event types (`context`, `fs`, `hook`, `review`, and `agent_progress`) rather
than copying raw payloads into memory.

Direct repair surface:

```bash
~/src/lab/afs/scripts/afs context repair --dry-run
~/src/lab/afs/scripts/afs context repair --rebuild-index
```

## Gemini / Claude / Codex Registration

Recommended command target:

```bash
~/src/lab/afs/scripts/afs mcp serve
```

Codex user config:

```toml
[mcp_servers.afs]
command = "$AFS_ROOT/scripts/afs"
args = ["mcp", "serve"]
```

Claude JSON config:

```json
{
  "mcpServers": {
    "afs": {
      "command": "$AFS_ROOT/.venv/bin/python",
      "args": ["-m", "afs.mcp_server"],
      "env": {
        "AFS_ROOT": "$AFS_ROOT",
        "AFS_VENV": "$AFS_ROOT/.venv",
        "PYTHONPATH": "$AFS_ROOT/src"
      }
    }
  }
}
```

Or let AFS write the user-level config for you:

```bash
~/src/lab/afs/scripts/afs claude setup --scope user
```

Claude session maintenance:

```bash
~/src/lab/afs/scripts/afs claude doctor --json
~/src/lab/afs/scripts/afs claude reap --limit 20          # dry-run
~/src/lab/afs/scripts/afs claude reap --limit 20 --apply  # archive candidates
```

Client bootstrap wrappers:

```bash
~/src/lab/afs/scripts/afs-gemini
~/src/lab/afs/scripts/afs-claude
~/src/lab/afs/scripts/afs-codex
```

Each wrapper:

- prefers the nearest repo-local `afs.toml`
- exports a shared `AFS_SESSION_ID`
- exports bootstrap, pack, skills, and combined session payload artifact paths
- refreshes the prepared session payload before launching the client
- runs `session_start` / `session_end` hooks around the client lifecycle
- never infers workspace roots on its own
- maps `AFS_<CLIENT>_MCP_ALLOWED_ROOTS` or `AFS_CLIENT_MCP_ALLOWED_ROOTS` into `AFS_MCP_ALLOWED_ROOTS` when you set them

Gemini registration helper:

```bash
~/src/lab/afs/scripts/afs gemini setup
~/src/lab/afs/scripts/afs gemini setup --scope project
```

The default Gemini entry uses `scripts/afs mcp serve` with repo runtime env so
the MCP server resolves local source checkouts reliably. Use `--python-module`
only when you explicitly want `python -m afs.mcp_server`.

Antigravity raw config example:

```json
{
  "mcpServers": {
    "afs": {
      "command": "$AFS_ROOT/scripts/afs",
      "args": ["mcp", "serve"]
    }
  }
}
```

If the client requires a Python module entrypoint instead, use a Python
environment where `afs` is installed and run `python3 -m afs.mcp_server`.

For the VS Code extension, `AFS: Register MCP Server` checks the configured
Antigravity context-root candidates first. Set `afs.mcp.configPath` when your
fork stores the raw MCP config in a nonstandard location.

`afs health` checks Gemini, Claude, and Codex registration files, recognizes
wrapper-style `afs mcp serve` processes, and surfaces context mount drift such
as broken symlinks, missing profile-managed mounts, provenance drift, and
maintenance service/report state.
