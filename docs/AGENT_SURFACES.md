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
~/src/lab/afs/scripts/afs profile current
~/src/lab/afs/scripts/afs skills list --profile work
~/src/lab/afs/scripts/afs health
```

Warm context/cache:

```bash
~/src/lab/afs/scripts/afs-warm
```

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

Paths are scoped to:

- `~/.context`
- configured `general.context_root`
- configured `general.agent_workspaces_dir`
- configured `general.workspace_directories`
- configured `general.mcp_allowed_roots`
- `AFS_MCP_ALLOWED_ROOTS`
- local project `.context`

`context.init` is intended for Gemini-style project bootstrap:

- local project init when the target project is under the current working directory
- init under configured workspace roots such as `/google`
- explicit `context_root` under an allowed root for centralized/shared contexts

Gemini-friendly prompts/resources are also exposed over MCP:

- prompts: `afs.context.overview`, `afs.query.search`, `afs.scratchpad.review`
- resources: `afs://contexts`, `afs://context/<path>/metadata`, `.../mounts`, `.../index`

Gemini work setup example:

```toml
[general]
mcp_allowed_roots = ["/google"]

[[general.workspace_directories]]
path = "/google"
description = "Mercurial cloud workspaces"
```

Session-only override:

```bash
export AFS_MCP_ALLOWED_ROOTS=/google
```

Gemini background agent surfaces:

```bash
~/src/lab/afs/scripts/afs agents run gemini-workspace-brief --stdout
~/src/lab/afs/scripts/afs services start gemini-workspace-brief
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
--rebuild-stale-indexes` by default, so it can seed provenance, remap missing
sources conservatively, and reapply profile-managed mounts when the configured
source still exists.

`context-watch` is the event-driven companion surface. It uses `context-warm
--watch` to watch the context root and mounted source paths, then reruns repair
and index maintenance only for affected contexts. If the optional `watchfiles`
package is unavailable, it degrades to polling.

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
command = "/Users/scawful/src/lab/afs/scripts/afs"
args = ["mcp", "serve"]
```

Claude JSON config:

```json
{
  "mcpServers": {
    "afs": {
      "command": "/Users/scawful/src/lab/afs/scripts/afs",
      "args": ["mcp", "serve"]
    }
  }
}
```

Antigravity raw config example:

```json
{
  "mcpServers": {
    "afs": {
      "command": "/Users/scawful/src/lab/afs/scripts/afs",
      "args": ["mcp", "serve"]
    }
  }
}
```

If the client requires a Python module entrypoint instead, use a Python
environment where `afs` is installed and run `python3 -m afs.mcp_server`.

`afs health` checks Gemini, Claude, and Codex registration files, recognizes
wrapper-style `afs mcp serve` processes, and surfaces context mount drift such
as broken symlinks, missing profile-managed mounts, provenance drift, and
maintenance service/report state.
