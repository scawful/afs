# AFS MCP Server

AFS provides a lightweight stdio MCP server for context operations.

For Gemini CLI, prefer the MCP server from the repo or project root you want to
work in. That keeps repo-local `context.init` available while preserving the
allowed-root guardrails on all file and context operations.

## Run

```bash
<afs-root>/scripts/afs mcp serve
# or, from an environment where `afs` is installed
afs mcp serve
# or
python3 -m afs.mcp_server
```

## Gemini CLI Registration

Preferred — use the built-in setup command:

```bash
afs gemini setup
afs gemini setup --scope project              # writes ./.gemini/settings.json
afs gemini setup --python-module              # opt into python -m afs.mcp_server
```

By default, `afs gemini setup` writes the local `scripts/afs mcp serve` wrapper
entry and preserves the repo runtime env (`AFS_ROOT`, `AFS_VENV`, `PYTHONPATH`,
and `AFS_PREFER_REPO_CONFIG=1`) so Gemini uses the same import/config path as
the rest of the AFS toolchain.

Use `--scope project` when you want Gemini CLI to keep MCP registration inside
the current repo at `./.gemini/settings.json`. `afs gemini status` detects both
user-level and project-level Gemini configs.

Manual alternative:

```bash
gemini mcp add afs <afs-root>/scripts/afs mcp serve
```

If Gemini is running inside an environment where `afs` is already installed,
`python3 -m afs.mcp_server` also works.

Verify registration:

```bash
afs gemini status
```

## Codex Registration

Codex uses `~/.codex/config.toml`:

```toml
[mcp_servers.afs]
command = "$AFS_ROOT/scripts/afs"
args = ["mcp", "serve"]
```

Project-local overrides also work via `./.codex/config.toml`.

## Claude Registration

Claude-compatible MCP configs use the same `mcpServers` JSON shape. Depending on
the client build, that is typically either `~/.claude/settings.json` or
`~/Library/Application Support/Claude/claude_desktop_config.json`.

For Claude Desktop, prefer launching AFS directly from the repo venv instead of
through the `scripts/afs` shell wrapper. Claude Desktop's MCP transport uses
newline-delimited JSON on stdio, and the direct Python module entrypoint has
been the most reliable path in practice.

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

For the bundled VS Code extension, `AFS: Register MCP Server` now checks the
existing Antigravity context-root candidates before falling back to
workspace `.cursor/mcp.json`. If your fork stores the raw config elsewhere, set
`afs.mcp.configPath` explicitly in editor settings and rerun the command.

AFS can write the user-level Claude config automatically:

```bash
afs claude setup --scope user
```

Project-local setup is also available:

```bash
afs claude setup --path /path/to/project
```

## Troubleshooting

### Claude Desktop `initialize` timeout

Symptom in `~/Library/Logs/Claude/mcp-server-afs.log`:

- `Message from client: {"method":"initialize"...}`
- followed about 60 seconds later by
  `notifications/cancelled ... MCP error -32001: Request timed out`

This usually means the Claude Desktop MCP entry is launching AFS with the wrong
transport assumptions or via a brittle shell wrapper.

Recommended fixes:

1. Use the direct venv Python entrypoint shown above:
   `command=$AFS_ROOT/.venv/bin/python`
   `args=["-m", "afs.mcp_server"]`
2. Preserve the repo env:
   `AFS_ROOT=$AFS_ROOT`
   `AFS_VENV=$AFS_ROOT/.venv`
   `PYTHONPATH=$AFS_ROOT/src`
3. Restart Claude Desktop fully with `Cmd+Q`, not just by closing the window.
4. Re-check `~/Library/Logs/Claude/mcp-server-afs.log`.

Healthy log sequence:

- `Message from client: {"method":"initialize"...}`
- `Message from server: {"jsonrpc":"2.0","id":0,"result":...}`
- `notifications/initialized`
- `tools/list`
- `prompts/list`
- `resources/list`

Notes:

- A warning like `context_index is stale` does not block MCP startup by itself.
- AFS now supports both `Content-Length` framing and newline-delimited JSON on
  stdio so it can interoperate with Claude Desktop and other MCP clients.

## Antigravity Custom Config

In Antigravity, open `MCP Servers -> Manage MCP Servers -> View raw config`, then add:

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

## Tools

Recommended default MCP/profile surface:

- `afs.session.bootstrap`
- `context.status`
- `context.query`
- `context.read`
- `context.write`
- `context.list`
- `context.diff`
- `context.index.rebuild`
- `handoff.create`

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

Optional tools for explicit workflows:

- `context.discover`
- `context.init`
- `context.mount`
- `context.unmount`
- `context.repair`
- `session.pack`
- `agent.spawn`
- `agent.ps`
- `agent.stop`
- `agent.logs`
- `events.query`
- `events.tail`
- `events.analytics`
- `events.replay`
- `hivemind.subscribe`
- `hivemind.unsubscribe`
- `hivemind.reap`
- `handoff.read`
- `handoff.list`
- `embeddings.index`
- `training.antigravity.status`

`context.query` uses a SQLite index with FTS ranking when available, and falls
back to `LIKE` matching if FTS is unavailable on the host SQLite build.
`context.write`/`fs.write`, `context.delete`/`fs.delete`, and
`context.move`/`fs.move` attempt incremental index sync so query results stay
fresh without a full rebuild. With `auto_index=true` (default),
`context.query` also auto-refreshes when it detects stale path/content metadata
via mount fingerprints, including external renames that keep file counts stable.

Gemini-facing MCP prompts:

- `afs.session.bootstrap`
- `afs.session.pack`
- `afs.workflow.structured`
- `afs.context.overview`
- `afs.query.search`
- `afs.scratchpad.review`

Gemini-facing MCP resources:

- `afs://contexts`
- `afs://schemas/<name>`
- `afs://claude/bootstrap`
- `afs://context/<path>/bootstrap`
- `afs://context/<path>/metadata`
- `afs://context/<path>/mounts`
- `afs://context/<path>/index`

`afs.session.bootstrap` is the recommended first call in a new session. It
packages health, cheap codebase orientation, drift, scratchpad notes, task
queue state, recent hivemind messages, and the latest durable memory summary
into one startup packet.

`afs.context.overview` now includes both mount structure and a cheap codebase
summary. When callers pass `path`/`project_path`, it can also summarize a raw
project tree before `.context` exists yet.

`session.pack` / `afs.session.pack` is the explicit follow-on surface for
model-specific working context. It builds a token-budgeted pack for Gemini,
Claude, Codex, or generic clients, respects `never_export` sensitivity rules
when including indexed content, and reuses the stored pack artifact on repeated
calls when the bootstrap snapshot and pack inputs have not changed. The prompt
and tool forms also accept optional `task`, `workflow`, and `tool_profile`
arguments so callers can encode a short execution contract and put the explicit
task at the end of the rendered pack. Returned pack JSON includes
`cache.prefix_hash` for stable-prefix cache experiments in adapters. `pack_mode`
supports `focused`, `retrieval`, and `full_slice` shaping for query-first vs
broader long-context packs. The `execution_profile` block now also carries a
prompt-only loop policy plus retry guidance so the host CLI keeps session
control while AFS still suggests narrower retries, schema-bound reruns, or
model escalation paths.

`afs://schemas/<name>` exposes compact response contracts for structured agent
workflows. Built-in names currently include:

- `plan`
- `file-shortlist`
- `review-findings`
- `edit-intent`
- `verification-summary`
- `handoff-summary`

These resources return `application/schema+json` so Gemini or other MCP clients
can request a tiny output contract before asking for a structured response.

`afs.workflow.structured` is the first higher-level rail built on top of those
schemas. It combines `session.pack` context shaping with an inline built-in
schema so a client can ask for a plan, review, edit intent, verification
summary, or handoff packet in one MCP prompt without hand-assembling the
contract each time.

For noisy command output, `operator.digest` provides a small compression step
before the text goes back into model context. It accepts raw `text` plus an
optional `kind` hint (`auto`, `pytest`, `traceback`, `grep`, `diffstat`,
`diagnostic`, or `generic`) and returns both structured fields and a compact
`digest_text` summary. `diagnostic` is the compiler/linter family for `tsc`,
ESLint, Ruff, and mypy-style output. This is intended for Gemini-style
sessions where raw terminal output often costs more context than it is worth.

Index behavior can be tuned in `afs.toml`:

```toml
[context_index]
enabled = true
db_filename = "context_index.sqlite3"
auto_index = true
auto_refresh = true
include_content = true
max_file_size_bytes = 262144
max_content_chars = 12000
```

Extensions can extend the MCP surface via `extension.toml`.

Bundle-installed extensions can also generate a wrapper MCP surface automatically
from bundled profile `mcp_tools`, so a packed profile can round-trip through
`afs bundle pack` / `afs bundle install` without hand-writing another manifest.

Legacy tool-only registration still works:

```toml
[mcp_tools]
module = "workspace_adapter.mcp_tools"
factory = "register_mcp_tools"
```

New server-surface registration can contribute tools, resources, and prompts
from one factory:

```toml
[mcp_server]
module = "workspace_adapter.mcp_surface"
factory = "register_mcp_server"
```

Example factory:

```python
def register_mcp_server(_manager):
    def workspace_status(_manager):
        return {"text": "{\"workspace\": \"ok\"}"}

    def workspace_prompt(arguments):
        return f"Review workspace: {arguments.get('query', '')}"

    return {
        "tools": [
            {
                "name": "workspace.echo",
                "description": "Echo extension payload",
                "inputSchema": {
                    "type": "object",
                    "properties": {"value": {"type": "string"}},
                    "additionalProperties": False,
                },
                "handler": lambda arguments: {"echo": arguments.get("value", "")},
            }
        ],
        "resources": [
            {
                "uri": "afs://workspace/status",
                "name": "Workspace status",
                "description": "Extension-owned workspace health",
                "mimeType": "application/json",
                "handler": workspace_status,
            }
        ],
        "prompts": [
            {
                "name": "workspace.review",
                "description": "Review extension workspace state",
                "arguments": [{"name": "query", "required": False}],
                "handler": workspace_prompt,
            }
        ],
    }
```

Core tool names, core prompt names, and reserved `afs://contexts` /
`afs://context/...` resources cannot be overridden by extensions.

Extension directory precedence matters when names collide:

- `AFS_EXTENSION_DIRS`
- configured `[extensions].extension_dirs`
- repo `extensions/`
- `~/.config/afs/extensions`
- `~/.afs/extensions`

Earlier directories win. That lets repo-local or context-local installs override
an older user-global extension with the same name.

Path operations are scoped to:

- `~/.context`
- configured `general.context_root`
- configured `general.workspace_directories`
- configured `general.mcp_allowed_roots`
- `AFS_MCP_ALLOWED_ROOTS` (path-separated env override)
- local project `.context` under the current working directory

`context.init` follows the same rule:

- repo-local initialization is allowed when `project_path` is under the current working directory
- initialization is also allowed when `project_path` is under a configured workspace root
- centralized initialization is allowed when `context_root` is explicitly set under an allowed root

Recommended Gemini work configuration for Mercurial cloud workspaces:

```toml
[general]
mcp_allowed_roots = ["~/workspaces/company"]

[[general.workspace_directories]]
path = "~/workspaces/company"
description = "Managed workspace root"
```

Temporary shell override:

```bash
export AFS_MCP_ALLOWED_ROOTS=~/workspaces/company
```

`context.diff` reports added, modified, and deleted files relative to the last
index build. `context.status` reports mount counts, mount health, provenance
health, index health, the active profile, and suggested repair actions for the
target context. `context.repair` seeds missing provenance records, prunes stale
provenance, remaps missing mount sources conservatively across configured
workspace roots declared in `general.workspace_directories`, and optionally
rebuilds the index.

Gemini background brief surfaces:

```bash
<afs-root>/scripts/afs agents run gemini-workspace-brief --stdout
<afs-root>/scripts/afs agents ps --all
<afs-root>/scripts/afs services start gemini-workspace-brief
<afs-root>/scripts/afs services start agent-supervisor
<afs-root>/scripts/afs services start context-warm
<afs-root>/scripts/afs services start context-watch
```

The brief agent requires `GEMINI_API_KEY` or `GOOGLE_API_KEY`.

For interactive clients, prefer the launcher wrappers:

```bash
<afs-root>/scripts/afs-gemini
<afs-root>/scripts/afs-claude
<afs-root>/scripts/afs-codex
```

They find the nearest `afs.toml`, refresh the session bootstrap artifact, and
export:

- `AFS_SESSION_BOOTSTRAP_JSON`
- `AFS_SESSION_BOOTSTRAP_MARKDOWN`
- `AFS_ACTIVE_CONTEXT_ROOT`

They never infer workspace roots automatically. If you want wrapper-local path
defaults, set `AFS_<CLIENT>_MCP_ALLOWED_ROOTS` or `AFS_CLIENT_MCP_ALLOWED_ROOTS`.

`context-warm` now audits discovered contexts for broken symlink mounts,
duplicate mount targets, missing profile-managed mounts, untracked/stale mount
provenance, and stale indexes. The built-in service runs with
`--repair-mounts --rebuild-stale-indexes --doctor-snapshot` by default.

`context-watch` is the on-change companion surface. It runs `context-warm
--watch`, monitors the context root and mounted source paths, and only reruns
repair/index work for the affected contexts. If `watchfiles` is not installed,
it falls back to polling.

`agent-supervisor` is the companion service for profile-defined background
agents. It reconciles `auto_start`, interval `schedule`, and `watch_paths`
settings and keeps state under `.context/scratchpad/afs_agents/supervisor/` by
default.

If you want maintenance services to stay on a repo-local config instead of the
user config, start them with `--config`. The service layer preserves that
explicit `AFS_CONFIG_PATH` when spawning the background process:

```bash
<afs-root>/scripts/afs services start --config /path/to/afs.toml context-warm
<afs-root>/scripts/afs services start --config /path/to/afs.toml agent-supervisor
```

`afs health` now reports AFS MCP registration across Gemini, Claude, and Codex
config surfaces, and it recognizes both `python -m afs.mcp_server` and
wrapper-style `afs mcp serve` processes. It also surfaces context mount drift so
agents can distinguish path access problems from index staleness quickly, and it
includes maintenance report/service state for `context-warm`, `context-watch`,
`agent-supervisor`, `doctor_snapshot`, and `gemini-workspace-brief`.

## Example Call Shape

`tools/call` expects:

```json
{
  "name": "context.read",
  "arguments": {
    "path": "~/.context/scratchpad/notes.md"
  }
}
```

Rebuild and query the SQLite context index:

```json
{
  "name": "context.query",
  "arguments": {
    "context_path": "~/.context",
    "mount_types": ["scratchpad", "knowledge"],
    "query": "Gemini",
    "limit": 20,
    "auto_index": true
  }
}
```

Prompt-oriented search for Gemini clients:

```json
{
  "name": "afs.query.search",
  "arguments": {
    "context_path": "~/.context",
    "query": "Gemini",
    "mount_types": "scratchpad,knowledge",
    "relative_prefix": "work",
    "limit": 10
  }
}
```
