# AFS MCP Server

AFS provides a lightweight stdio MCP server for context operations.

Work-assistant state for people, project relationships, review routes,
approvals, and activity is native AFS state. Keep MCP thin: do not expose full
people, docs, sheets, ticket, or permission administration through the default
server. External writes should flow through an approved AFS work request and
one explicit connector executor command.

For Antigravity CLI and Gemini CLI compatibility, prefer the MCP server from the repo or project root you want to
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

## Antigravity and Gemini CLI Registration

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

Use `afs antigravity setup --scope project` for the public `agy` CLI path. New
`agy` builds use `~/.gemini/config/mcp_config.json` for migrated MCP config, and
AFS detects the older Antigravity CLI/IDE paths as compatibility fallbacks. Use
`--scope project` when you want the compatibility Gemini CLI setup to keep MCP
registration inside the current repo at `./.gemini/settings.json`.
`afs gemini status` detects both user-level and project-level Gemini configs.

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
        "PYTHONPATH": "$AFS_ROOT/src",
        "AFS_MCP_TOOL_NAME_STYLE": "claude"
      }
    }
  }
}
```

`AFS_MCP_TOOL_NAME_STYLE=claude` exposes tool names with Claude-safe
underscores, while routing calls back to the canonical AFS names internally
(`context.status` is listed as `context_status`, for example). Leave it unset
for clients that already accept dotted MCP tool names.

For the bundled VS Code extension, `AFS: Register MCP Server` now checks
workspace `.cursor`, `.vscode`, and `.antigravity` MCP configs plus existing
Antigravity context-root candidates before falling back to workspace
`.cursor/mcp.json`. If your fork stores the raw config elsewhere, set
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

Default `tools/list` is deliberately focused. It exposes normal scoped context,
message, note, handoff, and skill work; administration remains in prompts, CLI
hints, or an explicit full-catalog/debug launch:

- `context.status`
- `context.query`
- `context.search`
- `context.read`
- `context.write`
- `context.list`
- `messages.send`
- `messages.read`
- `note.create`
- `note.read`
- `note.list`
- `handoff.create`
- `handoff.read`
- `handoff.list`
- `skill.match`
- `skill.read`

In a version 2 central context, `project_path` authorizes the registered
current-project scope. Normal reads include that scope plus `common`.
`context_path` alone does not grant project access, and cross-project search
requires `all_projects=true`.

`context.search` reads the version 2 hybrid index and filters scope before
ranking. It is local text/symbol retrieval by default. `semantic=true`
explicitly permits the configured embedding provider for the query; build the
index with `afs search --semantic --rebuild` first. The default Gemini
collection uses stable `gemini-embedding-2` at 768 dimensions.

`skill.match` ranks configured, profile-eligible skills against a task prompt.
Prompts are capped at 8,000 characters and `top_k` must be from 1 through 10.
Bodies are opt-in; when requested, at most three bodies are returned, each body
is capped at 2,000 characters, and the combined body budget is 6,000
characters. `skill.read` resolves a skill by name only from configured skill
roots and returns at most 2,000 characters of its instruction body. Both tools
report truncation explicitly. Discovery rejects skill files over 64,000
characters, names over 256 characters, and metadata lists over 16 entries of
256 characters each, so safety metadata is not silently clipped.

When `AFS_MCP_TOOL_NAME_STYLE=claude` is set, advertised dotted names use
underscore aliases (for example, `context_search`, `messages_send`, and
`handoff_create`) to satisfy Claude's stricter tool-name schema. Calls using
either underscore aliases or canonical dotted names are accepted.

`afs.session.bootstrap`, `afs.session.pack`, and `afs.scratchpad.review` remain
available as prompts from `prompts/list`, not as tools. Work preflight,
approvals, repair, and verification should normally route through the
AFS CLI/framework hints rather than the default MCP tool catalog. For debugging,
migration, or a client that really needs every registered tool, start the MCP
server with the full catalog:

```bash
<afs-root>/scripts/afs mcp serve --tool-catalog full
# or:
AFS_MCP_TOOL_CATALOG=full <afs-root>/scripts/afs mcp serve
```

`AFS_ALLOWED_TOOLS` and `AFS_TOOL_PROFILE` are still stricter permission
filters. They narrow both `tools/list` and `tools/call`, even when the catalog
mode is `full`.

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
- `work.communication.list`
- `work.communication.add`
- `work.communication.guide`
- `work.approvals.show`
- `work.approvals.request`
- `agent.spawn`
- `agent.ps`
- `agent.stop`
- `agent.logs`
- `events.query`
- `events.tail`
- `events.analytics`
- `events.replay`
- `messages.subscribe`
- `messages.unsubscribe`
- `messages.clean`
- `handoff.revise`
- `handoff.threads`
- `handoff.ack`
- `handoff.close`
- `embeddings.index`
- `training.antigravity.status`

The public coordination surface is `messages.*`. Legacy `hivemind.*` tools
remain accepted in the full catalog for one compatibility cycle; new clients
should not discover or generate them.

`note.create/read/list` operates on immutable Markdown notes in the authorized
scope. Handoff content is also immutable: `handoff.revise` publishes a new
revision with a supersedes link, while `handoff.ack` and `handoff.close` append
separate lifecycle state.

`context.query` uses a SQLite index with FTS ranking when available, and falls
back to `LIKE` matching if FTS is unavailable on the host SQLite build.
`context.query` and the `context.*`/`fs.*` file tools enforce configured
`never_index`/`never_export` sensitivity rules before touching or exporting
matching paths or file contents.
`context.write`/`fs.write`, `context.delete`/`fs.delete`, and
`context.move`/`fs.move` attempt incremental index sync so query results stay
fresh without a full rebuild. With `auto_index=true` (default),
`context.query` also auto-refreshes when it detects stale path/content metadata
via mount fingerprints, including external renames that keep file counts stable.
In a v2 central context, `context.query` auto-indexing and
`context.index.rebuild` traverse and replace only the authorized current-project
and `common` prefixes. Queue-wide indexing is available only when the caller
sets the explicit `all_projects: true` administration boundary.

Work-context MCP tools stay deliberately narrow. `work.communication.preflight`
combines stored tone/style evidence, optional opt-in personal context, pending
approvals, and the approval rule for work writing. `work.communication.guide`
remains available as the smaller style-only summary. `work.approvals.request`
creates a local permission request for a drafted external write; it does not
approve or execute the write. Approval and connector execution still happen
through the explicit `afs work approvals ...` CLI flow.

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
queue state, recent scoped messages, and the latest durable memory summary
into one startup packet. Its optional `skills_prompt` and `skills_top_k`
arguments select bounded skill bodies explicitly. When no prompt is supplied,
bootstrap uses a bounded continuation signal from handoff next steps, active
missions, and open tasks; that untrusted state selects only instructions from
configured skill roots and is not itself treated as policy.

`afs.context.overview` now includes both mount structure and a cheap codebase
summary. When callers pass `path`/`project_path`, it can also summarize a raw
project tree before `.context` exists yet.

`session.pack` / `afs.session.pack` is the explicit follow-on surface for
model-specific working context. It builds a token-budgeted pack for Gemini,
Claude, Codex, or generic clients, respects `never_index`/`never_export`
sensitivity rules when including indexed content, applies `never_embed` to
embedding hits, and reuses the stored pack artifact on repeated calls when the
bootstrap snapshot, pack inputs, and sensitivity rules have not changed. The
prompt and tool forms also accept optional `task`, `workflow`, and `tool_profile`
arguments so callers can encode a short execution contract and put the explicit
task at the end of the rendered pack. Returned pack JSON includes
`cache.prefix_hash` for stable-prefix cache experiments in adapters. `pack_mode`
supports `focused`, `retrieval`, and `full_slice` shaping for query-first vs
broader long-context packs. The `execution_profile` block now also carries a
prompt-only loop policy plus retry guidance so the host CLI keeps session
control while AFS still suggests narrower retries, schema-bound reruns, or
model escalation paths. Guidance is rendered once as fixed pack overhead, not
again as a selectable section, so query/embedding hits can displace generic
session boilerplate in tight budgets.
Both the `session.pack` tool and `afs.session.pack` prompt expose an optional
`semantic` boolean. It defaults to `false`; only an explicit true value permits
the configured provider to receive the pack query.

`afs://schemas/<name>` exposes compact response contracts for structured agent
workflows. Built-in names currently include:

- `plan`
- `file-shortlist`
- `review-findings`
- `edit-intent`
- `verification-summary`
- `handoff-summary`

Versioned protocol schemas are also exposed through the same resource surface:

- `v1/optimization/evaluation`, `v1/optimization/policy`, and
  `v1/optimization/decision`
- `v1/execution/request`, `v1/execution/inspection`, and
  `v1/execution/record`

For example, an MCP client can read
`afs://schemas/v1/execution/request`. Execution schemas describe portable
request and audit records; reading or validating one never launches a process.

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
# Optional. Default is "full"; "slim" opts inherited tools into default tools/list.
catalog = "slim"
```

The `[mcp_tools].catalog` default applies only to that tool-only factory.
Per-tool dictionaries may set `catalog = "slim"` or `catalog = "full"` to
override it. Tools from `[mcp_server]` default to `"full"` and can opt in only
per tool. Invalid catalog values reject the affected extension surface instead
of making it discoverable accidentally.

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
settings and keeps state under
`.context/scratchpad/common/afs_agents/supervisor/` in v2 or
`.context/scratchpad/afs_agents/supervisor/` in v1 by default.

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
    "context_path": "~/.context",
    "project_path": "~/src/project-a",
    "path": "notes/investigation.md"
  }
}
```

Rebuild and query the SQLite context index:

```json
{
  "name": "context.query",
  "arguments": {
    "context_path": "~/.context",
    "project_path": "~/src/project-a",
    "mount_types": ["scratchpad", "knowledge"],
    "query": "Gemini",
    "limit": 20,
    "auto_index": true
  }
}
```

Search the scoped v2 hybrid index without remote embeddings:

```json
{
  "name": "context.search",
  "arguments": {
    "context_path": "~/.context",
    "project_path": "~/src/project-a",
    "query": "cache invalidation",
    "mode": "symbol",
    "semantic": false,
    "limit": 10
  }
}
```

Create scoped human-readable records:

```json
{
  "name": "note.create",
  "arguments": {
    "context_path": "~/.context",
    "project_path": "~/src/project-a",
    "title": "Cache decision",
    "body": "Keep invalidation local to the repository boundary."
  }
}
```

```json
{
  "name": "handoff.create",
  "arguments": {
    "context_path": "~/.context",
    "project_path": "~/src/project-a",
    "title": "Cache cleanup",
    "agent_name": "codex",
    "accomplished": ["Added scoped invalidation"],
    "next_steps": ["Run the integration suite"]
  }
}
```

Send a current-project message:

```json
{
  "name": "messages.send",
  "arguments": {
    "context_path": "~/.context",
    "project_path": "~/src/project-a",
    "from": "codex",
    "topic": "status",
    "payload": {"summary": "integration suite passed"}
  }
}
```

The older prompt-oriented search remains for version 1 clients:

```json
{
  "name": "afs.query.search",
  "arguments": {
    "context_path": "~/src/project-a/.context",
    "query": "Gemini",
    "mount_types": "scratchpad,knowledge",
    "relative_prefix": "work",
    "limit": 10
  }
}
```

That prompt uses the SQLite compatibility index and does not accept v2 project
scope arguments. New central-context clients should call `context.search`
instead. The older full-catalog `afs.search`, `afs.codebase.index`, and
`afs.codebase.symbols` tools are likewise unavailable in v2 because their
legacy index formats carry no project-scope authorization metadata.
