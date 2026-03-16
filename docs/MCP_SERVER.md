# AFS MCP Server

AFS provides a lightweight stdio MCP server for context operations.

For Gemini CLI, prefer the MCP server from the repo or project root you want to
work in. That keeps repo-local `context.init` available while preserving the
allowed-root guardrails on all file and context operations.

## Run

```bash
~/src/lab/afs/scripts/afs mcp serve
# or, from an environment where `afs` is installed
afs mcp serve
# or
python3 -m afs.mcp_server
```

## Gemini CLI Registration

```bash
gemini mcp add afs /Users/scawful/src/lab/afs/scripts/afs mcp serve
```

If Gemini is running inside an environment where `afs` is already installed,
`python3 -m afs.mcp_server` also works.

## Antigravity Custom Config

In Antigravity, open `MCP Servers -> Manage MCP Servers -> View raw config`, then add:

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

## Tools

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

`context.query` uses a SQLite index with FTS ranking when available, and falls
back to `LIKE` matching if FTS is unavailable on the host SQLite build.
`fs.write`, `fs.delete`, and `fs.move` attempt incremental index sync so query
results stay fresh without a full rebuild. With `auto_index=true` (default),
`context.query` also auto-refreshes when it detects stale path/content metadata
via mount fingerprints, including external renames that keep file counts stable.

Gemini-facing MCP prompts:

- `afs.context.overview`
- `afs.query.search`
- `afs.scratchpad.review`

Gemini-facing MCP resources:

- `afs://contexts`
- `afs://context/<path>/metadata`
- `afs://context/<path>/mounts`
- `afs://context/<path>/index`

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

Extensions can register additional tools via `extension.toml`:

```toml
[mcp_tools]
module = "afs_google.mcp_tools"
factory = "register_mcp_tools"
```

Path operations are scoped to:

- `~/.context`
- configured `general.context_root`
- configured `general.agent_workspaces_dir`
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
mcp_allowed_roots = ["/google"]

[[general.workspace_directories]]
path = "/google"
description = "Mercurial cloud workspaces"
```

Temporary shell override:

```bash
export AFS_MCP_ALLOWED_ROOTS=/google
```

`context.diff` reports added, modified, and deleted files relative to the last
index build. `context.status` reports mount counts, index health, and the active
profile for the target context.

Gemini background brief surfaces:

```bash
~/src/lab/afs/scripts/afs agents run gemini-workspace-brief --stdout
~/src/lab/afs/scripts/afs services start gemini-workspace-brief
```

The brief agent requires `GEMINI_API_KEY` or `GOOGLE_API_KEY`.

## Example Call Shape

`tools/call` expects:

```json
{
  "name": "fs.read",
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
