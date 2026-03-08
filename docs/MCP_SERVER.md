# AFS MCP Server

AFS provides a lightweight stdio MCP server for context operations.

## Run

```bash
python -m afs.mcp_server
# or
afs mcp serve
```

## Gemini CLI Registration

```bash
gemini mcp add afs python -m afs.mcp_server
```

## Antigravity Custom Config

In Antigravity, open `MCP Servers -> Manage MCP Servers -> View raw config`, then add:

```json
{
  "mcpServers": {
    "afs": {
      "command": "python",
      "args": ["-m", "afs.mcp_server"]
    }
  }
}
```

## Tools

- `fs.read`
- `fs.write`
- `fs.list`
- `context.discover`
- `context.mount`

Extensions can register additional tools via `extension.toml`:

```toml
[mcp_tools]
module = "afs_google.mcp_tools"
factory = "register_mcp_tools"
```

Path operations are scoped to:

- `~/.context`
- configured `general.context_root`
- local project `.context` under the current working directory

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
