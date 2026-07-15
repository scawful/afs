# AFS Extensions

`extensions/` defines installable adapters that layer project-specific behavior onto core AFS without forking core code.

## Layout

Each extension lives in its own directory and must include `extension.toml`.

```text
extensions/
  workspace_adapter/
    extension.toml
    skills/
    knowledge/
```

## Manifest Schema

```toml
name = "workspace_adapter"
description = "Private workspace adapter"

knowledge_mounts = ["knowledge/work"]
skill_roots = ["skills"]
model_registries = ["config/chat_registry.toml"]
cli_modules = ["workspace_adapter.cli"]
policies = ["no_zelda"]

[hooks]
before_context_read = ["scripts/hooks/before_context_read.sh"]

[mcp_tools]
module = "workspace_adapter.mcp_tools"
factory = "register_mcp_tools"
# Optional extension-wide default for this factory; defaults to "full".
catalog = "slim"
```

## Config + Env

- `afs.toml`:

```toml
[extensions]
auto_discover = true
enabled_extensions = ["workspace_adapter"]
extension_dirs = ["./extensions"]
```

- Environment overrides:
- `AFS_EXTENSION_DIRS` (`:` separated)
- `AFS_ENABLED_EXTENSIONS` (comma/space separated)

## MCP Tool Extension Point

If an extension needs custom MCP tools, implement a Python module and declare it in
`[mcp_tools]`.

Factory contract:

- callable name defaults to `register_mcp_tools`
- return value is `list[dict]`
- each dict requires:
  - `name`
  - `description`
  - `inputSchema` (or `input_schema`)
  - `handler` callable (`handler(arguments)` or `handler(arguments, manager)`)
  - optional `catalog`: `"slim"` opts into the default `tools/list`; `"full"`
    keeps the tool full-catalog-only

Catalog behavior is fail-closed:

- `[mcp_tools].catalog` defaults to `"full"` and applies only to tools from the
  `[mcp_tools]` factory.
- A per-tool `catalog` overrides that default. This is how one tool opts out of
  an extension-wide `"slim"` setting.
- `[mcp_server]` and profile-contributed tools default to `"full"`; they opt in
  per tool.
- Only `"full"` and `"slim"` are valid manifest and per-tool values. An invalid
  value rejects the affected manifest or MCP surface rather than exposing it.

## Notes

Core `afs` should only ship generic primitives. Domain-specific content belongs in separate extension repos and is mounted via manifests.
