# AFS Extensions

`extensions/` defines installable adapters that layer project-specific behavior onto core AFS without forking core code.

## Layout

Each extension lives in its own directory and must include `extension.toml`.

```text
extensions/
  afs_google/
    extension.toml
    skills/
    knowledge/
```

## Manifest Schema

```toml
name = "afs_google"
description = "Google-internal workspace adapter"

knowledge_mounts = ["knowledge/work"]
skill_roots = ["skills"]
model_registries = ["config/chat_registry.toml"]
cli_modules = ["afs_google.cli"]
policies = ["no_zelda"]

[hooks]
before_context_read = ["scripts/hooks/before_context_read.sh"]

[mcp_tools]
module = "afs_google.mcp_tools"
factory = "register_mcp_tools"
```

## Config + Env

- `afs.toml`:

```toml
[extensions]
auto_discover = true
enabled_extensions = ["afs_google"]
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

## Notes

Core `afs` should only ship generic primitives. Domain-specific content (e.g. Zelda, internal work adapters) belongs in extension repos such as `afs-scawful` or `afs_google` and is mounted via manifests.
