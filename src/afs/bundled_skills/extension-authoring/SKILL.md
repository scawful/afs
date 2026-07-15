---
name: extension-authoring
triggers:
  - afs extension
  - extension.toml
  - afs extension manifest
  - mcp extension
  - mcp tools factory
  - slim catalog
profiles:
  - general
requires:
  - afs
enforcement:
  - Load extension modules and hook scripts only from trusted local sources.
  - Slim-catalog visibility never grants permission to invoke a tool.
---

# Extension Authoring

Layer project-specific behavior onto core AFS via `extensions/<name>/extension.toml`
instead of forking core code.

## Manifest

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
factory = "register_mcp_tools"   # default name
catalog = "slim"                 # list all tools by default
```

## MCP Tool Contract

Factory returns `list[dict]`; each dict needs `name`, `description`,
`inputSchema`, and a `handler(arguments)` or `handler(arguments, manager)`
callable. Optional per-tool `catalog`: `"slim"` lists it in the default
`tools/list`; `"full"` keeps it callable but hidden unless
`AFS_MCP_TOOL_CATALOG=full`; omitted inherits the manifest default (or `"full"`
when no default is declared). Per-tool values override the manifest default.
Catalog selection affects discovery only; normal call-time tool permissions
still apply.

## Enabling

```toml
# afs.toml
[extensions]
auto_discover = true
enabled_extensions = ["workspace_adapter"]
extension_dirs = ["./extensions"]
```

Env overrides: `AFS_EXTENSION_DIRS` (`:`-separated), `AFS_ENABLED_EXTENSIONS`
(comma/space separated). `afs plugins --details --json` proves manifest
discovery only; it does not import the declared MCP factory. Verify MCP wiring
with an isolated server initialization plus `tools/list`, and use
`afs skills list` to confirm extension skill roots. Keep domain content in the
extension; core `afs` ships only generic primitives.
