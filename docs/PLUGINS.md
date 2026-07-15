# Plugins and Extensions

For a step-by-step extension contract and hello-world template, see the [Extension Authoring Guide](EXTENSION_AUTHORING.md).

AFS supports two opt-in extension mechanisms:

- Python plugins for lightweight runtime hooks
- Manifest-based extensions for repo-owned context, commands, agents, and MCP surfaces

Core AFS stays neutral. Personal, work, or domain-specific behavior should live in
a companion extension repo such as `afs_example` or `afs_company` instead of
being hardcoded into core AFS.

## Python Plugins

Discovery inputs:

- Name prefix (`afs_plugin`, optionally a configured prefix)
- Configured `plugin_dirs`
- `AFS_PLUGIN_DIRS`
- Default directories: `~/.config/afs/plugins`, `~/.afs/plugins`

Enable explicitly:

```bash
export AFS_ENABLED_PLUGINS="afs_plugin_hello"
```

Supported plugin hooks:

- `register_cli(subparsers)`
- `register_parsers(subparsers)`
- `register_backend()`

Inspect resolved plugin state:

```bash
./scripts/afs plugins --details
./scripts/afs plugins --json
```

## Manifest Extensions

Extensions are discovered from manifest files. The default manifest filename is
`extension.toml`.

Discovery roots:

- `AFS_EXTENSION_DIRS`
- configured `extension_dirs`
- repo-local `extensions/`
- `~/.config/afs/extensions`
- `~/.afs/extensions`

Companion repo discovery:

- `AFS_EXTENSION_REPO_ROOTS`
- configured `extension_repo_roots`
- `[general].workspace_directories` when a full `AFSConfig` is active
- repos whose directory name matches `extension_repo_prefixes` (defaults:
  `afs_`, `afs-`)
- manifest names from `manifest_filenames` (default: `extension.toml`)

That means a user can create a sibling repo like this:

```text
~/src/lab/afs_example/
  extension.toml
  src/afs_example/
    __init__.py
    cli.py
    agents.py
    mcp_surface.py
```

Then enable it from core AFS without copying implementation code:

```toml
[extensions]
auto_discover = false
enabled_extensions = ["afs_example"]
extension_repo_roots = ["~/src/lab"]
# optional overrides:
extension_repo_prefixes = ["afs_", "team_"]
manifest_filenames = ["extension.toml", "afs-extension.toml"]
```

Or with environment variables:

```bash
export AFS_EXTENSION_REPO_ROOTS="$HOME/src/lab"
export AFS_ENABLED_EXTENSIONS="afs_example"
```

Config and env:

- `[extensions]` in `afs.toml`
- `AFS_EXTENSION_DIRS`
- `AFS_EXTENSION_REPO_ROOTS`
- `AFS_EXTENSION_REPO_PREFIXES`
- `AFS_EXTENSION_MANIFEST_FILENAMES`
- `AFS_ENABLED_EXTENSIONS`

Manifest fields:

- `knowledge_mounts`
- `skill_roots`
- `model_registries`
- `python_paths` / `import_paths` (optional; `src/` is added automatically
  when present)
- `cli_modules`
- `agent_modules`
- `policies`
- `[hooks]`
- `[mcp_tools]`
- `[mcp_server]`

Example `extension.toml`:

```toml
name = "afs_example"
description = "Example context and approval helpers"

knowledge_mounts = ["knowledge"]
skill_roots = ["skills"]
python_paths = ["src"]
cli_modules = ["afs_example.cli"]
agent_modules = ["afs_example.agents"]

[mcp_server]
module = "afs_example.mcp_surface"
factory = "register_mcp_server"
```

`agent_modules` let an extension register extra `afs agents run ...` entries
without putting personal, work-specific, or domain-specific agent code into core
AFS.

When two extension roots contain the same manifest `name`, the first discovery
root wins. That means repo-local or context-local extension dirs can safely
override an older user-global install with the same name.

`[mcp_tools]` example:

```toml
[mcp_tools]
module = "my_extension.mcp_tools"
factory = "register_mcp_tools"
# Optional; defaults to "full".
catalog = "slim"
```

`[mcp_tools].catalog` is an extension-wide default for that factory only.
Individual tool definitions can override it with `catalog = "full"` or
`catalog = "slim"`. `[mcp_server]` and profile tools default to `"full"` and
require a per-tool opt-in to enter the slim catalog. Values outside `"full"`
and `"slim"` reject the affected manifest or MCP contribution.

`[mcp_server]` is the broader surface. It can register tools, resources, and
prompts from one extension module:

```toml
[mcp_server]
module = "my_extension.mcp_surface"
factory = "register_mcp_server"
```

Expected factory shape:

```python
def register_mcp_server(_manager):
    return {
        "tools": [...],
        "resources": [...],
        "prompts": [...],
    }
```

Legacy `[mcp_tools]` remains supported for tool-only extensions. Core MCP
resource URIs and prompt names are reserved and cannot be overridden.

## Bundles

`afs bundle install` materializes a bundle as a normal manifest extension. In
addition to copying `knowledge/` and `skills/`, it can generate:

- an `agent_modules` shim from bundled profile `agent_configs`
- an `[mcp_server]` shim from bundled profile `mcp_tools`
- `profile-snippet.toml` for safe profile reactivation

That keeps bundles portable without requiring manual extension glue after
install.


## Context Source Providers

Extensions can declare provider-neutral context source adapters with `[[context_sources]]`. See `docs/CONTEXT_SOURCES.md` for the record schema and CLI flow.
