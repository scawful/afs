# Plugins and Extensions

AFS supports two extension mechanisms:

- Python plugins (runtime hooks)
- Manifest-based extensions (`extension.toml`)

## Python Plugins

Discovery inputs:

- Name prefix (`afs_plugin`, optionally `afs_scawful`)
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
afs plugins --details
afs plugins --json
```

## Manifest Extensions

Extensions are discovered from `extension.toml` files.

Discovery roots:

- `extensions/`
- `~/.config/afs/extensions`
- `~/.afs/extensions`

Config and env:

- `[extensions]` in `afs.toml`
- `AFS_EXTENSION_DIRS`
- `AFS_ENABLED_EXTENSIONS`

Manifest fields:

- `knowledge_mounts`
- `skill_roots`
- `model_registries`
- `cli_modules`
- `policies`
- `[hooks]`
- `[mcp_tools]`

`[mcp_tools]` example:

```toml
[mcp_tools]
module = "my_extension.mcp_tools"
factory = "register_mcp_tools"
```
