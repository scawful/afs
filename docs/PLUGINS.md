# Plugins

AFS plugins are regular Python modules/packages that expose optional hooks.
They are discovered by name prefix and can live outside the repo.

## Discovery Rules

AFS discovers plugins by:
- Name prefix: `afs_plugin` (or `afs_scawful` if enabled)
- Configured `plugin_dirs`
- `AFS_PLUGIN_DIRS` (colon-separated on macOS/Linux)
- Default directories: `~/.config/afs/plugins` and `~/.afs/plugins`

To enable specific plugins regardless of auto-discovery, set:

```bash
export AFS_ENABLED_PLUGINS="afs_plugin_hello"
```

## Quickstart (no build required)

Use the skeleton in `examples/plugin_skeleton`:

```bash
export AFS_PLUGIN_DIRS="$PWD/examples/plugin_skeleton"
./scripts/afs plugins --details
./scripts/afs hello
```

## Supported Hooks

- `register_cli(subparsers)` or `register_parsers(subparsers)`
- `register_backend()` (generator backends)
- `register_converter()` (training converters)

Run `afs plugins --json` to inspect resolved plugin config.
