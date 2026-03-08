# AFS API Reference (Core)

This page documents core AFS APIs and CLI surfaces in `lab/afs`.

## Python Modules

### Context and Mounts

- `afs.manager.AFSManager`
- `afs.models.MountType`
- `afs.context_fs.ContextFileSystem`

### Config and Profiles

- `afs.config.load_config_model`
- `afs.profiles.resolve_active_profile`
- `afs.profiles.apply_profile_mounts`

### Extensions and Plugins

- `afs.extensions.load_extensions`
- `afs.plugins.load_enabled_plugins`
- `afs.plugins.load_enabled_extensions`

### MCP

- `afs.mcp_server.serve`
- `afs.mcp_server.build_mcp_registry`
- `afs.mcp_server.get_mcp_status`

### Health

- `afs.health.afs_status.collect_afs_health`
- `afs.health.afs_status.render_afs_health`

## CLI Surface

### Context

```bash
afs context init
afs context ensure
afs context list
afs context mount <mount_type> <source>
afs context unmount <mount_type> <alias>
afs context profile-show --profile <name>
afs context profile-apply --profile <name>
```

### Profiles

```bash
afs profile current
afs profile list
afs profile switch <name>
```

### MCP

```bash
afs mcp serve
```

### Health

```bash
afs health
afs health check --level standard
afs health status
```

### Skills and Embeddings

```bash
afs skills list --profile <name>
afs skills match "<query>" --profile <name>
afs embeddings index --knowledge-dir <path>
afs embeddings search "<query>" --knowledge-dir <path>
```

## Config Keys (Core)

- `[general]`
- `[profiles]`, `[profiles.<name>]`
- `[extensions]`
- `[hooks]`
- `[plugins]`

For examples, see:

- [Profiles and Hooks](PROFILES.md)
- [MCP Server](MCP_SERVER.md)
- [Extensions](PLUGINS.md)
