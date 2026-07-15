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

### Policy-Checked Execution

- `afs.execution.ExecutionRequest`
- `afs.execution.ExecutionPolicy`
- `afs.execution.ExecutionInspection`
- `afs.execution.ExecutionRecord`
- `afs.execution.inspect_execution`
- `afs.execution.execute_checked`

## CLI Surface

Use `./scripts/afs` during local development unless `afs` is installed in the
active environment.

### Context

```bash
./scripts/afs context init
./scripts/afs context ensure
./scripts/afs context list
./scripts/afs context mount <mount_type> <source>
./scripts/afs context unmount <mount_type> <alias>
./scripts/afs context profile-show --profile <name>
./scripts/afs context profile-apply --profile <name>
```

### Profiles

```bash
./scripts/afs profile current
./scripts/afs profile list
./scripts/afs profile switch <name>
```

### MCP

```bash
./scripts/afs mcp serve
```

### Health

```bash
./scripts/afs health
./scripts/afs health check --level standard
./scripts/afs health status
```

### Execution Inspection

```bash
./scripts/afs execution inspect --request request.json --allowed-root "$PWD" \
  --allowed-executable python3 --json
```

Inspection is read-only. AFS does not expose a generic execution command; use
the typed Python API from trusted code. The CLI blocks when executable
permission is omitted; pass `--allowed-env NAME` for each non-baseline
environment key. See [Policy-Checked Execution](EXECUTION_BROKER.md).

### Skills and Embeddings

```bash
./scripts/afs skills list --profile <name>
./scripts/afs skills match "<query>" --profile <name>
./scripts/afs embeddings index --knowledge-dir <path> --source <path>
./scripts/afs embeddings search "<query>" --knowledge-dir <path>
```

## Config Keys (Core)

- `[general]`
- `[profiles]`, `[profiles.<name>]`
- `[extensions]`
- `[hooks]`
- `[plugins]`
- `[verification]`, `[verification.profiles.<name>]`

For examples, see:

- [Profiles and Hooks](PROFILES.md)
- [MCP Server](MCP_SERVER.md)
- [Extensions](PLUGINS.md)
