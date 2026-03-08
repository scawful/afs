# Agentic File System (AFS)

AFS is a core context platform for agent workflows.

It provides filesystem-native primitives for:

- context roots (`.context`)
- typed mounts (`memory`, `knowledge`, `tools`, `scratchpad`, `history`, `monorepo`, ...)
- profile-driven context injection
- extension manifests
- grounding hooks
- MCP tool surfaces

## Quick Start

```bash
# Initialize context for current project
afs context init

# Show active profile and switch if needed
afs profile current
afs profile switch work

# Inspect profile mounts
afs context profile-show --profile work
afs context profile-apply --profile work

# Run MCP server
afs mcp serve

# Diagnose context/mount/extension state
afs health
```

## Core Docs

- [Architecture](ARCHITECTURE.md)
- [Profiles and Hooks](PROFILES.md)
- [Extensions](PLUGINS.md)
- [MCP Server](MCP_SERVER.md)
- [Memory and Context Layout](MEMORY_SYSTEM.md)
- [CLI Reference](CLI_REFERENCE.md)

## Scope Boundary

Domain/model-training documentation has moved out of core AFS.

See [AFS Scawful Migration](SCAWFUL_MIGRATION.md).
