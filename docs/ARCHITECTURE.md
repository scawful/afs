# AFS Core Architecture

## Design Goal

AFS treats context as first-class filesystem state so agents can operate with explicit,
inspectable, and mountable working memory.

## Core Layers

### 1. Context Layer

- `AFSManager` manages `.context` roots and mount operations.
- Mount roles are typed (`MountType`) and policy-governed.
- Metadata defines directory role mapping and manual-protection behavior.
- Runtime surfaces resolve mount roots by role, so remapped directories remain
  valid across CLI, MCP, health, and background services.

Primary modules:

- `src/afs/manager.py`
- `src/afs/models.py`
- `src/afs/mapping.py`
- `src/afs/context_paths.py`
- `src/afs/context_fs.py`

### 2. Config Layer

- `afs.toml` + user config are merged into typed schema.
- Profiles resolve inheritance, env overrides, and extension overlays.
- Extensions are loaded from `extensions/*/extension.toml` and from opt-in
  companion repos named like `afs_<name>` / `afs-<name>`.

Primary modules:

- `src/afs/config.py`
- `src/afs/schema.py`
- `src/afs/profiles.py`
- `src/afs/extensions.py`

### 3. Policy and Grounding Layer

- Directory policy enforces read/write/execute capabilities per mount role.
- Hook events guard context IO and agent dispatch.
- Extension hooks can be composed with profile policy.

Primary modules:

- `src/afs/policy.py`
- `src/afs/grounding_hooks.py`

### 4. Interface Layer

- CLI (`afs`) for context, profile, workspace, skills, embeddings, health.
- MCP server for structured tool access (`context.read`, `context.write`, and
  compatibility aliases like `fs.read`/`fs.write`).
- Extension-provided MCP tools can be registered from manifests.

Primary modules:

- `src/afs/cli/`
- `src/afs/mcp_server.py`

## Extensibility Model

AFS extension points are explicit:

- Plugins: Python module discovery
- Extensions: `extension.toml` mount/policy/CLI/agent declarations
- Companion repos: user-owned `afs_<name>` repos with their own manifest,
  import paths, MCP surface, and overrides
- Hooks: pre/post operation command hooks
- MCP tools/resources/prompts: extension module factories via `[mcp_tools]`
  or `[mcp_server]`

## Operational Diagnostics

`afs health` reports:

- active profile and mount state
- monorepo bridge freshness
- embedding index age summary
- extension and hook registration state
- MCP registration and runtime status

## Scope Boundary

Core AFS docs and code avoid domain-specific model-training, assistant, or
private connector workflows. Those belong in companion extensions.

Core AFS may still own generic training/feedback orchestration primitives when
they are reusable across repos and model families.
