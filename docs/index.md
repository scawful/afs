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
./scripts/afs context init

# Diagnose and repair common setup/runtime issues
./scripts/afs doctor
./scripts/afs doctor --fix

# Show active profile and switch if needed
./scripts/afs profile current
./scripts/afs profile switch work

# Inspect profile mounts
./scripts/afs context profile-show --profile work
./scripts/afs context profile-apply --profile work

# Run MCP server
./scripts/afs mcp serve

# Diagnose context/mount/extension state
./scripts/afs health
```

## Core Docs

- [Agent Surfaces](AGENT_SURFACES.md)
- [Agent Operations](AGENT_OPERATIONS.md)
- [Architecture](ARCHITECTURE.md)
- [Profiles and Hooks](PROFILES.md)
- [Extensions](PLUGINS.md)
- [MCP Server](MCP_SERVER.md)
- [Embeddings](EMBEDDINGS.md)
- [Knowledge System & Gemini Setup Guide](KNOWLEDGE_SYSTEM_GUIDE.md)
- [VSCode Extension Review](VSCODE_EXTENSION_REVIEW.md)
- [Memory and Context Layout](MEMORY_SYSTEM.md)
- [CLI Reference](CLI_REFERENCE.md)
- [Training and Feedback RFC](TRAINING_FEEDBACK_RFC.md)
- [Roadmap](ROADMAP.md)
- [Emacs Integration](EMACS_INTEGRATION.md)

## Agent Note

For local automation, prefer `./scripts/afs <command>` over bare `python -m afs`
unless the package is already installed in the Python environment the agent uses.

## Scope Boundary

Domain/model-training workflows live in extensions and downstream repos.
Generic training and feedback orchestration primitives may live in core AFS when
they are reusable across repos.
