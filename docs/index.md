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

- [Executive Summary](EXECUTIVE_SUMMARY.md) — concise non-domain overview for sharing
- [Research Lineage](LINEAGE.md) — how AFS maps to the Agentic File System paper and where it goes beyond it
- [Agent Surfaces](AGENT_SURFACES.md)
- [Agent Operations](AGENT_OPERATIONS.md)
- [Agent Integration Upgrade](AGENT_INTEGRATION_UPGRADE.md)
- [Guided Setup](SETUP_GUIDE.md)
- [Architecture](ARCHITECTURE.md)
- [Profiles and Hooks](PROFILES.md)
- [Extensions](PLUGINS.md)
- [Extension Authoring](EXTENSION_AUTHORING.md)
- [MCP Server](MCP_SERVER.md)
- [Work Assistant](WORK_ASSISTANT.md)
- [Work Assistant Connectors](WORK_ASSISTANT_CONNECTORS.md)
- [Work Assistant Upgrade Guide](WORK_ASSISTANT_UPGRADE.md)
- [Embeddings](EMBEDDINGS.md)
- [Knowledge System & Gemini Setup Guide](KNOWLEDGE_SYSTEM_GUIDE.md)
- [VSCode Extension Review](VSCODE_EXTENSION_REVIEW.md)
- [Memory and Context Layout](MEMORY_SYSTEM.md)
- [CLI Reference](CLI_REFERENCE.md)
- [Training and Feedback RFC](TRAINING_FEEDBACK_RFC.md)
- [Autonomous Optimization Protocol](OPTIMIZATION_PROTOCOL.md)
- [Policy-Checked Execution](EXECUTION_BROKER.md)
- [Emacs Integration](EMACS_INTEGRATION.md)

## Agent Note

For local automation, prefer `./scripts/afs <command>` over bare `python -m afs`
unless the package is already installed in the Python environment the agent uses.

## Scope Boundary

Domain/model-training workflows live in extensions and downstream repos.
Generic training and feedback orchestration primitives may live in core AFS when
they are reusable across repos.
