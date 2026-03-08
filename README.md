# Agentic File System (AFS)

**Everything is Context.**

AFS is an experimental orchestration layer for managing multi-agent swarms and context loops directly within the filesystem. It treats documentation, tools, and memory as mountable context nodes, allowing AI agents to "live" and operate within the repository.

## Core Concepts

- **Context Mounting:** Dynamically load relevant documentation, code, and memories into the agent's working context.
- **Agent Swarms:** Orchestrate multiple specialized agents (Planner, Coder, Critic) to solve complex tasks.
- **File-Based Memory:** Store long-term memories and project knowledge in a structured, human-readable file system.
- **Tool Integration:** Expose command-line tools and scripts as callable functions for agents.

## Architecture

AFS is built on a modular architecture:

- **Core:** The central engine that manages context, agents, and tool execution.
- **Services:** Pluggable modules for specific functionalities (e.g., LLM providers, vector databases).
- **Tools:** Scripts and executables that agents can invoke to perform actions.
- **Context:** The structured file system where agents operate and store information.

## Profiles and Extensions

Core `afs` now supports profile-driven context injection via `afs.toml`.

- Use `[profiles]` + `[profiles.<name>]` to control `knowledge_mounts`, `skill_roots`, and `model_registries`.
- Use `[extensions]` + `extensions/*/extension.toml` to load external adapters (e.g. `afs_google`) without forking core files.
- Use `[hooks]` (`before_context_read`, `after_context_write`, `before_agent_dispatch`) for grounding policies.

Inspect and apply profiles:

```bash
afs context profile-show --profile work
afs context profile-apply --profile work
afs profile current
afs profile switch work
afs health
```

Run the MCP server for Gemini/other MCP clients:

```bash
python -m afs.mcp_server
# or
afs mcp serve
```

## Getting Started

1.  **Installation:**
    ```bash
    git clone https://github.com/scawful/afs.git
    cd afs
    pip install -r requirements.txt
    ```

2.  **Configuration:**
    Configure your LLM providers and other settings in `config/`.

3.  **Running Agents:**
    Use the CLI tools to interact with agents and manage context.

## AFS Studio variants

There are two "AFS Studio" implementations:

- **Python TUI (this repo):** Run with `afs-studio` (shell function in `config/dotfiles/zsh/65-afs.zsh`) or `python -m afs studio run --build` from `lab/afs`. Best for development and context-heavy workflows.
- **C++ ImGui (afs_suite):** Built from `lab/afs_suite/apps/studio/`. Launched via the unified `afs` CLI (`afs studio` or `afs launch afs_studio`) and from Barista (⌘⌥S). Defined in `shared/cpp/afs_core/resources/afs_apps.json`. Best for a standalone desktop operations app.

The unified CLI in `tools/afs/` uses the manifest for app launch; "studio" there always means the C++ app. Use the shell function `afs-studio` for the Python TUI.

## lab/afs vs lab/afs-scawful

- **lab/afs** — Core research prototype: orchestration, context mounts, Python CLI and TUI. Shared upstream-friendly code.
- **lab/afs-scawful** — Personal fork: scripts (e.g. afs-studio wrappers, afs-warm), policies, agent instructions, chat registry. Overrides and machine-specific config live here; the AFS CLI and Barista can read from it for "my defaults."

## Documentation

See the `docs/` directory for detailed guides and architectural overviews. Workspace map: `~/src/docs/SOURCE_UNIVERSE_MAP.md`.

## License

MIT
