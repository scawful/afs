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
./scripts/afs context profile-show --profile work
./scripts/afs context profile-apply --profile work
./scripts/afs profile current
./scripts/afs profile switch work
./scripts/afs health
```

Run the MCP server for Gemini/other MCP clients:

```bash
./scripts/afs mcp serve
# or, after installing the package into the active environment
afs mcp serve
```

## Gemini Integration

AFS has first-class Gemini support: embeddings, MCP registration, and context generation.

```bash
# One-time setup — registers AFS MCP in ~/.gemini/settings.json
afs gemini setup

# Check readiness (API key, SDK, MCP, embeddings)
afs gemini status
afs gemini status --project afs --context-root ~/src/lab/.context

# Index knowledge with Gemini embedding vectors
afs embeddings index --knowledge-path ~/.context/knowledge --provider gemini --include "*.md"

# Semantic search
afs embeddings search --knowledge-path ~/.context/knowledge --provider gemini "how to debug a sprite"

# Generate context for a Gemini session
afs gemini context "sprite development"                    # search mode
afs gemini context "sprite development" --include-content  # with full content
afs gemini context --project afs "sqlite index"            # search a specific project subtree
afs gemini context                                         # full knowledge index

# Skip Google Workspace lookups in the morning briefing
afs briefing --no-gws

# Optional Google Workspace helper commands
afs gws status
afs gws agenda
```

Install the optional Gemini dependency: `pip install -e ".[gemini]"`

See `docs/EMBEDDINGS.md` for the full embedding system reference.

Tune context index defaults in `afs.toml`:

```toml
[context_index]
enabled = true
db_filename = "context_index.sqlite3"
auto_index = true
auto_refresh = true
include_content = true
max_file_size_bytes = 262144
max_content_chars = 12000
```

## Getting Started

1.  **Installation:**
    ```bash
    git clone https://github.com/scawful/afs.git
    cd afs
    python3 -m venv .venv
    source .venv/bin/activate
    python3 -m pip install -e .
    ```

2.  **Preferred local entrypoint:**
    ```bash
    ./scripts/afs --help
    ```
    The wrapper sets `AFS_ROOT` and `PYTHONPATH`, which is more reliable for agents than
    depending on the shell's default `python`.

3.  **Configuration:**
    Initialize config and context:
    ```bash
    ./scripts/afs init
    ./scripts/afs context init
    ```

4.  **Running Agents:**
    Use the CLI tools to manage context, profiles, skills, embeddings, and MCP.

## AFS Studio variants

There are two "AFS Studio" implementations:

- **Python TUI (this repo):** Run with `afs-studio` (shell function in `config/dotfiles/zsh/65-afs.zsh`) or `python -m afs studio run --build` from `lab/afs`. Best for development and context-heavy workflows.
- **C++ ImGui (afs_suite):** Built from `lab/afs_suite/apps/studio/`. Launched via the unified `afs` CLI (`afs studio` or `afs launch afs_studio`) and from Barista (⌘⌥S). Defined in `shared/cpp/afs_core/resources/afs_apps.json`. Best for a standalone desktop operations app.

The unified CLI in `tools/afs/` uses the manifest for app launch; "studio" there always means the C++ app. Use the shell function `afs-studio` for the Python TUI.

## lab/afs vs lab/afs-scawful

- **lab/afs** — Core research prototype: orchestration, context mounts, Python CLI and TUI. Shared upstream-friendly code.
- **lab/afs-scawful** — Personal fork: scripts (e.g. afs-studio wrappers, afs-warm), policies, agent instructions, chat registry, persona gateway, and MoE/orchestration surfaces. Overrides and machine-specific config live here; the AFS CLI and Barista can read from it for "my defaults."

## Documentation

See the `docs/` directory for detailed guides and architectural overviews.
Recommended starting points:

- `docs/index.md`
- `docs/AGENT_SURFACES.md`
- `docs/CLI_REFERENCE.md`
- `docs/PROFILES.md`

Workspace map: `~/src/docs/SOURCE_UNIVERSE_MAP.md`.

## License

MIT
