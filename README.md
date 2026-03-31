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
- Use `[extensions]` + `extensions/*/extension.toml` to load external adapters (for example, a private workspace adapter) without forking core files.
- Use `[hooks]` for grounding policies and client-session lifecycle hooks, including `session_start`, `session_end`, `user_prompt_submit`, `turn_*`, and `task_*`.

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
# for Claude Desktop, prefer the direct venv module entrypoint
/Users/scawful/src/lab/afs/.venv/bin/python -m afs.mcp_server
```

For Claude Desktop MCP registration, prefer the direct Python module entrypoint
over the shell wrapper. See `docs/MCP_SERVER.md` for the recommended config and
for the `initialize` timeout troubleshooting sequence.

## Gemini Integration

AFS has first-class Gemini support: embeddings, MCP registration, and context generation.

```bash
# One-time setup — user-level Gemini MCP registration
afs gemini setup
afs gemini setup --scope project                     # write ./.gemini/settings.json for the current repo

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

`afs gemini setup` writes the repo wrapper entry by default, so Gemini CLI uses
the same `AFS_ROOT`/`PYTHONPATH` assumptions as `./scripts/afs`. Use
`--python-module` only when you explicitly want `sys.executable -m afs.mcp_server`.

Install the optional Gemini dependency: `pip install -e ".[gemini]"`

See `docs/EMBEDDINGS.md` for the full embedding system reference.
See `docs/EMACS_INTEGRATION.md` for the bundled Emacs/Spacemacs helper.

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
    ./scripts/afs doctor
    ```

4.  **Running Agents:**
    Use the CLI tools to manage context, profiles, skills, embeddings, and MCP.

## Operational Baseline

Use these first when bringing up AFS on a new machine or workspace:

```bash
./scripts/afs session bootstrap --json
./scripts/afs doctor
./scripts/afs health
./scripts/afs services status --system
```

`afs doctor --fix` is the fastest path for repairing missing context roots,
missing required mount directories, untracked/stale mount provenance, and stale
context indexes.

`afs session pack` is the explicit, token-budgeted follow-on surface for
Gemini, Claude, Codex, or generic clients when you intentionally need a
handoff/export packet. Repeated calls with the same bootstrap snapshot and
pack inputs reuse the stored `session_pack_<model>` artifact instead of
rebuilding from scratch.

```bash
./scripts/afs session pack "current task" --model gemini --json
```

### Client Session Harness

Use the session harness when a wrapper, IDE adapter, or child shell script
needs a stable contract for AFS context and lifecycle events.

```bash
./scripts/afs session prepare-client --client codex --json
./scripts/afs session event user_prompt_submit --client codex --session-id "$AFS_SESSION_ID" --prompt "current task"
./scripts/afs-session-notify task_progress --task-id bg-1 --summary "Indexing symbols"
```

- `session prepare-client` writes the shared session payload together with cached bootstrap, pack, and skill-match artifacts.
- Wrappers such as `afs-codex`, `afs-claude`, and `afs-gemini` export `AFS_SESSION_BOOTSTRAP_*`, `AFS_SESSION_PACK_*`, `AFS_SESSION_SKILLS_JSON`, and `AFS_SESSION_CLIENT_PAYLOAD_JSON`.
- When launched with `--prompt`, `--prompt-file`, or `--turn-id`, wrappers also export `AFS_SESSION_EVENT_BIN` / `AFS_SESSION_DEFAULT_TURN_ID` and emit `user_prompt_submit`, `turn_started`, and `turn_completed` / `turn_failed` around the client process.
- Child scripts can call `afs-session-notify` to append `task_*` lifecycle events without rebuilding session context.

## Training Integrations

AFS also exposes training-oriented surfaces backed by the same context and
session systems:

```bash
./scripts/afs training freshness-gate --path ~/src/project-a
./scripts/afs training extract-sessions --path ~/src/project-a --output ./session_replay_training.jsonl
./scripts/afs training generate-router-data --config ~/src/project-a/afs.toml
./scripts/training_watch.sh --debounce 45
```

These cover pre-training freshness checks, extraction of training samples from
recorded AFS sessions, router-dataset generation from declared agent
capabilities, and an optional watch wrapper for local dataset QA workflows.

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
