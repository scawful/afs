# AFS — Agentic File System

AFS is an orchestration layer for managing multi-agent systems and context directly within the filesystem. It treats documentation, tools, and memory as mountable context nodes, providing a structured surface for AI agents to operate within a repository.

## Install

```bash
git clone https://github.com/scawful/afs.git
cd afs
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

Use the wrapper script for reliable agent invocation (sets `AFS_ROOT` and `PYTHONPATH`):

```bash
./scripts/afs --help
```

## Quick Start

```bash
./scripts/afs init                    # Initialize AFS configuration
./scripts/afs context init            # Create .context directory structure
./scripts/afs doctor                  # Diagnose and auto-fix issues
./scripts/afs health                  # Health check
```

## Branching

AFS uses a staged integration flow across `features`, `development`, and `main`.
See `docs/development.md` for PR target and promotion guidance.

## Core Concepts

**Context Mounting** — Structured `.context/` directories with typed mounts (knowledge, skills, scratchpad, memory, tasks) that agents can read and write.

**Session System** — Token-budgeted context packs, bootstrap summaries, and client harness for Gemini, Claude, and Codex integrations.

**Agent Supervisor** — Background agent lifecycle management with dependency graphs, restart with exponential backoff, mutex groups, and agent-to-agent handoff.

**Hivemind** — Inter-agent message bus for coordination, with pub/sub channels and structured message routing.

**Memory Consolidation** — Event history rolled up into durable memory entries, with optional LLM-assisted summarization.

**Profiles & Extensions** — Profile-driven context injection via `afs.toml`. Extensions add domain-specific functionality without forking core.

## Architecture

```
src/afs/
├── cli/              # 30+ CLI command groups
├── agents/           # 17 background agents + supervisor
├── mcp_server.py     # 40+ MCP tools for external clients
├── context_index.py  # SQLite-backed context indexing and search
├── context_pack.py   # Token-budgeted context packs with caching
├── session_*.py      # Session bootstrap, harness, workflows
├── memory_*.py       # Memory consolidation and LLM summarization
├── hivemind.py       # Inter-agent message bus
├── handoff.py        # Structured session handoff protocol
├── embeddings.py     # Embedding index with Gemini provider
├── services/         # launchd/systemd service adapters
├── gates/            # Quality gates and CI integration
├── continuous/       # A/B testing, triggers, continuous learning
└── ...
```

## CLI

### Context & Workspace

```bash
afs context discover                  # Find .context roots
afs context mount <path>              # Mount a context directory
afs context status                    # Show mount status and health
afs context query "search term"       # Search the context index
afs context diff                      # Changes since last session
afs context pack --model gemini       # Token-budgeted context export
```

### Agents

```bash
afs agents list                       # Available agents
afs agents ps                         # Running agents
afs agents run <name> [--prompt ...]  # Run an agent
afs agents capabilities               # Agent capability matrix
afs agent-manifest validate           # Validate harness/skill/MCP manifest
afs agent-runs start "task"           # Record a replayable agent run
afs agent-jobs create "task"          # Queue a markdown background job
```

`session bootstrap` includes manifest, run, and job state; MCP clients can use
`agent.manifest.show`, `agent.run.*`, and `agent.job.*` directly.

### Session

```bash
afs session bootstrap --json          # Full session context summary
afs session pack "task" --model gemini --json
afs session prepare-client --client codex --json
```

### Memory & Events

```bash
afs memory status                     # Memory consolidation stats
afs memory consolidate                # Roll history into memory
afs events query --last 50            # Recent events
afs events analytics                  # Event statistics
```

### Profiles

```bash
afs profile list                      # Available profiles
afs profile switch work               # Activate a profile
afs profile current                   # Show active profile
```

### Embeddings

```bash
afs embeddings index --provider gemini --include "*.md"
afs embeddings search "how to debug a sprite"
```

### Health & Diagnostics

```bash
afs doctor --fix                      # Diagnose and repair
afs health                            # System health check
afs services status --system          # Service status
```

## MCP Server

AFS exposes 40+ tools via MCP for use by Gemini, Claude, Codex, and other MCP clients.

```bash
afs mcp serve                         # Start MCP server
# Or via direct module entrypoint (preferred for Claude Desktop):
.venv/bin/python -m afs.mcp_server
```

**Tool categories:** `fs.*`, `context.*`, `session.*`, `memory.*`, `handoff.*`, `agent.*`, `hivemind.*`, `task.*`, `review.*`, `events.*`

See [docs/MCP_SERVER.md](docs/MCP_SERVER.md) for configuration and tool reference.

## Agents

| Agent | Purpose |
|-------|---------|
| `agent-supervisor` | Lifecycle management, dependency graph, restart with backoff |
| `context-warm` | Background context warming and embedding indexing |
| `mission-runner` | TOML mission definitions with OODA execution phases |
| `journal-agent` | Draft hybrid weekly reviews from thoughts and active tasks |
| `workspace-analyst` | Codebase health, git drift, dependency scanning |
| `gemini-workspace-brief` | Gemini-powered workspace briefings |
| `dashboard-export` | Data export for status bar and dashboard surfaces |
| `tether-bridge` | Agent findings to ADHD-friendly capture items |
| `history-memory` | Consolidate event history into durable memory |
| `context-audit` | Audit contexts for missing directories |
| `context-inventory` | Inventory contexts and mount counts |
| `scribe-draft` | Draft responses via configured chat model |
| `researcher` | Research agent with structured output |

## Gemini Integration

```bash
afs gemini setup                      # Register MCP for Gemini CLI
afs gemini status                     # Check API key, SDK, embeddings
afs gemini context "search query"     # Generate context for Gemini session
afs gemini context --include-content  # With full file content
```

Install: `pip install -e ".[gemini]"`

## Client Wrappers

AFS ships wrapper scripts that inject session context into native clients:

```bash
./scripts/afs-claude --prompt "task description"
./scripts/afs-codex --prompt "task description"
./scripts/afs-gemini --prompt "task description"
```

Wrappers export `AFS_SESSION_BOOTSTRAP_*`, `AFS_SESSION_PACK_*`, `AFS_SESSION_SYSTEM_PROMPT_*` and emit lifecycle events (`user_prompt_submit`, `turn_started`, `turn_completed`).

## Configuration

`afs.toml` in the project root:

```toml
[general]
context_root = "~/.context"

[profiles.work]
knowledge_mounts = ["~/docs/work"]
skill_roots = ["~/skills"]

[context_index]
enabled = true
auto_index = true
include_content = true

[hooks]
session_start = ["echo 'session started'"]
```

## Extensions

Domain-specific functionality (model training, persona configurations, deployment playbooks) goes in extension packages. See [docs/EXTENSION_MIGRATION.md](docs/EXTENSION_MIGRATION.md).

## Documentation

- [docs/index.md](docs/index.md) — Documentation index
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — System architecture
- [docs/CLI_REFERENCE.md](docs/CLI_REFERENCE.md) — Full CLI reference
- [docs/AGENT_SURFACES.md](docs/AGENT_SURFACES.md) — Agent system design
- [docs/MCP_SERVER.md](docs/MCP_SERVER.md) — MCP server setup and tools
- [docs/PROFILES.md](docs/PROFILES.md) — Profile system
- [docs/EMBEDDINGS.md](docs/EMBEDDINGS.md) — Embedding system
- [docs/MEMORY_SYSTEM.md](docs/MEMORY_SYSTEM.md) — Memory consolidation
- [docs/EMACS_INTEGRATION.md](docs/EMACS_INTEGRATION.md) — Emacs/Spacemacs helper

## License

MIT
