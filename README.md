# AFS — Agentic File System

[![CI](https://github.com/scawful/afs/actions/workflows/ci.yml/badge.svg)](https://github.com/scawful/afs/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.2.0-blue.svg)](CHANGELOG.md)

AFS is an orchestration layer for managing multi-agent systems and context directly within the filesystem. It treats documentation, tools, and memory as mountable context nodes, providing a structured surface for AI agents to operate within a repository.

AFS is an independent implementation and extension of the agentic file system abstraction from ["Everything is Context" (arXiv:2512.05470)](https://arxiv.org/abs/2512.05470) — see [docs/LINEAGE.md](docs/LINEAGE.md) for the concept mapping and where AFS goes beyond the paper.

## Install

Fast path for a fresh checkout:

```bash
git clone https://github.com/scawful/afs.git
cd afs
make setup
./scripts/afs --help
```

Equivalent manual setup:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -e .
```

Use the wrapper script for reliable local development and agent invocation; it sets repo-local environment defaults before dispatching to the package:

```bash
./scripts/afs --help
```

## Quick Start

```bash
make check                            # lint, tests, package smoke
./scripts/afs setup                   # Guided setup wizard
./scripts/afs guide                   # Friendly workflow menu
./scripts/afs init                    # Initialize AFS configuration
./scripts/afs context init            # Create .context directory structure
./scripts/afs status --start-dir "$PWD"  # Show context, mount, and index health
./scripts/afs doctor                  # Diagnose and auto-fix issues
./scripts/afs health                  # Health check
```

Refresh local agent harnesses, MCP setup, copied skills, hooks, and context
indexes with a dry-run first:

```bash
./scripts/afs-upgrade-agent-setup --workspace ~/src
./scripts/afs-upgrade-agent-setup --workspace ~/src --apply --all
```

## Branching and releases

AFS uses a staged integration flow across `features`, `development`, and `main`.
See `docs/development.md` for PR target and promotion guidance, and `RELEASE.md` for the release/tag checklist.

Current release line: `0.2.x` pre-1.0 core platform.

## Core Concepts

**Context Mounting** — Structured `.context/` directories with typed mounts (knowledge, skills, scratchpad, memory, tasks) that agents can read and write.

**Session System** — Token-budgeted context packs, bootstrap summaries, and client harness for Gemini, Claude, and Codex integrations.

**Agent Operations** — Optional run records, safe background job queues, and
handoffs for work that spans turns or harnesses.

**Workflow Assistant** — Context-local people, relationship, review-route,
approval, and activity records for documents, sheets, tickets, planning, and
other non-technical workflows.

**Hivemind** — Optional inter-agent message bus for tasks that explicitly need
cross-agent coordination.

**Memory Consolidation** — Event history rolled up into durable memory entries, with optional LLM-assisted summarization.

**Profiles & Extensions** — Profile-driven context injection via `afs.toml`. Extensions add domain-specific functionality without forking core.

**Context Sources** — Provider-neutral adapters for tasks, tickets, reviews, docs, messages, tests, hooks, and traces. Core AFS owns the normalized records; concrete source connectors live in extensions.

**Optimization Evidence** — Versioned, language-neutral evaluation and policy
records plus a pure decision gate for bounded hill-climbing experiments. The
gate can recommend human review but cannot execute or promote a candidate.

**Policy-Checked Execution** — Typed, hash-bound execution requests inspected
against trusted policy before a portable process backend launches them. The
current backend scrubs environment state and bounds time/output, but is not a
security sandbox.

## Professional/project docs

- [Executive Summary](docs/EXECUTIVE_SUMMARY.md)
- [Lineage](docs/LINEAGE.md)
- [Setup Guide](docs/SETUP_GUIDE.md)
- [Extension Authoring](docs/EXTENSION_AUTHORING.md)
- [Autonomous Optimization Protocol](docs/OPTIMIZATION_PROTOCOL.md)
- [Policy-Checked Execution](docs/EXECUTION_BROKER.md)
- [Contributing](CONTRIBUTING.md)
- [Security](SECURITY.md)
- [Release Process](RELEASE.md)
- [Roadmap](ROADMAP.md)
- [Changelog](CHANGELOG.md)

## Architecture

```
src/afs/
├── cli/              # 30+ CLI command groups
├── agents/           # optional background agents + supervisor
├── execution/        # typed policy checks + bounded process backend
├── mcp_server.py     # MCP prompts/tools/resources for external clients
├── context_index.py  # SQLite-backed context indexing and search
├── context_pack.py   # Token-budgeted context packs with caching
├── session_*.py      # Session bootstrap, harness, workflows
├── memory_*.py       # Memory consolidation and LLM summarization
├── hivemind.py       # Inter-agent message bus
├── handoff.py        # Structured session handoff protocol
├── embeddings.py     # Embedding index with Gemini provider
├── services/         # launchd/systemd service adapters
├── training/         # Generic dataset/run/eval/feedback primitives
├── sources/          # Provider-neutral context source interfaces
├── mcp/              # MCP extension registry and shared schemas
├── protocols/        # Versioned, language-neutral JSON Schema contracts
└── ...
```

## CLI

### Context & Workspace

```bash
afs context discover                  # Find .context roots
afs context mount <path>              # Mount a context directory
afs status --start-dir "$PWD"         # Show mount status and index health
afs context query "search term"       # Search the context index
afs sources list                      # Extension-owned context source providers
afs sources sync --provider NAME      # Preview provider records into .context/items
afs context diff                      # Changes since last session
afs session pack --model gemini       # Token-budgeted context export
```

### Execution & Verification

```bash
afs execution inspect --request request.json --allowed-root "$PWD" \
  --allowed-executable python3 --json
afs verify plan --cwd "$PWD" --json   # Inspect selected structured checks
afs verify run --cwd "$PWD" --json    # Run checks through the broker
```

Execution inspection never launches the request, and AFS intentionally exposes
no generic execution CLI. Executable permission is explicit; omitting
`--allowed-executable` returns a blocked inspection. See [Policy-Checked
Execution](docs/EXECUTION_BROKER.md) for the typed Python API and backend limits.

### Agents

```bash
afs agents list                       # Available agents
afs agents ps                         # Running agents
afs agents run <name> [--prompt ...]  # Run an agent
afs agents capabilities               # Agent capability matrix
afs agent-manifest validate           # Validate harness/skill/MCP manifest
afs agent-manifest sync --apply       # Copy shared skills and write harness exports
afs agent-hooks install-shell --apply # Route harness commands through AFS wrappers
afs agent-hooks install-worker --apply --load  # Run queued jobs automatically
afs agent-runs start "task"           # Record a replayable agent run
afs agent-jobs create "task"          # Queue a markdown background job
afs agent-jobs status                 # Queue, worker, run, and watchdog status
afs agent-jobs inbox                  # Review completed, failed, stale, or blocked jobs
afs agent-jobs review <job-id>        # Inspect one job and its linked run record
afs agent-jobs promote <job-id> --to-handoff  # Save a job review into scratchpad/handoffs
afs agent-jobs archive <job-id>       # Archive a handled job without deleting it
afs agent-jobs seed                   # Idempotently queue safe maintenance jobs
afs agent-jobs work --agent codex --command '...'  # Claim and execute queued jobs
```

`session bootstrap` includes manifest, run, and job state; MCP clients can use
`agent.manifest.show`, `agent.run.*`, and `agent.job.*` directly.

### Work Assistant

```bash
afs work                              # People/review/approval summary
afs work people list                  # Known work-scoped people
afs work reviewers --target-type docs # Suggested reviewers
afs work approvals list               # Pending external-write approvals
afs work approvals execute <id> --dry-run
afs work approvals execute <id> --executor "python3 scripts/afs-work-gws-executor.py"
afs work activity list                # Recent work-assistant activity
```

Work-assistant state is native to AFS and backed by
`.context/global/work_assistant.sqlite3`. It creates approval records for
external writes instead of editing shared docs, sheets, tickets, or messages
directly. Approved actions can be handed to explicit local connector commands
with `afs work approvals execute`.

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

AFS exposes a small recommended MCP surface for normal agent work, with broader
agent, hivemind, events, embeddings, and training tools available for harnesses
that explicitly need them.

```bash
afs mcp serve                         # Start MCP server
# Or via direct module entrypoint (preferred for Claude Desktop):
.venv/bin/python -m afs.mcp_server
```

Recommended default tools/prompts: `afs.session.bootstrap`, `context.status`,
`context.query`, `context.read`, `context.write`, `context.list`,
`context.diff`, `context.index.rebuild`, and `handoff.create`.

Optional categories: `agent.*`, `hivemind.*`, `task.*`, `review.*`,
`events.*`, `embeddings.*`, and `training.*`.

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
| `index-rebuild` | Refresh the knowledge/memory SQLite index after source changes |
| `skills-mine` | Mine repeated successful traces into reviewable skill candidates |
| `morning-briefing` | Write a network-free daily digest to the configured scratchpad |
| `scribe-draft` | Draft responses via configured chat model |
| `researcher` | Research agent with structured output |

When the active profile has no `agent_configs`, the supervisor supplies a
conservative four-agent default set. It performs a network-free daily context
audit, watches the configured knowledge/memory roots, mines skills weekly, and
writes a daily briefing. Existing custom agent lists are never augmented.
Disable defaults with `[agents] default_set = false` or
`AFS_DEFAULT_AGENTS=off`. Starting the supervisor remains explicit.

## Gemini Integration

```bash
afs antigravity setup --scope project    # Preview Antigravity CLI MCP setup
afs gemini setup                         # Gemini CLI compatibility/API helper
afs antigravity models --json            # Parse the installed agy model list
```

Gemini CLI compatibility is retained for API-key/enterprise workflows, but the
individual/free/Pro/Ultra public path moved to Antigravity CLI (`agy`) on
2026-06-18. AFS does not auto-install `agy`; run `afs antigravity status` or
`afs antigravity setup --json` to inspect the local state. Current `agy` builds
use `~/.gemini/config/mcp_config.json` as the migrated MCP config path.

```
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

`skill_roots` are an instruction trust boundary: task-matched `SKILL.md`
bodies can enter generated system prompts. Configure only roots whose content
you trust. AFS delivers at most three bodies, capped at 2,000 characters each
and 6,000 characters in aggregate; additional matches remain metadata-only.
Skill files over 64,000 characters, names over 256 characters, or metadata
lists over 16 items of 256 characters each are rejected during discovery so
enforcement and verification rules are never silently clipped.

## Extensions

Domain-specific functionality (model training, persona configurations, deployment playbooks) goes in extension packages. See [docs/EXTENSION_MIGRATION.md](docs/EXTENSION_MIGRATION.md).

## Documentation

- [docs/index.md](docs/index.md) — Documentation index
- [docs/EXECUTIVE_SUMMARY.md](docs/EXECUTIVE_SUMMARY.md) — Shareable non-domain executive summary
- [docs/LINEAGE.md](docs/LINEAGE.md) — Research lineage and concept mapping to arXiv:2512.05470
- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) — System architecture
- [docs/SETUP_GUIDE.md](docs/SETUP_GUIDE.md) — Guided setup, shell helpers, and approachable onboarding
- [docs/CLI_REFERENCE.md](docs/CLI_REFERENCE.md) — Full CLI reference
- [docs/AGENT_INTEGRATION_UPGRADE.md](docs/AGENT_INTEGRATION_UPGRADE.md) — Upgrade agent harnesses, MCP setup, hooks, and copied skills
- [docs/AGENT_SURFACES.md](docs/AGENT_SURFACES.md) — Agent system design
- [docs/MCP_SERVER.md](docs/MCP_SERVER.md) — MCP server setup and tools
- [docs/PROFILES.md](docs/PROFILES.md) — Profile system
- [docs/EMBEDDINGS.md](docs/EMBEDDINGS.md) — Embedding system
- [docs/MEMORY_SYSTEM.md](docs/MEMORY_SYSTEM.md) — Memory consolidation
- [docs/EMACS_INTEGRATION.md](docs/EMACS_INTEGRATION.md) — Emacs/Spacemacs helper

## License

MIT
