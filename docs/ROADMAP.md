# ROADMAP

## Current State

AFS now has a usable core operator loop:

- repo-local and shared `.context` roots
- typed mount roles with remapping support
- profile and extension loading
- SQLite-backed context indexing and query
- MCP tools/prompts/resources for Gemini, Codex, Claude, and other clients
- background maintenance agents (`context-warm`, `context-watch`, `agent-supervisor`, `history-memory`)
- session bootstrap, task queue, hivemind, review, and durable memory consolidation
- `afs doctor` for operator-facing diagnostics and repair

## Priority Next

1. Real service installation and lifecycle management
   `launchd` and `systemd` adapters still mainly render units. AFS should be
   able to install, enable, disable, tail logs, and reconcile services without
   dropping to raw system commands.

2. MCP/server refactor
   `src/afs/mcp_server.py` is too large. Split transport, built-in tools,
   prompts/resources, extension loading, and diagnostics into separate modules.

3. Gemini-oriented workflow scaffolding on top of session packs
   The basic model-aware context pack builder exists now. The next layer is
   workflow profiles, tool-profile narrowing, task-at-end prompt shaping, and
   later plan/verify contracts that help Gemini stay disciplined without making
   core AFS Google-specific.

4. Better agent observability
   Extend `afs status` / `afs health` to show what agents produced, what is
   awaiting review, what is stale, and whether clients are actually using the
   bootstrap/status/diff workflow.

5. Stronger sensitivity controls
   Add explicit path-level rules for "never index", "never embed", and "never
   export", especially for governed workspace roots configured in
   `general.workspace_directories`.

## Maintenance Rules

- Prefer tightening the operational core over adding more surface area.
- New public surfaces should reuse existing context/path/policy resolution.
- Docs should describe the current operator workflow, not a historical one.
