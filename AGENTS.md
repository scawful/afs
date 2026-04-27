# AGENTS.md

Purpose: make AFS itself get used in an AFS-first way.

Startup Contract
1. Start with `./scripts/afs session bootstrap --json` or the MCP prompt `afs.session.bootstrap`.
2. If bootstrap is unavailable, call `context.status`, then `context.diff`, then `afs.scratchpad.review`.
3. Read scratchpad state and deferred notes before major edits.
4. Use `context.query` before asking for context that may already be in memory, knowledge, or scratchpad.
5. Prefer scratchpad/task/hivemind updates for handoff instead of ad hoc summaries.

Working Rules
1. Clarify goals, constraints, and done criteria before major edits.
2. Prefer the smallest working change over architecture churn.
3. Touch only task-related files.
4. Keep hygiene high: no dead code or commented-out leftovers.
5. Run the fastest relevant verification command before finishing.
6. If checks cannot run, report exactly why and residual risk.
7. Ask before destructive actions (`rm`, force-push, history rewrite).

Delivery Contract
- Report what changed.
- Report what was verified.
- Report known gaps or follow-ups.

AFS Defaults
- Treat `scratchpad` as the default writable working area.
- Treat `memory` and `knowledge` as deliberate, durable updates.
- Use `items` for queued work and `hivemind` for cross-agent handoffs only when a task spans turns or tools.
- Do not start training, embeddings, background agents, or domain MCP tooling unless the task explicitly needs that surface.

Reference Material
- Agent harness upgrade guide: `docs/AGENT_INTEGRATION_UPGRADE.md`
- Detailed agent/runtime docs: `docs/AGENT_SURFACES.md`
- MCP surface docs: `docs/MCP_SERVER.md`
- CLI docs: `docs/CLI_REFERENCE.md`
