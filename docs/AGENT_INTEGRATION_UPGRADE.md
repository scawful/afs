# Agent Integration Upgrade Guide

Use this when refreshing Codex, Claude, Gemini, hcode, z3cli, or another local
agent harness to follow AFS without adding unnecessary tool noise.

## Upgrade Command

Preview first:

```bash
cd ~/src/lab/afs
scripts/afs-upgrade-agent-setup --workspace ~/src
```

Apply the common local setup:

```bash
cd ~/src/lab/afs
scripts/afs-upgrade-agent-setup --workspace ~/src --apply --all
```

The script keeps dry-run mode as the default. `--apply --all` performs the
normal local upgrade:

- refreshes the repo venv
- validates `configs/agent_manifest.toml`
- copies shared skills and writes harness manifest exports
- repairs the selected workspace context and rebuilds its SQLite index
- installs idempotent shell hooks for `codex`, `claude`, `gemini`, `hcode`, and
  `z3cli`
- installs the background agent-job LaunchAgent
- writes project-scoped Claude and Gemini MCP setup
- prints the exact status, inbox, and bootstrap commands to run next

Narrow examples:

```bash
# Copy skills/exports for only Codex and Claude.
scripts/afs-upgrade-agent-setup --apply --harness codex --harness claude

# Refresh MCP setup only, without worker installation.
scripts/afs-upgrade-agent-setup --workspace ~/src/project-a --apply \
  --setup-claude --setup-gemini --rebuild-index

# Inspect hooks and context health without writing anything.
scripts/afs-upgrade-agent-setup --workspace ~/src/project-a --skip-venv
```

## Minimal Agent Contract

An AFS-aware harness should do this at session start:

1. Run `afs session bootstrap --json`, or call MCP prompt
   `afs.session.bootstrap`.
2. If bootstrap is unavailable, read MCP `context.status`, `context.diff`, and
   then query with `context.query`.
3. Prefer `context.query` before asking the user for context that may already be
   in `scratchpad`, `memory`, or `knowledge`.
4. Write routine working notes to `scratchpad` only.
5. Treat `memory` and `knowledge` as deliberate durable updates.
6. Create a handoff with `handoff.create` or a scratchpad handoff file when work
   spans turns, agents, or tools.

Do not start background agents, hivemind coordination, embeddings, training
workflows, or domain MCP servers just because AFS is present. Those are opt-in
surfaces for tasks that explicitly need them.

## Default MCP Surface

Keep the default MCP set small:

- `afs.session.bootstrap`
- `context.status`
- `context.query`
- `context.read`
- `context.write`
- `context.list`
- `context.diff`
- `context.index.rebuild`
- `handoff.create`

Optional surfaces should be profile-gated or harness-specific:

- `agent.*` and `agent.job.*` for background work
- `hivemind.*` for cross-agent coordination
- `events.*` for audits and telemetry
- `embeddings.*` for semantic indexing
- `training.*` for reusable training/eval workflows
- domain servers such as `hyrule-historian`, `book-of-mudora`, `yaze-mcp`,
  `yaze-debugger`, and `yaze-editor`

## Skills

`afs agent-manifest sync` copies canonical skill directories into harness skill
roots. It intentionally does not rely on symlinks, because not every harness
loads symlinked skill folders consistently.

Current shared skills are declared in `configs/agent_manifest.toml`. Refresh
them with:

```bash
cd ~/src/lab/afs
scripts/afs agent-manifest sync --apply
```

Validate after editing skills or manifest entries:

```bash
scripts/afs agent-manifest validate --check-paths
scripts/afs skills list
```

## Context Placement

Use repo-local `.context/` when the repo can own its context. This is preferred
for normal `~/src` development because project scratchpad, memory, and handoffs
stay near the code.

Use global `~/.context` when the workspace cannot contain `.context/`, such as
large managed work codebases. In that case, keep `AFS_CONTEXT_ROOT` or
`general.context_root` explicit so agents do not silently drift between context
trees.

Useful repair commands:

```bash
scripts/afs status --start-dir ~/src/project-a
scripts/afs context repair --path ~/src/project-a --rebuild-index --json
scripts/afs index rebuild --path ~/src/project-a --json
scripts/afs query "handoff" --path ~/src/project-a --mount scratchpad
```

## Harness Notes

Codex, Claude, Gemini, hcode, and z3cli should all launch through the repo
wrappers when shell hooks are enabled:

```bash
scripts/afs agent-hooks install-shell --apply
```

After opening a new shell, normal commands route through:

- `scripts/afs-codex`
- `scripts/afs-claude`
- `scripts/afs-gemini`
- `scripts/afs-hcode`
- `scripts/afs-z3cli`

Bypass functions remain available in that shell:

- `codex-raw`
- `claude-raw`
- `gemini-raw`
- `hcode-raw`
- `z3cli-raw`

The hook status command always prints what to run next:

```bash
scripts/afs agent-hooks status --path ~/src/project-a
```
