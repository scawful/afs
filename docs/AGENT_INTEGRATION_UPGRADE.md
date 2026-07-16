# Agent Integration Upgrade Guide

Use this when refreshing Codex, Claude, Gemini compatibility, Antigravity, hcode, or another local
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

For a full local harness refresh, keep the default catalog slim and let the setup
script sync hcode/OpenCode commands plus the usual Codex/Claude/Gemini/Antigravity harness
state:

```bash
cd ~/src/lab/afs
scripts/afs-upgrade-agent-setup --workspace ~/src --full --setup-hcode
scripts/afs-upgrade-agent-setup --workspace ~/src --full --setup-hcode --apply
```

The script keeps dry-run mode as the default. `--apply --all` performs the
normal local upgrade:

- refreshes the repo venv
- validates `configs/agent_manifest.toml`
- copies shared skills and writes harness manifest exports
- repairs the selected workspace context and rebuilds its SQLite index
- installs idempotent shell hooks for generic harnesses such as `codex`,
  `claude`, `gemini`, `antigravity`, and `hcode`
- installs the background agent-job LaunchAgent
- writes project-scoped Claude and Gemini MCP setup
- syncs the default hcode/OpenCode AFS slash-command pack when hcode setup is
  requested
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

# Preview hcode/OpenCode command sync and bootstrap smoke.
scripts/afs-upgrade-agent-setup --workspace ~/src/project-a --setup-hcode
```

## Minimal Agent Contract

An AFS-aware harness should do this at session start:

1. Run `afs session bootstrap --json`, or call MCP prompt
   `afs.session.bootstrap`.
2. If bootstrap is unavailable, read MCP `context.status`, then query with
   `context.query`; use `context.read`/`context.list` for scratchpad follow-up.
3. Prefer `context.query` before asking the user for context that may already be
   in `scratchpad`, `memory`, or `knowledge`.
4. Write routine working notes to `scratchpad` only.
5. Treat `memory` and `knowledge` as deliberate durable updates.
6. Create a scratchpad handoff file when work spans turns, agents, or tools.
   Use `handoff.create` only in a full-catalog/client-specific flow.
7. Run `afs work --path . --json` when the task involves docs, sheets, tickets,
   planning, people, or review routing.
8. For work-context writing, run `afs work communication preflight` before
   matching the user's tone. Use MCP `work.communication.preflight` only when a
   full-catalog client explicitly exposes it.
9. For external writes, create or reuse an AFS work approval request and execute
   exactly one approved action with `afs work approvals execute`.

Do not start background agents, hivemind coordination, embeddings, training
workflows, or domain MCP servers just because AFS is present. Those are opt-in
surfaces for tasks that explicitly need them.

## Default MCP Surface

Keep the default MCP set small:

- `context.status`
- `context.query`
- `context.read`
- `context.write`
- `context.list`

`afs.session.bootstrap`, `afs.session.pack`, and scratchpad review are prompts,
not default `tools/list` entries. Work preflight, approvals, repair, handoff,
and verification should route through CLI/framework hints unless a client is
explicitly launched with `afs mcp serve --tool-catalog full` or
`AFS_MCP_TOOL_CATALOG=full`.

Optional surfaces should be profile-gated or harness-specific:

- `agent.*` and `agent.job.*` for background work
- `hivemind.*` for cross-agent coordination
- `events.*` for audits and telemetry
- `embeddings.*` for semantic indexing
- `training.*` for reusable training/eval workflows
- companion-repo domain servers, for example the MCP surfaces supplied by a
  local `afs_example` or `afs_company` repo

## Skills

`afs agent-manifest sync` copies canonical skill directories into harness skill
roots. It intentionally does not rely on symlinks, because not every harness
loads symlinked skill folders consistently.

The same manifest sync can copy default OpenCode slash-command packs into
harness command roots such as `~/src/company-agent/.opencode/command`. These
commands keep models on the slim MCP surface by default and route heavier
actions through CLI/framework commands. Command packs are additive by default:
existing customized command files are reported as `customized` and left
untouched unless a pack explicitly sets `overwrite = true`.

Current shared skills are declared in `configs/agent_manifest.toml`. Refresh
them with:

```bash
cd ~/src/lab/afs
scripts/afs agent-manifest sync --apply
scripts/afs agent-manifest sync --harness hcode --apply
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

Codex, Claude, Gemini compatibility, Antigravity, hcode, and any companion-repo harnesses should launch
through the repo wrappers when shell hooks are enabled:

```bash
scripts/afs agent-hooks install-shell --apply
```

After opening a new shell, normal commands route through:

- `scripts/afs-codex`
- `scripts/afs-claude`
- `scripts/afs-gemini`
- `scripts/afs-hcode`

Bypass functions remain available in that shell:

- `codex-raw`
- `claude-raw`
- `gemini-raw`
- `hcode-raw`

The hook status command always prints what to run next:

```bash
scripts/afs agent-hooks status --path ~/src/project-a
```

## Work Assistant Upgrade

The work-assistant layer is native AFS state, not a broad MCP administration
surface. Upgrade agents by teaching them the small command contract:

```bash
scripts/afs work --path .
scripts/afs work communication list --path .
scripts/afs work communication guide --path .
scripts/afs work communication preflight --path .
scripts/afs work approvals list --path .
scripts/afs work approvals request --path . ...
scripts/afs work approvals approve <approval-id> --path . \
  --because "preview and target verified"
scripts/afs work approvals execute <approval-id> --path . --dry-run --json
scripts/afs work approvals execute <approval-id> --path . --executor "<connector command>"
```

Use `docs/WORK_ASSISTANT_UPGRADE.md` as the copy-paste guide for harness
instructions and connector setup. `afs-client-session` also exports
`AFS_SESSION_WORK_HINT` and `AFS_SESSION_WORK_APPROVALS_HINT` so wrappers can
show the exact commands at startup. `AFS_SESSION_WORK_COMMUNICATION_HINT`
points editor and harness surfaces at the communication preflight command.
