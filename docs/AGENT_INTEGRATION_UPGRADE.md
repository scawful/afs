# Agent Integration Upgrade Guide

See [Compact startup and extension contracts](AGENT_CONTRACTS.md) for the short
bootstrap, conditional writes, content-bound approvals, and extension migration.

Use this when refreshing Codex, Claude, Gemini compatibility, Antigravity, hcode, or another local
agent harness to follow AFS without adding unnecessary tool noise.

## Upgrade Command

Preview first:

```bash
cd /path/to/afs
scripts/afs-upgrade-agent-setup --workspace /path/to/workspace
```

Apply the common local setup:

```bash
cd /path/to/afs
scripts/afs-upgrade-agent-setup --workspace /path/to/workspace --apply --all
```

For a full local harness refresh, keep the default catalog slim. Hcode/OpenCode
is an explicit opt-in because its checkout may live anywhere or may not be
installed on a given computer:

```bash
cd /path/to/afs
scripts/afs-upgrade-agent-setup --workspace /path/to/workspace --full
scripts/afs-upgrade-agent-setup --workspace /path/to/workspace --full --apply

# Optional OpenCode integration
scripts/afs-upgrade-agent-setup --workspace /path/to/workspace \
  --setup-hcode --halext-code /path/to/halext-code --apply
```

The script keeps dry-run mode as the default. `--apply --all` performs the
normal local upgrade:

- refreshes the repo venv
- validates `configs/agent_manifest.toml`
- copies explicitly targeted shared skills/commands and writes explicitly
  targeted manifest exports
- repairs the selected workspace context and rebuilds its SQLite index
- installs idempotent shell hooks for the selected local harnesses
- installs the background agent-job LaunchAgent
- writes project-scoped Claude and Gemini MCP setup
- syncs the default hcode/OpenCode AFS slash-command pack when hcode setup is
  requested
- prints the exact status, inbox, and bootstrap commands to run next

Narrow examples:

```bash
# Copy the AFS skill to one explicitly chosen harness directory.
scripts/afs agent-manifest sync --harness codex \
  --skill-root codex=/path/to/codex/skills --apply

# Refresh MCP setup only, without worker installation.
scripts/afs-upgrade-agent-setup --workspace /path/to/project --apply \
  --setup-claude --setup-gemini --rebuild-index

# Inspect hooks and context health without writing anything.
scripts/afs-upgrade-agent-setup --workspace /path/to/project --skip-venv

# Preview hcode/OpenCode command sync and bootstrap smoke.
scripts/afs-upgrade-agent-setup --workspace /path/to/project \
  --setup-hcode --halext-code /path/to/halext-code
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

The same manifest sync can copy default OpenCode slash-command packs into an
explicit harness command root. These
commands keep models on the slim MCP surface by default and route heavier
actions through CLI/framework commands. Command packs are additive by default:
existing customized command files are reported as `customized` and left
untouched unless a pack explicitly sets `overwrite = true`.

The default manifest contains repo-owned sources but deliberately leaves
machine-specific destination roots empty. Supply destinations at sync time, or
point `AFS_AGENT_MANIFEST` at a user/organization manifest:

```bash
cd /path/to/afs
scripts/afs agent-manifest sync --harness hcode \
  --skill-root hcode=/path/to/halext-code/.opencode/skills \
  --command-root hcode=/path/to/halext-code/.opencode/commands \
  --apply
```

Validate after editing skills or manifest entries:

```bash
scripts/afs agent-manifest validate --check-paths
scripts/afs skills list
```

## Context Placement

Use repo-local `.context/` when the repo can own its context and local placement
fits the environment.

Use a configured central context root when the workspace cannot contain
`.context/`, such as a managed work codebase. AFS does not require that root to
be `~/.context`; keep `AFS_CONTEXT_ROOT` or `general.context_root` explicit so
agents do not silently drift between context trees.

Useful repair commands:

```bash
scripts/afs status --start-dir /path/to/project
scripts/afs context repair --path /path/to/project --rebuild-index --json
scripts/afs index rebuild --path /path/to/project --json
scripts/afs query "handoff" --path /path/to/project --mount scratchpad
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
scripts/afs agent-hooks status --path /path/to/project
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
