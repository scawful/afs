# Agent Operations

AFS owns three repo-native surfaces for cross-harness agent coordination.

These surfaces are visible in:

- `afs session bootstrap`
- AFS MCP tools
- `scripts/afs-client-session` wrappers for Codex, Claude, and Gemini
- `afs doctor`

## Manifest

`configs/agent_manifest.toml` is the single source of truth for harnesses, shared skills, MCP servers, startup hints, and shared instruction files.

```bash
./scripts/afs agent-manifest show
./scripts/afs agent-manifest validate
./scripts/afs agent-manifest export codex
./scripts/afs agent-manifest sync --apply
./scripts/afs agent-manifest sync --harness hcode --apply
./scripts/afs agent-hooks install-shell --apply
./scripts/afs agent-hooks install-worker --apply --load
./scripts/afs agent-hooks status --path "$PWD"
./scripts/afs-upgrade-agent-setup --workspace ~/src --apply --all
./scripts/afs-upgrade-agent-setup --workspace ~/src --full --setup-hcode --apply
```

Use this before editing Codex, Claude, Gemini compatibility, Antigravity, hcode, or another
harness-specific config. Harness-specific files can still exist, but they
should point back to this manifest or derive their local view from it.

`scripts/afs-upgrade-agent-setup` is the operator wrapper for a full local
refresh. It defaults to dry-run, then with `--apply` can update the venv, copy
skills and slash-command packs, write manifest exports, repair/rebuild context
state, install hooks, run hcode bootstrap smoke, and write Claude/Gemini MCP
setup. `--full --setup-hcode` is the convenient full local path; it keeps the
MCP default catalog slim and routes richer flows through commands/framework
hints.

`sync` copies manifest-declared shared skill directories into harness skill
roots, copies slash-command packs into command roots, and writes per-harness
JSON export files. It deliberately uses copied files/directories rather than
symlinks. Slash-command packs are additive by default and leave existing
customized commands untouched unless a pack explicitly opts into overwrite.

`agent-hooks install-shell` adds a marked, idempotent block to the shell profile
that sources AFS shell helpers. Use `--helpers-only` for aliases, colors, and
completion without routing AI harness commands. Without `--helpers-only`, a new
shell routes normal generic harness commands such as `codex`, `claude`,
`gemini`, `antigravity`, and `hcode` through the AFS wrappers. Companion extension repos can
add their own local harness wrappers. Run `afs-agent-hooks-off` inside a shell
to disable the functions for that shell. Use raw bypass functions such as
`codex-raw`, `claude-raw`, `gemini-raw`, `antigravity-raw`, or `hcode-raw` when you want the
underlying command directly.

`agent-hooks install-worker` installs a user LaunchAgent for
`agent-jobs work --loop`, so queued background jobs are claimed and executed
without a manual worker command. The worker is intentionally permissive for
normal repo work, but it skips obvious destructive prompts such as broad deletes,
history rewrites, force pushes, or data wipes unless the job or worker is
created with `--allow-destructive`.

`agent-hooks status --path <workspace>` reports whether shell hooks and the
LaunchAgent are installed, then prints exact follow-up commands for installing
missing pieces, checking the watchdog, and opening the review inbox.

MCP:

- `agent.manifest.show`

Doctor:

- `afs doctor` validates the manifest, checks declared paths, compares copied
  skills against canonical skills, and confirms declared MCP server names appear
  in known harness/client config.

## Run Recorder

Agent runs use `scratchpad/common/agent_runs/` in a v2 context and retain the
v1 `scratchpad/agent_runs/` path for compatibility.

```bash
run_id="$(./scripts/afs agent-runs start "Fix settings drift" --harness codex)"
./scripts/afs agent-runs event "$run_id" verification --summary "pytest passed"
./scripts/afs agent-runs finish "$run_id" --summary "patched and verified" --verify "pytest=passed"
./scripts/afs agent-runs list
```

Each record captures task, harness, workspace, prompt, changed files, commands, verification, handoff path, and timestamped events.

The AFS client wrapper starts and finishes these records automatically unless
`AFS_CLIENT_RECORD_RUNS=0` is set.

MCP:

- `agent.run.start`
- `agent.run.list`
- `agent.run.show`
- `agent.run.event`
- `agent.run.finish`

## Background Jobs

Markdown jobs live under `items/agent_jobs/{queue,running,done,failed,archived}/`.

```bash
job_id="$(./scripts/afs agent-jobs create "Review stale instructions" --prompt "Scan docs and report stale model aliases.")"
./scripts/afs agent-jobs claim "$job_id" --agent reviewer
./scripts/afs agent-jobs move "$job_id" done --result "No stale aliases found."
./scripts/afs agent-jobs list
./scripts/afs agent-jobs status
./scripts/afs agent-jobs inbox
./scripts/afs agent-jobs review "$job_id"
./scripts/afs agent-jobs promote "$job_id" --to-handoff
./scripts/afs agent-jobs archive "$job_id"
./scripts/afs agent-jobs seed --profile repo-maintenance
./scripts/afs agent-jobs work --agent local-worker --command 'codex exec < "$AFS_AGENT_JOB_PROMPT_FILE"'
```

Use jobs for background work whose output is independently useful. Each job should include a concrete prompt, scope, and expected output.

Use `agent-jobs status` for a read-only queue and worker watchdog view. It shows
queue counts, runnable jobs, destructive opt-in blockers, stale running jobs,
recent run failures, and LaunchAgent state. The default command reports issues
without blocking automation; add `--strict` when a script should fail on
watchdog warnings.

Use `agent-jobs inbox` for review. It shows completed reports, failed jobs,
stale running jobs, and queued jobs blocked on destructive opt-in, with exact
`agent-jobs review <job-id>` commands. Use `agent-jobs promote <job-id>
--to-handoff` to create a durable common-scope handoff (or the v1
`scratchpad/handoffs/` compatibility file), then
`agent-jobs archive <job-id>` when the output has been handled.

Use `agent-jobs seed` to queue safe report-only maintenance jobs. The
`repo-maintenance` profile covers stale docs/reference scans, skill drift, MCP
tool-name drift, TODO/FIXME summaries, focused verification suggestions, and
uncommitted-change review. Seeded jobs use daily dedupe keys and skip existing
open jobs, so shell/session hooks can call the command without creating
duplicates. Use `--dry-run` to preview, `--force` to intentionally reseed, and
`AFS_CLIENT_SEED_JOBS=1` or wrapper `--seed-jobs` to opt client-session wrappers
into seeding.

The worker command claims queued jobs, writes the prompt to
`scratchpad/agent_job_prompts/`, runs the configured local command with
`AFS_AGENT_JOB_ID`, `AFS_AGENT_JOB_PROMPT_FILE`, and `AFS_AGENT_RUN_ID` in the
environment, then records both job status and an agent run.

MCP:

- `agent.job.create`
- `agent.job.status`
- `agent.job.inbox`
- `agent.job.review`
- `agent.job.promote`
- `agent.job.archive`
- `agent.job.seed`
- `agent.job.list`
- `agent.job.show`
- `agent.job.claim`
- `agent.job.move`

Harness wrappers:

- `scripts/afs-codex`
- `scripts/afs-claude`
- `scripts/afs-gemini`
- `scripts/afs-hcode`

## Supervised Agents

The `agent-supervisor` service reconciles profile-defined background agents:

```bash
afs services start agent-supervisor
afs agents ps --all
```

State lives under `.context/scratchpad/common/afs_agents/supervisor/` in v2;
version 1 retains `.context/scratchpad/afs_agents/supervisor/`.

### Start Conditions

Each `AgentConfig` can declare any combination of:

| Field | Behavior |
| ----- | -------- |
| `auto_start` | Keep the agent running; restart it (with backoff) when it exits. |
| `schedule` | Run as a one-shot on an interval: `hourly`/`daily`/`weekly` (`@`-prefixed forms accepted) or `<number><s\|m\|h\|d>` such as `30m`. |
| `watch_paths` | Run when a watched file or directory changes. |
| `triggers` | Named lifecycle triggers; the supervisor loop fires `on_boot`. |

Scheduled and watched agents that are not `auto_start` run as one-shots. All
start paths share the same circuit breaker, backoff, and dependency checks.

### Default Agent Set

A fresh supervisor is useful without any profile configuration: a profile
with an *empty* agent list receives a conservative shipped set, tagged
`afs-default`. A profile that configures any agents at all is never augmented
implicitly.

| Agent | Starts | Does |
| ----- | ------ | ---- |
| `context-warm` | `daily` | Audit workspace contexts, without repair or network calls. |
| `index-rebuild` | `watch_paths` on the knowledge + memory mounts | Rebuild the context SQLite index. |
| `skills-mine` | `weekly` | Mine repeated session traces into reviewable skill candidates. |
| `morning-briefing` | `daily` | Write a briefing digest under `scratchpad/common/briefings/` in v2 (`scratchpad/briefings/` in v1). |

Disable the set with `[agents] default_set = false` in `afs.toml`, or
per-environment with `AFS_DEFAULT_AGENTS=off` (`on` force-enables; an
unrecognized value disables with a warning). Starting the supervisor itself
is always an explicit operator action.
