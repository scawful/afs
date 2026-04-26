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
./scripts/afs agent-hooks install-shell --apply
./scripts/afs agent-hooks install-worker --apply --load
./scripts/afs agent-hooks status --path "$PWD"
```

Use this before editing Codex, Claude, Gemini, hcode, or z3cli-specific config. Harness-specific files can still exist, but they should point back to this manifest or derive their local view from it.

`sync` copies manifest-declared shared skill directories into harness skill
roots and writes per-harness JSON export files. It deliberately uses copied
directories rather than symlinks.

`agent-hooks install-shell` adds a marked, idempotent block to the shell profile
that sources `afs-shell-init.sh` and `afs-agent-hooks.sh`. After a new shell,
normal `codex`, `claude`, `gemini`, `hcode`, and `z3cli` commands route through
the AFS wrappers. Run `afs-agent-hooks-off` inside a shell to disable the
functions for that shell. Use `codex-raw`, `claude-raw`, `gemini-raw`,
`hcode-raw`, or `z3cli-raw` when you want the underlying command directly.

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

- `afs doctor` validates the manifest, checks declared paths, compares copied skills against canonical skills, and confirms declared MCP server names appear in known harness/client config.

## Run Recorder

Agent runs are stored under `scratchpad/agent_runs/` in the active context.

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
--to-handoff` to save a useful report under `scratchpad/handoffs/`, then
`agent-jobs archive <job-id>` when the output has been handled.

Use `agent-jobs seed` to queue safe report-only maintenance jobs. The
`repo-maintenance` profile covers stale docs/reference scans, skill drift, MCP
tool-name drift, TODO/FIXME summaries, focused verification suggestions, and
uncommitted-change review. Seeded jobs use daily dedupe keys and skip existing
open jobs, so shell/session hooks can call the command without creating
duplicates. Use `--dry-run` to preview, `--force` to intentionally reseed, and
`AFS_CLIENT_SEED_JOBS=0` or wrapper `--no-seed-jobs` to disable automatic
client-session seeding.

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
- `scripts/afs-z3cli`
