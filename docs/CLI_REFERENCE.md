# AFS CLI Reference

## Invocation

Preferred during local development:

- `./scripts/afs <command>`

Also supported once installed into the active environment:

- `afs <command>`

## Quickstart

- `./scripts/afs`
- `./scripts/afs next --intent continue`
- `./scripts/afs manager`
- `./scripts/afs setup`
- `./scripts/afs guide`
- `./scripts/afs help context`
- `./scripts/afs init --context-root ~/.context --workspace-name src`
- `./scripts/afs status`
- `./scripts/afs status --json`
- `./scripts/afs doctor`
- `./scripts/afs context init --path ~/src`
- `./scripts/afs context discover --path ~/src`
- `./scripts/afs context ensure-all --path ~/src`
- `./scripts/afs start --path "$PWD"`
- `./scripts/afs search "current task" --path "$PWD"`
- `./scripts/afs graph export --path ~/src`

Top-level `afs init` keeps the legacy v1 directory contract for a new root.
When its target is an existing authorized v2 root, it preserves that scaffold
without creating top-level `hivemind`, `global`, or `items` directories; a
damaged v2 marker fails closed. The guided `afs setup` flow uses the same
layout-aware initialization. To create a new v2 root explicitly, run
`afs context init --layout-version 2 --path <project>`.

## Plain-language core

The short command names are the preferred entry points for normal agent and
operator work. Existing nested commands remain available.

| Command | Purpose / compatibility |
|---|---|
| `afs start` | Scoped session bootstrap |
| `afs search` | Local-first search of current project plus `common` |
| `afs files` | Alias for `afs fs` |
| `afs notes` | Durable notes and temporary draft lifecycle |
| `afs handoff` | Immutable cross-session handoff threads |
| `afs messages` | Scoped inter-agent messages |
| `afs projects` | Central v2 project registry |
| `afs jobs` | Alias for `afs agent-jobs` |
| `afs missions` | Alias for `afs mission` |
| `afs check` | Alias for `afs health` |
| `afs repair` | Alias for `afs doctor` |

Common flows:

```bash
./scripts/afs start --path "$PWD"
./scripts/afs search "cache invalidation" --path "$PWD"
./scripts/afs files list knowledge --path "$PWD"
./scripts/afs notes draft "Cache investigation" --body-file notes.md
./scripts/afs notes promote <draft-id>
./scripts/afs handoff threads --path "$PWD"
./scripts/afs messages list --path "$PWD"
./scripts/afs projects current --path "$PWD"
./scripts/afs jobs status
./scripts/afs missions list
./scripts/afs check
./scripts/afs repair
```

Durable `afs mission` records use `.context/.afs/compat/items/missions/` in
v2. The separate legacy mission-runner agent still reads TOML definitions from
`.context/scratchpad/missions/`; its v2 run output is written under
`.context/scratchpad/common/missions/`.

## Profiles

```bash
./scripts/afs profile current
./scripts/afs profile list
./scripts/afs profile switch work
```

## Bundles

```bash
./scripts/afs bundle pack work --output ./dist
./scripts/afs bundle inspect ./dist/work
./scripts/afs bundle install ./dist/work --install-dir ./.afs/extensions
./scripts/afs bundle list
```

`bundle install` now writes:

- an installable extension under the chosen extension root
- generated MCP/agent shims when the bundled profile defines `mcp_tools` or `agent_configs`
- `profile-snippet.toml` you can merge into `afs.toml` to re-enable the bundled profile semantics safely

## Agent Operations

```bash
./scripts/afs agent-manifest show
./scripts/afs agent-manifest validate
./scripts/afs agent-manifest export codex
./scripts/afs agent-manifest sync --apply
./scripts/afs agent-manifest sync --harness hcode --apply
./scripts/afs-upgrade-agent-setup --workspace ~/src
./scripts/afs-upgrade-agent-setup --workspace ~/src --apply --all
./scripts/afs-upgrade-agent-setup --workspace ~/src --full --setup-hcode
./scripts/afs-upgrade-agent-setup --workspace ~/src --full --setup-hcode --apply

./scripts/afs agent-hooks show
./scripts/afs agent-hooks install-shell --apply
./scripts/afs agent-hooks install-worker --apply --load
./scripts/afs agent-hooks status --path "$PWD"

run_id="$(./scripts/afs agent-runs start "Fix settings drift" --harness codex)"
./scripts/afs agent-runs event "$run_id" verification --summary "pytest passed"
./scripts/afs agent-runs finish "$run_id" --summary "patched" --verify "pytest=passed"

job_id="$(./scripts/afs agent-jobs create "Review stale instructions" --prompt "Scan docs and report stale model aliases.")"
./scripts/afs agent-jobs claim "$job_id" --agent reviewer
./scripts/afs agent-jobs move "$job_id" done --result "No stale aliases found."
./scripts/afs agent-jobs status
./scripts/afs agent-jobs inbox
./scripts/afs agent-jobs review "$job_id"
./scripts/afs agent-jobs promote "$job_id" --to-handoff
./scripts/afs agent-jobs archive "$job_id"
./scripts/afs agent-jobs seed --profile repo-maintenance
./scripts/afs agent-jobs work --agent local-worker --command 'codex exec < "$AFS_AGENT_JOB_PROMPT_FILE"'
```

`agent-manifest` reads `configs/agent_manifest.toml`, the repo-owned source of
truth for harnesses, shared skills, slash-command packs, MCP servers, and
startup hints.
`afs-upgrade-agent-setup` wraps the common local upgrade path and stays dry-run
unless `--apply` is provided. Use `--full --setup-hcode` to preview or apply
the full local path that syncs Codex/Claude/Gemini/Antigravity/hcode manifest state,
OpenCode slash commands, shell hooks, and index freshness without making the
MCP catalog noisy.
`agent-manifest sync` copies manifest-declared shared skills into harness skill
roots, copies slash-command packs into harness command roots, and writes
per-harness export JSON. It uses real copied directories/files, not symlinks.
Slash-command packs are additive by default: existing customized command files
are reported as `customized` and are not overwritten unless the manifest pack
opts into `overwrite = true`.
`agent-hooks install-shell` adds an idempotent block to the shell profile that
sources `afs-shell-init.sh` and `afs-agent-hooks.sh`, making normal generic
harness commands such as `codex`, `claude`, `gemini`, `antigravity`, and `hcode` route
through AFS wrappers. Companion extension repos can add more local harnesses.
Raw bypass functions are also exposed for installed wrappers, such as
`codex-raw`, `claude-raw`, `gemini-raw`, `antigravity-raw`, and `hcode-raw`.
`agent-hooks install-worker` writes a user LaunchAgent that runs
`agent-jobs work --loop` for automatic queued-job execution. The worker skips
obvious destructive prompts unless the job or worker uses `--allow-destructive`.
`agent-hooks status --path <workspace>` prints exact next commands for missing
hook setup, watchdog status, and the agent job review inbox.
`agent-runs` writes replayable shared records under
`scratchpad/common/agent_runs/` in v2 and `scratchpad/agent_runs/` in v1.
`agent-jobs` writes markdown prompt jobs through the legacy `items` role:
`items/agent_jobs/{queue,running,done,failed,archived}/` in v1 and the
corresponding `.afs/compat/items/` subtree in v2.
`agent-jobs status` provides a read-only watchdog summary of queue counts,
runnable jobs, destructive opt-in blockers, stale running jobs, recent run
failures, and LaunchAgent state. It exits successfully by default so it can be
used for visibility without blocking agents; use `--strict` for scripts that
should fail when watchdog checks need attention.
`agent-jobs inbox` is the review surface for completed reports, failed jobs,
stale running jobs, and destructive opt-in blockers. Use `agent-jobs review
<job-id>` to inspect a job with its linked run record, `agent-jobs promote
<job-id> --to-handoff` to create a durable common-scope handoff in v2 (or a
v1 `scratchpad/handoffs/` compatibility file), and `agent-jobs archive
<job-id>` after handling it.
`agent-jobs seed` idempotently queues safe report-only background jobs. The
`repo-maintenance` profile creates daily-deduped stale docs/reference, skill
drift, MCP/tool drift, TODO/FIXME, verification suggestion, and uncommitted
change review jobs. Client-session wrappers do not seed these jobs by default;
set `AFS_CLIENT_SEED_JOBS=1` or pass `--seed-jobs` to opt in.
`agent-jobs work` claims queued jobs, runs a local command with
`AFS_AGENT_JOB_*` environment variables, moves jobs to `done` or `failed`, and
records an `agent-runs` entry.

`session bootstrap` includes the manifest summary, open agent jobs, and recent
run records. AFS client wrappers start and finish run records automatically
unless `AFS_CLIENT_RECORD_RUNS=0` is set. MCP exposes the same surfaces through
`agent.manifest.show`, `agent.run.*`, and `agent.job.*` tools, including
`agent.job.status` for the watchdog payload, `agent.job.inbox/review/promote/archive`
for review handling, and `agent.job.seed` for safe maintenance job seeding.

## GUI Manager

```bash
./scripts/afs manager
./scripts/afs manager open --path ~/src/project-a
./scripts/afs manager snapshot --path ~/src/project-a --json
./scripts/afs-manager
```

`manager` launches the friendly Python GUI surface for normal users. It
summarizes context health, mount counts, task queue state, project client
config like `.gemini/settings.json`, enabled extensions, extension hooks, and
suggested setup commands. The `snapshot` form returns the same read model
without opening a window.

The manager and `context.status` both expose the same deterministic agent
discovery path: status, query, exact read/list, scratchpad write, then named
CLI/slash-command routes for heavier flows. This keeps agents proactive without
forcing the whole AFS catalog into their first tool choice.

## Next Action Router

```bash
./scripts/afs next --intent continue --json
./scripts/afs next --intent work-writing --json
./scripts/afs next --intent verify --json
./scripts/afs next report --json
```

`next` is the deterministic funnel for agents. It turns a common intent into
the first cheap MCP step, the exact CLI/slash-command route, a stop condition,
and a short list of surfaces to avoid. Supported intents include `continue`,
`context`, `review`, `ship`, `work-writing`, `verify`, `handoff`, `setup`,
`refresh`, and `pack`.

Every `afs next` route records a small `afs.next` history event. `next report`
summarizes recent route use, MCP tool calls, and any heavy MCP calls that
bypassed the default funnel.

Extensions can add manager-visible commands with:

```toml
[manager]
actions = ["afs status", "afs tasks list --path ."]
```

## Guided Setup

```bash
./scripts/afs setup
./scripts/afs setup --yes --dry-run --shell helpers --mcp none --google-workspace skip
./scripts/afs setup --yes --apply --shell helpers --mcp none --google-workspace skip
./scripts/afs guide
./scripts/afs guide context
./scripts/afs guide shell
./scripts/afs guide mcp
```

`setup` asks for config scope, context placement, shell integration level,
optional MCP registration, optional Google Workspace public API handling, and background
worker installation. It prints a plan before writing. `--shell helpers` installs
aliases, colors, and zsh completion without routing AI harness commands;
`--shell agent-hooks` enables the full wrapper routing.

## Context

```bash
./scripts/afs context init
./scripts/afs context init --layout-version 2 --path "$PWD"
./scripts/afs context ensure
./scripts/afs context list
./scripts/afs context overview --path "<afs-root>"
./scripts/afs context validate
./scripts/afs context repair --dry-run
./scripts/afs context query "startup guidance"
./scripts/afs query "startup guidance"
./scripts/afs context mount knowledge ~/src/docs --alias docs
./scripts/afs context unmount knowledge docs
./scripts/afs index rebuild --mount scratchpad
./scripts/afs projects register "$PWD"
./scripts/afs layout audit --context-root ~/.context --json
./scripts/afs layout plan --context-root /path/to/v1-context \
  --destination-root /path/to/new-v2-context \
  --mapping-file /private/path/layout-mappings.json \
  --output /private/path/migration-plan.json \
  --rollback-output /private/path/source-retention.json
./scripts/afs layout migrate --plan /private/path/migration-plan.json
./scripts/afs layout migrate --plan /private/path/migration-plan.json \
  --apply --because "Create a verified v2 candidate for review"
./scripts/afs layout activate --plan /private/path/migration-plan.json \
  --state-dir /private/path/activation-state
./scripts/afs layout activate --plan /private/path/migration-plan.json \
  --state-dir /private/path/activation-state \
  --apply --because "Activate the fresh candidate during maintenance"
./scripts/afs layout rollback --state-dir /private/path/activation-state
```

An explicitly marked version 2 root uses six human-facing categories:
`history`, `memory`, `scratchpad`, `knowledge`, `tools`, and `human`. Project
records, messages, and indexes are internal state under `.afs/`. Normal v2
access is limited to the current registered project plus `common`; a central
context path by itself does not authorize every project.

`layout audit` is read-only. `layout plan` writes an atomic, mode-`0600`,
hash-bound plan and does not move data. Copy-only plans use schema v2; a plan
with reviewed source-only exclusions uses schema v3. Existing schema-v2 plans
remain loadable. The destination must be separate from the source and must not
exist. Unknown top-level entries stay blocking unless `--mapping-file` maps
their exact names to `<category>/common/...` or `.afs/compat/imported/...`, or
records them as source-only. Generic mappings cannot select nested paths or
use globs.

The mapping file, plan output, and optional rollback output must use three
distinct paths outside both the source and candidate roots. This prevents a
planning artifact from invalidating the source fingerprint or leaking into
the copied candidate.

The strict schema-v1 mapping format remains accepted for copy-only decisions:

```json
{
  "schema_version": 1,
  "mappings": {
    "exact-top-level-name": "knowledge/common/imported/exact-top-level-name"
  }
}
```

Mapping schema v2 adds exact, reason-bearing exclusions:

```json
{
  "schema_version": 2,
  "mappings": {
    "AFS_SPEC.md": "knowledge/common/specs/AFS_SPEC.md"
  },
  "retained_sources": {
    "legacy-projects": "Requires project-scoped import before cutover"
  },
  "retained_paths": {
    "knowledge/skills": "Recreate this link from the v2 tool registry"
  }
}
```

`retained_sources` accepts exact unknown top-level names. `retained_paths`
accepts normalized nested paths below a copied top-level operation. Each value
is a non-empty reviewed reason. These are **source-only exclusions**: the
named file or subtree remains in the untouched v1 source and is not copied
into the candidate. Entries must exist and may not overlap.

Exclusions remain part of the whole-source fingerprint, so any content or
metadata drift still invalidates the plan. Explicitly excluded links are
hashed as link metadata without following them. Unreviewed links and copied
non-portable names fail closed; a link or non-portable name is accepted only
inside an exclusion. Hard links and special files remain blocked. Stop active
writers before planning, previewing, and applying.

Plan fields `source_file_count` and `source_bytes` cover the whole source.
Schema-v3 `copy_file_count` and `copy_bytes` cover only candidate data;
preflight capacity uses `copy_bytes`. Text and JSON preview distinguish these
totals and list the reviewed exclusions. Schema-v1 mapping files and
schema-v2 migration plans remain backward compatible.

`layout migrate --plan PLAN` is also read-only by default: it verifies and
previews the transaction without creating the destination. Applying requires
both `--apply --because "..."` and confirmation through the controlling
terminal. The confirmation separates whole-source totals from candidate copy
totals and lists every path and reason that will remain source-only. Piped
input and headless agents cannot confirm. Unreviewed links, copied
non-portable names, hard links, and special files block; the v1 source is never
modified or deleted; and the v2 layout marker is published only after every
copy verifies.
A caught pre-marker apply failure is moved to an adjacent `.failed-*` path
when that quarantine rename succeeds. A hard process or host interruption may
instead leave an unmarked partial tree at the requested destination; inspect
and move it explicitly before retrying.

On Windows, `layout audit` and `layout plan` remain available, but
`layout migrate` preview and apply fail preflight closed. The executor cannot
use `chmod` to establish or verify the required private DACLs; Windows
migration remains unavailable until explicit DACL support lands.

A successful migrate apply produces a separate verified candidate; it does
not activate or swap roots. `layout activate` is a separate preview-first
command. Apply requires `--apply --because`, an exact controlling-terminal
token, a private external `--state-dir`, stable configured root, zero retained
source/path exclusions, no open processes below either root, and a supported
same-filesystem atomic directory exchange. It puts v2 at the stable source
path and preserves v1 at the old candidate path. There is no sequential-rename,
symlink, copy, merge, or delete fallback.

`layout rollback` previews from the external activation state. Its separately
authorized apply verifies the activation receipt, inode topology, and
preserved v1 fingerprint, then exchanges the roots back. V2-era writes remain
at the inactive candidate path. A `receipt_pending` preview means an exchange
committed but its immutable receipt still needs a fresh human-authorized
finalization; it does not perform a second exchange.

The rollback manifest emitted by `layout plan` remains informational and is
not an executable rollback receipt. `layout audit` validity also does not mean
activation readiness: central v2 project access needs registry records, and
repo-local `.context` roots can still shadow the central root. See
[Central Context Layout v2](CONTEXT_LAYOUT_V2.md) for the full safety
contract. These commands do not imply that live `~/.context` is ready or
migrated.

`layout migrate` exit codes:

| Code | Meaning |
|---|---|
| `0` | Preview ready, candidate applied, or the exact completed transaction recognized as already applied. |
| `2` | Invalid or stale plan, blocked preflight, invalid rationale, or refused/unavailable human confirmation. |
| `3` | Authorized apply failure; output includes the retained `.failed-*` destination when a pre-marker tree was quarantined. |
| `1` | Unexpected internal failure handled by the generic CLI; never treat it as migration evidence. |

`layout activate` and `layout rollback` use the same code classes. Exit `0`
means preview-ready or a verified completed state, `2` means blocked/refused,
`3` means an authorized transition started but did not fully record success,
and `1` remains an unexpected internal error. After exit `3`, keep services
stopped and rerun preview to distinguish unchanged, `receipt_pending`, and
conflicting topology.

Run migration in a controlled maintenance window with active producers and
other writers stopped. Its checks fail closed against accidental drift,
static links, and copy failures, but do not sandbox a hostile concurrent
same-user process that swaps paths between checks. Already-completed
recognition also requires the source to continue matching the reviewed plan.
See the threat boundary in
[Central Context Layout v2](CONTEXT_LAYOUT_V2.md#threat-boundary).

Indexed query usage:

- `./scripts/afs query <text> --path <workspace>` is the fast top-level shortcut.
- `./scripts/afs context query <text> --path <workspace>` is the canonical form.
- `./scripts/afs context overview --path <workspace>` gives a cheap structural repo summary before deeper grep/query passes.
- `context overview` also works on a raw project path before `.context` exists, which is useful for new repos such as training/eval workspaces.
- Use `--mount` repeatedly to narrow search to specific mounts such as `scratchpad`, `knowledge`, or `tools`.
- Use `--prefix` to keep search under a relative subtree like `docs/sqlite/` or `public/`.
- `--include-content --json` returns full indexed content for each hit; plain text mode shows a compact excerpt.
- JSON output includes `count`, `entries`, and `index_rebuild` when the command auto-built or auto-refreshed the SQLite index.
- In v2, query, freshness, and index rebuild scan only the current registered
  project plus `common`. Cross-project query or rebuild requires the explicit
  `--all-projects` flag; an ordinary scoped refresh preserves other projects'
  existing index rows without traversing their directories. Use `--common`
  when operating directly from the central context root without a project.

Examples:

```bash
./scripts/afs query sqlite --path "<afs-root>"
./scripts/afs context query sqlite --path "<afs-root>" --mount scratchpad --mount knowledge
./scripts/afs context query sqlite --path "<afs-root>" --prefix docs/sqlite/ --limit 10 --include-content --json
./scripts/afs index rebuild --path "<afs-root>" --mount scratchpad
./scripts/afs index rebuild --path "$PWD" --mount knowledge --all-projects
```

For version 2 scoped hybrid retrieval, prefer `afs search`:

```bash
./scripts/afs search "release checklist" --path "$PWD"
./scripts/afs search "parser symbol" --path "$PWD" --mode symbol
./scripts/afs search "similar failure" --path "$PWD" --semantic --rebuild
./scripts/afs search "shared convention" --path "$PWD" --all-projects
```

The default is local text/symbol retrieval. `--semantic` explicitly enables
embeddings for the rebuild/query; the default provider is Gemini, stable model
`gemini-embedding-2`, at 768 dimensions. `--all-projects` is the explicit
cross-project authorization boundary.

Readable notes and handoffs use immutable filenames of the form
`YYYY-MM-DDTHHMMSSZ--slug--10charid.md`. `notes promote` copies a draft into
durable memory with provenance and leaves the draft in place; `notes archive`
explicitly moves it out of the active scratchpad. `handoff revise` appends a
superseding revision, while `ack` and `close` record lifecycle state separately.

## Insights

`insights` provides scoped research and deterministic reflection:

```bash
./scripts/afs insights research "retry policy" --path "$PWD"
./scripts/afs insights research "similar failure" --path "$PWD" \
  --semantic --provider ollama
./scripts/afs insights reflect --path "$PWD"
./scripts/afs insights list --path "$PWD"
./scripts/afs insights show <candidate-id> --path "$PWD"
./scripts/afs insights accept <candidate-id> --path "$PWD" \
  --because "The attributed evidence supports this pattern."
./scripts/afs insights reject <candidate-id> --path "$PWD" \
  --because "The sample is too small."
```

Research searches only the current registered project plus `common` and
refreshes its local index unless `--reuse-index` is set. `--semantic` is
explicit: Ollama keeps embedding input local, while Gemini transmits content
and the query to Gemini. Internet research separately requires an enabled
extension selected with `--internet-provider` and one or more
`--allow-domain` values.

The internet provider is trusted extension code responsible for DNS,
redirect, private-IP, rebinding, and transport-timeout enforcement. Core AFS
only bounds the subprocess and validates returned HTTPS URLs, record count,
and bytes; it does not mediate provider sockets. API-key environment variables
are scrubbed. Providers should read only a credential file explicitly named by
trusted configuration, never inherit the full parent environment.

Reflection uses no model or network. It consumes only exactly attributed,
payload-free failure metadata and creates deterministic pending candidates in
the current project, or in common with `--common`. Successful completions and
general activity are ignored to avoid rolling candidate spam. Accept/reject
require a rationale and human confirmation; there is no automatic promotion.
The `insights-reflect` and `insights-research` scheduled agents are opt-in,
not shipped defaults. Scheduled reflection reads only the newest 1,000 raw
JSONL history records before attribution/filtering and the evidence limit, so
its recurring scan stays bounded; interactive `insights reflect` retains
complete-history behavior. Research-agent internet access additionally requires
the literal profile setting `network_allowed = true`, a selected provider,
and an explicit domain allowlist. Reports remain in scratchpad and are never
promoted automatically.

See [Insights](INSIGHTS.md) for storage, trust boundaries, and configuration.

## Review

`review` now operates on the active context instead of a separate
`~/.context/projects/...` tree. Pending drafts live under the context
scratchpad at `review/<category>/`.

```bash
./scripts/afs review list --path ~/src/project-a
./scripts/afs review approve --path ~/src/project-a draft.md
./scripts/afs review reject --path ~/src/project-a draft.md --reason "needs revision"
```

Approved plans move into `memory/reviewed/plans/`. Other approved review
documents move into `history/reviewed/<category>/`. Rejections are archived
under `history/rejected/<category>/`.

For compatibility, `./scripts/afs review approve project-a draft.md` still
works when `project-a` can be resolved from configured
`general.workspace_directories`.

## Work Assistant

`work` manages non-technical work-assistant state in the active context:
people, project relationships, review routes, communication samples, approval
requests, and activity.
This state is native to AFS and backed by the legacy `global` role:
`.context/global/work_assistant.sqlite3` in v1 and
`.context/.afs/compat/global/work_assistant.sqlite3` in v2.
It is intentionally not exposed as a broad MCP CRUD surface.

```bash
./scripts/afs work --path .
./scripts/afs work people list --path .
./scripts/afs work relationships list --path .
./scripts/afs work reviewers --path . --target-type docs
./scripts/afs work approvals list --path .
./scripts/afs work approvals show <approval-id> --path .
./scripts/afs work approvals request \
  --path . \
  --target-system zendesk \
  --target-id ticket-123 \
  --action post_ticket_comment \
  --summary "Send drafted support reply" \
  --preview "Thanks for the report..." \
  --permission-required "ticket comment approval"
./scripts/afs work approvals request \
  --path . \
  --target-system gmail \
  --target-id "email:person@example.com" \
  --action send_email \
  --summary "Send approved follow-up email" \
  --preview-json '{"to":"person@example.com","subject":"Follow-up","body":"Thanks for the update."}'
./scripts/afs work approvals approve <approval-id> --path . \
  --because "preview and target verified"
./scripts/afs work approvals execute <approval-id> --path . --dry-run --json
./scripts/afs work approvals execute <approval-id> --path . \
  --executor "python3 scripts/afs-work-approval-echo.py"
./scripts/afs work approvals execute <approval-id> --path . \
  --executor "python3 scripts/afs-work-gws-executor.py"
./scripts/afs work communication list --path .
./scripts/afs work communication list --path . --purpose responding_to_comments --json
./scripts/afs work communication add --path . \
  --purpose responding_to_comments \
  --style-note "direct" \
  --text "Short, concrete reply with exact file evidence."
./scripts/afs work communication guide --path . --purpose responding_to_comments
./scripts/afs work communication preflight --path . \
  --purpose responding_to_comments \
  --personal-mode work
./scripts/afs work activity list --path .
```

When context/history events include work metadata such as `owner`,
`reviewers`, `relationships`, `review_routes`, `communication_sample`,
`approval_request`, or `requires_approval`, AFS enriches the work-assistant
database automatically.
External writes should be executed only from one approved action at a time.
`approvals execute` passes an approved request JSON file to an explicit local
connector command and marks the request `applied` only when that command exits
successfully.
Applied communication actions also seed `communication_samples` from the
approved preview/body so future work-writing guidance learns only from text the
user permitted to send or post.

See `docs/WORK_ASSISTANT.md` and `docs/WORK_ASSISTANT_UPGRADE.md`.
Google Workspace connector examples are in `docs/WORK_ASSISTANT_CONNECTORS.md`.

## Personal Context

`personal` loads an explicit personal-context mode from
`$AFS_PERSONAL_CONTEXT_ROOT` or `~/.config/afs/personal`.

```bash
./scripts/afs personal modes
./scripts/afs personal load work
./scripts/afs personal load work --json
```

For work modes, `manifest.toml` may include `work_context = true`,
`style_instructions`, `communication_sources`, and `posting_policy`. Rendered
context then tells agents to inspect the user's actual communication samples
before matching tone and to ask for explicit permission before posting on the
user's behalf.

## Memory

```bash
./scripts/afs memory consolidate --path ~/src/project-a
./scripts/afs memory consolidate --path ~/src/project-a --json
./scripts/afs agents run history-memory --stdout
./scripts/afs services start history-memory
```

`memory consolidate` is the canonical history-to-memory step. It reads new
metadata-first history events, writes durable summaries into
`memory/common/entries.jsonl`, writes markdown summaries into
`memory/common/history_consolidation/`, and checkpoints incremental progress
under `.context/scratchpad/common/afs_agents/history_memory_checkpoint.json`
in v2. Version 1 retains the paths directly under `memory/` and
`.context/scratchpad/afs_agents/`.

## Journal Agent

```bash
# Draft this week's review
./scripts/afs agents run journal-agent

# Draft a specific week (overwrite existing AI draft section)
./scripts/afs agents run journal-agent -- --week 2026-W11 --overwrite

# Override paths explicitly
./scripts/afs agents run journal-agent -- \
  --thoughts ~/notes/thoughts.org \
  --active-tasks ~/notes/tasks/active.md \
  --weekly-dir ~/notes/weekly
```

See `docs/JOURNAL_AGENT.md` for full argument reference and JSON output shape.

## Session

```bash
./scripts/afs session bootstrap
./scripts/afs session bootstrap --json
./scripts/afs session bootstrap --skills-prompt "review this Python refactor"
./scripts/afs session pack
./scripts/afs session pack "sqlite indexing" --model gemini
./scripts/afs session pack "sqlite indexing" --model gemini --workflow scan_fast --task "Find the three most relevant SQLite files"
./scripts/afs session pack "sprite" --model gemini --pack-mode retrieval
./scripts/afs session pack "sprite" --model gemini --semantic  # explicit remote query embedding
./scripts/afs session pack --model gemini --pack-mode full_slice
./scripts/afs session pack "runtime bug" --model codex --token-budget 12000 --json
```

`session bootstrap` is the preferred start-of-session surface. It combines:

- `context.status`
- `context.diff`
- cheap codebase orientation from `context overview`
- scratchpad state and deferred notes
- queued tasks from the legacy `items` storage role
- work-assistant people, activity, and pending approval-gated external writes
- recent scoped messages
- latest durable memory summary
- bounded bodies for skills matched by `--skills-prompt`, or by the current
  handoff, active missions, and open tasks when no explicit focus is supplied

In v2 it also refreshes these files under the active scoped agent artifact
directory, `.context/scratchpad/projects/<project-id>/afs_agents/` (or
`.context/scratchpad/common/afs_agents/` for explicit common scope):

- `session_bootstrap.json`
- `session_bootstrap.md`

`session pack` is the explicit follow-on surface when an agent needs a bounded
working set for Gemini, Claude, Codex, or a generic client. It builds a
token-budgeted packet from bootstrap state, scratchpad, queued tasks, messages,
durable memory, and indexed retrieval hits, then writes or reuses:

- `session_pack_<model>.json`
- `session_pack_<model>.md`

Version 1 retains `.context/scratchpad/afs_agents/` for all session artifacts.

`never_index`/`never_export` sensitivity rules are applied to indexed content
included in the pack, and `never_embed` is applied to embedding hits, so blocked
paths do not leak into session exports. When the bootstrap snapshot, pack
inputs, and sensitivity rules have not changed, repeated calls reuse the stored
artifact instead of rebuilding from scratch.

Session packs are local keyword retrieval by default. `--semantic` is the
explicit permission boundary for sending the pack query to the configured
embedding provider. The same opt-in is available on `session prepare-client`;
without it, neither command creates or invokes a remote query embedder.

Rendered packs keep model-control guidance as a single top-level block instead
of duplicating it as a normal section. The selectable section budget accounts
for that fixed guidance, and query/embedding hits are prioritized ahead of
generic session boilerplate when a query is present.

`session pack` now also accepts:

- `--task` to append an explicit task block after the context sections
- `--workflow` to encode a generic execution profile such as `scan_fast`,
  `edit_fast`, `review_deep`, or `root_cause_deep`
- `--tool-profile` to encode a preferred AFS surface mix such as
  `context_readonly`, `context_repair`, `edit_and_verify`, or `handoff_only`
- `--pack-mode` to choose `focused`, `retrieval`, or `full_slice` context
  shaping depending on whether Gemini needs a narrow query-first pack or a
  broader long-context slice
- `--semantic` to explicitly permit remote query embeddings; default is local
  retrieval only

The rendered pack guidance now points at the human CLI surfaces too:

- `afs query <text> --path <workspace>` for cheap follow-on indexed retrieval
- `afs context query <text> --path <workspace>` as the canonical equivalent
- `afs index rebuild --path <workspace>` when the pack warns that indexed search may be stale

Pack JSON now includes `execution_profile` metadata and `cache.prefix_hash` so
Gemini-side adapters can distinguish a stable context prefix from a changing
task suffix. It also records `pack_mode` and `pack_mode_summary`. The
`execution_profile` now carries prompt-only loop policy and retry guidance, so
Gemini-facing wrappers can rerun with narrower context, schema-bound prompts,
or different model tiers without AFS owning the turn loop.

## Events

```bash
./scripts/afs events tail --json
./scripts/afs events list --type mcp_tool --limit 25
./scripts/afs events list --path ~/src/project-a --source afs.mcp
./scripts/afs events analytics --hours 24 --json
./scripts/afs events replay --session-id "$AFS_SESSION_ID"
./scripts/afs session event user_prompt_submit --client codex --session-id "$AFS_SESSION_ID" --prompt "current task"
./scripts/afs-session-notify task_progress --task-id bg-1 --summary "Indexing symbols"
./scripts/afs messages clean --all-projects --json
```

`messages clean` is a dry-run unless `--apply` is supplied. The legacy
`afs hivemind ...` command remains available for one compatibility cycle; new
automation should use `afs messages ...`.

`events` reads the active context history log with the same config/context
resolution as the rest of the CLI. `events analytics` summarizes recent tool
usage, durations, and error rates; `events replay` reconstructs a session
timeline from the shared `AFS_SESSION_ID` propagated by the client launch
wrappers. `session event` is the harness-facing write surface for prompt, turn,
and task lifecycle updates that should appear in those replays.
`afs-session-notify` is the wrapper-friendly shell helper for child scripts: it
fills in `--client`, `--session-id`, `--payload-file`, `--cwd`, and the current
`--turn-id` from the exported session environment. `afs-client-session` wrappers
also export `AFS_SESSION_SYSTEM_PROMPT_*` and, by default, wire the prompt
artifact into the native client entrypoint when one exists: Codex via
`-c model_instructions_file=...`, Claude via `--append-system-prompt-file`,
and Gemini/Antigravity via `GEMINI_SYSTEM_MD`. Hook-only host integrations
receive a compact pointer plus a 1,000-character top-skill excerpt. Set
`AFS_CLIENT_NATIVE_PROMPT=0` or
`AFS_<CLIENT>_NATIVE_PROMPT=0` to disable that handoff. They also accept
`--prompt`, `--prompt-file`, and `--turn-id`; when present, they emit
`user_prompt_submit`, `turn_started`, and `turn_completed` / `turn_failed`
around the client process.

`session prepare-client` payloads now also include a `cli_hints` block with:

- `workspace_path`
- `query_shortcut`
- `query_canonical`
- `index_rebuild`
- `agent_jobs_inbox`
- `work_summary`
- `work_approvals`
- `work_communication`
- `notes`

Prepared sessions match skills against `--skills-prompt`, then `--task`, then
`--query`. Match records and generated system prompts may include bodies for
the first three skills, bounded to 2,000 characters per body and 6,000 total.
Compact enforcement and verification rules remain a higher-priority prompt
section so a small token budget sheds bodies before rules.

`afs-client-session` exports the same follow-up hints as:

- `AFS_SESSION_QUERY_HINT`
- `AFS_SESSION_CONTEXT_QUERY_HINT`
- `AFS_SESSION_INDEX_REBUILD_HINT`
- `AFS_SESSION_AGENT_JOBS_INBOX_HINT`
- `AFS_SESSION_WORK_HINT`
- `AFS_SESSION_WORK_APPROVALS_HINT`
- `AFS_SESSION_WORK_COMMUNICATION_HINT`

Client-session wrappers can call `agent-jobs seed --profile repo-maintenance`
when opted in with `AFS_CLIENT_SEED_JOBS=1`, a client-specific
`AFS_<CLIENT>_SEED_JOBS=1`, or `--seed-jobs`. This queues report-only
maintenance jobs with daily dedupe keys and skips existing open jobs. Wrappers
still print the agent job inbox command at startup so completed background
output has an obvious review path.

## Optimization Evidence

```bash
./scripts/afs optimize decide \
  --baseline examples/optimization_gate/baseline.json \
  --candidate examples/optimization_gate/candidate.json \
  --policy examples/optimization_gate/policy.json \
  --json

./scripts/afs schema show v1/optimization/evaluation
./scripts/afs schema show v1/optimization/policy
./scripts/afs schema show v1/optimization/decision
```

`optimize decide` is a pure evidence comparator. It never executes, writes,
activates, or promotes a candidate. Exit codes are `0` for
`eligible_for_human_review`, `1` for `rejected`, `2` for invalid input, `3`
for `inconclusive`, and `4` for an internal gate error (not an evidence
verdict). See `docs/OPTIMIZATION_PROTOCOL.md` for the versioned contracts and
safety boundary.

## Policy-Checked Execution

```bash
./scripts/afs execution inspect \
  --request ./request.json \
  --allowed-root "$PWD" \
  --allowed-executable python3 \
  --json

./scripts/afs schema show v1/execution/request
./scripts/afs schema show v1/execution/inspection
./scripts/afs schema show v1/execution/record
```

`execution inspect` validates and resolves a typed request against trusted
policy but never executes it. Exit codes are `0` when allowed, `2` for invalid
input, and `3` when blocked. AFS intentionally exposes no generic execution CLI;
trusted Python callers use `execute_checked(...)`. The portable backend
supports only `isolation=process` with `network=inherit` and fails closed for
unsupported sandbox or network restrictions. Omitted executable permission also
blocks; pass repeated `--allowed-executable` and `--allowed-env` options to
construct the trusted read-only inspection policy. See
`docs/EXECUTION_BROKER.md`.

## Verification

```bash
./scripts/afs verify plan --cwd "$PWD" --json
./scripts/afs verify run --cwd "$PWD" --json
./scripts/afs verify run --cwd "$PWD" --allow-legacy-shell
```

Structured verification `executions` use argv arrays and run through the
policy-checked broker. String `commands` are deprecated shell input and blocked
unless configuration or the explicit CLI flag enables the migration path.
Warnings go to stderr so `--json` stdout remains machine-readable. Legacy shell
verification commands are scheduled for removal in AFS `0.4.0`.

## Training

```bash
./scripts/afs training dataset stats ./data/output/tooling
./scripts/afs training dataset outliers ./data/output/tooling --limit 5
./scripts/afs training run start ./training/jobs/qwen35-tools-local.toml
./scripts/afs training run status <run-id>
./scripts/afs training run stop <run-id>
./scripts/afs training memory-export --path ~/src/project-a --output ./memory.jsonl

./scripts/afs training freshness-gate --path ~/src/project-a
./scripts/afs training freshness-gate --path ~/src/project-a --warn-only --json
./scripts/afs training antigravity-status --json
./scripts/afs training extract-sessions --path ~/src/project-a --output ./session_replay_training.jsonl
./scripts/afs training generate-router-data --config ~/src/project-a/afs.toml --output ./router_from_capabilities.jsonl
./scripts/training_watch.sh --debounce 45
```

`training dataset stats` writes dataset summary artifacts under the active
context scratchpad and reports split counts, average row size, max row size,
role counts, and tool-call counts.

`training dataset outliers` writes the largest rows into scratchpad artifacts so
operators and agents can prune disruptive samples before launching a run.

`training run start` launches a detached job from a JSON/TOML spec and writes
status, event, artifact, and log paths under
`.context/scratchpad/common/training/runs/<run-id>/` in v2. Version 1 retains
`.context/scratchpad/training/runs/<run-id>/`.

See `docs/examples/training_run.example.toml` for a minimal spec layout.

`training run status` refreshes the stored status snapshot against the live
process table. `training run stop` terminates the recorded process group and
updates the run artifact.

In v2, `training memory-export` requires `--path` (or the current directory) to
resolve to a registered project and exports only that project's memory plus
common memory. `--memory-root` is an explicit administrative override that
bypasses project scope; v1 retains its recursive memory-root behavior.

`training freshness-gate` checks per-mount context freshness before training and
returns a blocking or warning-only readiness report.

`training antigravity-status` reports whether the local Antigravity capture DB
exists, when it last changed, and whether the expected trajectory-summary
payloads are present before you attempt an export.

`training extract-sessions` prefers explicit `AFS_SESSION_ID`-based replay data
when present, and falls back to older date-grouped timelines only when the
history log does not contain recorded sessions yet.

`training generate-router-data` derives routing examples from the live agent
capability registry, including extension agents enabled by the resolved runtime
config.

## Claude

```bash
./scripts/afs claude setup --path ~/src/project-a
./scripts/afs claude context --path ~/src/project-a
./scripts/afs claude session-report --session <uuid> --write-scratchpad
```

`claude setup` writes `project/.claude/settings.json` and `project/CLAUDE.md`
for the resolved project path, not just the current shell directory. When an
`afs.toml` is present, the generated Claude MCP entry pins `AFS_CONFIG_PATH`
and `AFS_PREFER_REPO_CONFIG=1` so Claude uses the repo-local AFS config.

## Workspace

```bash
./scripts/afs workspace list
./scripts/afs workspace add ~/src/project-a --description "project-a"
./scripts/afs workspace remove ~/src/project-a
./scripts/afs workspace sync --root ~/src
```

## Plugins and Extensions

```bash
./scripts/afs plugins --details
./scripts/afs plugins --json
```

## Skills

```bash
./scripts/afs skills list --profile work
./scripts/afs skills list --profile work --json
./scripts/afs skills match "mcp context mount" --profile work
./scripts/afs skills mine --path ~/src/project-a
./scripts/afs skills review --path ~/src/project-a --status pending
./scripts/afs skills promote --path ~/src/project-a --candidate workflow-example
./scripts/afs skills reject --path ~/src/project-a --candidate workflow-example
./scripts/afs skills archive --path ~/src/project-a --candidate workflow-example
```

Skill discovery is fail-soft per entry and directory: malformed or unreadable
`SKILL.md` files and failed directory scans do not hide valid skills from other
configured roots or readable sibling directories. The human list and match
commands print warnings, while their `--json` output includes
`diagnostic_count` and structured `diagnostics` records with the affected
root/path and warning code. Diagnostic fields are bounded; `truncated_fields`
names any code, message, root, or path shortened for output. Session bootstrap
carries the same bounded warnings; prepare-client prompts include a compact
count plus an `afs skills list --json` follow-up, and `afs doctor` summarizes
them using the same runtime config precedence.

Structured diagnostic parity for the MCP `skill.match` and `skill.read` tools
is a separate, ratchet-gated `mcp_server.py` cleanup slice; this CLI/session
change intentionally does not broaden that legacy module.

## Embeddings

```bash
# Recommended scoped retrieval (local by default)
./scripts/afs search "sprite RAM tables" --path "$PWD"

# Explicit semantic rebuild/query with stable Gemini embeddings
./scripts/afs search "how to debug a sprite" --path "$PWD" --semantic --rebuild

# Index with keyword-only (no embedding provider needed)
./scripts/afs embeddings index --knowledge-path ~/.context/knowledge --provider none --include "*.md"

# Index with Gemini vectors (requires GEMINI_API_KEY)
./scripts/afs embeddings index --knowledge-path ~/.context/knowledge --provider gemini --include "*.md"

# Index with other providers
./scripts/afs embeddings index --knowledge-path ~/.context/knowledge --provider ollama
./scripts/afs embeddings index --knowledge-path ~/.context/knowledge --provider openai --model text-embedding-3-small
./scripts/afs embeddings index --knowledge-path ~/.context/knowledge --provider hf --model nomic-embed-text

# Semantic search (auto-uses RETRIEVAL_QUERY for Gemini asymmetric retrieval)
./scripts/afs embeddings search --knowledge-path ~/.context/knowledge --provider gemini "how to debug a sprite"

# Keyword search (no provider needed if index exists)
./scripts/afs embeddings search --knowledge-path ~/.context/knowledge --provider none "sprite RAM tables"

# Evaluate retrieval quality
./scripts/afs embeddings eval --knowledge-path ~/.context/knowledge --provider gemini --query-file eval_cases.jsonl
```

Embedding providers: `none` (keyword-only), `ollama`, `hf` (HuggingFace), `openai`, `gemini`.

`afs search` is the version 2 user-facing API. It filters scope before ranking,
uses local retrieval unless `--semantic` is present, and defaults semantic
Gemini collections to stable `gemini-embedding-2` at 768 dimensions. The
`afs embeddings ...` commands remain the lower-level collection and evaluation
API.

For Gemini, the system auto-selects the correct task type: `RETRIEVAL_DOCUMENT` for
indexing, `RETRIEVAL_QUERY` for search queries (asymmetric retrieval). Override with
`--gemini-task-type`.

## Context Sources

```bash
./scripts/afs sources list --json
./scripts/afs sources status --path . --json
./scripts/afs sources sync --provider example_tasks --path . --json
./scripts/afs sources sync --provider example_tasks --path . --apply
```

Context-source providers are extension-owned and provider-neutral. See
`docs/CONTEXT_SOURCES.md`. `sources sync` materializes only in v1 under
`.context/items/sources/`. In v2 it fails before provider invocation until
scoped ingestion can route project records to
`knowledge/projects/<project-id>/` or, with an explicit shared choice,
`knowledge/common/`. Provider `list` and `status` remain available in v2.

## Antigravity CLI

```bash
./scripts/afs antigravity status --json
./scripts/afs antigravity setup --scope project --project-path .
./scripts/afs antigravity setup --scope project --project-path . --apply
./scripts/afs antigravity models
./scripts/afs antigravity models --json
```

`setup` is dry-run by default and does not install `agy`. See
`docs/ANTIGRAVITY_CLI.md`.

## Gemini

```bash
# Set up Gemini settings.json with AFS MCP server
./scripts/afs gemini setup
./scripts/afs gemini setup --scope project          # write ./.gemini/settings.json
./scripts/afs gemini setup --force                    # overwrite existing entry
./scripts/afs gemini setup --settings-path ~/custom/settings.json
./scripts/afs gemini setup --python-module          # use python -m afs.mcp_server

# Check integration health
./scripts/afs gemini status
./scripts/afs gemini status --json
./scripts/afs gemini status --skip-ping               # skip live embedding test
./scripts/afs gemini status --project afs             # inspect one project subtree
./scripts/afs gemini status --context-root "<workspace-root>/.context"

# Generate context from knowledge base
./scripts/afs gemini context                           # dump full INDEX.md
./scripts/afs gemini context "sprite development"      # search for relevant docs
./scripts/afs gemini context "debugging" --top-k 3     # limit results
./scripts/afs gemini context "hooks" --include-content  # include full doc text
./scripts/afs gemini context "training" --json          # machine-readable output
./scripts/afs gemini context --project afs "sqlite"     # search one project subtree
./scripts/afs gemini context --knowledge-path ~/.context/knowledge/afs "hooks"
```

`afs antigravity setup` previews or writes the AFS MCP entry for Antigravity CLI. New `agy` builds use `~/.gemini/config/mcp_config.json` for MCP config by default. `afs gemini setup` remains as Gemini CLI compatibility/API-key setup and writes settings so
Gemini can discover AFS tools automatically. The default launch target is the
repo-local `scripts/afs mcp serve` wrapper, which preserves AFS runtime env and
repo-config preference automatically. Use `--scope project` for repo-local
`./.gemini/config/mcp_config.json` and `--python-module` only when you explicitly want
the direct Python module entrypoint.

`afs gemini status` checks: API key, google-genai SDK, settings.json, MCP registration,
embedding index, and live embedding ping.

`afs gemini context` generates context from the knowledge base using semantic search
(when embeddings are indexed) or dumps the full knowledge INDEX.md. When no
`--project` or `--knowledge-path` is given, it searches across every indexed
project subtree under the active context knowledge root.

The in-repo Gemini backend also supports configurable explicit cached-content
reuse for repeated long prompts. Use env vars for machine-wide defaults:

- `AFS_GEMINI_CACHE_MODE=off|try|required`
- `AFS_GEMINI_CACHE_TTL=3600s`
- `AFS_GEMINI_CACHE_MIN_CHARS=4000`

Or override per agent via `ModelConfig.extra`, for example:

```python
ModelConfig(
    provider=ModelProvider.GEMINI,
    model_id="gemini-1.5-flash-001",
    extra={
        "gemini_cache": {
            "mode": "try",
            "ttl": "600s",
            "min_chars": 2000,
        }
    },
)
```

`try` is best-effort and falls back to uncached generation if cache creation or
lookup fails. `required` treats cache failures as fatal. `min_chars` is a local
AFS heuristic for avoiding explicit cache creation on tiny prefixes.

## Briefing

```bash
./scripts/afs briefing
./scripts/afs briefing --short
./scripts/afs briefing --json
./scripts/afs briefing --org
./scripts/afs briefing --no-gws                        # skip Google Workspace lookups
```

## GWS

```bash
./scripts/setup_gws.sh --dry-run                     # preview install/auth
./scripts/setup_gws.sh --credentials ~/Downloads/client_secret.json
./scripts/afs gws status                               # gws auth status
./scripts/afs gws agenda                               # today's calendar agenda
./scripts/afs gws unread                               # unread primary inbox
./scripts/afs gws raw gmail +triage --output-format json
```

## MCP

```bash
./scripts/afs mcp serve                             # slim tools/list catalog
./scripts/afs mcp serve --tool-catalog full         # expose all registered tools
```

Useful Gemini-oriented MCP operations:

- `afs.session.bootstrap` for the full session-start packet
- `context.search` for scoped local-first hybrid retrieval
- `context.query` for SQLite path/content search
- `context.status` for mount counts, mount health, profile, and index health
- `context.read`, `context.write`, and `context.list` for context file access
- `messages.send` and `messages.read` for scoped coordination
- `note.create`, `note.read`, and `note.list` for durable Markdown notes
- `handoff.create`, `handoff.read`, and `handoff.list` for immutable revisions

The default MCP `tools/list` response is deliberately focused so models do not
have to choose among every administrative surface. Use the CLI/framework for
session packs, work preflight, approvals, repair, handoff, verification,
training, and diagnostics. Set `AFS_MCP_TOOL_CATALOG=full` or pass
`--tool-catalog full` when debugging, migrating, or using a client that needs
the legacy all-tools catalog.

Workspace-root override:

```bash
export AFS_MCP_ALLOWED_ROOTS=~/workspaces/company
```

Gemini brief agent:

```bash
./scripts/afs agents run gemini-workspace-brief --stdout
./scripts/afs agents ps --all
./scripts/afs agents ps --all --json
./scripts/afs services start gemini-workspace-brief
./scripts/afs services start agent-supervisor
./scripts/afs services start context-warm
```

Extension-owned agents, including local domain orchestrators, appear here only
after their companion repo is enabled via `[extensions]`.

`context-warm` now audits each discovered context for broken symlink mounts,
duplicate mount targets, missing profile-managed mounts, untracked/stale mount
provenance, and stale SQLite indexes. The built-in service now runs with
`--repair-mounts --rebuild-stale-indexes --doctor-snapshot` by default.

For continuous maintenance, start the watcher:

```bash
./scripts/afs services start context-watch
```

If you have profile-driven background agents with `auto_start`, `schedule`, or
`watch_paths`, start the supervisor too:

```bash
./scripts/afs services start agent-supervisor
./scripts/afs services start history-memory
```

Profiles with an empty `agent_configs` list receive a conservative default set:
a network-free daily context audit, configured knowledge/memory index refresh,
weekly skill mining, and a daily scratchpad briefing. Existing custom lists are
not augmented. Disable the set with `[agents] default_set = false` or
`AFS_DEFAULT_AGENTS=off`. `daily` is an elapsed interval from the first run,
not a wall-clock morning schedule, and starting the supervisor remains an
explicit operator action.

The supervisor stores state under
`.context/scratchpad/common/afs_agents/supervisor/` in v2 and
`.context/scratchpad/afs_agents/supervisor/` in v1, so repo- or context-scoped
configs do not get shadowed by a single global PID cache.

If you want background services to stay pinned to a repo-local config and
`.context`, start them with `--config`:

```bash
./scripts/afs services start --config /path/to/afs.toml context-warm
./scripts/afs services start --config /path/to/afs.toml agent-supervisor
```

OS-managed service lifecycle is also available:

```bash
./scripts/afs services install context-warm --enable
./scripts/afs services status --system
./scripts/afs services logs context-warm
./scripts/afs services disable context-warm
./scripts/afs services uninstall context-warm
```

`afs services render|install|enable|disable|start|stop|status|restart|logs` preserve that explicit
`AFS_CONFIG_PATH` for the spawned service process. `afs status` and
`afs health` also surface the `history-memory` maintenance report alongside
`context-warm`, `context-watch`, `agent-supervisor`, and the periodic
`doctor_snapshot`.

`context-watch` uses `context-warm --watch` and reacts to changes under the
context root and mounted source paths. If the optional `watchfiles` package is
not installed, it falls back to polling.

If you need the built-in maintenance daemons scoped to a subset of contexts
without overriding the full command, use `context_filters` in your service
config:

```toml
[services.services.context-watch]
context_filters = ["~/workspaces"]
```

Codex MCP config:

```toml
[mcp_servers.afs]
command = "$AFS_ROOT/scripts/afs"
args = ["mcp", "serve"]
```

Client launch wrappers:

```bash
./scripts/afs-gemini
./scripts/afs-claude
./scripts/afs-codex
```

These wrappers prefer repo-local config, refresh the session bootstrap packet,
and export the bootstrap artifact paths before launching the client. They also
export a shared `AFS_SESSION_ID`, so MCP tool calls, embeddings, message
traffic, and CLI actions can be replayed later with `afs events replay`.
Set `AFS_GEMINI_MCP_ALLOWED_ROOTS` or `AFS_CLIENT_MCP_ALLOWED_ROOTS` if you want
wrapper-local path defaults without exporting `AFS_MCP_ALLOWED_ROOTS` globally.

## Doctor

```bash
./scripts/afs doctor                               # diagnose all common issues
./scripts/afs doctor --fix                          # auto-apply available fixes
./scripts/afs doctor --json                         # machine-readable output
```

`afs doctor` runs diagnostic checks across the full AFS stack and reports
actionable results. Checks include: Python environment, config loading,
context root integrity, context mount/provenance health, optional dependencies,
MCP registration, embedding indexes, extension loading, context index freshness,
configured skill loading, configured auto-start service state, and MCP server
build.

When `--fix` is passed, the doctor auto-applies fixes for issues that have
automated remediation (e.g., creating missing context directories, rebuilding
mount structures, seeding/pruning mount provenance, and rebuilding stale
indexes). Issues without auto-fix include suggested manual commands.

The CLI also catches common runtime errors (missing dependencies, file not
found, permission denied) and suggests running `afs doctor` instead of showing
raw tracebacks.

The MCP server runs startup diagnostics on launch and logs warnings/errors to
stderr, so Gemini and other MCP clients can surface context/index/runtime
problems without the server crashing. The startup subset is lighter than the
full `doctor` run and skips operator-only checks like service registration
state.

`context-warm` and `context-watch` now write
`.context/scratchpad/common/afs_agents/doctor_snapshot.json` in v2 (or
`.context/scratchpad/afs_agents/doctor_snapshot.json` in v1) so `afs health`
can surface the latest maintenance-time diagnosis even when you have not run
the doctor manually.

## Health

```bash
./scripts/afs health
./scripts/afs health --json
./scripts/afs health check --level standard
```

`afs health` reports AFS MCP registration for Gemini, Claude, and Codex, and it
detects both `python -m afs.mcp_server` and `afs mcp serve` processes. It also
reports broken mounts, duplicate mount targets, provenance drift, repair/remap
activity, recent MCP workflow usage (`afs.session.bootstrap`, `context.status`,
`context.diff`, `context.query`, `session.pack`), maintenance service state,
and supervisor agent state so you can see context drift or failed background
agents without opening the directory manually.

Repair a context directly when you want an explicit fix instead of waiting for a
background service:

```bash
./scripts/afs context repair --dry-run
./scripts/afs context repair --rebuild-index
```
