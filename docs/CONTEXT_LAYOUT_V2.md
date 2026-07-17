# Central Context Layout v2

Status: opt-in. Unmarked version 1 contexts remain supported and are not
silently converted.

## The namespace

Version 2 keeps AFS-managed context in one central root, normally
`~/.context`, with six human-facing categories:

```text
~/.context/
├── history/       # chronology and provenance
├── memory/        # durable learned context and handoffs
├── scratchpad/    # temporary work and drafts
├── knowledge/     # reference material
├── tools/         # trusted skills and executable resources
├── human/         # human-authored intent, decisions, and approvals
├── .afs/          # registries, queues, indexes, health, runtime state
└── README.md
```

`.afs/layout.toml` marks a valid v2 root. `.afs` is implementation state, not
a seventh context category. Messages live at `.afs/queue/messages`, project
records at `.afs/projects`, and the scoped hybrid index at `.afs/search`.

Project source code stays in its checkout. AFS registers and indexes it; it
does not copy a repository into `~/.context`.

Create a fresh v2 root explicitly:

```bash
afs context init --layout-version 2 --path /path/to/project
```

Without `--context-root`, v2 uses `general.context_root`. Scaffolding refuses
to overlay a non-empty unmarked directory.

## Stable scopes

Every registered project has a stable `prj_<uuid>` ID and a
`project:<project-id>` scope. Checkout paths and worktrees are aliases, not
identity. Cwd resolution chooses the most-specific registered project.

Each category separates shared and project-owned context:

```text
memory/
├── common/
└── projects/
    └── prj_<uuid>/
```

The same `common/` and `projects/<id>/` shape applies to the other five
categories as scoped data is created. Normal reads see only the current
project and `common`. Cross-project reads require an explicit
`--all-projects` option (or its API equivalent). Supplying the central context
path alone never grants access to every project.

Register paths deliberately:

```bash
afs projects current --path "$PWD"
afs projects register /path/to/project --name project-name
afs projects import --workspace-root ~/src          # dry-run
afs projects import --workspace-root ~/src --apply  # register previewed paths
```

## Readable artifacts

New notes, drafts, and handoff revisions are immutable Markdown with TOML
front matter. AFS owns filename allocation:

```text
YYYY-MM-DDTHHMMSSZ--readable-title-up-to-60-chars--10charid.md
```

The full UUID remains in metadata. The short suffix makes filenames readable
without making it the identity. Creation is exclusive; updating an artifact
means publishing a new one.

Durable notes live under `memory/<scope>/notes`. Drafts live under
`scratchpad/<scope>/notes`:

```bash
afs notes draft "Investigate cache drift" --body-file findings.md
afs notes promote <draft-id>  # copies to memory and records provenance
afs notes archive <draft-id>  # explicitly moves it out of the active scratchpad
```

Promotion is idempotent and leaves the draft in place. Archival is always an
explicit action.

## Handoff threads

A handoff is a logical thread containing immutable revisions. `revise`
publishes a new revision with a `supersedes` link; it never edits the old
revision. Acknowledgement and closure are separate append-only lifecycle
events, so status changes do not rewrite content.

```bash
afs handoff create --title "Parser cleanup" --agent-name codex \
  --accomplished "Added scoped parser" --next "Run integration tests"
afs handoff revise <revision-id> --title "Parser cleanup follow-up" \
  --agent-name codex --next "Review the final diff"
afs handoff ack <revision-id> --by reviewer
afs handoff close <thread-or-revision-id> --by reviewer --reason "merged"
```

Handoffs live under `memory/<scope>/handoffs`. Version 1/2 JSON handoffs remain
readable in the `common` compatibility scope for one transition cycle.

## Search and messages

`afs search` is the v2 entry point for scoped retrieval. It searches the
current project plus `common` and applies scope filtering before ranking:

```bash
afs search "where is cache invalidation handled" --path "$PWD"
afs search "parser ownership" --path "$PWD" --mode symbol
afs search "similar retry failures" --path "$PWD" --semantic --rebuild
```

The default is local text/symbol retrieval. `--semantic` is the explicit
permission boundary for embeddings. Gemini uses stable
`gemini-embedding-2` at 768 dimensions by default; Ollama is also supported.
Use `--all-projects` only when cross-project results are intended.

The public coordination name is **messages**:

```bash
afs messages list --path "$PWD"
afs messages send --path "$PWD" --from codex --topic status \
  --payload '{"summary":"tests pass"}'
```

The old `hivemind` CLI, MCP tools, persisted class names, and mount role remain
compatibility surfaces for one cycle. New integrations should use
`messages.*` and `.afs/queue/messages`.

## Safe inspection and migration planning

These commands inspect and write manifests only; they do not migrate data:

```bash
afs layout audit --context-root ~/.context --json
afs layout plan --context-root ~/.context --json
afs layout plan --context-root ~/.context \
  --output /private/path/migration.json \
  --rollback-output /private/path/rollback.json
```

Unknown top-level entries block readiness instead of being guessed. A plan is
hash-bound to the audited inventory and records deterministic copy-and-verify
operations plus a paired rollback manifest. Manifests are atomically written
with mode `0600`.

There is intentionally no layout `apply` command. Migrating a live root still
requires a separately reviewed transaction implementation and an approved
dry-run against that exact root.
