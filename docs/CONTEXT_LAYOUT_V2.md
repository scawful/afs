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

The shared append-only event ledger is stored at
`history/common/events_YYYYMMDD.jsonl`; large event payloads stay beneath the
same directory. Readers retain a link-safe transition read of pre-fix event
files directly under `history/`, while all new writes use `history/common`.

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

## Safe inspection, planning, and destination-only migration

Auditing is read-only. Planning writes a private schema-v2 manifest;
it does not copy data:

```bash
afs layout audit --context-root /path/to/v1-context --json
afs layout plan --context-root /path/to/v1-context \
  --destination-root /path/to/new-v2-context \
  --mapping-file /private/path/layout-mappings.json \
  --output /private/path/migration-plan.json \
  --rollback-output /private/path/source-retention.json
```

The destination must be separate from the source and must not already exist.
The plan is hash-bound to the exact source inventory and records the reviewed
destination and mapping decisions. Plans are written atomically with mode
`0600`; treat the mapping file as private input too. If the source changes,
build and review a new plan rather than reusing the stale one.

The mapping file, migration-plan output, and optional rollback-manifest output
must resolve to distinct paths outside both the v1 source and the candidate
destination trees. This keeps private planning artifacts out of the source
fingerprint and prevents them from being copied into or exposed through the
candidate.

Unknown top-level entries remain blocking unless the mapping file names each
source entry exactly. Generic mappings do not accept globs, nested source
paths, or inferred categories. Their destinations are limited to either:

- `<category>/common/...`, where `<category>` is one of the six human-facing
  categories; or
- `.afs/compat/imported/...` for preserved compatibility data that does not
  belong in a human-facing category.

A minimal mapping file has this exact shape:

```json
{
  "schema_version": 1,
  "mappings": {
    "exact-top-level-name": "knowledge/common/imported/exact-top-level-name"
  }
}
```

Mapping-file schema v1 is distinct from migration-plan schema v2. The mapping
document contains exactly `schema_version` and `mappings`. Each mapping key
names one exact unknown top-level entry; each value is one allowed destination
relative to the new v2 root.

This deliberately narrow contract is suitable for simple files and
single-purpose directories. Mixed stores, nested project contexts, active
queues, and registries that need merging require a dedicated importer or must
remain quarantined; a generic mapping must not guess how to split them.

### Preview and apply

`layout migrate` previews a reviewed plan by default. Preview verifies the
plan, source fingerprint, destination preconditions, links, special files,
and proposed operations without creating the destination:

```bash
afs layout migrate --plan /private/path/migration-plan.json
```

Applying is an explicit human-gated operation:

```bash
afs layout migrate --plan /private/path/migration-plan.json \
  --apply --because "Create a verified v2 candidate for review"
```

Audit and planning remain available on Windows, but the current migration
executor does not run there. `layout migrate` fails preflight closed because
POSIX `chmod` semantics cannot establish or verify the private Windows DACLs
required for candidate directories, copied files, receipts, and markers.
Preview/apply support on Windows remains blocked until explicit DACL handling
lands.

`--apply` requires a non-empty rationale and confirmation through the
controlling terminal; piped input and headless callers cannot satisfy the
gate. The migration copies and verifies into the separate destination. It
never modifies or deletes the v1 source. A symbolic link or special file
anywhere in a planned source subtree blocks the transaction rather than being
followed or copied.

The destination remains an unmarked candidate while data is copied and
verified. `.afs/layout.toml` is published last, so an interrupted copy cannot
be mistaken for an authorized v2 root. On a caught apply failure before marker
publication, AFS attempts to rename the partial tree to an adjacent
`.failed-*` name and reports that retained path when the rename succeeds.

A hard process or host interruption can bypass that failure handler and leave
an unmarked partial tree at the requested destination. It is never treated as
v2. Inspect it and explicitly move it out of the destination path before
retrying; AFS does not silently delete, resume, or quarantine it on the next
run.

### Threat boundary

Migration fails closed for stale reviewed evidence, accidental source
changes, links or special files found during inspection, and interrupted or
failed copies. It runs as a cooperative process with the invoking user's
filesystem authority, however; it is not a sandbox against a hostile
concurrent process running under the same user and swapping paths between
checks. Stop active producers and other writers, then plan, preview, and apply
in a controlled maintenance window. Recognition of an already completed
candidate is valid only while the source still matches the identity and
fingerprint captured by the reviewed plan.

Successful migration creates a verified v2 candidate only. Activating or
swapping it into use, rolling back an activation, and cleaning up either the
source or a `.failed-*` tree are separate future human-gated operations.

Any rollback manifest emitted for compatibility is informational and records
source preservation. It does not restore, overwrite, or delete data. Because
the source remains untouched and activation is not automatic, there is no
implicit restore step in `layout migrate`.

### Exit codes

`layout migrate` uses evidence-bearing exit codes:

| Code | Meaning |
|---|---|
| `0` | Preview is ready, the candidate was applied, or the exact completed transaction was recognized as already applied. |
| `2` | The plan is invalid or stale, preflight is blocked, the rationale is invalid, or human confirmation was refused or unavailable. |
| `3` | An authorized apply started and failed. Output identifies the retained `.failed-*` path when a pre-marker tree was created and quarantined. |
| `1` | An unexpected internal error escaped to the generic AFS CLI handler. It is not migration evidence. |

These examples describe the command contract; they do not assert that any
particular live `~/.context` is migration-ready or has been migrated.
