# AFS Memory and Context Layout

## Two supported layouts

AFS reads both context generations:

- **Version 1** is an unmarked, usually project-local `.context` with legacy
  mount roles such as `hivemind`, `global`, and `items`.
- **Version 2** is an explicitly marked central namespace, normally
  `~/.context`, with six human-facing categories and stable project scopes.

Version 2 is opt-in. AFS never treats a non-empty v1 root as v2 merely because
it has similarly named directories. See [Central Context Layout
v2](CONTEXT_LAYOUT_V2.md) for the exact marker and migration guardrails.

## Version 2 layout

```text
~/.context/
├── history/       # chronology and provenance
├── memory/        # durable learned context, notes, and handoffs
├── scratchpad/    # temporary work and drafts
├── knowledge/     # reference material
├── tools/         # trusted skills and executable resources
├── human/         # human intent, decisions, and approvals
└── .afs/          # internal registries, queues, indexes, and health
```

`.afs` is not user context. In particular:

- `.afs/projects/` holds stable project records.
- `.afs/queue/messages/` holds the inter-agent message queue.
- `.afs/search/` holds rebuildable hybrid-search generations.

Within each human category, new scoped data uses `common/` or
`projects/<project-id>/`. The current project plus `common` is the default
visibility boundary. Project source stays in its checkout and is indexed from
there rather than copied into the central root.

## Durable memory and temporary notes

Human-facing artifacts are Markdown with TOML front matter and immutable,
sortable filenames:

```text
YYYY-MM-DDTHHMMSSZ--readable-title-up-to-60-chars--10charid.md
```

Use the lifecycle rather than editing generated files in place:

```bash
afs notes create "Release rationale" --body-file rationale.md
afs notes draft "Investigate flaky test" --body "Initial observations"
afs notes promote <draft-id>
afs notes archive <draft-id>
```

`create` writes a durable note to `memory/<scope>/notes`. `draft` writes to
`scratchpad/<scope>/notes`. `promote` copies the draft into durable memory with
source provenance and is idempotent; the source remains active until an
explicit `archive` moves it under the scoped scratchpad archive.

Handoffs use the same artifact format under `memory/<scope>/handoffs`, but add
logical thread and revision IDs. `handoff revise` appends a superseding
revision. Acknowledgement and closure are separate append-only state, not
content edits.

## History-to-memory consolidation

AFS also has a legacy-compatible event consolidation loop:

- `history/common/` is the v2 shared append-only event ledger.
- `memory/common/entries.jsonl` stores v2 summarized memory entries.
- `memory/common/history_consolidation/` stores readable v2 summaries.
- `scratchpad/common/afs_agents/history_memory_checkpoint.json` stores the v2
  incremental checkpoint.

Version 1 keeps the corresponding paths directly under `.context/history/`,
`.context/memory/`, and `.context/scratchpad/afs_agents/`.

Run it manually or through the maintenance agent:

```bash
afs memory consolidate --path ~/src/project-a
afs agents run history-memory --stdout
afs services start history-memory
```

The default consolidator is intentionally low-sensitivity. It summarizes
selected event metadata and does not copy raw history payloads unless payload
capture was explicitly enabled. Pre-fix v2 entries at the memory category root
remain readable but all new writes use the explicit shared scope.

Optional config:

```toml
[memory_consolidation]
interval_seconds = 1800
auto_start = true
include_event_types = ["context", "fs", "review"]
max_events_per_run = 200
max_events_per_entry = 50
write_markdown = true
```

## Access and retrieval

Plain-language commands cover normal work:

```bash
afs start --path ~/src/project-a
afs search "release checklist" --path ~/src/project-a
afs files list knowledge --path ~/src/project-a
afs notes list --path ~/src/project-a
afs handoff threads --path ~/src/project-a
afs messages list --path ~/src/project-a
```

`files` is an alias for `fs`; existing `afs fs ...` calls remain valid. In a
v2 root, scoped file and search operations require the project path (or an
explicit authorized scope). A central `context_path` by itself is not an
all-project capability.

`afs search` is local-first and filters scope before ranking. Pass
`--semantic` to explicitly enable embeddings for that rebuild/query. Gemini
defaults to stable `gemini-embedding-2` with 768-dimensional vectors. Without
`--semantic`, no remote embedding call is made.

The older `afs embeddings ...` collection commands remain available for
direct index management and evaluation. See [Embeddings](EMBEDDINGS.md).

## Version 1 compatibility

Version 1 may contain:

```text
.context/{memory,knowledge,tools,scratchpad,history,hivemind,global,items}/
```

Those names remain readable through `MountType` and existing CLI/MCP aliases.
The public v2 term for `hivemind` is **messages**. Compatibility does not imply
that unscoped legacy records become visible inside every project scope: v2
imports legacy handoffs only into `common`, and legacy unscoped messages are
excluded unless explicitly requested.

Inspect before considering migration:

```bash
afs layout audit --context-root ~/.context --json
afs layout plan --context-root ~/.context --json
```

Both are non-migrating operations. There is no v2 layout apply command.
