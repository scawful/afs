# Context Source Providers

AFS core keeps context-source integration provider-neutral. Concrete adapters for
issue trackers, task systems, review tools, document stores, chat logs, test
systems, hooks, or traces should live in extensions and emit normalized source
records. Materialization currently targets only the version 1
`.context/items` layout.

In a version 2 context, `afs sources list` and `afs sources status` remain
available, but `afs sources sync` fails before loading or invoking a provider.
Scoped v2 ingestion is pending: project records will belong under
`knowledge/projects/<project-id>/`, while shared records will require an
explicit `knowledge/common/` choice. This prevents provider data from silently
entering the unscoped compatibility store.

## Record kinds

Core AFS recognizes these broad categories:

- `task`
- `ticket`
- `review`
- `doc`
- `message`
- `test`
- `hook`
- `trace`

These categories are intentionally generic. Avoid encoding a company's internal
systems or private workflow names in core AFS.

## Extension manifest

Declare providers in an extension `extension.toml`:

```toml
name = "afs_example"
description = "Example context source providers"
python_paths = ["src"]

[[context_sources]]
name = "example_tasks"
module = "afs_example.sources"
factory = "register_context_source_provider"
description = "Example task/review source"
kinds = ["task", "review"]
```

The factory should return an object with:

```python
name = "example_tasks"
kinds = ("task", "review")

def status() -> dict:
    return {"ok": True}

def sync(*, query: str = "", limit: int = 50) -> list[ContextSourceRecord | dict]:
    ...
```

## CLI

Version 1 sync:

```bash
afs sources list --json
afs sources status --path . --json
afs sources sync --provider example_tasks --path . --json
afs sources sync --provider example_tasks --path . --apply
```

In a v1 context, `sync` writes markdown records under:

```text
.context/items/sources/<provider>/<kind>-<id>.md
```

The v1 context index can then search these records like other AFS item files.
Do not use this path as a manual workaround in v2; wait for scoped ingestion.
