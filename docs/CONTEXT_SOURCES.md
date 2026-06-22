# Context Source Providers

AFS core keeps context-source integration provider-neutral. Concrete adapters for
issue trackers, task systems, review tools, document stores, chat logs, test
systems, hooks, or traces should live in extensions and emit normalized source
records into `.context/items`.

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

```bash
afs sources list --json
afs sources status --path . --json
afs sources sync --provider example_tasks --path . --json
afs sources sync --provider example_tasks --path . --apply
```

`sync` writes markdown records under:

```text
.context/items/sources/<provider>/<kind>-<id>.md
```

The context index can then search these records like other AFS item files.
