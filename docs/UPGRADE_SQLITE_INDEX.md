# Upgrading AFS: SQLite Context Index

This guide covers the new SQLite-backed context index added alongside the existing
filesystem layer. No migration is required — the index is additive and opt-out.

## What changed

AFS now maintains an optional SQLite index (`context_index.sqlite3`) that mirrors
the contents of your `.context/` mount tree. The filesystem is still the source of
truth for all reads, writes, and mounts. The SQLite layer adds:

- **Full-text search** (FTS5) across file paths and content
- **Structured queries** with mount type, path prefix, and content filters
- **Fingerprint-based staleness detection** to know when the index is out of date
- **Incremental sync** — writes and mount/unmount operations automatically update
  the index without a full rebuild

The index lives at `.context/global/context_index.sqlite3` by default.

## Architecture

```
Writes/Mounts ──> Filesystem (.context/)  ← source of truth
                      │
                      ├── _sync_index_for_write()    (on every write_text)
                      └── _sync_context_index_mount() (on mount/unmount)
                              │
                              ▼
                   SQLite Index (context_index.sqlite3)
                              │
                              ▼
            MCP tools: context.query, context.index.rebuild, context.diff, context.status, context.repair
```

If the index fails or is deleted, all primary AFS operations continue to work.
The index can be rebuilt at any time.

## Configuration

The index is controlled by the `[context_index]` section in your AFS config
(`afs.toml` or programmatic `AFSConfig`):

```toml
[context_index]
enabled = true                   # Master switch (default: true)
db_filename = "context_index.sqlite3"  # Relative to .context/global/
auto_index = true                # Auto-build index on first query
auto_refresh = true              # Auto-rebuild when staleness detected
include_content = true           # Index file contents (not just paths)
max_file_size_bytes = 262144     # Skip files larger than 256 KB
max_content_chars = 12000        # Truncate indexed content at this limit
```

### Disabling the index

If you don't need search and want zero overhead:

```toml
[context_index]
enabled = false
```

When disabled, `context.query` and `context.index.rebuild` MCP tools still exist
but return empty results. Writes and mounts skip the sync step entirely.

## New MCP tools

### `context.index.rebuild`

Force a full rebuild of the index for a context path.

| Parameter           | Type     | Default   | Description                     |
|---------------------|----------|-----------|---------------------------------|
| `context_path`      | string   | cwd       | Path to `.context/` directory   |
| `mount_types`       | string[] | all       | Filter to specific mount types  |
| `include_content`   | boolean  | true      | Index file contents             |
| `max_file_size_bytes`| integer | 262144    | Skip files larger than this     |
| `max_content_chars` | integer  | 12000     | Truncate content at this length |

### `context.query`

Search the index with text queries, path prefixes, and mount type filters.

| Parameter           | Type     | Default | Description                          |
|---------------------|----------|---------|--------------------------------------|
| `context_path`      | string   | cwd     | Path to `.context/` directory        |
| `query`             | string   | ""      | Search text (matches paths + content)|
| `relative_prefix`   | string   | —       | Filter by path prefix                |
| `mount_types`       | string[] | all     | Filter to specific mount types       |
| `limit`             | integer  | 25      | Max results (1–500)                  |
| `include_content`   | boolean  | false   | Return full file content in results  |
| `auto_index`        | boolean  | true    | Build index if empty                 |
| `auto_refresh`      | boolean  | true    | Rebuild if stale                     |
| `refresh`           | boolean  | false   | Force rebuild before querying        |

Results include `relevance_score` when FTS matches, plus `content_excerpt` with
context around the match.

The same query path now powers the Gemini-facing `afs.query.search` MCP prompt,
including automatic refresh when the index is stale.

### `context.diff`

Show added, modified, and deleted files relative to the current index.

| Parameter      | Type     | Default | Description                         |
|----------------|----------|---------|-------------------------------------|
| `context_path` | string   | cwd     | Path to `.context/` directory       |
| `mount_types`  | string[] | all     | Filter to specific mount types      |

### `context.status`

Return mount counts, active profile, and index health for a context.

| Parameter      | Type   | Default | Description                   |
|----------------|--------|---------|-------------------------------|
| `context_path` | string | cwd     | Path to `.context/` directory |

This is useful for Gemini-style agents that need a quick “context health”
snapshot before deciding whether to query or rebuild.

### `context.repair`

Repair a context by seeding missing provenance records, pruning stale
provenance, conservatively remapping missing mount sources across configured
workspace roots, reapplying profile-managed mounts when possible, and
optionally rebuilding the index.

| Parameter               | Type    | Default | Description                              |
|-------------------------|---------|---------|------------------------------------------|
| `context_path`          | string  | cwd     | Path to `.context/` directory            |
| `profile_name`          | string  | active  | Profile override                         |
| `dry_run`               | boolean | false   | Report planned repairs only              |
| `reapply_profile`       | boolean | true    | Reapply profile-managed mounts           |
| `remap_missing_sources` | boolean | true    | Try conservative workspace remapping     |
| `rebuild_index`         | boolean | false   | Rebuild SQLite index if stale or empty   |

## Upgrading from a previous AFS version

1. **Pull the latest code** and reinstall:
   ```bash
   cd /path/to/afs
   pip install -e .
   ```

2. **No config changes required.** The index is enabled by default. Your existing
   `.context/` directories and mounts are untouched.

3. **Build the initial index** (optional — it auto-builds on first query):
   ```bash
   # Via MCP tool
   afs mcp  # then call context.index.rebuild

   # Or it happens automatically on the first context.query call
   ```

4. **Verify** the index was created:
   ```bash
   ls -la .context/global/context_index.sqlite3
   ```

## Deleting and rebuilding the index

The index is fully disposable. To reset:

```bash
rm .context/global/context_index.sqlite3
```

The next `context.query` call (with `auto_index=true`, the default) will rebuild
it from the current filesystem state.

## Supported file types for content indexing

The index reads content from files with these extensions:
`.asm`, `.bash`, `.c`, `.cc`, `.cfg`, `.conf`, `.cpp`, `.css`, `.csv`,
`.h`, `.html`, `.inc`, `.ini`, `.java`, `.js`, `.json`, `.lua`, `.md`,
`.ps1`, `.py`, `.rs`, `.s`, `.sh`, `.sql`, `.toml`, `.ts`, `.tsx`,
`.txt`, `.xml`, `.yaml`, `.yml`, `.zsh`, `.65c`, `.65s`

Files without an extension are also checked (binary detection via null-byte scan).
Binary files and files exceeding `max_file_size_bytes` are indexed by path only.
