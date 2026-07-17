# Central Context Layout v2

Status: opt-in foundation. Existing version 1 contexts remain the default and
continue to be readable.

## Namespace

Version 2 keeps AFS-managed state in one central context root (normally
`~/.context`) with six human-facing categories:

```text
~/.context/
├── history/
├── memory/
├── scratchpad/
├── knowledge/
├── tools/
├── human/
├── .afs/
└── README.md
```

`.afs/layout.toml` is the version marker. Internal registries, compatibility
state, queues, search data, and runtime data belong below `.afs`; they are not
additional context categories. The canonical v2 messages path is
`.afs/queue/messages`. Legacy `MountType.HIVEMIND` resolution points to that
path, so existing consumers can read messages while adopting the new naming.

Create a new v2 root explicitly:

```bash
afs context init --layout-version 2 --path /path/to/project
```

Without `--context-root`, version 2 uses `general.context_root`. Version 1
initialization behavior is unchanged.

## Projects and authorization scopes

Project records live at `.afs/projects/prj_<uuid>.toml`. A record owns a stable
project ID and `project:<id>` scope; checkout paths and worktrees are aliases,
not identity. Cwd resolution selects the most-specific registered project.

Artifact lookups through `ProjectRegistry.resolve_scoped_path` are authorized
for only `common` and the current project by default. Cross-project lookup must
set `allow_all_projects=True` at an already-authorized boundary. An
unregistered directory does not inherit the central context merely because it
is located below the home directory.

## Inspection and migration planning

These commands do not migrate data:

```bash
afs layout audit --context-root ~/.context --json
afs layout plan --context-root ~/.context --json
afs layout plan --context-root ~/.context \
  --output /private/path/migration.json \
  --rollback-output /private/path/rollback.json
```

The audit reports missing categories and unknown top-level entries. Unknown
entries block migration instead of being guessed. A plan records an inventory
fingerprint, byte/file counts, deterministic copy-and-verify operations, and a
paired rollback manifest. Manifests are written atomically with mode `0600`.
There is intentionally no apply command in this foundation; execution requires
a separately reviewed transaction implementation and a dry-run against the
real context root.
