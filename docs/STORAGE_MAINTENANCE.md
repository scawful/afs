# Storage Maintenance

AFS separates **seeing storage pressure** from **deleting data**. The default
workflow stays online, never stops an agent or model server, and never treats a
large directory as permission to remove it.

## Read-only model retention review

Model stores need lifecycle evidence rather than generic age-based cleanup.
Inspect them with the separate model-retention audit:

```bash
# Inspect known local model roots with a seven-day recent-artifact grace period.
afs storage models

# Make roots, runtime/config references, and reviewed lifecycle intent explicit.
afs storage models \
  --root ~/models/gguf \
  --root ~/models/mlx \
  --registry ~/src/lab/afs-scawful/config/chat_registry.toml \
  --policy ~/models/model-retention.toml \
  --recent-days 14 \
  --json
```

The report classifies each discovered model artifact:

- `keep` — an explicit policy keep, active process reference, registry
  reference, or recent modification protects it.
- `review` — an explicit policy review names a discovered, protected
  replacement, both reference scans succeeded, measurement is complete, and
  there is neither an unobserved hard link nor a retained hard-link peer.
- `unknown` — every other artifact, including one with incomplete evidence or
  no explicit lifecycle policy.

Every artifact includes reasons. The review total uses filesystem-allocated
bytes and is an **upper bound**, not a reclaim promise: hard links, APFS clones,
shared cache blobs, or other storage sharing may reduce actual savings. Age and
size alone never authorize model removal.

Policies are TOML and require a rationale for every decision:

```toml
schema = "afs.model-retention.v1"
roots = ["~/models/gguf", "~/models/mlx"]

[[artifacts]]
path = "~/models/gguf/current.gguf"
decision = "keep"
because = "Referenced by the current deployment."

[[artifacts]]
path = "~/models/gguf/older.gguf"
decision = "review"
because = "Superseded after evaluation."
superseded_by = "~/models/gguf/current.gguf"
```

Policy artifact and root paths must stay inside the audited home. Unlisted
artifacts remain `unknown`; filenames and apparent version numbers are never
used to infer supersession.

The active scan reads accessible process arguments and, when installed,
queries `lms ps --json` and `ollama ps` so models hidden behind those runtimes
are not mistaken for idle files. It also requires `lsof` to establish that
reviewed artifacts are not open or memory-mapped. An unreadable process,
ambiguous `lsof` result, missing `lsof`, or failed runtime status query makes
explicit review entries `unknown`.

Likewise, `review` requires at least one validated chat-registry source. Pass
`--registry` explicitly or configure `model_registries` in the active AFS
profile. A missing, empty-source, malformed, or structurally invalid registry
scan fails closed; an intentionally empty registry file is valid evidence.

The inventory is an online, read-only observation rather than an atomic
filesystem snapshot. Directory enumeration uses no-follow handles and detected
changes, links, truncation, or read errors fail closed, but model-store
directories should not be renamed or relinked during a run. Model servers and
agents can remain online. Re-run the audit after any concurrent model download,
conversion, or store reorganization before acting on its report.

`storage models` does not write, rename, stop, unload, or delete anything. Its
output is not accepted by `storage plan` or `storage apply`; those commands
cannot remove model artifacts. A later lifecycle action must use the model
store's owning tool and a separately reviewed decision.

## Three-step workflow

```bash
# 1. Read-only: disk pressure, safe candidates, protected footprints, snapshots.
afs storage audit

# 2. Write one plan artifact; do not mutate any cleanup candidate.
afs storage plan --output ~/.afs/storage-plan.json

# 3. Mutating and human-gated: review the JSON, then run this in a terminal.
afs storage apply \
  --plan ~/.afs/storage-plan.json \
  --confirm storage_<exact-token-from-plan> \
  --because "Remove reviewed rebuildable artifacts."
```

`audit` is read-only. `plan` never changes a cleanup candidate, but it writes
the requested JSON plan and creates missing parent directories for that
artifact. `apply` requires both the exact hash-derived transaction and a
non-empty rationale, then asks the person at the controlling terminal to
re-type that transaction. Piped input, background agents, and API callers
cannot satisfy this confirmation. After confirmation, AFS re-discovers every
path, checks type/device/inode/size/timestamps/tree digest, checks for open
files again, and refuses the whole preflight if anything drifted.

Plans expire after 24 hours. A claimed transaction cannot be replayed. The
immutable plan copy, claim, per-candidate intent/outcome journal, transaction
quarantine, and final receipt are stored under
`~/.afs/storage/transactions/<transaction>/`. Candidates are first atomically
moved into that same-filesystem quarantine, verified by identity and recorded
tree metadata, and only then removed there. If a race or crash interrupts
cleanup, the durable journal and any retained quarantine entry identify the
recovery point, and the transaction stays blocked from replay.

## What can enter a plan

Only old, bounded, rebuildable entries are candidates:

- Yaze nightly directories named `build-nightly*`
- OpenCode snapshot temporary pack files named `tmp_pack_*` and stale `gc.pid`
- dated LM Studio server logs and `*.old.log` desktop logs
- broken LM Studio model symlinks (the link only, never a model target)

The default minimum age is 30 days. Change it explicitly with
`--stale-days`, from 1 through 3650 days.

Open or ambiguous entries are blocked. If `lsof` is unavailable, normal files
and directories fail closed rather than becoming eligible. A live PID in an
OpenCode `gc.pid` file also blocks that entry.

## What never enters a plan

These stay protected and never become cleanup candidates. The audit measures
the storage roots it can inspect safely so a human can see where space went:

- model stores (local, Ollama, Hugging Face, LM Studio)
- caches outside the narrow temporary-file contract above
- `~/Archives` and `~/.Trash`
- AFS context roots and migration candidates
- APFS and Time Machine snapshots

Applications and application model bundles are also protected, but the storage
audit does not recursively measure app bundles; use macOS Storage Settings or
the owning app's uninstaller for that inventory.

Use `afs storage models` to inspect model-retention evidence, then use the
owning tool and a separate reviewed lifecycle decision for any action. Use the
owning tool directly for the other protected categories. AFS does not stop
processes or silently turn an audit into a cleanup.

## Disk-pressure language

The default free-space bands are intentionally plain:

| Free space | Status |
|---:|---|
| at least 450 GiB | `healthy` |
| 400–450 GiB | `watch` |
| 300–400 GiB | `warning` |
| below 300 GiB | `critical` |

Reported sizes use filesystem-allocated bytes. Hard links inside one candidate
are counted once, and externally linked regular files are excluded from the
reclaim estimate. APFS clones and snapshots can still share physical blocks,
so every reclaim number is an estimate rather than a promise.

## Local trust boundary

The cleanup protocol is designed to fail closed around ordinary concurrent
writers, path drift, symlinks, mount changes, crashes, and stale plans. It is
not a sandbox against a hostile process running as the same operating-system
user and deliberately racing private AFS state directories after human
confirmation. Keep `~/.afs` private to the account and do not grant untrusted
same-user code access during an apply.

## Automation boundary

It is safe to schedule `afs storage audit --json` or `afs storage models
--json`, and to alert on disk pressure or changed model classifications. Do not
schedule `storage apply`: a human must review each exact plan, supply its
transaction plus rationale, and confirm it at the controlling terminal.
