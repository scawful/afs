# AFS Retool Plan (2026-10-27)

Status: in progress. Built from a read-only audit of `src/afs/` (333 files, ~132K LOC)
and the live context roots. Numbers marked (verified) were re-checked by hand;
the rest come from the audit with file:line references below.

Progress (2026-10-27):

- Prerequisite: done. The dirty BLE001 cleanup landed as #125; `49ab30d`
  (`--request-id`) landed inside #124.
- Phase 1 (approvals → SQLite): code merged as #124; Barista reader merged as
  barista#37. Live once the runtime checkout is updated.
- Phase 2 (storage hygiene): code merged as #123. The live DB compaction,
  registry reap, and events archive still need an owner go-ahead.
- Phase 3, first step (dead packages): merged as #122. The remaining Phase 3
  rows are open.
- Decisions 1 and 3 below are resolved (land the cleanup; SQLite now).

## Prerequisite: get `main` clean

- Local `main` is 19 commits behind `origin/main` and 1 ahead (`49ab30d`,
  approvals `--request-id`, unpushed). (verified)
- 36 uncommitted files on that stale base: a BLE001 lint cleanup from another
  agent, last touched 2026-10-27 13:01. `tooling_watch.py` is untracked locally
  but already merged upstream in #119. (verified)
- Do all retool work in worktrees cut from `origin/main` until the owner of the
  dirty tree rebases or drops it.

## Evidence

| Problem | Number |
|---|---|
| `context_index.sqlite3` free pages, lab root | 1,490 of 1,586 MB (verified) |
| `context_index.sqlite3` free pages, src root | 613 of 930 MB (verified) |
| Embedding JSON payloads with `embedding: []` | 17,754 files, ~71 MB; 200/200 sampled empty (verified) |
| History jsonl + payloads, no retention | 988 MB `~/.context/history`, 119 MB lab (verified) |
| Agent registry rows from pytest temp roots | 438 of 462; 145 stuck `running` (verified) |
| Unreachable code (no import or entry point) | ~13.8K LOC |
| MCP tools / CLI leaf commands | 95 / 286 |

## Phase 1 — Approvals (about 4 hours)

Current faults (`agents/guardrails.py`, `cli/approvals.py`, `agents/mission_runner.py`):

1. The only terminal states are human `approved`/`rejected`. No expiry, withdraw,
   or supersede, so orphan gates live forever. The 5 July gates were orphans:
   mission TOMLs set `requires_approval = false` on decide on 2026-09-09.
2. Rejection is not durable: the next `check()` with the same
   (agent, action, detail) queues a new request (`guardrails.py:733-741`).
3. Approval is a permanent blanket grant, yet `clear` archives it and silently
   revokes it because `check()` only reads the active file (`:560-562`).
4. The gate carries no payload; approving a decide phase approves a blank check.
5. Two siloed systems: agent gates (`~/.config/afs/agents/approvals.json`) and
   work approvals (`work_assistant.sqlite3`). Bootstrap and MCP only show work
   approvals; Barista only shows agent gates and reads the JSON with jq.
6. One malformed row raises `ApprovalStateError` and halts all gating (`:532-546`).
7. `require_approval_for` is parsed but never enforced (`mission_runner.py:123,162`).

Target design:

1. States: `pending` → `approved` | `rejected` (human, tty ack) | `expired` |
   `withdrawn` | `superseded` (system or agent; recorded with `actor`,
   `human_confirmed=false`, excluded from calibration). This is the honest label
   the current code lacks: an agent can withdraw, it cannot reject.
2. `expires_at` on every request; default TTL 72h for mission gates. A reaper
   marks rows `expired` on every list/check.
3. `source` field (mission id + run id + config hash). A new run, or a config
   change that no longer requires the gate, supersedes older requests.
4. Approval scope is explicit: `once` (consumed by one run) or `standing`
   (until revoked). Reject applies to the run, not forever.
5. Request carries a payload (plan/scan summary or path) shown at decision time.
6. Batch decision: `afs approvals reject --stale` lists N rows, one tty ack.
7. One store and one surface: agent gates move into SQLite next to work
   approvals; `afs approvals list --json` is the only reader. Barista calls it
   instead of jq on the file. Add MCP `approvals.list`.
8. Malformed rows are skipped and reported, never fatal.

## Phase 2 — Storage hygiene quick wins (about 2 hours)

1. `VACUUM` then FTS `rebuild` on both `context_index.sqlite3` files: recovers
   ~2.1 GB. FTS joins on implicit rowid, so the rebuild is required after VACUUM.
2. Exclude `embeddings/`, `*.bak*`, `embedding_index.json`, `.afs/`, and
   `~/Music` from every indexer.
3. Stop delete-all-and-reinsert on mount refresh (`context_index.py:~190-200`);
   update only rows whose hash changed.
4. History retention: keep 30 days of events, gzip older daily files.
5. Isolate `AFS_AGENT_REGISTRY_PATH` in `test_event_reactor.py` and
   `test_review_gate.py`; reap rows whose context root no longer exists.

Delete the 17,754 empty embedding JSON files only with explicit approval.

## Phase 3 — Bloat removal (about 1 day)

| Step | LOC | Risk |
|---|---|---|
| Delete dead packages: benchmark, benchmarks, evaluation, quality, deployment, comparison, pretraining, registry; 9 unregistered `cli/` modules; `agents/researcher` | ~13.8K | low |
| Merge `health/` into `diagnostics.py`; drop `notifications/` | ~2.8K | low |
| Purge MCP/CLI aliases (`fs.*`, `messages.*`, 3 cleanup tools → 1) | ~0.8K | low-medium (skills reference names) |
| Move training stack (`cli/training`, generators, training, knowledge, tokenizer, `agent/`, gateway) to afs-scawful | ~9.5K | medium (afs-scawful imports it back) |
| Move personal/work code (storage_doctor, model_retention, gws, work_*, briefing) to an extension | ~10K | medium |
| Decide v2 layout, then delete the losing side | ~4-6K | high (data layout) |

`afs/__init__.py:44-48` eagerly imports the dead `registry` package; fix first.

## Phase 4 — One runtime SQLite db + coordination primitives (2-3 days)

`.afs/afs.sqlite3` per context root, WAL, `busy_timeout`, `user_version` migrations:

- `events` (replaces history jsonl; retention is a `DELETE`)
- `work_items` (one model for tasks + agent jobs + mission steps; claim is
  `UPDATE ... WHERE status='pending'`, atomic)
- `messages` (one bus; hivemind and messages tools become one)
- `runs` (merges agent_registry + agent_runs + supervisor state; `heartbeat_at`,
  lease expiry, reaper)
- `approvals` (Phase 1 lands here or migrates here)
- `memory_entries` + FTS5
- `claims` (agent, repo, branch, paths, lease) — the missing "who is editing
  what" record that would have flagged the dirty `main` above

Markdown stays as files and the source of truth: scratchpad `state.md`,
handoffs, memory summaries, knowledge docs, job prompts, project TOML.
`context.write` gets compare-and-swap (`expected_sha256`).

## Phase 5 — Embeddings (1-2 days)

1. Tables in the same db: `files`, `chunks` (heading/symbol-aware, ~300-500
   tokens), `chunks_fts` (FTS5), `chunk_vec` (sqlite-vec), `embed_meta`.
2. Local embedder by default (nomic-embed-text v1.5 with task prefixes, or
   Qwen3-Embedding-0.6B via MLX/ollama), batched. Gemini stays opt-in; today
   it is the default in `search_service.py:44` and sends content off-machine.
3. Incremental by sha256; delete orphans.
4. One fusion: keep `rrf_fuse`; retire `mcp_server._fuse_search_results`, the
   legacy JSON index, and `vectors.npy`.
5. Measure recall before and after with the existing harness
   (`embeddings.py:1270-1350`).

## Decisions needed

1. Who owns the 36 dirty files on `main`: rebase or drop.
2. v2 layout: activate `~/.context-v2-candidate-2026-07-20` or abandon v2.
3. Approval store: SQLite now (Phase 1) or JSON now, migrate in Phase 4.
