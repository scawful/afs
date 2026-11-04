# Agent coordination: observation and operating guidance

Automate accepted bookkeeping while keeping negotiation, debugging hypotheses, peer challenge, replanning, and human choices with the agents and user. A board should be a projection of accepted records, so accepting a handoff does not leave a separate manual board edit to remember.

This report summarizes a completed, selected observation of a Lyra authoring and verification episode. It minimizes private coordination details and contains no raw transcripts or agent identifiers. Findings describe the episode as observed, not the current branch or live team state.

## What the observation established

The ledger contains **97 records with 97 unique event IDs**. Six bounded accepted outcome units were covered: **one accepted at baseline and five newly accepted during observation**. This is selected coverage, not a complete work census or a productivity rate.

| Accepted outcome | Evidence and boundary |
| --- | --- |
| Save and fresh-process reopen | Each named scenario passed 1/1 with exit 0; saved generation, stable identifiers, bindings, originals, and cleanup were checked. |
| Typed-resource planner | 27/27 focused host methods passed. Compiler and runtime refusal paths remained active. |
| Schema verifier and independent inspection | 96 unique map/state rows had zero binding mismatches, and the coordinator accepted the inspection. |
| Immutable saved-generation transfer and input review | 133 read-only snapshot files and 127 input roles were checked. Admission and candidate-ROM flags remained false. |
| Historical fixture repair | 12/12 focused controls passed, followed by exact-head CI. Frozen source hashes and current-tree refusal remained enforced. |
| Alias emitter repair | Five focused tests and whole-source command-line assembly passed, followed by final exact-head CI. Payload placement and authority flags were preserved. |

At the episode's declared stopping point, **2026-10-31 01:53:54 UTC**, committed-source workflows had **1,466 passed, 86 skipped, and zero failures**. A separate uncommitted private measurement fixture still **failed its supplementary metadata stage** and was excluded from that green CI result. Allocation, ordinary all-32-map Build/Test, and Wolf pickup persistence remained open. No source or ROM retries were queued; the episode was explicitly parked for a human metadata decision.

## Coordination worth retaining

One coordinator reused existing owners and released bounded work with distinct writing scopes. Reviewed hashes were frozen, and writers preserved unrelated changes. No writer collision was confirmed in the sampled scopes. Intentional idle was a useful owner state, not evidence that a replacement should be spawned.

Peer challenge corrected a selector assumption: a physical child map and gameplay parent can legitimately differ. Another challenge kept an output-containment guard intact and changed the backend/output layout instead. Deterministic records must preserve the accepted result of these discussions without deciding the discussion itself.

Evidence reuse stayed scoped. A reopen missing its exact child exit remained unqualified despite internal success. The retry reused the same binary and saved files without another Save or build. A repaired test reused **63 unchanged passing cases plus one rerun repaired case**; this was composed evidence, not a fresh exact-head 64/64 run.

Two parallel phases overlapped for approximately **597 seconds and 403 seconds**. These are second-granularity native turn windows, including waits. They demonstrate overlap, not active-work time or measured speedup.

## Failures and limits to preserve

One historical source-binding family produced **two CI failures and four errors**: changed working-tree consumers were checked against frozen historical hashes before their intended assertions. The affected offline stage lasted **1,110.157 seconds**. That is a stage duration, not active agent effort, wasted cost, or measured avoidable time. The repair staged the exact historical bytes and retained a control refusing current-tree substitution.

Seven private measurement attempts reached different stages: synthetic pointer validation, alias syntax, assembler-version correspondence, library loadability, output containment, compiled-consumer profile, and supplementary manifest generation. They are not seven equivalent compile failures. Keep each attempt's inputs, stage, reason, component results, and overall result.

The final measurement retained useful compiled checks and zero E3 intersections, but its overall receipt remained failed. The manifest scanner could not resolve macro `org <site>`, and the hook scanner omitted **20 macro-generated callsites**. Passing component checks did not complete metadata acceptance or grant allocation. The human choice among deferral, literal hook expansion, and invocation-aware scanning remained unresolved; this bookkeeping work does not select one.

Early board sections lagged newer receipts and later needed manual reconciliation. Cursor resets also replayed completed messages. Stable event IDs should deduplicate those deliveries; transport replay does not establish new work or compaction.

## Operating contract for agents

1. Propose bounded work with its existing task and handoff identifiers, intended owner, writing scope, dependencies, input roles, and explicit acceptance conditions. A proposal is not accepted ownership.
2. Record recipient acceptance before treating a transfer as complete. Retain ownership until a valid transfer or release is recorded. Never infer consent from silence, a name, an embedding, or an elapsed timeout.
3. Keep one writer per accepted scope, following existing lease and conditional-write rules. A handoff does not expand execution, messaging, or takeover authority.
4. Bind evidence to immutable source, input, tool, configuration, and relevant environment fingerprints. Record scope and completeness, including missing predicates. A present tool is not proof that it loads or supports the full workload.
5. Propose completion against the agreed conditions. Preserve failed and partial checks; acceptance requires the relevant predicates and authorized acceptance, not conversation sentiment or test counts.
6. Version accepted agreements and record supersession and release. Changed known inputs invalidate only affected evidence and dependent outcomes. Unknown or volatile inputs cannot be claimed cache-valid.
7. Use the derived board and durable receipts at handoff boundaries. Keep transport cursors separate from stable event IDs. Replayed events must not duplicate ownership or completion transitions.
8. Leave hypotheses, negotiation, design choices, permission expansion, and unresolved gameplay strategy to agents and humans. Embeddings retrieve linked context; they never decide authority or status.

## Implementation contract

The implementation is ready for review in an isolated AFS worktree, based on source commit `3affd6291b9a9cea9ff6fef3df75424521ef8f2d`. Live adoption is explicitly deferred; active boards and owners were not changed. No commit, push, PR, deployment, or gameplay change was made.

Supported commands in that checkout:

```sh
afs handoff agree --path PROJECT --event-file event.json
afs handoff agreement TASK_ID --path PROJECT --revision 1
afs board accepted --path PROJECT --json
```

An agreement event names its stable event ID, existing task ID, expected task version, explicit operation, and actor. The agreed reviewer proposes bounded predicates, immutable input roles, intended owner, canonical writing scopes, optional existing handoff revision, and pinned task dependencies. The recipient explicitly accepts. The same transition returns its newly derived board. Evidence records exact bindings and a durable receipt reference; the owner proposes completion and the reviewer confirms only after all agreed predicates and dependencies are valid. The CLI inspection command exposes the exact bindings required for a receipt.

The existing SQLite work-assistant activity table stores each explicit event and resulting task state. Immediate transactions, stable event deduplication, expected versions, scope-conflict checks, and replay replace a separately editable projection. Insert failure leaves the previous board intact; identical stable-ID retries recover uncertain responses without another transition. No new database or identity registry was introduced.

Task-queue writers and agreement events share the existing AFS directory writer lock. Opt-in requires an existing unassigned pending task; claimed legacy ownership cannot be taken over. Once opted in, legacy claim/status/completion writes refuse loudly until owners release, proposals are resolved, and the reviewer explicitly retires the agreement. V1 requires persisted context mount routing; v2 uses registered project scopes. Retirement invalidates dependent outcomes before legacy writing resumes.

Known input changes invalidate only predicates bound to those roles and transitive declared dependent tasks. Tool and environment fingerprints bind every predicate. Unknown/null and declared volatile inputs cannot count as valid reusable evidence. A released confirmed result may still receive scoped invalidation without reclaiming its writing scope. All failed and superseded receipts remain recorded. Dependencies are task-level and must be declared; source dependencies are not discovered automatically.

Existing machine-wide leases and conditional file-write rules still apply. The cooperative scope guarantee covers accepted agreements in the same context database; it does not reserve OS resources, reconcile undisclosed legacy scopes, authenticate actor labels, or coordinate separate databases. Hosts remain responsible for caller identity and execution permissions. The code does not run tests or model calls on behalf of an agreement.

Verification completed: **95 focused checks passed**, including 25 new agreement checks and legacy task, board, handoff, scope, SQLite, and lease compatibility. Coverage includes duplicate/reordered events, conflicting/concurrent accepts, owner transfer, scoped and transitive invalidation, unknown/volatile inputs, insert failure and recovery, v2 isolation, custom persisted routing, legacy-claim/opt-in races, retirement, and real CLI entry points. Repository-wide Ruff lint passed. Whole-package mypy passed with no issues in 363 source files; the MCP task compatibility check also passed (one selected test). No full repository suite or live-team adoption was claimed.

The detailed event examples, acceptance contract, limitations, migration, and rollback are in `docs/ACCEPTED_AGREEMENTS.md`; this report is also saved as `docs/AGENT_COORDINATION_REPORT.md`. The worktree is retained with uncommitted reviewable changes. For rollback after a pilot, release owners, resolve proposals, retire managed tasks, and preserve the database/receipts before returning to an older runtime. Older binaries do not enforce the opt-in task guard, so do not downgrade an active managed context silently.

## Measurement limits and source evidence

Prompt, completion, cache, provider, harness, and observer token totals were unavailable. Actual model, reasoning effort, service tier, compaction events, dollars, quota, and active-work time were also unavailable. Do not infer them from names, text, elapsed windows, or handoffs. No quota-efficiency or speedup conclusion is supported.

The observer read supported status deltas, receipts, and acceptance records; it did not independently rerun the team tests or decode all raw evidence. Observation append order is not a complete occurrence-time timeline. No continuing monitor is implied by this report.

Task-owned supporting sources are `observation/report.md`, `observation/events.jsonl`, `observation/metrics.json`, `observation/measurement-attempts.md`, and `observation/cursor.json`. Consult those local records when fuller evidence is needed; do not copy the long ledger or private transcripts into general agent instructions.
