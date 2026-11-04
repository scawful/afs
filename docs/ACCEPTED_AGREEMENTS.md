# Accepted task agreements (opt-in)

Accept a bounded handoff and get its updated board in the same operation. This
slice extends the existing task queue, handoff CLI, and SQLite work-assistant
activity log. It does not activate itself or rewrite `agent_board.md`.

Read [the observation report](AGENT_COORDINATION_REPORT.md) for why the accepted
bookkeeping is deterministic while negotiation, peer challenge, replanning,
debugging hypotheses, gameplay decisions, and permission choices remain with
agents and humans.

## Supported entry points

```sh
afs handoff agree --path PROJECT --event-file event.json
afs handoff agreement TASK_ID --path PROJECT
afs board accepted --path PROJECT --json
```

`handoff agree` accepts an explicit JSON event and always returns JSON containing
`applied`, the event's resulting task `version`, and the current derived `board`.
`handoff agreement` returns the current version, latest proposed agreement,
active revision, stored receipts, and exact `evidence_bindings` per predicate.
Use `--revision N` to inspect an older revision or the still-active agreement
while a successor proposal is pending. `board accepted` separates accepted
records from pending proposals. These surfaces share the usual `--config`,
`--context-root`, `--context-dir`, and `--path` resolution; there is no arbitrary
scope override. Registered v2 projects see only their own agreements.

Python consumers use `afs.commitments.CommitmentStore(context_root,
requester_path=project)` and its `apply`, `inspect`, and `board` methods.

## An explicit handoff

Start with an existing, unassigned, pending `TaskQueue` task created through the
supported task API/MCP. A v2 task must carry `context.scope_id` matching its
registered project scope. V1 contexts must already have persisted `global` and
`items` directory routes in context metadata (use the existing context initializer);
configuration-only custom routes fail closed so queue guards cannot inspect a
different database. This first slice does not migrate claimed or completed
legacy work: release/reconcile that work through its owner and use a new bounded
pending task. Existing task, session, run, and agent IDs remain the identifiers;
there is no additional agent registry.

The agreed reviewer proposes `offer.json` (replace the sample identifiers and
fingerprints with real ones):

```json
{
  "event_id": "offer-001",
  "task_id": "EXISTING_TASK_ID",
  "expected_version": 0,
  "op": "propose",
  "actor": "coordinator",
  "owner": "worker",
  "handoff_revision": "EXISTING_HANDOFF_REVISION",
  "writer_scopes": ["/absolute/owned/worktree/src/component"],
  "inputs": {
    "source": "sha256:immutable-reviewed-inputs",
    "tool": "sha256:resolved-tool-and-configuration",
    "environment": "sha256:relevant-known-environment"
  },
  "conditions": {"reopen-child-exit": ["source"]},
  "dependencies": {},
  "volatile": []
}
```

`handoff_revision` is optional; when present it must resolve to an existing
immutable revision in this scope. A handoff acknowledgement alone never accepts
ownership or completion. Conditions name the agreed predicates, not a test-count
threshold; put their full meaning and receipt scope in the linked handoff.

Run `afs handoff agree --event-file offer.json --path PROJECT`. The proposed
recipient then submits `accept.json`:

```json
{
  "event_id": "accept-001",
  "task_id": "EXISTING_TASK_ID",
  "expected_version": 1,
  "revision": 1,
  "op": "accept",
  "actor": "worker"
}
```

Acceptance returns the updated board. No second board-write command is needed.
The SQLite transaction checks overlapping absolute writer paths, including
ancestor/descendant scopes across projects in this database. Even two tasks
owned by the same agent cannot hold overlapping scopes. Existing machine-wide
resource rules remain in force: use `afs lease run` for the heavy lane and
`afs fs write --if-match` for shared files. Commitments are cooperative ownership
records; they neither acquire OS locks on source files nor coordinate across
separate context databases. Reconcile outstanding legacy/manual scopes before
opting a team into these guarantees.

For evidence, inspect the agreement and copy its exact bindings. Submit an
`evidence` event with `condition`, `bindings`, boolean `passed`, and a durable
`reference`. For example, at version 2:

```json
{
  "event_id": "evidence-001",
  "task_id": "EXISTING_TASK_ID",
  "expected_version": 2,
  "revision": 1,
  "op": "evidence",
  "actor": "worker",
  "condition": "reopen-child-exit",
  "bindings": {
    "source": "sha256:immutable-reviewed-inputs",
    "tool": "sha256:resolved-tool-and-configuration",
    "environment": "sha256:relevant-known-environment"
  },
  "passed": true,
  "reference": "receipt:immutable-run/child-exit"
}
```

Only the agreed owner records evidence. Recording a receipt is a report, not an
independent execution or verification of its contents. The reviewer evaluates
that receipt against the agreed predicate. `complete` by the owner proposes
completion; a subsequent `confirm` by the agreed reviewer accepts it. Both
require all predicates to have current passing evidence and all pinned
dependencies to be confirmed with valid evidence. They cannot be inferred from
conversation, an embedding, elapsed time, or successful component checks.

## Revisions, invalidation, and release

Every event includes `event_id`, `task_id`, `actor`, `op`, and an integer
`expected_version`. After the offer, include the integer `revision` too. Each
successful event advances the task version; stale or reordered requests refuse
without an event. Replaying an identical stable ID is harmless; reusing its ID
with different content refuses. A transport cursor is not an event ID.

| Operation | Required actor and effect |
| --- | --- |
| `propose` | Reviewer offers revision 1, or a successor after resolving the preceding proposal. Existing accepted ownership remains until the successor is accepted. |
| `accept` | Proposed owner accepts. A different previous owner must explicitly release first. Same-owner acceptance can supersede that owner's previous revision. |
| `withdraw` | Reviewer withdraws a pending proposal; it creates no ownership. |
| `inputs` | Active agreement's owner or reviewer supplies an `inputs` map changing known roles. This can invalidate released evidence without reacquiring ownership. |
| `evidence` | Owner records the predicate, bindings, passing/failing result, and receipt reference. Failures and superseded receipts remain in history. The latest receipt for a predicate controls its current validity. |
| `complete` | Owner proposes completion after every agreed predicate and dependency is valid. |
| `confirm` | Reviewer accepts a proposed completion after rechecking those conditions. |
| `release` | Owner supplies `reason` and releases writing scope. A confirmed outcome remains usable by dependents until its evidence changes. |
| `retire` | Latest reviewer explicitly returns the task to legacy queue control after proposals are resolved and all writing scopes are released. |

`inputs` updates invalidate receipts that bind the changed roles. Every predicate
also binds `tool` and `environment`. A null fingerprint or a role named in
`volatile` is never reported valid or reusable, even if a passing receipt was
recorded. Capture a known reproducible fingerprint, or revise the agreement;
this slice has no volatile-evidence acceptance exception.

Dependencies map existing task IDs to accepted revision integers. Exact computed
`dependency:ID` bindings are exposed by inspection. A changed input or accepted
revision invalidates only that task's affected predicates and its transitive
dependent tasks. Independent task evidence remains valid. Dependencies are
currently task-level: all predicates in a dependent agreement bind every
explicit dependency. Split bounded tasks when finer reuse boundaries matter.
No source dependency is discovered automatically; consumers must declare it.
Cycles and nonexistent revision pins refuse.

A board can show `needs_review` after a previously confirmed result loses valid
evidence. It retains the original receipts and accepted owner; invalidation does
not invent a new owner, restore released scopes, or authorize an automatic retry.

## Persistence, compatibility, and rollback

Records live in the existing `work_assistant.sqlite3` `activity` table, with
`activity_type=commitment.v1`, a scope-specific target system, stable event key,
explicit request, and resulting task state. SQLite `rowid` gives append order;
wall clocks and observation order never resolve ownership. An immediate
transaction validates and appends one transition and generates its board. Reads
replay the stored snapshots. There is no separately mutable projection to repair.
A failed insert leaves the prior board intact; after an uncertain response,
resubmit the same event ID and identical request before attempting another.

Task-queue mutations and agreement transitions use the existing AFS directory
writer lock in the same order, then SQLite. Once a task opts in, legacy
`TaskQueue.claim`, `update_status`, and `complete` refuse loudly and direct the
caller to `handoff agree`. Unmanaged tasks retain legacy behavior and do not
create an agreement database. Legacy task status is retained as the pre-opt-in
snapshot; consumers of opted-in work use the accepted board. A queue listing is
not evidence that managed work is unowned.

CLI actor labels and registry rows are not identity authentication. An authorized
local host must bind actor labels to its own caller identity and permissions.
This protocol grants no execution, messaging, allocation, or approval authority.
It adds no network, model calls, background watcher, test execution, or global
configuration. Keep admission, allocation, review, and execution permission as
separate predicates when they matter.

Adoption is explicitly deferred: this implementation is retained for review in an
isolated worktree. Do not switch active teams or boards automatically. A pilot
requires consumers to invoke this checkout's CLI and opt individual new tasks in.
For rollback with that code still available, release owners, resolve proposals,
and `retire` opted-in tasks first. Retirement invalidates dependent outcomes before
legacy writers can change their prerequisites. Preserve the database and receipts; do not
remove event rows. Then use the prior runtime. Older binaries do not enforce the
opt-in queue guard, so never downgrade an active opted-in context silently.
The additive activity records require no destructive schema migration.
