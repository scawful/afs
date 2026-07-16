# Event Reactor

The write-only history log and hivemind bus become actionable through
`on_event` agent start conditions: the supervisor matches new entries against
per-agent patterns each reconcile cycle and starts the agent (or enqueues a
job) when one fires.

## Configuration

```toml
[[profiles.work.agent_configs]]
name = "index-rebuild"
module = "afs.agents.index_rebuild"
on_event = ["hivemind:context:repair", "error:index_*"]
on_event_action = "spawn"   # or "job"
event_debounce = "5m"
```

Pattern grammar is `"<kind>"` or `"<kind>:<detail>"`; both sides are fnmatch
globs (`"fs:*"`, `"hivemind:context:*"`). Only the first colon splits kind
from detail, so `hivemind:context:repair` matches topic `context:repair`.
Grammar caveats: a leading colon (`":repair"`) leaves the kind side empty and
therefore widens it to `*` (any kind); matching is case-sensitive
(`error:*` does not match type `Error`); source records with an empty kind are
malformed and never reach matching.

Two sources feed the reactor:

- History events (`history/events_YYYYMMDD.jsonl`): kind is the event `type`,
  detail is the `op`.
- Hivemind messages: kind is `hivemind`, detail is the topic. The bus is the
  canonical source: history records of type `hivemind` (the send-op mirror of
  each bus message) are excluded so one message never yields two events.

The shipped default `index-rebuild` agent consumes the `context:repair` topic
that `afs watch` publishes, so watch-driven repair loops close out of the box.

## Delivery Semantics

Delivery is transactional, at-least-once:

- Each cycle reads events oldest-first under an exclusive lock, routes
  matches, and atomically commits source checkpoints plus a coalesced
  `pending_routes` outbox (at most one route per agent). Successful or
  intentionally coalesced routes leave the outbox; failed dispatches and
  retryable gates stay parked while unrelated source backlog advances.
  `reactor_dispatch_failures` counts those parked attempts. A crash or failed
  commit loses neither side: source matches reconstruct routes when offsets
  did not advance, while already parked routes retry without a new event.
- Job delivery is acknowledged through a durable receipt handshake. The
  markdown item is written through a flushed same-directory temporary file,
  atomically published with a POSIX directory sync or Windows write-through
  rename, and recorded in the reactor outbox. Only a later cycle that reloads
  the committed receipt may confirm the live job record and consume the route.
  A missing record is recreated; a read or sync failure leaves the receipt
  parked. Active jobs adopted through the dedupe key use the same handshake.
  Windows never adopts an unmarked pre-v5 job because its original plain
  publication cannot be proven durable; a duplicate reaction is safer than a
  lost one. Job state moves publish the new state durably before hiding the old
  state behind an ignored tombstone.
- Complete version-5 records are delivered when durably visible, regardless
  of caller timestamps. History byte positions and exact hivemind identities
  make future/skewed timestamps unnecessary as a delivery gate; timestamp
  watermarks remain compatibility and diagnostic metadata. Hivemind's built-in
  writer publishes with a same-directory atomic temp-file rename; external
  writers that expose partial JSON are retried.
- The lock covers the whole read → dispatch → ack window, so two supervisors
  sharing a state directory never double-deliver a batch. A contended lock
  defers events one cycle (reported as `reactor_busy` in supervisor
  metrics). The lock is advisory and host-local: supervisors pointed at
  different `AFS_AGENT_STATE_DIR`s, or hosts sharing a synced filesystem,
  are outside its reach.
- Reads retain at most 500 candidate payloads per source per cycle. History
  JSONL is streamed from durable byte offsets with a 1 MiB target scan budget
  and a 256 KiB per-record ceiling. A valid record already started may finish
  past the target budget (up to that ceiling), preventing a large line from
  livelocking at the same offset. Newline-complete history records above the
  ceiling are scanned with bounded memory across cycles, warned, counted, and
  skipped so they cannot brick later source work. Hivemind payloads have the
  same ceiling and use the stable-malformed quarantine.
  Source discovery still enumerates the extant history log filenames and all
  final hivemind `*.json` metadata each cycle. Those filesystem walks (and the
  hivemind identity map) are proportional to live files, not bounded to 500.
  The exact hivemind inventory is required to find a newly copied file even
  when its filename and mtime are older than every previously delivered file.
  Finite, exact discovery epochs plus an independent round-robin retry cursor
  reserve capacity for new identities and failed reads, so multiple unreadable
  oldest files cannot starve later candidates, copied/backdated ingress cannot
  hide an existing identity, and sustained ingress cannot starve retries.
  History uses finite file rounds for the same reason. Partial tails remain at
  their original byte offset while later independent logs still get a turn,
  even when repeated partial reads consume the cycle byte budget.
  Payload contents remain bounded and larger backlogs resume across cycles.
  History checkpoints are append-position based, so a newly appended event is
  delivered even if its caller supplied a backdated timestamp. Fresh-state
  prime snapshots round history offsets down to the last complete newline, so
  an in-flight partial JSONL append is never checkpointed or skipped.
- Timestamp-only v1/v2 state migrates through a version-5 shell persisted
  before its first dispatch with empty history and hivemind checkpoints.
  Timestamp cursors cannot prove whether a backdated record landed before or
  after the legacy state, and filesystem mtimes cannot safely supply the
  missing per-record arrival time, so extant source content is conservatively
  replayed once rather than silently consumed. Version-3 upgrades preserve its
  positional history offsets but likewise replay the extant hivemind inventory
  because tuple checkpoints cannot prove file identity. Strict bounds resume
  either replay across cycles, and an unacked first batch reuses the already
  persisted shell. One caveat: hivemind messages carry an optional TTL enforced
  at read time, so a message that expires while waiting out a long drain or an
  unacked crash window is gone when its redelivery cycle arrives.
- The version-5 cursor state (`supervisor/event_reactor/cursor.json`) keeps
  independent per-source watermarks, history offsets, exact identities for
  extant hivemind files, bounded scan progress, and pending routes. It is
  primed to "now" only for
  a genuinely new state, so enabling the reactor never replays historical
  events as a spawn storm. Once initialized, a missing, unreadable, malformed,
  or partial cursor fails closed with `reactor_state_error`; repair is explicit
  and no backlog is silently skipped. Only explicit v1/v2 state may enter the
  timestamp-to-positional replay upgrade; version-3 state may enter the one-way
  exact-identity upgrade. Existing version-4 state upgrades in place to the
  receipt-capable version-5 format; its marker makes a rollback reader fail
  closed instead of silently ignoring v5 receipt fields.
  Replacement data is fsynced before rename; publication uses a parent
  directory fsync on POSIX and a write-through move on Windows. The complete
  state hierarchy is established from an existing common ancestor before the
  reactor lock is opened. If a POSIX cursor rename is already visible but its
  directory sync reports an error, that atomic replacement is treated as
  committed rather than falsely reporting unchanged cursors; the earlier
  durable marker still makes a cursor lost after host failure fail closed
  instead of silently priming as fresh state.
  Unsupported versions, incomplete current positional maps or migration
  pairs, and history offsets that are not newline boundaries are rejected. An
  adjacent `initialized` marker distinguishes first use from accidental cursor
  deletion. The marker is committed before cursor offsets, so failure to
  persist it cannot consume a batch. Marker and legacy checkpoint integers use
  bounded ASCII parsing. Cursor and source JSON reject duplicate members,
  non-finite numbers, overlong integers, invalid UTF-8, and excessive nesting;
  parse failures normalize to `ReactorStateError` or the malformed-record
  policy instead of escaping the reconcile loop. State timestamps must be
  strings that safely normalize to UTC, and source filenames used as durable
  checkpoint keys must be UTF-8; unrepresentable POSIX names fail closed with
  state unchanged until they are renamed.
- If a history mount becomes unavailable after positional checkpoints exist,
  the cycle fails closed with state unchanged rather than pruning offsets and
  replaying a restored mount from zero. An available empty history directory
  remains a normal inventory.
- Complete malformed history records are skipped and counted. A malformed
  hivemind identity (including an invalid provided expiry) is retried for one
  acked cycle in case an external writer was still copying it; only the same
  unchanged identity on the next cycle is classified as durable, warned,
  skipped, and counted
  (`reactor_skipped_malformed`). Transient open/stat failures never mark a file
  seen. History payload read/seek failures and lock-setup failures normalize to
  `ReactorStateError`, leaving the previous durable state intact.

## Actions and Gates

- `on_event_action = "spawn"` (default) routes through the normal reconcile
  path; `"job"` enqueues a background agent-job instead, deduped per agent
  while one is queued or running. Both actions pass the same gates: circuit
  breaker, manual stop, and dependency checks — `job` is a delivery mode, not
  an authorization bypass.
- Manual stops, open circuits, agents awaiting review, missing modules, and
  unmet dependency or mutex gates are retryable: the route stays in the
  durable outbox until the gate clears. Recovery edits to modules,
  dependencies, or debounce do not erase it; removing the config or changing
  its trigger authorization/action terminally rejects the stale route. A currently running agent, an
  active deduped job, or the configured debounce window intentionally
  coalesces the trigger and permits the cursor to advance.
- A process-launch failure records against the supervisor's existing restart
  budget. Event routes remain parked during exponential backoff, when restart
  is disabled, and while the circuit is open; cooldown re-enables a launch
  attempt. Loaded `max_restarts` values and persisted launch-failure counts are
  bounded at 64 so corrupt state cannot drive unbounded backoff arithmetic.
  The shipped `index-rebuild` reaction opts into a bounded failure budget
  (`max_restarts = 3`). Only a process that actually starts advances event
  debounce state.
- Any other `on_event_action` value fails closed: the trigger is skipped with
  a warning and recorded as a terminal rejection; nothing spawns or enqueues.
- Job prompts contain operator config plus an opaque route audit ID. Raw event
  kind, detail, source, and payload text never reach the prompt or title; the
  title separately carries the bounded hexadecimal source fingerprint for
  post-hoc correlation.
- Debounce (`event_debounce`, schedule grammar, default `5m`) is keyed off
  actual dispatches and persisted at ack, so it covers job actions and
  survives supervisor restarts. In the crash window before an ack, the job
  dedupe key and the started agent's own state are what suppress duplicates
  — at-least-once delivery means a redelivered batch can enqueue a second
  job if the first already completed. Failed starts never provide the fallback
  debounce clock, and future clock values do not coalesce a route indefinitely.

## Threat Model

The reactor operates inside a **cooperative same-user boundary**, not an
operating-system security boundary. A process running as the same account can
write event logs, hivemind messages, agent-job files, or reactor state. Within
that boundary:

- Events can trigger only the actions authorized by human-authored `on_event`
  configuration, and those actions still pass the supervisor's circuit,
  manual-stop, dependency, mutex, and debounce gates. These gates constrain
  the configured action space; they do not authenticate the event writer.
- Event `source` is informational and writer-supplied. Raw kind, detail, and
  source text never reaches launch reasons, job titles, or job prompts.
  Persisted reasons carry separate bounded route and source fingerprints;
  titles expose the source fingerprint, while prompts use an opaque audit ID.
  Matching uses only `kind:detail`, and rules must not treat any fingerprint
  as proof of identity or authority.
- Job dedupe keys (`on_event:<agent>`) are predictable. An active job with the
  same key intentionally coalesces a reaction, whether it is the expected job
  or another same-user writer's collision. The supervisor exposes those
  suppressions as `reactor_jobs_coalesced` so they are observable. The metric
  is a per-cycle count of confirmed coalesce attempts, not a unique or
  cumulative event counter; a later cursor-ack failure can make a retry count
  the same suppression again.
- Positional history offsets and exact hivemind file identities govern
  delivery. Writer-supplied timestamps are diagnostic and compatibility data,
  not authorization or a delivery watermark for complete version-5 records.
- Cursor integrity is fail-closed, not tamper-proof. Once the atomic
  `initialized` marker exists, missing, unreadable, malformed, or incomplete
  state reports `reactor_state_error` and leaves source progress unchanged
  until an operator repairs it.
