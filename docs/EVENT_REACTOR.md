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

- Each cycle reads events oldest-first under an exclusive lock, dispatches
  matches, and only then advances the cursors (the commit is an atomic
  temp-file rename). A crash anywhere before ack redelivers the batch next
  cycle; a failed dispatch (spawn or enqueue) defers the ack itself
  (`reactor_dispatch_failures`), so an event is never consumed with zero
  deliveries. If the commit cannot be persisted the cycle reports
  `reactor_state_error` and the cursors stay put — again redelivery, not
  loss.
- Events stamped within the last ~5 seconds are deferred one cycle: writers
  stamp before their write lands, so consuming right up to "now" could move
  the watermark past an in-flight write. The residual assumption is that a
  writer lands within that grace window of its stamp.
- The lock covers the whole read → dispatch → ack window, so two supervisors
  sharing a state directory never double-deliver a batch. A contended lock
  defers events one cycle (reported as `reactor_busy` in supervisor
  metrics). The lock is advisory and host-local: supervisors pointed at
  different `AFS_AGENT_STATE_DIR`s, or hosts sharing a synced filesystem,
  are outside its reach.
- Reads are bounded to 500 events per source per cycle. A larger backlog
  drains across cycles in order — never dropped, never read unbounded.
  History scans skip daily files more than one day older than the cursor
  date (the day of grace absorbs writer-local filename skew). One caveat:
  hivemind messages carry an optional TTL enforced at read time, so a
  message that expires while waiting out a long drain or an unacked crash
  window is gone when its redelivery cycle arrives.
- The cursor state (`supervisor/event_reactor/cursor.json`) keeps
  independent per-source cursors and is primed to "now" only for a genuinely
  new state, so enabling the reactor never replays historical events as a
  spawn storm. Once initialized, a missing, unreadable, malformed, or partial
  cursor fails closed with `reactor_state_error`; repair is explicit and no
  backlog is silently skipped. An adjacent `initialized` marker distinguishes
  first use from accidental cursor deletion.
- Malformed log or hivemind records are skipped and counted
  (`reactor_skipped_malformed`), never crash ingestion.

## Actions and Gates

- `on_event_action = "spawn"` (default) routes through the normal reconcile
  path; `"job"` enqueues a background agent-job instead, deduped per agent
  while one is queued or running. Both actions pass the same gates: circuit
  breaker, manual stop, and dependency checks — `job` is a delivery mode, not
  an authorization bypass.
- Any other `on_event_action` value fails closed: the trigger is skipped with
  a warning, nothing spawns.
- Job prompts are built from operator config plus a sanitized event label;
  event payload text never reaches a job prompt.
- Debounce (`event_debounce`, schedule grammar, default `5m`) is keyed off
  actual dispatches and persisted at ack, so it covers job actions and
  survives supervisor restarts. In the crash window before an ack, the job
  dedupe key and the started agent's own state are what suppress duplicates
  — at-least-once delivery means a redelivered batch can enqueue a second
  job if the first already completed.
