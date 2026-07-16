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
- Hivemind messages: kind is `hivemind`, detail is the topic.

The shipped default `index-rebuild` agent consumes the `context:repair` topic
that `afs watch` publishes, so watch-driven repair loops close out of the box.

## Delivery Semantics

Delivery is transactional, at-least-once:

- Each cycle reads events oldest-first under an exclusive per-context lock,
  dispatches matches, and only then advances the cursors. A crash between
  read and dispatch redelivers the batch next cycle; it never loses events.
- The lock covers the whole read → dispatch → ack window, so two supervisors
  reconciling the same context never double-deliver a batch. A contended lock
  defers events one cycle (reported as `reactor_busy` in supervisor metrics).
- Reads are bounded to 500 events per source per cycle. A larger backlog
  drains across cycles in order — never dropped, never read unbounded.
  History scans skip daily files older than the cursor date.
- The cursor state (`supervisor/event_reactor_cursor.json`) keeps independent
  per-source cursors and is primed to "now" on first run, so enabling the
  reactor never replays historical events as a spawn storm.
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
- Debounce (`event_debounce`, schedule grammar, default `5m`) is persisted
  per agent and keyed off actual dispatches, so it covers job actions and
  survives supervisor restarts.
