# Engagement Primitives

Binary approve/reject gates atrophy human judgment: clicking "yes" is
mechanically identical to thinking. These primitives put friction on the
*decisions* — generation, prediction, rationale — while keeping the mechanics
frictionless for agents and headless callers. They share one trust model:
judgment-bearing state enters a store only with a decision-scoped capability
minted by the human-decision broker after input on the controlling terminal.
POSIX uses the process controlling terminal (`os.ctermid()`); Windows uses
`CONIN$`/`CONOUT$`. Piped stdin cannot satisfy the broker. Reviewer identity is
derived from the process UID on POSIX or token SID on Windows rather than
`USER`, `LOGNAME`, or a claimable flag; when neither is available the record
is explicitly labeled `unauthenticated` and the broker refuses to mint human
authorization.

This is a **cooperative same-user boundary, not an OS security boundary**. A
hostile process already running as the same account can open the same terminal,
import private Python internals, or edit AFS state files directly. The broker
prevents ordinary public-API callers from forging provenance and prevents
accidental trust escalation; isolating hostile code requires an external
sandbox or separate OS account. Capabilities are scoped to the exact store,
decision, and record/content (including rationale or score note), and are
consumed once, so one confirmation cannot be replayed or relabeled as a
different judgment.

## Rationale-Gated Approvals

Both approval surfaces require a `--because` rationale, refused when empty
(exit 2):

```bash
afs approvals approve <agent> <action> --because "diff reviewed, release-gated"
afs work approvals approve <approval_id> --path <ws> --because "content verified"
```

Both CLI surfaces require re-typing the displayed decision token on the
terminal (`agent:action` for the agent gate, approval id for work approvals) —
a headless agent cannot label its own decision human-reviewed. The decision
is recorded with UID/SID provenance and `reviewed_via: controlling_terminal`;
the confirmation prompt shows, and the capability binds, the exact rationale.
Programmatic rejection remains available as a fail-safe denial, but it is
explicitly unauthenticated and is not a human calibration judgment.
Rationales appear in `approvals history` and are resurfaced by calibration
review.

`afs approvals clear` compacts the active queue but first appends every
completed record to an immutable JSONL history beside the queue. Approved and
rejected rationales, provenance, calibration refs, and decision timestamps
therefore survive clearing.

Every gate request carries a unique `request_id` (`gate_...`) used as its
calibration ref; legacy records are backfilled with stable ids on first load.

## Mission Acceptance

Missions carry a human-authored `acceptance` field — what does done look
like — prompted for at `mission create` on a terminal and settable with
`--acceptance`. Setting, changing, or clearing it requires a typed terminal
confirmation; a headless caller passing `--acceptance` is refused (exit 2).
Provenance (`acceptance_set_by`, `acceptance_set_at`) is stored with a log
entry. Agents must leave it unset and surface the follow-up nudge instead:
acceptance is the calibration anchor the outcome is later scored against, so
an agent authoring it would be grading its own homework.

The public `MissionStore` compatibility API cannot turn a caller string into
that anchor. Acceptance text supplied without a broker capability is stored as
`acceptance_suggestion`, with `acceptance` left empty; legacy unconfirmed
records fail closed into the same suggestion field when loaded.

## Calibration Review

```bash
afs calibration review --path <ws> [--days 7] [--markdown]
afs calibration score <ref> --outcome hit|miss|unclear [--note "..."]
```

`review` resurfaces the window's approval decisions (with their rationales),
closed missions (next to their acceptance), and predictions, each with a
ready-to-paste score command. `--markdown` emits a digest section for a
weekly review document. `score` records outcomes to an append-only JSONL
trail under `scratchpad/calibration/` and rejects refs that no known store
contains — a typo'd ref cannot silently poison the trail.

The score is itself a judgment: recording one requires re-typing the outcome
on the terminal (headless callers are refused, exit 2), and each entry
carries `scored_by`/`scored_via` provenance. Agent-gate decisions (`gate_…`
refs) are global rather than per-context, so their outcomes land in a global
`approval_outcomes.jsonl` next to the gate store — a decision scored in one
context never resurfaces as unscored in another.

The public programmatic outcome API remains useful for annotations and tests,
but stamps `human_confirmed: false`, `scored_via: programmatic`, and
`scored_by: unauthenticated`; those entries never satisfy or hide the human
calibration score requested by `review`.

## Predict-Before-Reveal

```bash
afs session bootstrap --engage --path <ws>
```

Before the queue is revealed, `--engage` asks you to name the top queued
item, then logs prediction vs actual to the calibration trail. Skipping
(empty input, no controlling terminal) is always allowed; `--json` mode never
prompts. The answer is collected by the same broker and carries UID/SID
provenance. Direct `record_prediction(...)` calls are labeled programmatic
and excluded from calibration review and match-rate aggregates.

## Skeleton-First Planning

The `implementation-plan` schema has a `human_intent` section (goal,
non-goals, done-when) that agents must never write, fill, or edit:

```bash
afs schema validate --schema implementation-plan --file plan.json \
  --skeleton human_plan.json
```

Validation fails when the expansion modified, removed, or fabricated
`human_intent`. The skeleton is parsed strictly (no lenient fence coercion),
its `human_intent` must satisfy the schema contract (a malformed anchor is an
error, not "absent"), and comparison is canonical-JSON so Python equality
quirks (`True == 1`) cannot smuggle edits through.
When `human_intent` is present it must contain at least one declared intent
field; an empty object is rejected rather than accepted as a vacuous trust
anchor.

## Store API Boundary

`ApprovalGate`, `WorkAssistantStore`, `MissionStore`, and calibration expose
separate authoritative methods that require a broker-minted capability.
Legacy methods such as `approve(...)`, `reject(...)`, acceptance arguments
without a capability, `record_prediction(...)`, and `record_outcome(...)` are
intentionally non-authoritative. They may retain compatibility data, but it is labeled
`programmatic`/`unauthenticated`, cannot authorize guarded execution, and is
excluded from human calibration.
