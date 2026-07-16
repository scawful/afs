# Engagement Primitives

Binary approve/reject gates atrophy human judgment: clicking "yes" is
mechanically identical to thinking. These primitives put friction on the
*decisions* — generation, prediction, rationale — while keeping the mechanics
frictionless for agents and headless callers. They share one trust model:
judgment-bearing state can only enter through the controlling terminal
(`/dev/tty`), which piped stdin cannot satisfy, with the reviewer identified
by the OS user rather than a claimable flag.

## Rationale-Gated Approvals

Both approval surfaces require a `--because` rationale, refused when empty
(exit 2):

```bash
afs approvals approve <agent> <action> --because "diff reviewed, release-gated"
afs work approvals approve <approval_id> --path <ws> --because "content verified"
```

`afs approvals approve` additionally requires re-typing the `agent:action`
pair on the terminal — a headless agent cannot self-approve; the decision is
recorded with the OS user and `reviewed_via: tty`. Rejection is the fail-safe
direction and works headlessly, still with provenance. Rationales appear in
`approvals history` and are resurfaced by calibration review.

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

## Predict-Before-Reveal

```bash
afs session bootstrap --engage --path <ws>
```

Before the queue is revealed, `--engage` asks you to name the top queued
item, then logs prediction vs actual to the calibration trail. Skipping
(empty input, no terminal) is always allowed; `--json` mode never prompts.

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
