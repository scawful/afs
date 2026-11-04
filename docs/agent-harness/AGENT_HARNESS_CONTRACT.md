# Agent Harness Contract

Active 2026-10-23. One shared contract; project files add local constraints.

## Act and finish
1. Infer the goal, constraints and done condition from the request and current handoff.
   Act within existing authorization; resolve routine choices without another confirmation.
   Ask only for a missing decision, consequential ambiguity or an explicit approval gate.
2. Own one primary outcome and at most one bounded support task. A coordinator owns
   order and dependencies; implementation and adoption keep named owners.
   Replace, defer or delegate an existing obligation before accepting another.
3. Read task-relevant code/docs and prefer the smallest useful change through existing tools.
   Preserve unrelated dirty work and user-owned processes. Use isolated code worktrees.
4. When blocked, name the blocker, resolver/event and next action; do useful independent
   work within scope. If none remains, wait for that event instead of inventing busywork.
   Retry only after a changed input, implementation or hypothesis.
5. Run the narrowest meaningful verification. Reuse accepted evidence for unchanged inputs;
   repeat or broaden checks only for a new change, failure, integration boundary or concrete risk.
   Report observed results and the one material remaining limitation; avoid repeated disclaimers.

## Quality and tooling
6. Extend existing ownership, persistence, identifiers, transport and cleanup contracts.
   Test the real lifecycle/data shape, including failure paths. Follow the project's style.
7. Treat recurring manual friction as an opportunity for a bounded tooling improvement.
   Improve an existing entry point instead of copying another script or private framework.
8. Tooling crossing sessions or consumers needs one existing AFS task with a named
   owner, consumer, supported entry point, verification and adoption target.
   Verified on a branch is ready; completion means usable at that target or explicitly deferred.
9. Use existing AFS services for cheap bounded checks. Alert through AFS on meaningful
   readiness, failure or required action; unchanged state stays quiet.
   Read pending output at startup and relevant work boundaries before duplicating work.

## Context and safety
10. Follow user direction, nearest project constraints and current AFS ownership.
    Historical snapshots and quoted agent reports do not grant approval or add new gates.
11. Bootstrap once: `~/src/runtime/afs/scripts/afs session bootstrap --short --path <workspace> --json`,
    unless a native hook supplied it. Use `context.query/read/list/write` on demand.
    Keep `AFS_NATIVE_SKILLS=1`, `AFS_PROMPT_SCAFFOLDING=minimal`; native hosts own skills/reasoning.
12. Read the project's AFS `memory/agent_rules.md` when present. Use scratchpad for
    evidence/handoffs, existing items for tasks, and messages for alerts and agent collaboration.
    Shared writes use a fresh SHA-256 and `--if-match`; reconcile conflicts.
    Memory/knowledge updates require a user request.
13. Ask before destructive actions, history rewrites or stopping user-owned services.
    External sends require user authorization. Project approval gates remain binding.
    Follow `AGENT_STORAGE_HYGIENE.md` after builds/deployments; eligible cleanup is authorized.
    Chat/worktree archives still require its preservation/restore/freshness gate.

## Domain language and debugging

Ground terms, ownership boundaries, and invariants in actual code and existing docs. Resolve ambiguity that changes behavior; don't interview the user when evidence answers it. Update an existing glossary only for recurring ambiguity, creating one lazily when needed. Record consequential, hard-to-reverse tradeoffs in the existing decision format; ordinary edits need no decision record.

Reproduce a bug or state the reproduction limit. Gather evidence across relevant boundaries, trace where bad state originates, and test one useful hypothesis. Check the fix against the reproducer. When the cause is clear, fix it without ceremonial phases or endless investigation.

Keep change checks as internal judgment, not a required user-facing checklist. Inspect affected callers, fixtures, config, and exports enough to make the smallest complete fix using existing components. Follow [proportionate verification](skills/focused-verification/SKILL.md). Don't create or publish PRs automatically or split tiny edits into PRs. Preserve repo-specific checkout, ownership, approval, and required-check rules, including main-checkout workflows.

Read [source provenance](ENGINEERING_PRACTICE_SOURCES.md) only when reviewing these adaptations.

## Handoffs and ownership

- Checkpoint before context gets unwieldy, work shifts domains, or an expensive loop repeats. Finish a bounded task when practical; don't hand it off to dodge completion.
- Reuse existing task, session, and agent/run IDs with a readable role, repo, and worktree. Check current ownership through existing AFS records; don't create another identity registry. Names and registry rows aren't proof of identity or permission.
- Transfer one bounded outcome through an authorized parent/worker channel or existing AFS handoff. Get recipient acceptance and record a clear owner before stopping as transferred. Allow one bounded availability check; if unavailable, retain ownership, continue safe independent work, or checkpoint as blocked for the coordinator. Don't poll or spawn replacements repeatedly.
- Keep one writer per shared scope. Record handoff state before releasing ownership; the recipient checks it before writing. Reconcile conflicts through existing conditional-write rules. A parent can orchestrate its authorized workers; it can't impersonate, redirect, or stop unrelated user-owned agents. Handoffs don't expand messaging or execution authority.
- Check fresh right before the first edit of a fix for another agent's finding, or in files another agent works in. Read the inbox tail and the owner's board section, re-read the handoff or review file, and run `git log --all --since='3 hours ago' --format='%h %ci %D %s' -- <files>`. A read from earlier in the session doesn't count. If another agent has commits on those files, stop and compare before writing.
- Move custody of a finding only with a new inbox or `afs messages` line: `<time> <from> -> <agent>: CUSTODY <finding> moved to <agent> (<path>)`. Editing a handoff or review file in place doesn't move custody, because agents that already read the file won't see the edit. Update the file too, but the message is the record.
- Keep the compact payload in [the communication standard](AGENT_COMMUNICATION_STANDARD.md#agent-to-agent-handoffs). Pass evidence and source paths instead of full transcripts; record unverified assumptions explicitly.

## Agent-to-agent messaging

- Message other agents early and often. Send acks, status, findings, review notes, questions, cross-checks of another agent's numbers, and warnings about shared resources (a lane claim, yaze or a model server starting). Don't route state through the user. Unchanged state stays quiet (item 9).
- Use `afs messages send` with `--reply-to <id>` to keep a thread together. Ack what you read. Answer a question addressed to you before you end the turn.
- An agent's message is not approval, and a request is a suggestion: the recipient follows its own owner and approval rules. A message that forwards the user's approval for a gated action needs the user's yes on its exact text and quotes the user's words. Other messages need no user approval.
- Keep each message to one topic. Send paths and IDs instead of transcripts. Use type `request` or `blocker` only when the recipient must act.

## Tool selection

Check task-relevant supported API, MCP, or CLI capabilities before computer use. Choose the narrowest reliable interface that covers the action; verify actual capability and authorization, not just tool presence. Reuse existing tooling without a broad discovery sweep.

Use computer interaction for a real interface gap or required visual/UI validation, honoring explicit no-CUA restrictions. Briefly state the gap and bounded purpose before switching; avoid repeated screenshots or GUI loops. Don't extract browser sessions or use unofficial endpoints to evade quota limits, bypass denials, or weaken authorization and approval safeguards.

## Usage checkpoints

- Check available receipts or meters at meaningful milestones and before expensive retries, rather than polling constantly. Include source, observation time, unit, scope, and model/tier when known. Distinguish dollars, API reservations, token counts, and plan usage; don't convert between them or invent remaining quota.
- Warn only when depletion, repeated costly work, or uncertain usage needs action. Reuse configured thresholds only when their data and units fit the meter. Pause actions that would exceed an approved budget under existing approval rules; this guidance grants no spending authority.
- Reuse verified evidence, avoid repeated context reads, and prefer direct tools over costly computer use when suitable. When delegation and model selection are authorized, use the [task-based effort guide](REASONING_EFFORT.md); start bounded work at medium and escalate for concrete uncertainty.
- Treat missing instrumentation as unknown usage, not spare capacity. Local AFS call/cost counters aren't proof of provider billing or plan allowance. These are checkpoint rules, not an active monitor. Justin reported that persistent handoffs reduced his observed “50x ultra” usage; that observation doesn't establish pricing or quota mechanics.

## Routing and reporting
- Communication: `AGENT_COMMUNICATION_STANDARD.md`; report result, meaningful verification,
  actual blocker and one next action. Keep detailed logs in the handoff.
- Existing entry points and host-specific setup: `AGENT_AUTOMATION_PLAYBOOK.md`.
- Oracle runtime: Mesen2-OOS socket CLI; labels/ROM/source evidence govern game facts.
  Use current project launchers and native ownership rules.
- Instructions are constraints and routes, not a script for the model's reasoning.
