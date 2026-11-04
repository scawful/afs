# Agent Automation Playbook

Last reviewed: 2026-04-26

Purpose: operational entrypoints for AI, automation, AFS, hooks, background agents, and context hygiene.

For non-technical document, sheet, ticket, planning, people, review, and approval workflows, see
`AFS_WORK_ASSISTANT_PLAN.md`.

## Scripts

- `agent-doctor [path]`
  - one-command preflight, skill sync dry run, hygiene audit, and config/script verification.
- `~/src/scripts/agent-harness-env`
  - shared environment sourced by local Claude, Gemini, hcode, and subagent launchers.
- `~/src/scripts/agent-preflight [path]`
  - compact repo, AFS, instruction, git, and skills summary before work.
- `~/src/scripts/agent-post-verify [files...]`
  - narrow parser/syntax/skill checks for changed files or explicit paths.
- `~/src/scripts/agent-skills-sync`
  - copies canonical Codex skills into Claude and Gemini, then validates.
- `~/src/scripts/agent-context-hygiene [--write-afs]`
  - audits skill drift, config parser failures, and stale agent terms.
- `~/src/scripts/afs-task-note --type handoff --title "..." --goal "..."`
  - writes a durable AFS scratchpad handoff/state/evidence note.

## Recommended Hooks

Preflight hook:
1. Source `~/src/scripts/agent-harness-env` from shell launchers.
2. Run `agent-preflight "$PWD"`.
3. Read the nearest `AGENTS.md` plus harness-specific file.
4. Use the listed AFS context before deep grep.

Doctor hook:
1. Run `agent-doctor "$PWD"` when a harness feels inconsistent or before a larger maintenance pass.
2. Read the nearest `AGENTS.md` plus harness-specific file.
3. Use the listed AFS context before deep grep.

Post-edit hook:
1. Run `agent-post-verify`.
2. If it skips a touched file type, decide whether a repo command is needed.
3. Report exact commands and residual risk.

Weekly hygiene:
1. Run `agent-skills-sync --dry-run`.
2. Run `agent-context-hygiene --write-afs`.
3. Review findings before allowing automatic rewrites.

## Background Agents

Use background agents only when their output is independently useful. Require:

- finding
- evidence
- confidence
- files inspected
- commands run
- next action

Good background tasks:

- stale documentation/config scan
- focused code review
- test failure triage
- related path inventory
- screenshot or UI regression capture
- model/artifact inventory

Do not delegate urgent blocking work when the main agent cannot proceed without it.

For work-assistant flows, background agents should enrich logged context with native AFS records:

- people and their work-scoped handles
- project relationships and review routes
- draft doc, sheet, ticket, and planning changes
- explicit approval requests for any proposed external write
- activity records for what was read, inferred, drafted, approved, or applied

Keep these concepts in AFS storage and background workers. MCP should stay thin: context overview,
relevant people, pending approvals, draft creation, approval status, and executing one approved action.

## Evidence Packs

For debugging, reviews, model work, and device deploys, write evidence under AFS scratchpad or repo docs:

```text
evidence/<date>-<topic>/
  summary.md
  commands.txt
  logs.txt
  screenshots/
  state.json
```

Every evidence summary should distinguish verified facts from inference.

## Done Criteria

For substantial work, completion requires:

- behavior changed or diagnosis proven
- touched files listed
- focused check run
- durable note/handoff written when state must survive the session
- known gaps reported
