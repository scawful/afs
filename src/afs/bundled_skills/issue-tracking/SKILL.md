---
name: issue-tracking
triggers:
  - ticket
  - issue tracker
  - github issue
  - backlog
  - file an issue
  - track work
  - link the PR
  - status update
profiles:
  - general
requires:
  - afs
enforcement:
  - Never create, comment on, close, or edit an external tracker item without explicit user approval; draft locally first.
  - Acceptance criteria are human-authored; propose them as a suggestion, never set them yourself.
  - Close nothing without evidence — a commit hash, PR link, or verification output in the closing note.
verification:
  - Confirm every tracked item links to its artifacts (branch, PR, commit) and each artifact links back.
---

# Issue Tracking

Team-visible work tracking: choosing the right surface, keeping tickets
actionable, and linking work to evidence.

## Choosing the Surface

| Work shape | Surface |
|------------|---------|
| Multi-day goal, survives sessions | `afs mission create` |
| Single queued item for an agent | `afs tasks` / `task.create` |
| Team-visible bug/feature/decision | external tracker (`gh issue`) — user-approved |
| In-flight coordination between agents | messages / handoffs, not tickets |

## Ticket Hygiene

- Title states the symptom or outcome, not the suspected fix.
- Body: reproduction steps or evidence, expected vs actual, severity,
  environment. A ticket someone must re-investigate to understand is debt.
- Acceptance criteria come from a human; agents may draft a suggestion
  clearly labeled as such.
- Link both directions: ticket ↔ branch/PR/commit; mission `--link-handoff`
  and `--note` with hashes on the AFS side.

## Read Commands (safe without approval)

```bash
gh issue list --search "<terms>"     # dedupe before filing
gh issue view <n> --comments
gh pr list / gh pr checks <n>
afs mission list / afs tasks list
```

Write operations (`gh issue create/comment/close`, `gh pr create`) are
drafted locally and executed only with the user's go-ahead.

## Status Discipline

- Update status when state changes, not in batches later.
- Blocked items say what unblocks them and who owns that.
- Duplicates get linked and closed, not re-litigated.
- Closing note carries the evidence: what shipped, where, how verified.
