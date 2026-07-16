---
name: mission-tracking
triggers:
  - afs mission
  - durable mission
  - cross-session goal
  - mission next step
  - track mission
profiles:
  - general
requires:
  - afs
---

# Mission Tracking

Durable, cross-session goals that render into every agent system prompt until
resolved. Use missions for multi-day work; use the task queue for single items.

## Commands

```bash
afs mission create --path <ws> --title "..." --summary "..." \
  --next-step "..." --next-step "..." --tag <tag>
afs mission list --path <ws>            # active missions
afs mission show <mission_id>           # full JSON
afs mission update <mission_id> --status done --note "commit abc123"
```

`--acceptance` is the human's definition of done — never author it as an
agent; leave it unset and let the human add it. Setting, changing, or
clearing it requires a typed confirmation on an interactive terminal, so a
headless agent passing `--acceptance` is refused (exit 2). Closed missions
are resurfaced by `afs calibration review` for outcome scoring against that
acceptance.
Direct store callers cannot forge this anchor with `acceptance_set_by`; text
without a broker capability is retained only as `acceptance_suggestion`.

## Lifecycle

Statuses: `active` -> `blocked` | `done` | `abandoned`.

Update flags: `--status`, `--summary`, `--owner`, `--next-step` (repeatable,
replaces the list), `--blocker`, `--note`, `--tag`, `--link-session`,
`--link-handoff`, `--actor`.

## Conventions

- One mission per coherent goal; encode concrete resumable steps in `--next-step`
- Record evidence (commit hashes, file paths) in `--note` when updating
- Link the closing handoff with `--link-handoff` so the trail survives compaction
- Mark `blocked` with a `--blocker` instead of letting a mission go stale
- Active missions appear automatically in session bootstrap and model prompts;
  never re-create a mission that already exists — update it
