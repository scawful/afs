# Journal Agent

Background agent that drafts the AI portion of a hybrid weekly review by
scanning a thoughts.org file and a tasks/active.md file.

## Overview

The `journal-agent` reads dated entries from a thoughts.org file (org-mode
headlines like `* 4 April 2026`) and items from a markdown task list. For
the current ISO week it writes (or refreshes) `<weekly_dir>/YYYY-WNN.org`,
preserving the human section above the divider untouched and only modifying
the AI draft below.

## Path resolution

Paths are not hardcoded to any workspace layout. Resolution order, highest
precedence first:

1. CLI flags `--thoughts`, `--active-tasks`, `--weekly-dir`
2. Per-field env vars `AFS_JOURNAL_THOUGHTS`, `AFS_JOURNAL_ACTIVE_TASKS`,
   `AFS_JOURNAL_WEEKLY_DIR`
3. Sub-paths derived from `AFS_JOURNAL_ROOT`
   (`thoughts.org`, `tasks/active.md`, `weekly/`)
4. Generic fallback: `~/.local/share/afs/journal/`

To point the agent at a custom writing folder, export e.g.:

```bash
export AFS_JOURNAL_ROOT="$HOME/notes"
```

The optional `AFS_JOURNAL_AUTHOR` env var sets the `#+AUTHOR:` line on
newly scaffolded weekly review files.

## Hybrid weekly review format

Newly scaffolded files use this skeleton:

```org
#+TITLE: Weekly Review — 2026-W14 (Mar 30–Apr 5)

* What happened
-
-
-

* What I want next week
-

---
(AI draft below — edit or ignore)
```

The human writes above the `---` divider; the agent writes below. On
re-runs, an existing AI section is preserved unless `--overwrite` is set.

## Usage

```bash
# Draft this week's review
afs agents run journal-agent

# Draft a specific week, refreshing any existing AI section
afs agents run journal-agent -- --week 2026-W12 --overwrite

# Override paths explicitly
afs agents run journal-agent -- \
  --thoughts ~/notes/thoughts.org \
  --active-tasks ~/notes/tasks/active.md \
  --weekly-dir ~/notes/weekly
```

## Arguments

| Flag | Default | Description |
|---|---|---|
| `--week` | current ISO week | Week to draft (`YYYY-WNN`) |
| `--thoughts` | env / fallback | Path to thoughts.org |
| `--active-tasks` | env / fallback | Path to tasks/active.md |
| `--weekly-dir` | env / fallback | Weekly review directory |
| `--overwrite` | off | Replace existing AI draft section instead of skipping |
| `--output` | — | Write JSON result to this path |
| `--stdout` | — | Force JSON to stdout even when non-interactive |
| `--pretty` | — | Pretty-print JSON output |
| `--quiet` | — | Suppress INFO logs |

## What goes into the AI draft

The agent scans:

- **Thoughts.org** for any `* DD Month YYYY` headline whose date falls in
  the current week. Each entry's body is summarized as a one-line preview.
- **tasks/active.md** for `- [ ]` (open) and `- [x]` (completed) items.
- **Recent agent activity** from the AFS context index, deduped per
  agent/op pair, last 7 days.

Output is appended below the divider in the weekly file as a single org
heading `* AI draft — YYYY-WNN` with subheadings for thoughts, tasks, and
agent activity.
