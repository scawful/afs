---
description: preview or apply the AFS local harness update path
---

Help update a local AFS checkout and harness setup.

Rules:

- Preview first unless the user explicitly asked to apply:
  `cd ~/src/lab/afs && scripts/afs-upgrade-agent-setup --workspace ~/src --full --setup-hcode`
- Apply only when requested:
  `cd ~/src/lab/afs && scripts/afs-upgrade-agent-setup --workspace ~/src --full --setup-hcode --apply`
- Explain that the default MCP catalog stays slim and heavier flows route
  through CLI/framework hints or `--tool-catalog full`.
- After applying, run `scripts/afs agent-manifest validate --check-paths` and
  `scripts/afs agent-hooks status --path ~/src`.

$ARGUMENTS
