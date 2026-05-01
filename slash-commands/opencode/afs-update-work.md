---
description: preview or apply the AFS harness update path for a work machine
---

Help update a work-machine AFS checkout and harness setup.

Rules:

- Preview first unless the user explicitly asked to apply:
  `cd ~/src/lab/afs && scripts/afs-upgrade-agent-setup --workspace ~/src --work --setup-hcode`
- Apply only when requested:
  `cd ~/src/lab/afs && scripts/afs-upgrade-agent-setup --workspace ~/src --work --setup-hcode --apply`
- Explain that the default MCP catalog stays slim and heavier flows route
  through CLI/framework hints or `--tool-catalog full`.
- After applying, run `scripts/afs agent-manifest validate --check-paths` and
  `scripts/afs agent-hooks status --path ~/src`.

$ARGUMENTS
