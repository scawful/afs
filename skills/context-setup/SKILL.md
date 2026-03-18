---
name: context-setup
triggers:
  - context
  - init
  - setup
  - bootstrap
profiles:
  - general
requires:
  - afs
---

# Context Setup

Initialize and validate an AFS context root for a project.

## Steps

1. Run `afs init --link-context` to create `.context/` with all mount dirs
2. Run `afs status` to verify mounts are present
3. Run `afs context discover --path .` to index the project
4. Switch profile if needed: `afs profile switch <name>`

## Quick Start

```bash
cd /path/to/project
afs init --link-context --workspace-path . --workspace-name my-project
afs status
```

## Hints

- Use `--json` on any command for machine-readable output
- The `.context/` symlink points to your context root (default: `~/.context`)
- Each mount dir (memory, knowledge, scratchpad, etc.) is auto-created
- Profiles control which knowledge/skills/tools are active
