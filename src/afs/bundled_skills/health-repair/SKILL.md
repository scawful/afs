---
name: health-repair
triggers:
  - afs health
  - afs doctor
  - context repair
  - broken context
  - afs health check
  - auto-heal
profiles:
  - general
requires:
  - afs
enforcement:
  - Start with read-only status, health, doctor, and dry-run repair commands.
  - Run doctor --fix, health --auto-heal, or repair without --dry-run only with explicit user direction.
  - Start an indefinite health monitor only with explicit user direction.
---

# Health & Repair

Diagnose and fix AFS contexts. Prefer these commands over ad-hoc symlink or
file surgery.

## Escalation Ladder

1. `afs status` — quick mount/index overview
2. `afs health check --level standard` — structured checks (`basic`, `standard`, `comprehensive`, `stress`)
3. `afs doctor` — cross-cutting diagnosis; `--fix` changes state and is user-directed
4. `afs context repair --path <ws> --dry-run` — preview mount/provenance repairs
5. After explicit approval, rerun without `--dry-run`; add `--rebuild-index`
   when the preview identifies a stale or empty index

## Other Commands

| Command | Description |
|---------|-------------|
| `afs health status` / `trend` / `history` | Current status, trends, recent reports |
| `afs health monitor` | Continuous monitoring loop |
| `afs context validate` | Validate a single context |
| `afs health check --auto-heal` | Checks with automatic healing |

## Rules

- Always `--dry-run` repair first; show the plan before applying it
- `repair` flags: `--no-profile-reapply` skips profile mounts, `--no-remap`
  skips workspace remapping for missing sources
- Run repair via CLI on the host that owns the context root, not through a
  sandboxed client
- `health monitor` runs indefinitely unless bounded with `--duration`; do not
  start it as background or persistent work without explicit user direction
- Report what a fix changed; never claim a repair that was not run
