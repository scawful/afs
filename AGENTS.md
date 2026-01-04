# Agent Instructions (AFS)

## Do not invent or market
- No marketing language or product claims.
- If something is unknown, state "Unknown / needs verification" and propose a test.

## Truth policy
- Only claim what is evidenced in this repo or cited notes.
- Do not guess roadmap, compatibility, or performance.

## Scope control
- Research-only; keep scope to core AFS primitives and APIs.

## CLI surface
- Preferred entrypoint: `afs` (installed) or `scripts/afs` (repo wrapper).
- Shell setup: `scripts/afs-shell-init.sh` (bash/zsh).
- Context warm: `scripts/afs-warm` or `afs agents run context-warm -- --interval 0`.

## Workspace integration
- If operating inside the Scawful workspace, consult `~/src/docs/NERV_INFRASTRUCTURE.md` and `~/src/docs/SRC_UNIVERSE_NETWORK.md`.
- Prefer SSH host aliases and mounts instead of hardcoded IPs.

## Provenance / separation
- Do not use employer or internal material.
- If provenance is unclear, leave it out.

## Output style
- Concise, engineering notebook tone.

## How to verify (tests/commands)
- `pytest`
