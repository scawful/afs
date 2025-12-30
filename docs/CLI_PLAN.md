# CLI Plan (AFS)

Scope: improve AFS CLI ergonomics for agent workflows. Research-only; no product
claims. See `docs/RESEARCH_SOURCES.md` for citations.

## Principles (with citations)
- Prefer a minimal, execution-aware tool surface over large ad hoc command sets. [R3]
- Make context flow explicit (state + errors + async boundaries). [R4]
- Treat context as a persistent filesystem surface, not transient prompt state. [R1]
- Add observability hooks as systems scale (logs, summaries, eval hooks). [R2]

## Phase 0 (now)
- Document existing commands and config defaults in `README.md`.
- Add `--json` output for `status`, `context validate`, `context discover`, and `graph export`.
- Add `afs config show` (effective config + defaults).

## Phase 1 (near-term)
- `afs context report` to emit a single summary JSON (roots, missing dirs, counts).
- `afs graph export --format dot|json` (JSON already present; define schema version).
- `afs services render --format` and `--output` for integration with runbooks.

## Phase 2 (later)
- `afs workspace lint` to catch missing `workspaces.toml` entries.
- `afs context diff` (two roots) to compare catalog drift.
- Optional: plugin registry summary to surface installed plugins and version pins.

## Unknown / needs verification
- Which commands agents actually use most in production flows.
- Whether a TUI is needed beyond `afs_studio`.

## Citations
- [R1] `docs/RESEARCH_SOURCES.md`
- [R2] `docs/RESEARCH_SOURCES.md`
- [R3] `docs/RESEARCH_SOURCES.md`
- [R4] `docs/RESEARCH_SOURCES.md`
