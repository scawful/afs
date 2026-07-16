# Changelog

All notable changes to AFS are documented here. AFS follows Semantic Versioning while it is pre-1.0: minor versions may refine public APIs, and patch versions are bugfix-only.

## [Unreleased]

### Added

- Engagement primitives that keep human judgment in the loop: approvals
  approve/reject on both surfaces now require a `--because` rationale and an
  interactive human confirmation (headless approval fails closed), with
  reviewer provenance recorded in history; missions carry a human-authored
  `acceptance` field with recorded provenance; `afs calibration review`
  resurfaces the window's decisions with their rationales for outcome
  scoring (`afs calibration score` is itself confirmation-gated with
  scored-by provenance, rejects unknown refs, and stores global gate-
  decision outcomes globally so they never resurface unscored in another
  context; `--markdown` emits a weekly-review digest);
  `afs session bootstrap --engage` asks for a top-priority prediction before
  revealing the queue (never in `--json` mode) and logs it to the calibration
  trail; and the `implementation-plan` schema gains a `human_intent` section
  agents must never author, enforced structurally by
  `afs schema validate --skeleton`.
- Human judgments now cross a decision-scoped store capability minted by a
  cross-platform controlling-terminal broker, with UID/SID-derived identity
  and fail-closed refusal when OS identity is unavailable. Capabilities bind
  the exact store, record, rationale/note, and process and are single-use.
  Programmatic store APIs cannot forge authorization or calibration
  provenance; programmatic predictions are excluded from human calibration;
  mission text becomes an `acceptance_suggestion` without the capability;
  approval clearing archives crash-repaired decision history; active approval
  state is corruption-failing, atomically replaced, refreshed across processes,
  and bound to exact action details; work approvals use an atomic execution
  claim and reopen legacy unconfirmed approvals for human review; calibration
  JSONL appends are locked, fsynced, and repair torn tails; mission updates are
  serialized and bounded human-confirmed acceptance now reaches bootstrap
  prompts; and malformed or changed `human_intent` anchors fail with correction
  guidance.
- Versioned, packaged JSON Schema contracts for optimization evaluation,
  policy, and decision records.
- Pure `afs optimize decide` evidence gate with deterministic hashes, stable
  reason codes, and no execution or promotion authority.
- Cross-language optimization protocol documentation and runnable fixtures.
- Versioned execution request, inspection, and record schemas plus a public
  typed Python policy-checking API.
- Read-only `afs execution inspect` and a portable process backend with bounded
  time/output, scrubbed environments, and redacted audit records.
- Structured verification executions routed through the execution broker.
- Focused execution-contract CI coverage on Linux, macOS, and Windows.
- Bounded matched-skill body delivery in session bootstrap and prepared client
  prompts, with optional CLI and MCP bootstrap focus arguments.
- Bundled operating skills included in wheel and source distributions.
- Bounded `skill.match` and root-contained `skill.read` MCP tools in the
  default slim catalog, including every session tool profile.
- Twelve bundled workflow skills covering sessions, missions, search, health,
  events, memory, approvals, schemas, verification, authoring, and CLI routing.
- Validated per-tool and `[mcp_tools]` catalog controls for extension discovery.
- A conservative default supervisor set for empty profiles: daily context
  audit, knowledge/memory index refresh, weekly skill mining, and a daily
  scratchpad briefing. Defaults are configurable and never augment a custom
  agent list implicitly.
- Versioned extension-manifest validation with bounded diagnostics, isolated
  CLI registration failures, and surfaced doctor/plugin reports.
- Event reactor: `on_event` agent start conditions match history events and
  hivemind messages with `"<kind>[:<detail>]"` fnmatch patterns. Delivery is
  transactional — events are read oldest-first under an exclusive lock with
  bounded per-cycle batches, dispatched, and only then acked via an atomic
  state commit, so a crash redelivers instead of losing events and
  supervisors sharing a state directory never double-deliver. A failed
  dispatch defers the ack (redelivery, not silent consumption); events
  stamped within a short grace window are deferred one cycle so the
  watermark cannot pass an in-flight write; cursors prime per source.
  Unknown `on_event_action` values fail closed; the `job` action passes the
  same supervisor gates as spawns and embeds only sanitized event labels in
  prompts; debounce (`event_debounce`, default 5m) is persisted at ack so it
  also covers job actions; future-stamped records are deferred without
  blocking later ripe work; hivemind sends publish atomically and exact file
  identities preserve newly copied/backdated messages; tuple-based v3 state
  conservatively replays the extant hivemind inventory once when upgrading to
  exact identities; stable malformed records, including invalid provided
  expiries, are skipped and counted; the hivemind bus is canonical (history
  mirrors of sends are excluded); and
  `AgentConfig` round-trips now preserve custom mapping keys.

### Changed

- Refreshed GitHub Actions dependencies to current Node-runtime-compatible releases.
- Narrowed CI type checking to the release-critical slice while broader type debt remains tracked in `ROADMAP.md`.
- Fixed hcode/OpenCode wrappers to request the supported generic session-pack
  model while retaining their client-specific prompt profile.
- Deprecated verification string commands; they are blocked by default, require
  an explicit legacy-shell opt-in during migration, and will be removed in
  `0.4.0`.
- Supervisor child processes now inherit an explicit `AFS_CONFIG_PATH`, watch
  signatures follow symlink-mounted sources, and missing completion records
  fail closed instead of looking like successful scheduled runs.

## [0.2.0] - 2026-07-09

### Added

- Executive-friendly overview in `docs/EXECUTIVE_SUMMARY.md`.
- Research lineage map in `docs/LINEAGE.md`.
- MCP tool-name compatibility for clients that require underscore-safe tool names.
- Release, contribution, security, and roadmap documentation.
- Version metadata in package code, packaging metadata, and release workflow.
- Extension authoring guide and hello-world extension example.

### Changed

- Core AFS is now framed as a generic agentic filesystem/context platform.
- Domain-specific training, cost, quality-gate, and private workflows are documented as extension-owned.
- README, setup, and development docs now prioritize quick onboarding and staged release flow.
- License metadata and root license file now agree on MIT.

### Removed

- Stale core examples for moved extension modules (`afs.continuous`, `afs.cost`, `afs.gates`) from the public example set.
- Main-branch CI references to project-specific model names and private deployment assumptions.

## [0.1.0] - 2026-01-14

### Added

- Initial core AFS runtime with context roots, MCP server, session bootstrap, context packs, memory consolidation, agent harness support, and guardrailed tooling.

[Unreleased]: https://github.com/scawful/afs/compare/v0.2.0...HEAD
[0.2.0]: https://github.com/scawful/afs/compare/v0.1...v0.2.0
[0.1.0]: https://github.com/scawful/afs/releases/tag/v0.1
