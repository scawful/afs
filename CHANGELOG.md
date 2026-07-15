# Changelog

All notable changes to AFS are documented here. AFS follows Semantic Versioning while it is pre-1.0: minor versions may refine public APIs, and patch versions are bugfix-only.

## [Unreleased]

### Added

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

### Changed

- Refreshed GitHub Actions dependencies to current Node-runtime-compatible releases.
- Narrowed CI type checking to the release-critical slice while broader type debt remains tracked in `ROADMAP.md`.
- Deprecated verification string commands; they are blocked by default, require
  an explicit legacy-shell opt-in during migration, and will be removed in
  `0.4.0`.

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
