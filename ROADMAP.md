# AFS Roadmap

This roadmap keeps the core repo focused on shareable platform capabilities.

## 0.2.x — share-ready core

- Keep setup and docs approachable for first-time users.
- Keep CI green across Python 3.10, 3.11, and 3.12.
- Harden packaging, release notes, and extension examples.
- Remove stale domain-specific examples from core docs.

## 0.3.x — extension developer experience

- Stabilize the v1 optimization evidence/policy/decision protocol and golden fixtures.
- Stabilize the v1 execution request/inspection/record protocol and portable
  process backend now used by verification.
- Freeze and publish cross-language canonical JSON conformance vectors covering
  duplicate members, Unicode scalar ordering/escaping, numeric spellings,
  negative zero, decimal rounding, and representative decision/request hashes;
  run the same fixtures against Python and future C++ clients.
- Before freezing execution v1, bind inspections and records to a trusted policy
  identifier/hash and encode or semantically validate invariants among
  `allowed`, reasons, outcome, timeout state, and return code.
- Add transactional job leases with owner, nonce, expiry, heartbeat, and
  compare-and-swap transitions; then migrate the background worker to
  per-argument templates through the execution broker and keep verification
  evidence distinct from a successful process exit.
- Migrate remaining generic agent shell/tool surfaces only after their callers
  can express structured commands and trusted policy explicitly.
- Version the extension manifest schema explicitly.
- Add richer extension validation and diagnostics.
- Add more complete example extensions for CLI, MCP tools, context sources, and manager actions.
- Document compatibility guarantees and deprecation policy for extension authors.

## 0.4.x — stable agent surfaces

- Remove deprecated verification string commands and the legacy-shell migration
  flag.
- Add a Docker backend before autonomous trials, with immutable candidate,
  mutation, trial, eval-pack, and search-state artifacts plus hard budgets and
  hidden holdouts.
- Add bounded local hill climbing first; keep GEPA- and AlphaEvolve-inspired
  proposers as untrusted extensions behind the same evaluation boundary.
- Add exact-request-hash approvals, an atomic champion pointer, canary rollout,
  and one-command rollback before activation can become autonomous.
- Modernize interoperability against MCP `2025-11-25`, then add a C++20 client
  over the stable JSON/MCP contracts rather than Python or SQLite internals.
- Stabilize the default MCP tool catalog and underscore-safe aliases.
- Improve `afs setup` and `afs doctor` for fresh machines.
- Add stronger release artifact checks and install smoke tests.
- Reduce advisory mypy/type-check debt until broader source surfaces can become blocking.
- Continue separating private/domain workflows into companion repos.

## 1.0 readiness criteria

AFS can be considered for `1.0.0` when:

- installation works from a release artifact without repo-local assumptions
- docs cover the default setup path and common extension path end to end
- CLI/MCP/context layout changes have a documented deprecation path
- extension authoring has a stable manifest contract
- security posture for filesystem, shell, and external writes is explicit and tested
