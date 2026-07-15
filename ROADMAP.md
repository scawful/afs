# AFS Roadmap

This roadmap keeps the core repo focused on shareable platform capabilities.

## 0.2.x — share-ready core

- Keep setup and docs approachable for first-time users.
- Keep CI green across Python 3.10, 3.11, and 3.12.
- Harden packaging, release notes, and extension examples.
- Remove stale domain-specific examples from core docs.

## 0.3.x — extension developer experience

- Stabilize the v1 optimization evidence/policy/decision protocol and golden fixtures.
- Add a centralized fail-closed executor before allowing autonomous candidate trials.
- Version the extension manifest schema explicitly.
- Add richer extension validation and diagnostics.
- Add more complete example extensions for CLI, MCP tools, context sources, and manager actions.
- Document compatibility guarantees and deprecation policy for extension authors.

## 0.4.x — stable agent surfaces

- Add immutable experiment/trial records, transactional leases, exact-hash review, and rollback.
- Add language adapters, beginning with a C++20 client over the stable JSON/MCP contract.
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
