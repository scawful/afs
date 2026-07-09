## Summary

Describe what changed and why.

## Branch Target Checklist

- [ ] This PR targets the correct base branch for AFS staged integration:
  - Feature work -> `features`
  - Integration/stabilization -> `development`
  - Release promotion -> `main` (from `development`)
- [ ] If this PR targets `main`, the head branch is `development` (or this PR documents why an exception is required).
- [ ] If `main` changed recently, sync/merge updates were considered for `features` and `development`.

## Validation

- [ ] Fastest relevant tests/checks were run locally.
- [ ] Validation commands and outcomes are included in the PR description.
- [ ] Package or setup changes ran `make package-check` or an equivalent build/install smoke test.

## Release / Docs Impact

- [ ] User-facing CLI, MCP, setup, config, or extension changes are documented.
- [ ] Version/release changes update `CHANGELOG.md`, `RELEASE.md`, and package metadata.
- [ ] Domain-specific examples remain in extensions, not core AFS.

## Notes

List risks, follow-ups, or rollout caveats.
