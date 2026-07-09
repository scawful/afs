# Release Process

AFS uses staged branches and SemVer-style tags.

## Versioning

- `0.x` means public surfaces are useful but still evolving.
- Minor releases (`0.2.0`, `0.3.0`) may refine CLI/MCP/extension APIs.
- Patch releases (`0.2.1`) are bugfix-only.
- `1.0.0` should wait until the CLI, MCP default surface, `.context` layout, and extension manifest contract are stable enough for external users.

## Branch flow

1. Topic PR -> `features`
2. Promotion PR: `features -> development`
3. Release PR: `development -> main`
4. Tag the final `main` commit.

## Pre-release checklist

From a clean checkout:

```bash
git checkout main
git pull --ff-only
make release-check
```

Confirm:

- `pyproject.toml` version matches `src/afs/version.py`
- `CHANGELOG.md` has the release entry
- README and setup docs mention the current release shape
- stale domain/private examples are not presented as core functionality
- CI is green on the release PR and on `main` after merge

## Tagging

After `development -> main` is merged and main CI is green:

```bash
VERSION=v0.2.0
git checkout main
git pull --ff-only
git tag -a "$VERSION" -m "AFS $VERSION"
git push origin "$VERSION"
```

The tag workflow builds source and wheel artifacts. GitHub release notes should summarize:

- what changed
- setup path
- compatibility notes
- extension migration notes
- verification status

## Package publishing

AFS does not publish to PyPI automatically yet. Build artifacts are suitable for GitHub Releases. Add PyPI publishing only after package naming, ownership, and long-term support expectations are explicit.
