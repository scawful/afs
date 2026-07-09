# Hello World Extension

This is a minimal AFS extension example. It demonstrates the shape of a companion extension repo without requiring private data or external services.

Preview from the AFS repo root:

```bash
export AFS_EXTENSION_DIRS="$PWD/examples/extension_hello_world"
export AFS_ENABLED_EXTENSIONS="afs_hello_extension"
./scripts/afs extensions list --details
```

A real companion repo can use the same layout with `extension_repo_roots` instead of copying files into core AFS.
