# Guided Setup

AFS has a guided setup flow for new machines and new workspaces:

```bash
afs setup
```

The wizard asks where configuration should live, where context files should
live, whether shell helpers should be installed, and whether optional MCP or
Google Workspace helpers should be configured. It prints the plan before
writing anything.

For a noninteractive preview:

```bash
afs setup --yes --dry-run --shell helpers --mcp none --google-workspace skip
```

Apply the same default shape:

```bash
afs setup --yes --apply --shell helpers --mcp none --google-workspace skip
```

## Setup Choices

Config scope:

- `project`: writes `./afs.toml` in the selected workspace.
- `user`: writes `~/.config/afs/config.toml`.

Context mode:

- `project`: keeps context files under the workspace `.context/`.
- `shared`: keeps context files under `~/.context`; useful for managed
  workspaces where repository-local files are not appropriate.

Shell mode:

- `helpers`: installs aliases, colors, and zsh completion only.
- `agent-hooks`: also routes supported AI harness commands through AFS wrappers.
- `none`: leaves shell startup files unchanged.

Optional integrations:

- `--mcp claude`, `--mcp gemini`, or `--mcp both`
- `--google-workspace check` or `--google-workspace setup`
- `--worker` for the background agent-job worker

## Friendly Guides

Use `afs guide` for a menu of workflow-oriented help:

```bash
afs guide
afs guide context
afs guide shell
afs guide mcp
afs guide google-workspace
afs guide agents
```

These guides are intentionally generic. Organization-specific MCP servers,
search tools, credentials, and policy should stay in local client config or
approved extension packages.
