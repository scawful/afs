# Guided Setup

AFS has a guided setup flow for new machines and new workspaces:

```bash
afs manager
afs next --intent setup
afs setup
```

`afs manager` opens the friendly Python GUI for people who do not want to learn
the CLI first. It shows context health, task queue state, project client config
such as `.gemini/settings.json`, loaded extensions, and extension manager
actions.

The setup wizard asks where configuration should live, where context files
should live, whether shell helpers should be installed, and whether optional
MCP or Google Workspace helpers should be configured. It prints the plan before
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

GUI manager:

- `afs manager` opens the manager for the current directory.
- `afs manager snapshot --json` prints the same read model for scripts.
- `scripts/afs-manager` is the repo-local launcher shortcut.
- `scripts/afs-manager.command` can be double-clicked on macOS.
- Extensions can add manager-visible actions with `[manager] actions = [...]`
  in `extension.toml`.

## Agent discovery path

Agents should not browse the entire AFS surface up front. Use this deterministic
ladder instead, or ask the router with `afs next --intent <intent> --json`:

1. `context.status`
2. `context.query`
3. `context.read` / `context.list`
4. `context.write` only for scratchpad notes or requested context files
5. CLI or slash commands for tasks, handoffs, work preflight, verification,
   refresh, repair, or session packs

`context.status` includes this discovery path in JSON/MCP output so clients can
surface it without expanding the default MCP catalog.
`afs next report --json` summarizes recent router use and flags heavy MCP calls
that bypassed the slim default path.

## Friendly Guides

Use `afs guide` for a menu of workflow-oriented help:

```bash
afs guide
afs guide next
afs guide manager
afs guide context
afs guide shell
afs guide mcp
afs guide google-workspace
afs guide agents
```

These guides are intentionally generic. Organization-specific MCP servers,
search tools, credentials, and policy should stay in local client config or
approved extension packages.
