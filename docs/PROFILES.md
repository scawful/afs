# Profiles, Skills, and Grounding Hooks

Profiles control what context is injected for a given environment.

## afs.toml Example

```toml
[extensions]
auto_discover = true
enabled_extensions = ["afs_google"]
extension_dirs = ["./extensions"]

[profiles]
active_profile = "work"
auto_apply = true

[profiles.default]
knowledge_mounts = []
skill_roots = []
model_registries = []
enabled_extensions = []
policies = []

[profiles.work]
inherits = ["default"]
knowledge_mounts = ["~/Journal/logs"]
skill_roots = ["~/src/lab/afs-scawful/skills"]
model_registries = ["~/src/lab/afs-scawful/config/chat_registry.toml"]
enabled_extensions = ["afs_google"]
policies = []

[hooks]
before_context_read = []
after_context_write = []
before_agent_dispatch = []
```

## Profile Commands

```bash
afs context profile-show --profile work
afs context profile-apply --profile work
afs profile current
afs profile switch work
```

`context init` and `context ensure` auto-apply the active profile when
`profiles.auto_apply = true`.

## Skill Frontmatter

`SKILL.md` files can include YAML frontmatter for filtering and auto-loading:

```yaml
---
name: workspace-navigator
triggers: ["workspace", "context"]
requires: ["mcp", "knowledge/work"]
profiles: ["work", "general"]
---
```

Use:

```bash
afs skills list --profile work
afs skills match "debug mcp tool registration" --profile work
```

## Monorepo Bridge

AFS provisions a `monorepo/` mount role and expects workspace switch automation to
write `monorepo/active_workspace.toml`.

Template hook:

- `extensions/afs_google/hooks/context-sync-active-workspace.sh`

`afs health` reports when the bridge file is stale (older than one hour).
