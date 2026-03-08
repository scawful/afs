# Profiles, Skills, and Grounding Hooks

AFS profiles let you control context injection per environment (e.g. `work` vs `zelda`) without hardwiring domain content into core.

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
policies = ["no_zelda"]

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

`context init` and `context ensure` auto-apply the active profile when `profiles.auto_apply = true`.

## Skill Frontmatter

`SKILL.md` files may include YAML frontmatter fields used for filtering and auto-loading:

```yaml
---
name: gemini-work
triggers: ["gemini-cli", "agent studio"]
requires: ["knowledge/work", "gemini mcp"]
profiles: ["work", "general"]
---
```

Use:

```bash
afs skills list --profile work
afs skills match "debug gemini-cli mcp setup" --profile work
```

## Grounding Policy

The built-in `no_zelda` policy blocks Zelda-related terms in profile-scoped hooks (`before_context_read`, `before_agent_dispatch`).
Use this for work profiles to prevent accidental cross-domain leakage.

## Monorepo Bridge

AFS now provisions a `monorepo/` mount role and expects workspace-switch automation to
write `monorepo/active_workspace.toml`. Use the template hook in
`extensions/afs_google/hooks/context-sync-active-workspace.sh`.

`afs health` reports when that bridge file is stale (older than one hour).
