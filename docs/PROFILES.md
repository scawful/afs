# Profiles, Skills, and Grounding Hooks

Profiles control what context is injected for a given environment.

## afs.toml Example

```toml
[extensions]
auto_discover = true
enabled_extensions = ["workspace_adapter"]
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
enabled_extensions = ["workspace_adapter"]
policies = []

[hooks]
before_context_read = []
after_context_write = []
before_agent_dispatch = []
session_start = []
session_end = []
user_prompt_submit = []
task_completed = []
```

## Profile Commands

```bash
./scripts/afs context profile-show --profile work
./scripts/afs context profile-apply --profile work
./scripts/afs profile current
./scripts/afs profile switch work
```

`context init` and `context ensure` auto-apply the active profile when
`profiles.auto_apply = true`.

`model_registries` are consumed by registry-aware runtime surfaces such as
`afs agents run scribe-draft` and any extension-provided agent or gateway entrypoints
that call the shared chat registry loader.

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
./scripts/afs skills list --profile work
./scripts/afs skills match "debug mcp tool registration" --profile work
```

## Monorepo Bridge

AFS provisions a `monorepo/` mount role and expects workspace switch automation to
write `monorepo/active_workspace.toml`.

Template hook:

- `extensions/workspace_adapter/hooks/context-sync-active-workspace.sh`

`session_start`, `session_end`, `user_prompt_submit`, `turn_*`, and `task_*`
are the harness-facing lifecycle seams for `afs-client-session` wrappers and
external adapters. They are the closest AFS analogue to Claude Code's
`SessionStart`, `UserPromptSubmit`, and task lifecycle hook events.

`./scripts/afs health` reports when the bridge file is stale (older than one hour).
