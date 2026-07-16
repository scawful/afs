# Profiles, Skills, and Grounding Hooks

Profiles control what context is injected for a given environment.

## afs.toml Example

```toml
[extensions]
auto_discover = true
enabled_extensions = ["workspace_adapter"]
extension_dirs = ["./extensions"]
# Optional sibling companion repos, e.g. ~/src/lab/afs_example.
extension_repo_roots = ["~/src/lab"]
extension_repo_prefixes = ["afs_", "afs-"]
manifest_filenames = ["extension.toml"]

[agents]
# A profile with an empty agent list receives the shipped default set
# (context-warm, index-rebuild, skills-mine, morning-briefing); profiles that
# configure any agents are never augmented. Disable here or per-environment
# with AFS_DEFAULT_AGENTS=off.
default_set = true

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
knowledge_mounts = ["~/work/logs"]
skill_roots = ["~/work/skills"]
model_registries = ["~/work/config/chat_registry.toml"]
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

Treat every configured `skill_roots` entry as an instruction trust grant.
Matched `SKILL.md` bodies can be injected into bootstrap and client system
prompts, so extension or shared roots should be reviewed before enabling them.

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

Companion repos use the same profile and extension controls as `extensions/`.
For example, a work setup can keep extension glue in `~/src/lab/afs_example`
and enable only that repo in the active profile:

```toml
[extensions]
enabled_extensions = ["afs_example"]
extension_repo_roots = ["~/src/lab"]

[profiles.work]
enabled_extensions = ["afs_example"]
policies = ["deny_keywords:customer-secret"]
```

## Skill Frontmatter

`SKILL.md` files can include YAML frontmatter for filtering and auto-loading:

```yaml
---
name: workspace-navigator
triggers: ["workspace", "context"]
requires: ["mcp", "knowledge/work"]
profiles: ["work", "general"]
enforcement:
  - Rules the agent must follow while the skill is active.
verification:
  - How to check the work.
---
```

Use:

```bash
./scripts/afs skills list --profile work
./scripts/afs skills match "debug mcp tool registration" --profile work
```

See `docs/SKILLS.md` for matching rules, session body injection, the
`skill.match`/`skill.read` MCP tools, and the bundled skill library.

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
