# Antigravity CLI Integration

AFS treats Antigravity CLI (`agy`) and Gemini CLI as separate supported agent
clients. Gemini API and Google Workspace public API helpers remain separate
surfaces as well.

## Commands

```bash
afs antigravity status --json
afs antigravity setup --scope project --project-path .
afs antigravity setup --scope project --project-path . --apply
afs antigravity models
afs antigravity models --json
```

`setup` is a dry run unless `--apply` is passed. AFS does not install `agy` or
add dangerous permission flags automatically.

Current `agy` builds use the shared migrated MCP config path
`~/.gemini/config/mcp_config.json`. AFS still detects older Antigravity CLI and
IDE config locations, but new setup writes the migrated MCP config by default.
The client also exposes `agy mcp add|remove|list|enable|disable`; use
`agy mcp list` to verify the effective registration after setup.

## Workspace skill

Gemini CLI and Antigravity both discover Agent Skills under
`.agents/skills/<name>/SKILL.md`. The portable setup helper uses this shared
workspace alias when either harness is selected:

```bash
scripts/afs-upgrade-agent-setup --workspace . --harness antigravity
scripts/afs-upgrade-agent-setup --workspace . --harness gemini --apply
```

Pass `--skill-root antigravity=/different/location` or
`--skill-root gemini=/different/location` when a managed machine uses another
layout. AFS copies a small `/afs` skill and does not select a model or reasoning
effort for the client.

## Install hint

If `agy` is missing, AFS reports the public install command:

```bash
curl -fsSL https://antigravity.google/cli/install.sh | bash
```

Then verify:

```bash
agy --version
agy models
```

`agy models` prints the models and reasoning labels enabled for the current
account and build. For example, a build may report:

```text
Gemini Flash (Medium)
Gemini Pro (High)
```

AFS parses these with `afs antigravity models --json` instead of hardcoding the
available model set.

## Gemini CLI compatibility

Use `afs gemini setup` and `afs gemini status` for Gemini CLI and Gemini API-key
workflows. Use `afs antigravity setup` and `afs antigravity status` for
Antigravity. Neither command silently configures the other client.
