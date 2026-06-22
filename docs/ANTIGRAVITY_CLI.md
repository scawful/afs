# Antigravity CLI Integration

AFS treats Antigravity CLI (`agy`) as the public successor path for Gemini CLI
style terminal-agent workflows. Gemini API and Google Workspace public API
helpers remain separate surfaces.

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

On `agy` 1.0.10, `agy models` prints labels such as:

```text
Gemini 3.5 Flash (Medium)
Gemini 3.1 Pro (High)
Claude Opus 4.6 (Thinking)
```

AFS parses these with `afs antigravity models --json` instead of hardcoding the
available model set.

## Gemini CLI compatibility

`afs gemini setup` and `afs gemini status` remain for Gemini CLI compatibility
and Gemini API-key workflows. Public individual/free/Pro/Ultra Gemini CLI
request serving moved to Antigravity CLI on 2026-06-18.
