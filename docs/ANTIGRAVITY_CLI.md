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
```

`setup` is a dry run unless `--apply` is passed. AFS does not install `agy` or
add dangerous permission flags automatically.

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

## Gemini CLI compatibility

`afs gemini setup` and `afs gemini status` remain for Gemini CLI compatibility
and Gemini API-key workflows. Public individual/free/Pro/Ultra Gemini CLI
request serving moved to Antigravity CLI on 2026-06-18.
