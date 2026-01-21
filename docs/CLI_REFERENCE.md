# AFS CLI Reference

*Moved from README.md*

## Quickstart
- `afs` (shows defaults + command tree)
- `afs help context`
- `afs init --context-root ~/.context --workspace-name src`
- `afs status`
- `afs workspace sync --root ~/src`
- `afs context init --path ~/src`
- `afs context validate --path ~/src`
- `afs context discover --path ~/src`
- `afs context ensure-all --path ~/src`
- `afs graph export --path ~/src`
- `afs services list`
- `afs orchestrator list`

## Installation
Recommended:
```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
afs status
```

Helper script: `scripts/afs-venv`

## Shell & Agent Access
- Source `scripts/afs-shell-init.sh` in bash/zsh.
- Use `scripts/afs` in non-interactive agents.
- Use `scripts/afs-warm` for periodic context warming.
- See `docs/AGENT_SURFACES.md` for CLI and MCP surfaces.

## Plugins
- Set `AFS_PLUGIN_DIRS` (colon-separated on macOS/Linux) to add local plugin folders.
- Set `AFS_ENABLED_PLUGINS` (comma or space separated) to load specific plugins.
- Defaults to `~/.config/afs/plugins` and `~/.afs/plugins` if present.

## JSON Output (Agent-Friendly)
```bash
afs status --json
afs context discover --path ~/src --json
afs context report --path ~/src --json
afs fs list memory --path ~/src --json
afs embeddings search --project afs --query "context root" --json
```

## Dataset Exports
**Memory/History:**
```bash
afs training memory-export --output ~/src/training/datasets/memory_export.jsonl
afs training history-export --output ~/src/training/datasets/history_export.jsonl
```

**Chat Logs:**
```bash
afs training antigravity-export --output ~/src/training/datasets/antigravity_export.jsonl
afs training gemini-export --output ~/src/training/datasets/gemini_export.jsonl
afs training claude-export --output ~/src/training/datasets/claude_export.jsonl
afs training codex-export --output ~/src/training/datasets/codex_export.jsonl
```

**Rebalance:**
```bash
afs training rebalance --input ... --output ...
```

## Background Services
```bash
afs services start memory-export
afs services stop memory-export
afs services start context-warm
```

## Agents
- `afs agents list`
- `afs agents run context-audit -- --path ~/src ...`
- `afs agents run scribe-draft -- --prompt "..."`

## Agentic Filesystem
- `afs fs read scratchpad state.md --path ~/src`
- `afs fs write scratchpad notes.md --path ~/src --content "..."`
- `afs fs list knowledge --path ~/src --glob "*.md"`

## Embeddings
- `afs embeddings index --project afs --source ~/src/lab/afs/docs --provider none`
- `afs embeddings search --project afs --query "context root"`

## Studio
- `afs-studio` (builds if needed)
- `afs studio install --prefix ~/.local`
- `afs studio alias`

## Domain Capabilities (ALTTP/65816)
- `afs.tokenizer` - Custom 65816 assembly tokenizer
```python
from afs.tokenizer import ASMTokenizer
print(ASMTokenizer().tokenize("LDA $7F00,X"))
```
