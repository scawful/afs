# AFS (Agentic File System)

Research-only. Not a product.

Scope: core AFS primitives + internal workflow tooling.

Provenance: avoid employer/internal sources; skip unclear origins.

Docs:
- `docs/ARCHITECTURE.md` - Module overview and training pipeline
- `docs/GLOSSARY.md` - Term definitions
- `docs/STATUS.md` - Current state
- `docs/ROADMAP.md` - Planned work
- `docs/REPO_FACTS.json`
- `docs/NARRATIVE.md`
- `docs/RESEARCH_SOURCES.md`
- `docs/CLI_PLAN.md`
- `docs/PORTING_MAP.md`
- `docs/WORKSPACE_INTEGRATION.md`

Quickstart:
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

Workspace catalog and navigation live in `ws`; `afs workspace sync` bridges
`WORKSPACE.toml` into AFS discovery.

Install CLI (recommended):
```bash
python3 -m venv .venv
. .venv/bin/activate
pip install -e .
afs status
```

Shell + agent access:
- Source `scripts/afs-shell-init.sh` in bash/zsh.
- Use `scripts/afs` in non-interactive agents.
- Use `scripts/afs-warm` for periodic context warming.
- See `docs/AGENT_SURFACES.md` for CLI and MCP surfaces.

Plugins (macOS/Linux friendly):
- Set `AFS_PLUGIN_DIRS` (colon-separated on macOS/Linux) to add local plugin folders.
- Set `AFS_ENABLED_PLUGINS` (comma or space separated) to load specific plugins.
- Plugins can be a simple package or single `.py` module; no compilation required.

Agent-friendly JSON output:
```bash
afs status --json
afs context discover --path ~/src --json
afs context report --path ~/src --json
afs fs list memory --path ~/src --json
afs embeddings search --project afs --query "context root" --json
```

Embedding eval:
```bash
afs embeddings eval --project afs --query-file examples/embedding_eval.jsonl --provider ollama --model nomic-embed-text --json
afs embeddings eval --project afs --query-file examples/embedding_eval.jsonl --provider hf --model google/embeddinggemma-300m --json
```

Eval JSONL format:
```json
{"query":"history export command","expected_path":"docs/MEMORY_SYSTEM.md"}
{"query":"embedding index CLI","expected":["src/afs/cli/embeddings.py","docs/AGENT_SURFACES.md"]}
```

Memory export:
```bash
afs training memory-export --output ~/src/training/datasets/memory_export.jsonl
```

History export:
```bash
afs training history-export --output ~/src/training/datasets/history_export.jsonl
```

Antigravity export:
```bash
afs training antigravity-export --output ~/src/training/datasets/antigravity_export.jsonl
```

Gemini export:
```bash
afs training gemini-export --output ~/src/training/datasets/gemini_export.jsonl
```

Claude export:
```bash
afs training claude-export --output ~/src/training/datasets/claude_export.jsonl
```

Codex export:
```bash
afs training codex-export --output ~/src/training/datasets/codex_export.jsonl
```

Codex history import:
```bash
afs training codex-history-import --history-root ~/.context/history
```

Rebalance datasets:
```bash
afs training rebalance --input ~/src/training/datasets/claude_export.jsonl \
  --input ~/src/training/datasets/gemini_export.jsonl \
  --input ~/src/training/datasets/codex_export.jsonl \
  --input ~/src/training/datasets/history_export.jsonl \
  --output ~/src/training/datasets/mix_export.jsonl \
  --weight gemini=0.35 --weight claude=0.30 --weight history=0.20 --weight codex=0.15
```

Background agent (manual control):
```bash
afs services start memory-export
afs services stop memory-export
afs services start context-warm
AFS_CONTEXT_WARM_INTERVAL=3600 afs services restart context-warm
```

Agents:
- `afs agents list`
- `afs agents run context-audit -- --path ~/src --output ~/.context/scratchpad/afs_agents/context_audit.json`
- `afs agents run context-inventory -- --path ~/src --output ~/.context/scratchpad/afs_agents/context_inventory.json`
- `afs agents run scribe-draft -- --prompt "Draft a concise changelog."`
- `afs agents run context-warm -- --workspace-root ~/src`

Discovery skips directory names in `general.discovery_ignore` (default: legacy, archive, archives).

Agentic filesystem:
- `afs fs read scratchpad state.md --path ~/src`
- `afs fs write scratchpad notes.md --path ~/src --content "Status update"`
- `afs fs list knowledge --path ~/src --glob "*.md"`

## Tooling Strategy: "Context as Files"
Agents interact with the world by reading/writing to `~/.context`.
- **Working Memory:** `~/.context/scratchpad/`
- **Tool Use:** Agents are trained to output shell commands (e.g., `ws find`) and file operations instead of just text.
- **Dataset:** Synthetic tooling examples in `training/datasets/afs_tooling_dataset.jsonl`.
- **See also:** `lab/afs/docs/TOOLING_STRATEGY.md` for the full architecture.

Embeddings:
- `afs embeddings index --project afs --source ~/src/lab/afs/docs --provider none`
- `afs embeddings search --project afs --query "context root"`
 - `--provider none` falls back to keyword scoring when no embedding runtime is available.

Studio:
- `afs-studio-build`
- `afs-studio` (builds if needed)
- `afs studio install --prefix ~/.local`
- `afs studio alias`
- Set `AFS_STUDIO_ROOT` to the standalone repo if it lives outside `apps/studio`.

Domain Capabilities (ALTTP/65816):
- `afs.generators` - Training data generation (CoT, augmentation)
- `afs.training` - Model training utilities (converters, registry)
- `afs.tokenizer` - Custom 65816 assembly tokenizer
- `afs.discriminator` - Quality filtering models
- `afs.knowledge` - ALTTP address tables

Example (ASM tokenizer):
```python
from afs.tokenizer import ASMTokenizer

tokenizer = ASMTokenizer()
encoded = tokenizer.encode("LDA $7F00,X")
print(tokenizer.tokenize("LDA $7F00,X"))  # ['LDA', '$7F00', ',X']
```
