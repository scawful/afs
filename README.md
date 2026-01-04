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
- `python -m afs init --context-root ~/src/context --workspace-name trunk`
- `python -m afs status`
- `python -m afs workspace sync --root ~/src`
- `python -m afs context init --path ~/src/trunk`
- `python -m afs context validate --path ~/src/trunk`
- `python -m afs context discover --path ~/src/trunk`
- `python -m afs context ensure-all --path ~/src/trunk`
- `python -m afs graph export --path ~/src/trunk`
- `python -m afs services list`
- `python -m afs orchestrator list`

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

Agent-friendly JSON output:
```bash
afs status --json
afs context discover --path ~/src --json
afs context report --path ~/src --json
afs fs list memory --path ~/src/trunk --json
afs embeddings search --project afs --query "context root" --json
```

Memory export:
```bash
afs training memory-export --output ~/src/training/datasets/memory_export.jsonl
```

Background agent (manual control):
```bash
afs services start memory-export
afs services stop memory-export
afs services start context-warm
AFS_CONTEXT_WARM_INTERVAL=3600 afs services restart context-warm
```

Agents:
- `python -m afs agents list`
- `python -m afs agents run context-audit -- --path ~/src --output ~/.context/scratchpad/afs_agents/context_audit.json`
- `python -m afs agents run context-inventory -- --path ~/src --output ~/.context/scratchpad/afs_agents/context_inventory.json`
- `python -m afs agents run scribe-draft -- --prompt "Draft a concise changelog."`
- `python -m afs agents run context-warm -- --workspace-root ~/src`

Discovery skips directory names in `general.discovery_ignore` (default: legacy, archive, archives).

Agentic filesystem:
- `afs fs read scratchpad state.md --path ~/src/trunk`
- `afs fs write scratchpad notes.md --path ~/src/trunk --content "Status update"`
- `afs fs list knowledge --path ~/src/trunk --glob "*.md"`

Embeddings:
- `afs embeddings index --project afs --source ~/src/lab/afs/docs --provider none`
- `afs embeddings search --project afs --query "context root"`

Studio:
- `python -m afs studio build`
- `python -m afs studio run --build`
- `python -m afs studio install --prefix ~/.local`
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
