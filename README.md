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
- `python -m afs workspace add --path ~/src/trunk --name trunk`
- `python -m afs context init --path ~/src/trunk`
- `python -m afs context validate --path ~/src/trunk`
- `python -m afs context discover --path ~/src/trunk`
- `python -m afs context ensure-all --path ~/src/trunk`
- `python -m afs graph export --path ~/src/trunk`
- `python -m afs services list`
- `python -m afs orchestrator list`

Agents:
- `python -m afs agents list`
- `python -m afs agents run context-audit -- --path ~/src --output ~/.context/scratchpad/afs_agents/context_audit.json`
- `python -m afs agents run context-inventory -- --path ~/src --output ~/.context/scratchpad/afs_agents/context_inventory.json`

Discovery skips directory names in `general.discovery_ignore` (default: legacy, archive, archives).

Studio:
- `python -m afs studio build`
- `python -m afs studio run --build`
- `python -m afs studio install --prefix ~/.local`

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
