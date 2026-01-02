# STATUS

Stage: Prototype

## Core AFS

**Done:**
- init/status/workspace commands
- context init/list/mount/validate/discover/ensure-all
- graph export
- minimal config + plugin discovery
- service + orchestrator skeletons
- render-only launchd/systemd adapters
- pytest coverage for core modules
- studio sources in apps/studio

**Not yet:**
- service adapters that install/start services
- full orchestration pipeline
- TUI

**Next:**
- improve orchestrator routing heuristics
- add TUI starter screen

**Issues:**
- service runtime not wired to system services

## Domain Capabilities (ALTTP/65816)

**Done:**
- `generators/` - CoT generation, augmentation, asar validation, data cleaning
- `training/` - Converters (MLX, Alpaca, ChatML), splitter, registry
- `tokenizer/` - Custom 65816 tokenizer with HuggingFace compatibility
- `knowledge/` - ALTTP address tables
- `discriminator/` - ELECTRA scaffolding, fake generators

**In Progress:**
- `asm_trainer.py` - Encoder training integration (needs torch testing)

**Not yet:**
- End-to-end encoder training on GPU
- Integration of trained encoders with core AFS agents
- Embedding service for semantic search
