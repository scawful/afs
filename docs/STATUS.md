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
- studio sources in afs_studio (standalone); legacy mirror in apps/studio

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

## Model Training Status

See [model-training-status.md](model-training-status.md) for detailed documentation.

**Summary:**

| Model | Intent | Base | Status | Ollama Tag |
|-------|--------|------|--------|------------|
| **Din** | Optimization | Qwen 7B + LoRA | ✅ Trained | `din-v3-fewshot:latest` |
| **Nayru** | Generation | Qwen 7B + LoRA | ✅ Trained | `nayru-v5:latest` |
| **Farore** | Debugging | Qwen 7B + LoRA | ✅ Trained | `farore-v1:latest` |
| **Veran** | Analysis | Qwen 7B + LoRA | ✅ Trained | Cloud adapters |

**Farore Training:**
| Version | Examples | Loss | Notes |
|---------|----------|------|-------|
| **v1** | 28 | 2.11 → 1.99 | Debugging examples (mode, DMA, stack, etc.) |

**Veran Training:**
| Version | Examples | Loss | Notes |
|---------|----------|------|-------|
| v1 | 90 | 3.28 → 0.67 | Basic coverage |
| v2 | 123 | 3.13 → 0.51 | Register-emphasis (failed - catastrophic forgetting) |

**Veran v1 Evaluation:**
- Basic 65816: 67%
- SNES Hardware: 30%
- Overall: 44%

**Veran v2 Status:** Failed experiment - model exhibited catastrophic forgetting, called all registers "CGADD". v1 remains current best.
