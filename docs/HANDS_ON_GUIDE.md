# AFS Hands-On Guide

_Created: 2026-01-11_

A practical guide to exploring and testing the finetuned expert models.

## System Overview

You've built a **Mixture of Experts** system for Oracle of Secrets development with:
- 4 GGUF models deployed to Mac + Windows
- AFS orchestrator for model invocation
- Network inference via LMStudio on Windows

### Infrastructure

| Component | Location | Status |
|-----------|----------|--------|
| Mac GGUFs | `~/src/lab/afs/models/gguf/` | Ready |
| Windows GGUFs | `C:\Users\starw\.lmstudio\models\` | Transferring |
| Orchestrator | `~/src/lab/afs/tools/orchestrator.py` | Ready |
| LoRA Adapters | `~/src/lab/afs/models/adapters/` | Backup only |

### Model Catalog

#### Production Models (Q8_0 GGUF)

| Model | Base | Size | Specialty |
|-------|------|------|-----------|
| **Nayru v9** | Qwen2.5-Coder-7B | 8.1GB | 65816 ASM generation |
| **Hylia v2** | Qwen2.5-7B | 8.1GB | Narrative, dialogue, lore |
| **Agahnim v2** | Qwen2.5-Coder-7B | 8.1GB | asar integration, namespaces |
| **Router v2** | Qwen2.5-3B | 3.3GB | Task classification/routing |

#### Adapter-Only Models (in `adapters/`)

| Model | Specialty | Notes |
|-------|-----------|-------|
| Farore v5 | Assembly debugging | Curated 70-sample dataset |
| Veran v4 | SNES hardware | PPU/DMA/Mode 7 expertise |
| Majora v2 | OoS codebase patterns | Architecture knowledge |
| Din v4 | Code optimization | STZ, loop optimization |

#### Experimental Adapters

| Model | Base | Purpose |
|-------|------|---------|
| echo-gemma2-2b-v3 | Gemma 2B | Personal style echo |
| memory-qwen25-7b-v1 | Qwen2.5-7B | Context compression |
| muse-v3-* | Various | Creative writing |

---

## Hands-On Testing Checklist

### Phase 1: Basic Connectivity

- [ ] **Test LMStudio local server**
  ```bash
  # On Mac (after loading model in LMStudio)
  curl http://localhost:1234/v1/models | jq
  ```

- [ ] **Test LMStudio remote server** (from Mac to Windows)
  ```bash
  curl http://medical-mechanica:1234/v1/models | jq
  ```

- [ ] **Test orchestrator basic call**
  ```bash
  python3 ~/src/lab/afs/tools/orchestrator.py --agent nayru-lm \
    --prompt "Write a simple LDA/STA loop"
  ```

### Phase 2: Model Capabilities

#### Nayru (Code Generation)
- [ ] Generate a simple sprite state machine
  ```bash
  python3 ~/src/lab/afs/tools/orchestrator.py --agent nayru-lm \
    --prompt "Generate 65816 code for a sprite that bounces between two positions"
  ```

- [ ] Generate item collection handler
  ```bash
  python3 ~/src/lab/afs/tools/orchestrator.py --agent nayru-lm \
    --prompt "Generate pickup code for a heart piece that checks if Link has all 4 pieces"
  ```

#### Hylia (Narrative/Creative)
- [ ] Dream sequence concept
  ```bash
  python3 ~/src/lab/afs/tools/orchestrator.py --agent hylia \
    --prompt "Write a dream sequence where Link glimpses a masked figure in a mirror"
  ```

- [ ] NPC dialogue
  ```bash
  python3 ~/src/lab/afs/tools/orchestrator.py --agent hylia \
    --prompt "Write dialogue for an old man who hints at the Time System"
  ```

#### Agahnim (asar/Build)
- [ ] Hook template
  ```bash
  python3 ~/src/lab/afs/tools/orchestrator.py --agent agahnim \
    --prompt "Create an asar hook that intercepts item collection at $09B110"
  ```

- [ ] Namespace setup
  ```bash
  python3 ~/src/lab/afs/tools/orchestrator.py --agent agahnim \
    --prompt "Set up namespace for a new dungeon with proper bank allocation"
  ```

### Phase 3: Remote Inference (Windows)

- [ ] **Start LMStudio server on Windows**
  1. Open LMStudio
  2. Load nayru-v9-q8_0.gguf
  3. Go to Local Server → Start

- [ ] **Test remote inference from Mac**
  ```bash
  python3 ~/src/lab/afs/tools/orchestrator.py --agent nayru-lm \
    --backend lmstudio-remote \
    --prompt "Generate OAM upload code for a 32x32 sprite"
  ```

- [ ] **Benchmark response time**
  ```bash
  time python3 ~/src/lab/afs/tools/orchestrator.py --agent nayru-lm \
    --backend lmstudio-remote \
    --prompt "Generate a DMA transfer routine"
  ```

### Phase 4: Multi-Model Workflows

- [ ] **Code + Review chain**
  ```bash
  # Generate with Nayru
  python3 ~/src/lab/afs/tools/orchestrator.py --agent nayru-lm \
    --prompt "Generate sprite collision detection" > /tmp/code.asm

  # Review with Agahnim (check namespaces)
  python3 ~/src/lab/afs/tools/orchestrator.py --agent agahnim \
    --prompt "Review this code for namespace issues: $(cat /tmp/code.asm)"
  ```

- [ ] **Narrative + Code chain**
  ```bash
  # Concept with Hylia
  python3 ~/src/lab/afs/tools/orchestrator.py --agent hylia \
    --prompt "Describe a puzzle mechanic involving masks and time"

  # Then implement with Nayru based on the concept
  ```

### Phase 5: Integration Tests

- [ ] **Test with yaze-debugger MCP** (if configured)
  - Set breakpoint on sprite routine
  - Step through generated code
  - Verify RAM writes

- [ ] **Test with book-of-mudora MCP**
  - Search for similar patterns in codebase
  - Validate generated code matches conventions

---

## Quick Reference

### Orchestrator Agents

| Agent | Backend | Model File |
|-------|---------|------------|
| `nayru-lm` | lmstudio | nayru-v9-q8_0.gguf |
| `hylia` | lmstudio | hylia-v2-q8_0.gguf |
| `agahnim` | lmstudio | agahnim-v2-q8_0.gguf |
| `majora` | lmstudio | (load manually) |
| `farore-lm` | lmstudio | (needs GGUF conversion) |
| `veran-lm` | lmstudio | (needs GGUF conversion) |
| `din-lm` | lmstudio | (needs GGUF conversion) |

### Backend Options

```bash
--backend lmstudio        # Local Mac LMStudio
--backend lmstudio-remote # Windows LMStudio (medical-mechanica)
--backend ollama          # Local Ollama (legacy)
--backend ollama-remote   # Windows Ollama (legacy)
```

### LMStudio Model Paths

- **Mac**: `~/.lmstudio/models/` or load from anywhere
- **Windows**: `C:\Users\starw\.lmstudio\models\`

---

## Troubleshooting

### Model not loading
- Check GGUF file exists at expected path
- LMStudio may need manual model load (My Models → Load)
- Check GPU memory (8GB+ recommended for 7B Q8_0)

### Connection refused
- Verify LMStudio server is running (Local Server → Start)
- Check Windows firewall allows port 1234
- Test with `curl http://medical-mechanica:1234/v1/models`

### Jinja template errors
- The orchestrator uses completions fallback automatically
- Or set model prompt template to "ChatML" in LMStudio UI

### Slow responses
- Consider Q4_K_M quantization (4GB vs 8GB)
- Check GPU utilization in LMStudio
- Network latency for remote inference

---

## Next Steps

After completing basic testing:

1. **Convert remaining adapters to GGUF** (Farore, Veran, Din, Majora)
2. **Create evaluation benchmarks** for each model specialty
3. **Build automated routing** via Router model
4. **Integrate with Oracle of Secrets workflow** (oracle.org tasks)

---

## Training History (2026-01-10/11)

| Model | Dataset | Samples | Base | Training Time |
|-------|---------|---------|------|---------------|
| Nayru v9 | 65816 generation | 2,444 | Qwen2.5-Coder-7B | ~75 min |
| Hylia v2 | Narrative/lore | 222 | Qwen2.5-7B | ~7 min |
| Agahnim v2 | asar integration | 518 | Qwen2.5-Coder-7B | ~15 min |
| Router v2 | Task routing | 120 | Qwen2.5-3B | ~6 min |

All trained on vast.ai RTX 3090/4090 instances. LoRA config: r=16, α=32, dropout=0.05.
