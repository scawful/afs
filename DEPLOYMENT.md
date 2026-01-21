# AFS Agents - Deployment Guide

## 🎉 Successfully Trained Models

All 5 agents trained on Qwen2.5-Coder-3B-Instruct base model:

| Agent | Expertise | Training Samples | Location |
|-------|-----------|------------------|----------|
| **Nayru v6** | 65816 assembly & SNES code | 256 | `~/Downloads/afs_models/nayru-lora` |
| **Majora v1** | Oracle of Secrets codebase | 423 | `~/Downloads/afs_models/majora-lora` |
| **Din v2** | Code optimization | 399 | `~/Downloads/afs_models/din-lora` |
| **Farore v6** | Debugging specialist | 178 | `~/Downloads/afs_models/farore-lora` |
| **Veran v5** | SNES hardware expert | 611 | `~/Downloads/afs_models/veran-lora` |

## 🚀 Quick Start

### Interactive Chat

Chat with any agent interactively:

```bash
cd ~/src/lab/afs
source venv/bin/activate

# Chat with Nayru (assembly expert)
python3 chat_agent.py --agent nayru

# Chat with Majora (Oracle codebase expert)
python3 chat_agent.py --agent majora

# Chat with Din (optimization expert)
python3 chat_agent.py --agent din

# Chat with Farore (debugging expert)
python3 chat_agent.py --agent farore

# Chat with Veran (SNES hardware expert)
python3 chat_agent.py --agent veran
```

### Single Query

Run a single query without interactive mode:

```bash
python3 serve_agents.py --agent nayru \
  --prompt "Write a 65816 routine to copy memory" \
  --max-tokens 256
```

## 📋 Agent Details

### Nayru - Assembly Expert
**Trained on:** 256 samples of 65816 assembly code
**Best for:**
- Writing SNES/65816 assembly routines
- Explaining assembly code
- Optimizing assembly

**Example prompts:**
- "Write a DMA transfer routine for SNES"
- "Explain this 65816 code: LDA $2137"
- "How do I set up Mode 7 on SNES?"

### Majora - Oracle Codebase Expert
**Trained on:** 423 samples from Oracle of Secrets codebase
**Best for:**
- Understanding Oracle of Secrets architecture
- C# and Unity patterns used in Oracle
- Explaining Oracle code structure

**Example prompts:**
- "Where is sprite instantiation handled in Oracle?"
- "How does Oracle structure dialogue systems?"
- "Explain the Unity patterns used in Oracle"

### Din - Optimization Expert
**Trained on:** 399 code optimization samples
**Best for:**
- Performance optimization suggestions
- Code efficiency improvements
- Identifying bottlenecks

**Example prompts:**
- "How can I optimize this loop?"
- "What's causing performance issues here?"
- "Suggest optimizations for this code"

### Farore - Debugging Expert
**Trained on:** 178 debugging pattern samples
**Best for:**
- Identifying bugs
- Debugging strategies
- Error analysis

**Example prompts:**
- "Why is this code crashing?"
- "Help me debug this function"
- "What's wrong with this implementation?"

### Veran - SNES Hardware Expert
**Trained on:** 611 SNES hardware samples
**Best for:**
- SNES PPU operations
- DMA and HDMA
- Memory mapping
- Hardware registers

**Example prompts:**
- "How does HDMA work on SNES?"
- "Explain PPU register $2105"
- "What's the difference between DMA modes?"

## 🔧 Technical Details

**Base Model:** Qwen/Qwen2.5-Coder-3B-Instruct
**LoRA Config:**
- Rank (r): 16
- Alpha: 32
- Dropout: 0.05
- Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj

**Training:**
- Epochs: 3
- Batch size: 2
- Gradient accumulation: 4
- Learning rate: 2e-4
- Optimizer: paged_adamw_8bit
- Quantization: 4-bit (nf4)

**Hardware Used:**
- Training: NVIDIA RTX A4000/4090 (16GB VRAM) on vast.ai
- Inference: MacBook Pro M5 (32GB RAM, 2TB storage) with 4-bit quantization

## 📦 Model Sizes

Each LoRA adapter: ~120MB
Total adapters: ~600MB
Base model (cached): ~3.2GB

### Memory Requirements

**M5 MacBook Pro (32GB RAM):**
- Single model (4-bit): ~8GB RAM
- Multiple models: Can load 2-3 models simultaneously (~20-24GB total)
- Overhead: ~8GB for system and applications
- Context windows: 8K-16K tokens comfortable, 32K+ with optimization

**Previous Macs (16GB RAM):**
- Single model only, limited context windows
- Frequent memory pressure

## 🎯 Next Steps

### Immediate (M5 Enabled)
1. **Test all agents** - Try each agent with domain-specific prompts
2. **Multi-model experiments** - Load 2-3 models simultaneously to compare outputs
3. **Larger context testing** - Test 16K-32K token contexts with enhanced memory
4. **Local fine-tuning** - Experiment with LoRA training on 3B-7B models locally

### Medium-term
5. **Create API server** - Build FastAPI server for multi-agent access
6. **Integrate with tools** - Connect to debuggers, emulators, etc.
7. **MLX optimization** - Optimize models specifically for M5 Neural Engine
8. **Fine-tune further** - Add more training data for specialized tasks

## 🐛 Troubleshooting

**Out of memory?**
- M5 with 32GB: Should handle multiple models comfortably
- Models use 4-bit quantization by default
- Close other applications if running 3+ models
- Use `--max-tokens 256` for shorter responses
- Monitor memory with Activity Monitor (should stay < 24GB)

**Slow responses?**
- First load downloads base model (~3GB) - subsequent loads are fast
- Base model is cached in `~/.cache/huggingface/hub/`

**Adapter not found?**
- Check adapters are in `~/Downloads/afs_models/`
- Each should have: adapter_model.safetensors, adapter_config.json, tokenizer files

## 📝 Training Stats

**Session:** 2026-01-15
**Total training time:** ~2 hours
**Cost:** ~$6-8 on vast.ai
**Success rate:** 5/5 models trained successfully
**Total training samples:** 1,867

---

Generated: 2026-01-15
