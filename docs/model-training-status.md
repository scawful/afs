# Model Training Status

*Last updated: January 2026*

This document tracks all AI model training efforts in the AFS project, focusing on 65816 assembly and SNES hardware understanding.

## Model Overview

### Triforce MoE System

The core system consists of three specialized 65816 assembly experts named after Zelda's creation goddesses, plus an additional Oracle-themed expert:

| Model | Intent | Specialty | Base | Status |
|-------|--------|-----------|------|--------|
| **Din** | Optimization | Code size/cycle reduction | Qwen 7B + LoRA | Trained |
| **Nayru** | Generation | Assembly code writing | Qwen 7B + LoRA | Trained |
| **Farore** | Debugging | Bug finding and fixing | Qwen 7B + LoRA | Trained |
| **Veran** | Analysis | ROM analysis, SNES hardware explanation | Qwen 7B + LoRA | Trained |

### Model Deployment (Ollama)

| Model | Ollama Tag | Description |
|-------|------------|-------------|
| din-v3-fewshot | `din-v3-fewshot:latest` | Optimization with few-shot examples |
| nayru-v5 | `nayru-v5:latest` | Code generation specialist |
| farore-v1 | `farore-v1:latest` | Debugging expert |

---

## Din (Optimization Expert)

Din specializes in making 65816 assembly code faster and smaller - reducing cycles and bytes.

### Training Versions

#### Din v2 (Current)
- **Base Model:** Qwen2.5-Coder-3B-Instruct
- **Training Method:** LoRA via MLX
- **Training Platform:** Mac
- **Training Data:** ~120 optimization examples

**Files:**
- Adapters: `models/din-lora-adapters-v2/`
- Fused: `models/din-lora-fused-v2/`
- Training data: `models/din_optimization_training_v2.jsonl`

### Pattern Keywords for Routing
- "optimize", "faster", "smaller", "reduce cycles", "tighten"
- "make more efficient", "loop unroll", "inline"

---

## Nayru (Generation Expert)

Nayru specializes in writing NEW 65816 assembly code from scratch.

### Training Versions

#### Nayru v5 (Current)
- **Base Model:** Qwen2.5-Coder-7B-Instruct
- **Training Method:** LoRA
- **Ollama Tag:** `nayru-v5:latest`
- **Training Data:** ~5,000 generation examples

**Modelfile:** `scripts/Modelfile.nayru`

### System Prompt
```
You are Nayru, an expert in Zelda: A Link to the Past ROM hacking and SNES 65816 assembly programming.
```

### Pattern Keywords for Routing
- "write", "generate", "create", "implement", "code for"
- "give me code", "how to write"

---

## Farore (Debugging Expert)

Farore specializes in finding and fixing bugs in 65816 assembly code.

### Training Versions

#### Farore Cloud v1 (Current)
- **Base Model:** Qwen2.5-Coder-7B-Instruct
- **Training Method:** QLoRA (4-bit quantization)
- **Training Platform:** Vast.ai RTX 4090
- **Training Data:** 28 debugging examples
- **Training Time:** ~24 seconds
- **Training Cost:** ~$0.01
- **Ollama Tag:** `farore-v1:latest`
- **Temperature:** 0.4 (more deterministic for debugging)

**LoRA Configuration:**
```python
r=16, lora_alpha=32
target_modules=["q_proj", "v_proj"]
lora_dropout=0.05
```

**Training Metrics:**
- Initial loss: 2.11
- Final loss: 1.99
- Loss reduction: 6%

**Training Data Categories:**
| Category | Examples |
|----------|----------|
| Mode mismatch (REP/SEP) | 3 |
| DMA issues | 3 |
| Addressing bugs | 3 |
| Stack bugs | 3 |
| Interrupt bugs | 2 |
| Branch bugs | 2 |
| Comparison bugs | 2 |
| Register preservation | 2 |
| Direct page bugs | 2 |
| Hardware timing | 2 |
| MVN/MVP bugs | 2 |
| Accumulator size | 1 |
| Zero flag | 1 |

**Files:**
- Cloud adapters: `models/farore-cloud-adapters/`
- Training data: `models/farore_debugging_training.jsonl`
- Modelfile: `scripts/Modelfile.farore`

### Pattern Keywords for Routing
- "bug", "fix", "debug", "crash", "wrong", "not working"
- "why doesn't this work", "find the problem"

---

## Veran (Analysis/Explanation Expert)

Veran specializes in explaining 65816 assembly code, particularly SNES hardware register interactions. Named after the Sorceress of Shadows from Oracle of Ages.

### Training Versions

#### Veran Cloud v1 (Current Best)
- **Base Model:** Qwen2.5-Coder-7B-Instruct
- **Training Method:** QLoRA (4-bit quantization)
- **Training Platform:** Vast.ai RTX 5090
- **Training Data:** 90 examples (SNES hardware focus)
- **Training Time:** ~3.5 minutes
- **Training Cost:** ~$0.05

**LoRA Configuration:**
```python
r=16, lora_alpha=32
target_modules=["q_proj", "v_proj"]
lora_dropout=0.05
```

**Training Metrics:**
- Initial loss: 3.28
- Final loss: 0.67
- Loss reduction: 79%

**Evaluation Results:**
| Category | Score | Tests |
|----------|-------|-------|
| Basic 65816 | 67% | 3 |
| SNES Hardware | 30% | 5 |
| **Overall** | **44%** | 8 |

**Known Issues:**
- Confuses register names (e.g., calls $2100 "VMAIN" instead of "INIDISP")
- Training data insufficient for accurate register name mapping

**Files:**
- Cloud adapters: `models/veran-cloud-adapters/`
- Training data: `models/veran_snes_hardware.jsonl`

#### Veran Cloud v2 (Failed Experiment)
- **Base Model:** Qwen2.5-Coder-7B-Instruct
- **Training Method:** QLoRA (4-bit quantization)
- **Training Platform:** Vast.ai RTX 4090
- **Training Data:** 123 examples (register-emphasis format)
- **Training Time:** ~4 minutes
- **Training Cost:** ~$0.06

**Training Metrics:**
- Initial loss: 3.13
- Final loss: 0.51
- Loss reduction: 84%

**Evaluation Results:**
| Category | Score | Tests |
|----------|-------|-------|
| Basic 65816 | 33% | 3 |
| SNES Hardware | 5% | 7 |
| **Overall** | **13%** | 10 |

**What Went Wrong:**
The "register-emphasis" training format caused catastrophic forgetting:
- Model latched onto "CGADD" as default register name for everything
- $2100, $420B, $4200, $2115 all incorrectly called "CGADD"
- Putting register name FIRST in outputs backfired
- Too much pattern repetition, not enough variety

**Lesson Learned:**
- Lower training loss (0.51 vs 0.67) does not mean better model
- Repetitive formats can cause the model to memorize patterns incorrectly
- Need diverse example structures, not just more examples

**Files:**
- Cloud adapters: `models/veran-cloud-adapters-v2/`
- Training data: `models/veran_snes_hardware_v2.jsonl`
- Register-emphasis data: `models/veran_register_emphasis.jsonl`

#### Veran Mac v1 (Baseline)
- **Base Model:** Qwen2.5-Coder-3B-Instruct
- **Training Method:** LoRA via MLX
- **Training Platform:** Mac M-series
- **Training Data:** 146 examples (basic 65816)
- **Training Time:** ~45 minutes

**Results:**
- Basic 65816: 67%
- SNES Hardware: 0% (not trained on this)

**Files:**
- Adapters: `models/veran-lora-adapters/`
- Fused: `models/veran-lora-fused/`

### Pattern Keywords for Routing
- "explain", "what does this do", "analyze", "understand"
- "disassemble", "reverse engineer", "ROM analysis"

---

## MoE Routing Architecture

The Triforce MoE router classifies incoming queries and routes to the appropriate expert:

```
User Query
    |
    v
[Intent Classifier] -- keyword-based routing
    |
    +---> din (optimization)
    +---> nayru (generation)
    +---> farore (debugging)
    +---> veran (analysis) [planned]
    +---> fallback (general)
```

### Orchestrator

The `src/afs/moe/orchestrator.py` provides a Gemini-powered planner that:
1. Analyzes tasks with thinking
2. Creates execution plans
3. Dispatches to din/nayru/farore experts
4. Calls file/debugger tools
5. Synthesizes responses

---

## Training Infrastructure

### Cloud Training (Vast.ai)

**Workflow:**
1. Search for GPU: `vastai search offers 'gpu_name=RTX_5090'`
2. Create instance with PyTorch image
3. Attach SSH key and upload training data
4. Run training script
5. Download adapters
6. Destroy instance

**Automation:** `scripts/cloud-train.sh`

**Documentation:** `docs/cloud-training-workflow.md`

**Cost Reference:**
| GPU | VRAM | $/hr | Best For |
|-----|------|------|----------|
| RTX 3090 | 24GB | $0.10-0.20 | Budget 7B QLoRA |
| RTX 4090 | 24GB | $0.25-0.35 | Fast 7B training |
| RTX 5090 | 32GB | $0.28-0.40 | 7B-14B training |
| A100 | 80GB | $0.50-1.50 | 14B+ full precision |

### Local Training (Mac)

**Stack:**
- MLX for Apple Silicon optimization
- LoRA for efficient fine-tuning
- Ollama for deployment

**Limitations:**
- 3B models max for reasonable training time
- 7B models too slow for iterative development

### Windows Inference

**Stack:**
- PyTorch with CUDA
- BitsAndBytes for 4-bit quantization
- PEFT for LoRA loading

**Tested Hardware:**
- RTX 5060 Ti (16GB VRAM)
- Qwen2.5-Coder-7B with 4-bit: ~8GB VRAM usage

---

## Deployment Options

### Ollama (Recommended for local)
1. Merge LoRA into base model
2. Convert to GGUF format
3. Create Modelfile
4. Import: `ollama create <model> -f Modelfile`

### PyTorch (Windows/Linux CUDA)
1. Load base model with 4-bit quantization
2. Load PEFT adapters
3. Run inference

### MLX (Mac)
1. Use fused model directly
2. Or convert PyTorch adapters to MLX format

---

## Training Data

### Current Datasets

| Dataset | Examples | Purpose | Location |
|---------|----------|---------|----------|
| veran_snes_hardware.jsonl | 90 | SNES hardware registers | models/ |
| veran_explanation_training.jsonl | 146 | Basic 65816 | models/ |
| veran_combined_v2.jsonl | ~200 | Combined | models/ |
| din_optimization_training_v2.jsonl | 120 | Optimization patterns | models/ |
| train_validated_cleaned.jsonl | — | Validated CoT | models/ |

### Data Generation Tools

- `generators/` - CoT generation, augmentation
- `training/` - Format converters (MLX, Alpaca, ChatML)
- `tokenizer/` - Custom 65816 tokenizer

---

## Evaluation Framework

### Test Categories

1. **Basic 65816** - Register operations, addressing modes
2. **Intermediate** - 16-bit operations, stack manipulation
3. **SNES Hardware** - PPU, DMA, CPU registers

### Scoring

Keyword matching against expected terms:
- Score = (found keywords / expected keywords) * 100%

### Test Cases

See `scripts/eval_veran_cloud.py` for current test suite.

---

## Future Plans

### Short Term
- [ ] Improve Veran SNES hardware accuracy (30% -> 70%+)
- [ ] Create additional training data with register name focus
- [ ] Integrate Veran into MoE router as `QueryIntent.ANALYSIS`

### Medium Term
- [ ] Scale Veran to 14B base for better reasoning
- [ ] Evaluate ensemble approaches for complex queries
- [ ] Add Zelda (architecture) and Ganon (vulnerability) experts

### Long Term
- [ ] End-to-end code analysis pipeline
- [ ] Interactive assistant for ROM hacking
- [ ] Integration with YAZE debugger and Oracle-of-Secrets

---

## References

- [Cloud Training Workflow](cloud-training-workflow.md)
- [Personal AI Models Brainstorm](personal-ai-models-brainstorm.md)
- [Triforce MoE Expansion](triforce-moe-expansion.md)
