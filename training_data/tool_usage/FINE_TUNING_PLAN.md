# Fine-Tuning Plan for MCP Tool Specialist Models

**Date:** 2026-01-08
**Dataset:** 854 training examples (yaze/z3ed/mesen2 MCP tools)
**Target Models:** VERAN-tools, FARORE-debug, NAYRU-editor

---

## Executive Summary

Fine-tune three specialist models (Qwen 2.5 Coder 32B base) for expert MCP tool calling:
- **VERAN-tools:** ROM analysis and inspection
- **FARORE-debug:** Debugging and emulator control
- **NAYRU-editor:** Code generation and ROM editing

**Expected improvements:**
- 15-25% better tool calling accuracy vs. base model
- 5-15x faster inference (32B vs. Claude Opus 4.5)
- 10-50x cheaper at scale
- >95% exact match on tool name + parameters

---

## 1. Dataset Preparation

### 1.1 Data Split

**Strategy:** Stratified split maintaining tool distribution

```
Total: 854 examples
├── Train: 683 examples (80%)
├── Val:    86 examples (10%)
└── Test:   85 examples (10%)
```

**Stratification criteria:**
- Tool type distribution
- Difficulty balance (simple/medium/complex)
- Source diversity (extracted vs. synthetic)

### 1.2 Format Conversion

Convert from custom JSON to training format:

**Input format (current):**
```json
{
  "id": "synthetic_yaze_read_001",
  "instruction": "Read OAM table to inspect sprite properties",
  "tool_calls": [{
    "tool": "yaze_debugger.read_memory",
    "parameters": {"address": "0x0300", "length": 64, "format": "hex"},
    "rationale": "Read ROM data to inspect sprite properties",
    "expected_output": "Hex bytes showing ROM content"
  }],
  "success_criteria": "Successfully read OAM property bytes"
}
```

**Output format (training):**
```json
{
  "messages": [
    {"role": "system", "content": "You are an expert ROM development assistant with access to MCP tools."},
    {"role": "user", "content": "Read OAM table to inspect sprite properties"},
    {"role": "assistant", "content": "", "tool_calls": [
      {
        "id": "call_1",
        "type": "function",
        "function": {
          "name": "yaze_debugger.read_memory",
          "arguments": "{\"address\": \"0x0300\", \"length\": 64, \"format\": \"hex\"}"
        }
      }
    ]}
  ]
}
```

### 1.3 Data Augmentation (Optional)

For coverage gaps identified:
- Generate 10-20 `yaze_debugger.assemble` examples
- Add 20-30 error handling scenarios
- Create 15-20 more complex workflows

**Augmentation script:**
```bash
python3 scripts/generate_synthetic_examples.py \
  --output examples/synthetic_batch3 \
  --schema schemas/mcp_tools_schema.json \
  --count 50 \
  --focus assemble,error_handling,complex
```

---

## 2. Model Selection & Architecture

### 2.1 Base Model

**Chosen:** Qwen/Qwen2.5-Coder-32B-Instruct

**Rationale:**
- Excellent code understanding (HumanEval: 92.7%)
- Strong tool calling capabilities
- Efficient 32B parameter count (fits on single RTX 4090)
- Apache 2.0 license (commercial use allowed)
- Already used for Triforce MoE agents

**Alternatives considered:**
- DeepSeek-Coder-33B: Similar performance, less tool-calling focus
- CodeLlama-34B: Older, lower tool-calling accuracy
- Mistral-Medium: Proprietary, API-only

### 2.2 Fine-Tuning Method

**Chosen:** LoRA (Low-Rank Adaptation)

**Configuration:**
```python
lora_config = {
    "r": 16,              # Rank (balance quality/efficiency)
    "lora_alpha": 32,     # Scaling factor
    "target_modules": [   # Which layers to adapt
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ],
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "CAUSAL_LM"
}
```

**Why LoRA:**
- 10-100x fewer trainable parameters (saves memory/time)
- Base model frozen (preserves general capabilities)
- Easy merging/switching between adapters
- Proven track record for tool calling tasks

---

## 3. Specialist Model Definitions

### 3.1 VERAN-tools (Analysis & Inspection)

**Purpose:** ROM analysis, structure inspection, data extraction

**Tool focus:**
- `yaze_debugger.read_memory` (61 examples)
- `z3ed_cli.inspect` (40 examples)
- `z3ed_cli.extract` (92 examples)
- `mesen2.read_memory` (137 examples)

**Training subset:** 330 examples (38.6% of dataset)

**System prompt:**
```
You are VERAN-tools, an expert ROM analysis assistant specializing in:
- Reading and interpreting ROM data structures
- Extracting graphics, text, and game data
- Inspecting ROM headers and metadata
- Memory analysis during emulation

Use MCP tools to provide accurate, detailed analysis of SNES ROM files.
```

**Evaluation metric:** Exact match on tool name + parameters (target: >95%)

### 3.2 FARORE-debug (Debugging & Emulation)

**Purpose:** Debugging workflows, emulator control, runtime testing

**Tool focus:**
- `mesen2.load_rom` (189 examples)
- `mesen2.run` (90 examples)
- `mesen2.screenshot` (189 examples)
- `yaze_debugger.read_memory` (61 examples)

**Training subset:** 529 examples (61.9% of dataset)

**System prompt:**
```
You are FARORE-debug, an expert debugging assistant specializing in:
- Loading ROMs into emulators for testing
- Controlling emulation (speed, frames, breakpoints)
- Capturing visual state via screenshots
- Debugging runtime behavior and crashes

Use MCP tools to efficiently debug and test SNES ROM modifications.
```

**Evaluation metric:** Exact match + emulation workflow correctness (target: >90%)

### 3.3 NAYRU-editor (Code Generation & Editing)

**Purpose:** ROM patching, code generation, data import/export

**Tool focus:**
- `yaze_debugger.write_memory` (157 examples)
- `yaze_debugger.assemble` (1 example - needs augmentation!)
- `z3ed_cli.import` (90 examples)
- `z3ed_cli.validate` (91 examples)

**Training subset:** 339 examples (39.7% of dataset)

**System prompt:**
```
You are NAYRU-editor, an expert ROM editing assistant specializing in:
- Writing patches to ROM memory
- Assembling 65816 code snippets
- Importing modified graphics and data
- Validating ROM integrity after changes

Use MCP tools to safely modify SNES ROM files with precision.
```

**Evaluation metric:** Exact match + validation pass rate (target: >95%)

---

## 4. Training Configuration

### 4.1 Hardware Requirements

**Per model training:**
- GPU: RTX 4090 (24GB VRAM) or A100 (40GB)
- RAM: 32GB system RAM
- Storage: 100GB for model + checkpoints
- Training time: ~2-4 hours per model

**Parallel training (all 3 models):**
- Use vast.ai or Lambda Labs GPU instances
- 3x RTX 4090 instances (~$3-4/hour total)
- Total cost: ~$12-16 for all three models

### 4.2 Hyperparameters

```python
training_args = {
    # Model & Data
    "model_name": "Qwen/Qwen2.5-Coder-32B-Instruct",
    "per_device_train_batch_size": 4,
    "per_device_eval_batch_size": 8,
    "gradient_accumulation_steps": 4,  # Effective batch size: 16

    # Optimization
    "learning_rate": 2e-5,
    "weight_decay": 0.01,
    "num_train_epochs": 3,
    "lr_scheduler_type": "cosine",
    "warmup_ratio": 0.1,

    # Precision & Efficiency
    "fp16": False,
    "bf16": True,  # Better for training stability
    "gradient_checkpointing": True,
    "optim": "adamw_torch_fused",

    # Logging & Checkpointing
    "logging_steps": 10,
    "eval_steps": 50,
    "save_steps": 100,
    "save_total_limit": 3,
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_exact_match",

    # Early Stopping
    "early_stopping_patience": 3,
    "early_stopping_threshold": 0.001
}
```

### 4.3 Training Script

**Location:** `training_data/tool_usage/scripts/train_specialist.py`

```bash
# Train VERAN-tools
python3 scripts/train_specialist.py \
  --model veran-tools \
  --data examples/veran_split/ \
  --output models/veran-tools-lora/ \
  --epochs 3

# Train FARORE-debug
python3 scripts/train_specialist.py \
  --model farore-debug \
  --data examples/farore_split/ \
  --output models/farore-debug-lora/ \
  --epochs 3

# Train NAYRU-editor
python3 scripts/train_specialist.py \
  --model nayru-editor \
  --data examples/nayru_split/ \
  --output models/nayru-editor-lora/ \
  --epochs 3
```

---

## 5. Evaluation Strategy

### 5.1 Metrics

**Primary metrics:**
1. **Exact Match (EM):** Tool name + all parameters match exactly
2. **Tool Accuracy:** Correct tool selected (ignoring parameters)
3. **Parameter F1:** Precision/recall on parameter correctness

**Secondary metrics:**
4. **Latency:** Time to first token (ms)
5. **Throughput:** Tokens per second
6. **Cost:** $/1M tokens (for production planning)

### 5.2 Test Scenarios

**Held-out test set (85 examples):**
- Stratified by tool type
- Includes simple, medium, complex examples
- Real-world workflows from Agahnim corpus

**Adversarial tests:**
- Ambiguous instructions (requires clarification)
- Invalid parameter values (should reject)
- Out-of-scope requests (should decline)

### 5.3 Comparison Baselines

| Model | Expected EM | Cost/1M | Latency |
|-------|-------------|---------|---------|
| Claude Opus 4.5 (baseline) | 74.2% | $15.00 | ~2000ms |
| Qwen 2.5 Coder 32B (base) | ~60% | $0.00 | ~200ms |
| **VERAN-tools (fine-tuned)** | **>90%** | **$0.00** | **~200ms** |
| **FARORE-debug (fine-tuned)** | **>90%** | **$0.00** | **~200ms** |
| **NAYRU-editor (fine-tuned)** | **>95%** | **$0.00** | **~200ms** |

---

## 6. Deployment Strategy

### 6.1 Model Serving

**Option A: Local (halext-nj server)**
- Ollama with LoRA adapters
- Swap adapters dynamically based on task
- Free (already own hardware)
- Latency: ~200ms

**Option B: Cloud (vast.ai on-demand)**
- Reserve GPU instance during development sessions
- Shutdown when idle
- Cost: ~$0.30/hour
- Latency: ~300ms (network overhead)

**Recommended:** Option A for development, Option B for scaling

### 6.2 Integration with Codex

**Router logic:**
```python
def route_to_specialist(task_description):
    # Analyze task intent
    if any(keyword in task_description.lower() for keyword in
           ['read', 'inspect', 'analyze', 'extract', 'examine']):
        return 'veran-tools'

    elif any(keyword in task_description.lower() for keyword in
             ['debug', 'test', 'emulate', 'run', 'screenshot']):
        return 'farore-debug'

    elif any(keyword in task_description.lower() for keyword in
             ['write', 'patch', 'edit', 'import', 'modify', 'assemble']):
        return 'nayru-editor'

    else:
        # Fallback to general model
        return 'default'
```

### 6.3 Fallback Strategy

If specialist fails (confidence < 0.8):
1. Try another specialist (overlap coverage)
2. Fall back to Claude Opus 4.5
3. Log failure for training data improvement

---

## 7. Timeline & Milestones

### Week 1: Data Preparation
- ✅ Day 1-2: Dataset validation (DONE)
- ✅ Day 2-3: Coverage analysis (DONE)
- ⏳ Day 4-5: Format conversion scripts
- ⏳ Day 6-7: Data augmentation (assemble examples)

### Week 2: Model Training
- ⏳ Day 8-9: Train VERAN-tools
- ⏳ Day 10-11: Train FARORE-debug
- ⏳ Day 12-13: Train NAYRU-editor
- ⏳ Day 14: Checkpoint evaluation

### Week 3: Evaluation & Refinement
- ⏳ Day 15-16: Run full test suite
- ⏳ Day 17-18: Hyperparameter tuning (if needed)
- ⏳ Day 19-20: Adversarial testing
- ⏳ Day 21: Final evaluation report

### Week 4: Deployment
- ⏳ Day 22-23: Deploy to Ollama on halext-nj
- ⏳ Day 24-25: Integrate with Codex router
- ⏳ Day 26-27: Production testing
- ⏳ Day 28: Documentation & handoff

**Total:** 4 weeks from data prep to production

---

## 8. Risk Mitigation

### Risk 1: Overfitting (small dataset)
**Mitigation:**
- Use LoRA (less prone to overfitting)
- Early stopping on validation loss
- Regularization (weight decay, dropout)
- Data augmentation to 1000+ examples

### Risk 2: Catastrophic forgetting
**Mitigation:**
- Freeze base model (LoRA adapters only)
- Include general examples in training
- Periodic evaluation on general benchmarks

### Risk 3: Tool hallucination (invalid tools)
**Mitigation:**
- Add validation layer (check against schema)
- Include negative examples (invalid tool usage)
- Fallback to base model if validation fails

### Risk 4: Hardware availability
**Mitigation:**
- Primary: halext-nj RTX 4090
- Backup: vast.ai on-demand instances
- Emergency: Google Colab Pro (slower but free)

---

## 9. Cost Analysis

### One-Time Costs

| Item | Cost |
|------|------|
| Data generation (vast.ai) | $0.05 (DONE) |
| Model training (3 models × 4 hours × $1/hr) | $12.00 |
| Evaluation compute | $2.00 |
| **Total one-time** | **$14.05** |

### Ongoing Costs (Self-Hosted)

| Item | Cost/Month |
|------|------------|
| Electricity (RTX 4090, 24/7) | ~$15 |
| halext-nj server maintenance | $0 (owned) |
| **Total monthly** | **$15** |

### Cost Savings vs. Claude Opus 4.5

**Scenario:** 10,000 tool calls/month

| Model | Cost/1M Tokens | Avg Tokens/Call | Monthly Cost |
|-------|----------------|-----------------|--------------|
| Claude Opus 4.5 | $15.00 | 500 | $75.00 |
| Specialist models (self-hosted) | $0.00 | 300 | $15.00 (electricity) |
| **Savings** | - | - | **$60/month (80% reduction)** |

**ROI:** Payback in <1 month at 10K calls/month

---

## 10. Success Criteria

### Minimum Viable Product (MVP)
- ✅ 854 training examples generated
- ✅ 100% schema validation
- ⏳ >85% exact match on test set
- ⏳ <500ms inference latency
- ⏳ Deployed to Ollama on halext-nj

### Production Ready
- ⏳ >90% exact match on test set
- ⏳ <300ms inference latency
- ⏳ Integrated with Codex router
- ⏳ Monitoring & logging in place
- ⏳ Fallback to Claude Opus 4.5 working

### Stretch Goals
- ⏳ >95% exact match (matches human expert)
- ⏳ Self-improvement loop (failure → training data)
- ⏳ Multi-turn tool calling support
- ⏳ Uncertainty quantification (confidence scores)

---

## 11. Next Steps

### Immediate (This Week)
1. Create format conversion script (`scripts/convert_to_training_format.py`)
2. Generate data splits (train/val/test)
3. Augment `yaze_debugger.assemble` examples (10-20 more)
4. Set up training environment (install dependencies)

### Short-Term (Next 2 Weeks)
1. Train VERAN-tools on vast.ai RTX 4090
2. Evaluate on held-out test set
3. Iterate based on failure analysis
4. Train FARORE-debug and NAYRU-editor

### Medium-Term (Next Month)
1. Deploy all three specialists to Ollama
2. Integrate with Codex MCP router
3. Production testing on Oracle of Secrets tasks
4. Collect production feedback for v2

---

**Document Status:** ✅ Complete
**Ready for:** Model training execution
**Author:** Claude Opus 4.5 (via afs training pipeline)
**Last Updated:** 2026-01-08
