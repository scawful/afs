# Training Environment Setup

Complete guide for setting up the training environment to fine-tune MCP tool specialist models.

## Overview

This guide covers training three specialist models based on Qwen 2.5 Coder 32B with LoRA adapters:
- **VERAN-tools**: ROM analysis and inspection specialist
- **FARORE-debug**: Debugging and emulation specialist
- **NAYRU-editor**: Code generation and ROM editing specialist

## Hardware Requirements

### Minimum Requirements
- **GPU**: 20GB VRAM (RTX 3090, RTX 4090, A5000, or better)
- **RAM**: 16GB system memory (32GB recommended)
- **Storage**: 50GB free disk space (100GB recommended)
- **OS**: Linux or macOS (Windows with WSL2)

### Recommended Hardware
- **GPU**: RTX 4090 (24GB) or A100 (40GB)
- **RAM**: 32GB+ system memory
- **Storage**: 100GB+ SSD
- **Training time**: 2-4 hours per model on RTX 4090

### CPU-Only Warning
Training on CPU is **not recommended** and will take 50-100+ hours per model.

## Prerequisites

### 1. Python Version
Python 3.9 or higher required:
```bash
python3 --version  # Should show >= 3.9
```

### 2. CUDA Toolkit (for GPU training)
CUDA 11.8 or 12.1+ recommended:
```bash
nvidia-smi  # Check GPU availability
nvcc --version  # Check CUDA version
```

### 3. Git LFS (for large model files)
```bash
git lfs install
```

## Installation Steps

### Step 1: Install Python Dependencies

From the `training_data/tool_usage/` directory:

```bash
cd ~/src/lab/afs/training_data/tool_usage/
pip install -r requirements-training.txt
```

This installs:
- **torch** (>= 2.1.0): PyTorch deep learning framework
- **transformers** (>= 4.36.0): HuggingFace model library
- **peft** (>= 0.7.0): LoRA implementation
- **accelerate** (>= 0.25.0): Multi-GPU training utilities
- **datasets** (>= 2.15.0): Dataset loading and processing
- **wandb** (>= 0.16.0): Experiment tracking (optional)
- **bitsandbytes** (>= 0.41.0): 8-bit quantization (optional)

**Optional dependencies:**
- `deepspeed` for multi-GPU distributed training
- `auto-gptq` for inference optimization

### Step 2: Verify Environment

Run the verification script to check all requirements:

```bash
cd scripts/
python3 verify_training_setup.py
```

**Expected output:**
```
============================================================
    MCP Tool Specialist Training Environment Check
============================================================

Python Version:
  ✅ Python 3.11.5 (OK)

GPU Check:
  ✅ CUDA available (1 GPU(s))
     GPU 0: NVIDIA GeForce RTX 4090
            VRAM: 24.0 GB
            Compute: 8.9

Required Packages:
  ✅ PyTorch
  ✅ HuggingFace Transformers
  ✅ LoRA/PEFT
  ✅ Accelerate
  ✅ HF Datasets

Disk Space:
  Free space: 250.3 GB
  ✅ Sufficient space (>100GB)

System Memory:
  Total RAM: 64.0 GB
  Available: 48.2 GB
  ✅ Sufficient RAM (>= 32GB)

Training Data:
  ✅ Training set        694 examples (521.3 KB)
  ✅ Validation set       81 examples (61.2 KB)
  ✅ Test set             99 examples (74.8 KB)
  ✅ Split statistics    (0.6 KB)

============================================================
                         Summary
============================================================

  Python          ✅ PASS
  GPU             ✅ PASS
  Packages        ✅ PASS
  Disk            ✅ PASS
  Memory          ✅ PASS
  Data            ✅ PASS

  Score: 6/6 checks passed

============================================================
                       Next Steps
============================================================

✅ All critical checks passed!

Ready to start training. Run:

  python3 train_specialist.py \
    --model veran-tools \
    --train-data ../training_formatted/train.jsonl \
    --val-data ../training_formatted/val.jsonl \
    --output ../models/veran-tools-lora \
    --epochs 3
```

**Verbose mode** (shows package versions):
```bash
python3 verify_training_setup.py --verbose
```

### Step 3: Configure Weights & Biases (Optional)

For experiment tracking with W&B:

```bash
wandb login
```

Enter your API key when prompted. Training will automatically log metrics to W&B if `--wandb` flag is used.

## Training Execution

### Basic Training Command

Train a specialist model:

```bash
python3 train_specialist.py \
    --model <specialist-name> \
    --train-data ../training_formatted/train.jsonl \
    --val-data ../training_formatted/val.jsonl \
    --output ../models/<model-output-dir> \
    --epochs 3
```

### Training Each Specialist

#### 1. VERAN-tools (Analysis & Inspection)

```bash
python3 train_specialist.py \
    --model veran-tools \
    --train-data ../training_formatted/train.jsonl \
    --val-data ../training_formatted/val.jsonl \
    --output ../models/veran-tools-lora \
    --epochs 3 \
    --wandb  # Optional: enable W&B logging
```

**Focus tools:**
- `yaze_debugger.read_memory` - Read ROM/RAM data
- `z3ed_cli.inspect` - Inspect ROM structure
- `z3ed_cli.extract` - Extract graphics/data
- `mesen2.read_memory` - Read emulator memory

**Expected training time:** 2-4 hours on RTX 4090

#### 2. FARORE-debug (Debugging & Emulation)

```bash
python3 train_specialist.py \
    --model farore-debug \
    --train-data ../training_formatted/train.jsonl \
    --val-data ../training_formatted/val.jsonl \
    --output ../models/farore-debug-lora \
    --epochs 3
```

**Focus tools:**
- `mesen2.load_rom` - Load ROM into emulator
- `mesen2.run` - Control emulation (frames, speed)
- `mesen2.screenshot` - Capture visual state
- `yaze_debugger.read_memory` - Debug runtime state

**Expected training time:** 2-4 hours on RTX 4090

#### 3. NAYRU-editor (Code Generation & Editing)

```bash
python3 train_specialist.py \
    --model nayru-editor \
    --train-data ../training_formatted/train.jsonl \
    --val-data ../training_formatted/val.jsonl \
    --output ../models/nayru-editor-lora \
    --epochs 3
```

**Focus tools:**
- `yaze_debugger.write_memory` - Write patches to ROM
- `yaze_debugger.assemble` - Assemble 65816 code
- `z3ed_cli.import` - Import modified graphics/data
- `z3ed_cli.validate` - Validate ROM integrity

**Expected training time:** 2-4 hours on RTX 4090

### Advanced Training Options

#### Custom Hyperparameters

```bash
python3 train_specialist.py \
    --model veran-tools \
    --train-data ../training_formatted/train.jsonl \
    --val-data ../training_formatted/val.jsonl \
    --output ../models/veran-tools-lora \
    --epochs 3 \
    --batch-size 4 \              # Per-device batch size
    --gradient-accumulation 4 \   # Effective batch = 4*4 = 16
    --learning-rate 2e-5 \        # Learning rate
    --warmup-ratio 0.1 \          # Warmup steps
    --weight-decay 0.01 \         # L2 regularization
    --max-seq-length 2048         # Max sequence length
```

#### 8-bit Quantization (for GPUs with <20GB VRAM)

```bash
python3 train_specialist.py \
    --model veran-tools \
    --train-data ../training_formatted/train.jsonl \
    --val-data ../training_formatted/val.jsonl \
    --output ../models/veran-tools-lora \
    --epochs 3 \
    --use-8bit  # Load model in 8-bit (saves ~50% VRAM)
```

**Note:** 8-bit training is slower but enables training on GPUs with 12-16GB VRAM.

#### Different Base Model

```bash
python3 train_specialist.py \
    --model veran-tools \
    --base-model "Qwen/Qwen2.5-Coder-7B-Instruct" \  # Smaller model
    --train-data ../training_formatted/train.jsonl \
    --val-data ../training_formatted/val.jsonl \
    --output ../models/veran-tools-7b-lora \
    --epochs 3
```

**Available base models:**
- `Qwen/Qwen2.5-Coder-7B-Instruct` (smaller, faster)
- `Qwen/Qwen2.5-Coder-32B-Instruct` (default, recommended)

## Monitoring Training

### TensorBoard (Default)

Training logs are saved to TensorBoard by default:

```bash
tensorboard --logdir ../models/veran-tools-lora/
```

Open browser to http://localhost:6006

### Weights & Biases (Optional)

If using `--wandb` flag:
1. Training metrics logged to https://wandb.ai
2. Real-time loss curves, learning rate schedule
3. Hardware utilization (GPU, memory)
4. Hyperparameter comparison across runs

### Training Logs

Console output shows:
- Training loss per step (every 10 steps)
- Validation loss per epoch (every 50 steps)
- Estimated time remaining
- GPU utilization

**Example output:**
```
Epoch 1/3, Step 100/500, Loss: 0.234, LR: 1.8e-5, GPU: 22.1GB/24.0GB
Validation Loss: 0.198 (Best: 0.198)
Checkpoint saved: ../models/veran-tools-lora/checkpoint-100/
```

## Model Outputs

After training, each model directory contains:

```
models/veran-tools-lora/
├── adapter_config.json       # LoRA configuration
├── adapter_model.bin         # LoRA weights (~100MB)
├── specialist_config.json    # Specialist metadata
├── tokenizer_config.json     # Tokenizer config
├── special_tokens_map.json   # Special tokens
└── checkpoints/              # Training checkpoints
    ├── checkpoint-100/
    ├── checkpoint-200/
    └── checkpoint-300/
```

**Key files:**
- `adapter_model.bin`: LoRA weights to merge with base model
- `specialist_config.json`: System prompt, focus tools, training args

## Troubleshooting

### CUDA Out of Memory (OOM)

**Symptoms:**
```
RuntimeError: CUDA out of memory. Tried to allocate X GB
```

**Solutions:**
1. **Reduce batch size:**
   ```bash
   --batch-size 2 --gradient-accumulation 8
   ```

2. **Enable 8-bit quantization:**
   ```bash
   --use-8bit
   ```

3. **Reduce sequence length:**
   ```bash
   --max-seq-length 1024
   ```

4. **Enable gradient checkpointing (already default):**
   ```bash
   --gradient-checkpointing
   ```

### Slow Training Speed

**Symptoms:** Training takes >8 hours per model

**Solutions:**
1. **Check GPU utilization:**
   ```bash
   watch -n 1 nvidia-smi
   ```
   Should show ~90%+ GPU utilization

2. **Increase batch size (if VRAM allows):**
   ```bash
   --batch-size 8 --gradient-accumulation 2
   ```

3. **Use bfloat16 precision (already default):**
   Training uses `bf16=True` automatically

4. **Disable wandb if network is slow:**
   Remove `--wandb` flag

### Package Installation Errors

**bitsandbytes installation fails:**
```bash
# Install from source
pip install bitsandbytes --no-binary bitsandbytes
```

**CUDA version mismatch:**
```bash
# Reinstall PyTorch with correct CUDA version
pip install torch --index-url https://download.pytorch.org/whl/cu118  # For CUDA 11.8
pip install torch --index-url https://download.pytorch.org/whl/cu121  # For CUDA 12.1
```

### Data Loading Errors

**FileNotFoundError:**
- Ensure paths are relative to `scripts/` directory
- Use absolute paths if needed:
  ```bash
  --train-data ~/src/lab/afs/training_data/tool_usage/training_formatted/train.jsonl
  ```

**JSON parsing errors:**
- Verify JSONL format (one JSON object per line)
- Run validation:
  ```bash
  python3 -c "
  import json
  with open('../training_formatted/train.jsonl') as f:
      for i, line in enumerate(f, 1):
          try:
              json.loads(line)
          except json.JSONDecodeError as e:
              print(f'Line {i}: {e}')
  "
  ```

## Evaluation

After training, evaluate model on test set:

```bash
python3 evaluate_specialist.py \
    --model ../models/veran-tools-lora \
    --test-data ../training_formatted/test.jsonl \
    --output ../evaluation/veran-tools-results.json
```

**Metrics:**
- **Exact Match**: Tool name + all parameters match exactly
- **Tool Accuracy**: Correct tool selected (ignoring parameters)
- **Parameter F1**: Precision/recall on parameter values
- **Per-tool breakdown**: Metrics for each MCP tool

**Target metrics:**
- Exact Match: >90% for VERAN/FARORE, >95% for NAYRU
- Tool Accuracy: >95% for all specialists
- Parameter F1: >92% for all specialists

## Next Steps

1. **Train all three specialists** (6-12 hours total)
2. **Evaluate on test set** (see above)
3. **Deploy as MCP server** (integration with yaze/mesen2)
4. **A/B test against base model** (measure improvement)
5. **Iterate on training data** (add more examples if needed)

## Reference

**Training data statistics:**
- Total examples: 874
- Train: 694 (79.4%)
- Validation: 81 (9.3%)
- Test: 99 (11.3%)
- Format: OpenAI function calling
- Random seed: 42

**Model architecture:**
- Base: Qwen 2.5 Coder 32B Instruct
- Method: LoRA (r=16, alpha=32)
- Target modules: All linear layers (q, k, v, o, gate, up, down)
- Dropout: 0.05
- Training: ~0.5% of base model parameters

**Hardware used for testing:**
- MacBook Pro M1 Max (CPU-only, not recommended)
- RTX 4090 24GB (recommended)
- A100 40GB (ideal)

For questions or issues, see:
- `FINE_TUNING_PLAN.md`: Complete training strategy
- `COVERAGE_REPORT.md`: Training data analysis
- `../README.md`: Project overview
