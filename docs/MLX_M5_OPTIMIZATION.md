# MLX Optimization for M5 MacBook Pro

**Hardware:** MacBook Pro M5 (Mac17,2) - 32GB RAM, 2TB storage
**Last Updated:** 2026-01-16

## Overview

The M5 chip's Neural Engine and 32GB unified memory enable significantly better local AI/ML performance compared to previous generations. This guide covers optimization strategies for running and training models using Apple's MLX framework.

## Hardware Capabilities

### M5 Neural Engine
- **Unified Memory:** 32GB shared between CPU, GPU, and Neural Engine
- **Memory Bandwidth:** ~400GB/s (estimated)
- **Neural Engine Cores:** ~16 cores optimized for transformer operations
- **GPU Cores:** ~20-40 depending on M5 variant

### Practical Limits

| Task | Model Size | Memory Usage | Performance |
|------|-----------|--------------|-------------|
| Single model inference | 3B-7B | ~8GB | Excellent (>40 tok/s) |
| Multi-model inference | 2-3 models (7B each) | ~20-24GB | Good (>20 tok/s each) |
| LoRA training | 3B | ~12GB | Fast (~1-2h for 1k samples) |
| LoRA training | 7B | ~20GB | Moderate (~3-4h for 1k samples) |
| LoRA training | 13B+ | >28GB | Use remote GPUs |

## MLX Installation & Setup

### Install MLX

```bash
cd ~/src/lab/afs
source venv/bin/activate
pip install mlx mlx-lm
```

### Verify Installation

```bash
python3 -c "import mlx.core as mx; print(mx.metal.is_available())"
# Should print: True
```

## Model Inference

### Single Model

```bash
# Download and cache model (first time only)
mlx_lm.generate --model mlx-community/Qwen2.5-Coder-7B-Instruct-4bit \
  --prompt "Write a function to reverse a string" \
  --max-tokens 256

# Subsequent runs use cached model
mlx_lm.generate --model mlx-community/Qwen2.5-Coder-7B-Instruct-4bit \
  --prompt "Explain quicksort" \
  --max-tokens 256
```

**Performance:** ~40-60 tokens/second for 7B models

### Multiple Models Simultaneously

With 32GB RAM, you can load multiple models for comparison:

```python
from mlx_lm import load, generate
import mlx.core as mx

# Load models
model_a, tokenizer_a = load("mlx-community/Qwen2.5-Coder-7B-Instruct-4bit")
model_b, tokenizer_b = load("mlx-community/deepseek-r1-distill-qwen-7b-4bit")

# Generate from both
prompt = "Write a binary search function"

response_a = generate(model_a, tokenizer_a, prompt=prompt, max_tokens=256)
response_b = generate(model_b, tokenizer_b, prompt=prompt, max_tokens=256)

print(f"Qwen: {response_a}")
print(f"DeepSeek: {response_b}")
```

**Memory:** Each 7B-4bit model uses ~8GB, total ~24GB including overhead

### Recommended Models

| Model | Size | Memory | Use Case |
|-------|------|--------|----------|
| `Qwen2.5-Coder-7B-Instruct-4bit` | 7B | ~8GB | Code generation, debugging |
| `deepseek-r1-distill-qwen-7b-4bit` | 7B | ~8GB | Reasoning, complex tasks |
| `gemma-2-9b-it-4bit` | 9B | ~10GB | General purpose |
| `Qwen2.5-Coder-3B-Instruct-4bit` | 3B | ~4GB | Fast iteration, testing |

## LoRA Fine-Tuning

### 3B Models (Recommended for Fast Iteration)

```bash
cd ~/src/lab/afs
source venv/bin/activate

python3 scripts/mlx_train.py \
  --model mlx-community/Qwen2.5-Coder-3B-Instruct-4bit \
  --data training_data/splits/train.jsonl \
  --output models/nayru-mlx-lora \
  --iters 1000 \
  --steps-per-eval 100 \
  --batch-size 4 \
  --learning-rate 1e-5 \
  --lora-layers 16
```

**Performance:**
- Training speed: ~5-10 samples/second
- Total time: 1-2 hours for 1000 samples
- Memory: ~12GB (leaves 20GB for system)

### 7B Models (Optimal for Quality)

```bash
python3 scripts/mlx_train.py \
  --model mlx-community/Qwen2.5-Coder-7B-Instruct-4bit \
  --data training_data/splits/train.jsonl \
  --output models/nayru-7b-mlx-lora \
  --iters 1000 \
  --steps-per-eval 100 \
  --batch-size 2 \
  --learning-rate 1e-5 \
  --lora-layers 16 \
  --grad-checkpoint
```

**Performance:**
- Training speed: ~2-4 samples/second
- Total time: 3-4 hours for 1000 samples
- Memory: ~20GB (leaves 12GB for system)

**Important:** Close heavy applications (browsers, IDEs) during 7B training

### LoRA Configuration

Recommended LoRA settings for M5:

```python
# In training script
lora_config = {
    "r": 16,              # LoRA rank (higher = more capacity)
    "alpha": 32,          # LoRA scaling factor
    "dropout": 0.05,      # Regularization
    "target_modules": [   # Which layers to adapt
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ]
}
```

## Memory Management

### Monitor Memory Usage

```bash
# Real-time memory monitoring
watch -n 1 'top -l 1 | grep PhysMem'

# Or use Activity Monitor.app
open -a "Activity Monitor"
```

### Memory Pressure Indicators

| Memory Used | Status | Action |
|-------------|--------|--------|
| < 24GB | ✅ Healthy | Continue |
| 24-28GB | ⚠️ High | Close apps |
| > 28GB | 🔴 Critical | Stop training |

### Free Memory Before Training

```bash
# Close background apps
killall "Google Chrome"
killall "Slack"

# Verify available memory
top -l 1 | grep PhysMem
# Target: ~25-28GB free
```

## Optimization Tips

### 1. Quantization

Always use 4-bit quantization for inference and training:

```python
# Load quantized model
model = load("mlx-community/model-name-4bit")

# Or quantize existing model
from mlx_lm.utils import quantize
quantized_model = quantize(model, group_size=64, bits=4)
```

### 2. Batch Sizes

| Model Size | Batch Size | Gradient Accumulation |
|------------|------------|----------------------|
| 3B | 4 | 4 (effective = 16) |
| 7B | 2 | 8 (effective = 16) |
| 9B | 1 | 16 (effective = 16) |

### 3. Context Windows

| Model Size | Max Context | Recommended | Performance |
|------------|-------------|-------------|-------------|
| 3B | 8K | 4K | Excellent |
| 7B | 16K | 8K | Good |
| 7B | 32K | 16K | Slower (use if needed) |

### 4. Gradient Checkpointing

Enable for large models to reduce memory:

```python
# In training config
use_gradient_checkpointing = True  # Saves ~20% memory
```

### 5. Mixed Precision

MLX handles this automatically, but you can configure:

```python
import mlx.core as mx

# Set precision (default is mixed)
mx.set_default_device(mx.cpu)  # Or mx.gpu
```

## Integration with AFS Agents

### Using MLX Models with AFS

```bash
# Configure AFS to use MLX backend
export AFS_INFERENCE_BACKEND=mlx
export AFS_MODEL_CACHE=~/.cache/mlx

# Run agent with MLX
python3 chat_agent.py --agent nayru --backend mlx
```

### Multi-Agent Workflow

Load multiple specialized agents:

```python
from afs.agents import Agent

# Each agent uses different LoRA adapter
nayru = Agent("nayru", base_model="Qwen2.5-Coder-7B", lora="nayru-lora")
majora = Agent("majora", base_model="Qwen2.5-Coder-7B", lora="majora-lora")
din = Agent("din", base_model="Qwen2.5-Coder-7B", lora="din-lora")

# All share same base model in memory (~8GB)
# LoRA adapters add minimal overhead (~120MB each)
```

## Benchmarking

### Inference Speed

```bash
# Test tokens/second
mlx_lm.generate --model mlx-community/Qwen2.5-Coder-7B-Instruct-4bit \
  --prompt "Write a quicksort implementation" \
  --max-tokens 512 \
  --verbose
```

Expected performance:
- 3B models: 60-80 tok/s
- 7B models: 40-60 tok/s
- 9B+ models: 30-40 tok/s

### Training Speed

```bash
# Benchmark training iterations
python3 scripts/mlx_train.py \
  --model mlx-community/Qwen2.5-Coder-3B-Instruct-4bit \
  --data training_data/splits/train.jsonl \
  --iters 100 \
  --batch-size 4 \
  --benchmark
```

Expected performance:
- 3B: 5-10 samples/sec
- 7B: 2-4 samples/sec

## Troubleshooting

### Slow Inference

**Symptoms:** < 20 tok/s on 7B model

**Solutions:**
1. Ensure Metal backend is active:
   ```python
   import mlx.core as mx
   print(mx.metal.is_available())  # Should be True
   ```

2. Close background applications

3. Use smaller model (3B instead of 7B)

4. Reduce context window length

### Training OOM

**Symptoms:** Crash with "Cannot allocate memory"

**Solutions:**
1. Reduce batch size to 1
2. Enable gradient checkpointing
3. Use 3B model instead of 7B
4. Close all background apps

### Model Loading Slow

**Symptoms:** First load takes >5 minutes

**Solutions:**
1. Models download on first use (~3-8GB)
2. Subsequent loads use cache (~10 seconds)
3. Pre-download models:
   ```bash
   mlx_lm.convert --hf-path Qwen/Qwen2.5-Coder-7B-Instruct \
     -q --q-bits 4 \
     --upload-repo mlx-community/Qwen2.5-Coder-7B-Instruct-4bit
   ```

## Performance Comparison

### M5 vs Previous Generations

| Task | M1/M2 (16GB) | M5 (32GB) | Improvement |
|------|-------------|-----------|-------------|
| 7B inference | Possible but slow | 40-60 tok/s | 2-3x faster |
| Multi-model | Not viable | 2-3 models | ✅ Enabled |
| 7B training | Frequent OOM | Comfortable | ✅ Viable |
| Context window | 4K max | 8-16K | 2-4x larger |

### M5 vs Cloud GPUs

| Task | M5 (Local) | A100 40GB | A4000 16GB |
|------|-----------|-----------|------------|
| 3B training | ✅ Excellent | ⚡ Faster | ⚡ Faster |
| 7B training | ✅ Good | ⚡ Much faster | ⚡ Faster |
| 13B+ training | ❌ Use cloud | ✅ Good | ⚠️ Tight |
| Cost | $0 | $1.50-3/hr | $0.30-0.50/hr |
| Latency | None | Network | Network |

**Recommendation:** Use M5 for 3B-7B models, cloud for 13B+

## Next Steps

1. **Install MLX:** Follow installation instructions above
2. **Test inference:** Try single model inference with 7B model
3. **Multi-model test:** Load 2-3 models simultaneously
4. **Local training:** Fine-tune 3B model on small dataset (100 samples)
5. **Production training:** Once comfortable, train full datasets locally

## Resources

- [MLX Documentation](https://ml-explore.github.io/mlx/)
- [MLX Examples](https://github.com/ml-explore/mlx-examples)
- [MLX Community Models](https://huggingface.co/mlx-community)
- [AFS Training Guide](training.md)
- [AFS Deployment Guide](../DEPLOYMENT.md)

---

**Generated:** 2026-01-16
**Hardware:** MacBook Pro M5 (32GB RAM, 2TB storage)
