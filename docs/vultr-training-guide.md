# Cloud GPU Training Guide for Large Language Models

A practical guide for training larger language models on Vultr cloud GPU infrastructure, with alternatives comparison.

**Context:** User has trained a 7B LoRA model locally on Apple Silicon (16GB). Now exploring training larger models or full fine-tunes in the cloud.

---

## Table of Contents

1. [Vultr GPU Offerings](#vultr-gpu-offerings)
2. [Training Requirements by Model Size](#training-requirements-by-model-size)
3. [Training Frameworks](#training-frameworks)
4. [Cost Estimates](#cost-estimates)
5. [Vultr Workflow](#vultr-workflow)
6. [Cloud GPU Alternatives](#cloud-gpu-alternatives)
7. [Decision Framework](#decision-framework)

---

## Vultr GPU Offerings

### Available GPUs

| GPU | VRAM | CUDA Cores | Tensor Cores | FP16 TFLOPS | Best For |
|-----|------|------------|--------------|-------------|----------|
| **A16** | 64 GB total (vGPU) | 5,120 | 160 | ~72 (TF32) | VDI, light inference |
| **A40** | 48 GB GDDR6 | 10,752 | 336 | 149.7 | Training, visualization |
| **L40S** | 48 GB GDDR6 | 18,716 | 568 | 366 (TF32) | Mixed LLM workloads |
| **A100** | 80 GB HBM2e | 6,912 | 432 | 312 | Heavy training |
| **H100** | 80 GB HBM2e | 16,896 | 528 | 1,979 | Extreme training |
| **GH200** | 282 GB HBM3e | - | - | ~8 PFLOPS | Massive models |
| **H200** | 141 GB HBM3e | - | - | - | Next-gen training |

### Pricing (On-Demand)

| GPU | Hourly Rate | 36-Month Prepaid |
|-----|-------------|------------------|
| L40S | ~$1.67/hr | $0.85/hr |
| A40 | ~$1.71/hr | - |
| A100 PCIe 80GB | ~$1.29/hr | $1.29/hr |
| A100 HGX (multi-GPU) | - | $1.49/hr per GPU |
| H100 (single) | ~$2.99/hr | $2.30/hr |
| 8x H100 cluster | ~$23.90/hr total | - |

### Availability Regions

Vultr operates 32 global data center regions with GPU availability. A100s fluctuate regionally, but aggregate availability is generally good. Other GPU types (L40S, A40) have more abundant stock.

**GPU-Rich Regions:** US locations, Amsterdam, Frankfurt, Singapore, Tokyo.

---

## Training Requirements by Model Size

### VRAM Requirements Summary

| Model Size | Full Fine-Tune (FP16) | LoRA (FP16) | QLoRA (4-bit) |
|------------|----------------------|-------------|---------------|
| **7B** | ~55-70 GB | ~19 GB | ~5 GB |
| **14B** | ~110-140 GB | ~33 GB | ~8.5 GB |
| **32B** | ~250-320 GB | ~76 GB | ~26 GB |
| **70B** | ~500-600 GB | ~164 GB | ~41 GB |

### 7B Model Training

**Full Fine-Tune:**
- VRAM needed: ~55-70 GB (with AdamW optimizer states)
- Recommended: 1x A100 80GB
- Training time: ~3 hours for 50k examples (3 epochs)
- Vultr cost estimate: ~$4-6 for complete training

**LoRA Fine-Tune:**
- VRAM needed: ~14-19 GB
- Feasible on: A40 (48GB), L40S (48GB), or even local 24GB GPUs
- Training time: 2-4 hours for typical dataset
- Vultr cost estimate: ~$3-7

**QLoRA Fine-Tune:**
- VRAM needed: ~5 GB
- Can run on any Vultr GPU (or continue locally on M1)
- Training time: Similar to LoRA, potentially slightly slower

### 14B Model Training

**Full Fine-Tune:**
- VRAM needed: ~110-140 GB
- Requires: 2x A100 80GB with model parallelism
- Consider: DeepSpeed ZeRO-3 or FSDP
- Vultr cost estimate: ~$15-30 for complete run

**LoRA/QLoRA:**
- LoRA (16-bit): ~33 GB - fits on 1x A100 or L40S with careful memory management
- QLoRA (4-bit): ~8.5 GB - fits easily on single GPU
- Training time: 6-12 hours depending on dataset

### 32B Model Training

**Full Fine-Tune:**
- VRAM needed: ~250-320 GB
- Requires: 4x A100 80GB minimum, or 2x H100
- Must use: DeepSpeed ZeRO-3 with CPU offloading, or FSDP
- Consider: Vultr H100 cluster

**LoRA/QLoRA:**
- LoRA (16-bit): ~76 GB - 1x A100 is tight, 2x A100 recommended
- QLoRA (4-bit): ~26 GB - fits on single A40/L40S
- Training time: 12-24 hours

### 70B Model Training

**Full Fine-Tune:**
- VRAM needed: ~500-600+ GB
- Requires: 8x A100 or 8x H100 cluster
- Essential: DeepSpeed ZeRO-3 + CPU offloading, or ZeRO-Infinity with NVMe
- Vultr cost estimate: $100-500+ depending on training duration

**LoRA (16-bit):**
- VRAM needed: ~164 GB
- Requires: 2-4x A100 80GB with DeepSpeed/FSDP

**QLoRA (4-bit):**
- VRAM needed: ~41 GB
- Can fit on: 1x A100 80GB or 1x L40S 48GB
- This is the practical path for 70B on a budget
- Training time: 24-48+ hours

---

## Training Frameworks

### Framework Comparison

| Framework | Best For | Multi-GPU | Memory Efficiency | Ease of Use |
|-----------|----------|-----------|-------------------|-------------|
| **Unsloth** | Single GPU, speed | No | Excellent (2-5x faster, 80% less VRAM) | High |
| **Axolotl** | Flexibility, production | Yes (DeepSpeed/FSDP) | Good | High |
| **DeepSpeed** | Large scale, extreme models | Yes | Best (ZeRO-Infinity) | Medium |
| **FSDP** | PyTorch native, multi-GPU | Yes | Good | Medium |
| **LLaMA-Factory** | Beginners, UI | Yes | Good | Highest |

### Unsloth

Best for single-GPU training. Offers 2-5x speedup and 80% VRAM reduction.

```bash
pip install unsloth
```

**Limitations:** No multi-GPU support. Use for 7B-32B QLoRA on single GPU.

**When to use:** Budget training, fast iteration, consumer hardware.

### Axolotl

Production-grade, flexible framework with excellent YAML-based configuration.

```bash
pip install axolotl
```

**Key features:**
- Built-in DeepSpeed ZeRO-2/3 and FSDP support
- Sequence parallelism for long contexts
- Pre-configured YAML templates for popular models
- LoRA optimizations (Feb 2025 update)

**Example QLoRA config (examples/llama-3/qlora.yml):**

```yaml
base_model: meta-llama/Meta-Llama-3.1-8B
model_type: LlamaForCausalLM
tokenizer_type: AutoTokenizer

load_in_4bit: true
adapter: qlora

datasets:
  - path: your-dataset
    type: alpaca

sequence_len: 4096
sample_packing: true
pad_to_sequence_len: true

lora_r: 32
lora_alpha: 16
lora_dropout: 0.05
lora_target_linear: true

gradient_accumulation_steps: 4
micro_batch_size: 2
num_epochs: 3
learning_rate: 2e-4

output_dir: ./output
```

Run training:
```bash
axolotl train config.yml
```

### DeepSpeed

Microsoft's library for training at extreme scale.

**ZeRO Stages:**
- **ZeRO-1:** Optimizer state partitioning
- **ZeRO-2:** + Gradient partitioning
- **ZeRO-3:** + Parameter partitioning (full sharding)
- **ZeRO-Infinity:** + CPU/NVMe offloading

**When to use:** 70B+ models, multi-node clusters, memory-constrained scenarios.

**DeepSpeed config example (ds_config.json):**

```json
{
  "bf16": {"enabled": true},
  "zero_optimization": {
    "stage": 3,
    "offload_optimizer": {"device": "cpu"},
    "offload_param": {"device": "cpu"},
    "overlap_comm": true,
    "contiguous_gradients": true
  },
  "gradient_accumulation_steps": 4,
  "train_micro_batch_size_per_gpu": 1
}
```

### FSDP (Fully Sharded Data Parallel)

PyTorch-native alternative to DeepSpeed ZeRO-3.

**Advantages:**
- Up to 5x faster per iteration than DeepSpeed ZeRO-3 in some benchmarks
- Simpler PyTorch integration
- Good for models that fit with sharding alone

**Disadvantages:**
- All-or-nothing offloading (no partial CPU offload)
- Less feature-rich than DeepSpeed for extreme cases

**When to use:** Multi-GPU training where DeepSpeed's advanced features aren't needed.

### FSDP vs DeepSpeed Decision Matrix

| Scenario | Recommended |
|----------|-------------|
| Single GPU | Neither (use standard training or Unsloth) |
| 2-4 GPUs, model fits with sharding | FSDP (faster, simpler) |
| 4+ GPUs, needs CPU offloading | DeepSpeed ZeRO-3 |
| 8+ GPUs, 70B+ models | DeepSpeed ZeRO-Infinity |
| NVMe offloading needed | DeepSpeed only |

---

## Cost Estimates

### Realistic Training Scenarios

**Scenario 1: 7B LoRA Fine-Tune (50k examples, 3 epochs)**
- GPU: 1x A100 80GB
- Time: ~3 hours
- Cost: **$3.87** (at $1.29/hr)

**Scenario 2: 7B Full Fine-Tune**
- GPU: 1x A100 80GB
- Time: ~6-8 hours
- Cost: **$7.74-$10.32**

**Scenario 3: 14B QLoRA Fine-Tune**
- GPU: 1x A100 80GB or L40S
- Time: ~8-12 hours
- Cost: **$10-$20**

**Scenario 4: 14B Full Fine-Tune**
- GPU: 2x A100 80GB (HGX)
- Time: ~12-18 hours
- Cost: **$35-$55**

**Scenario 5: 32B QLoRA Fine-Tune**
- GPU: 1x A100 80GB
- Time: ~18-24 hours
- Cost: **$23-$31**

**Scenario 6: 70B QLoRA Fine-Tune**
- GPU: 1x A100 80GB (tight) or 2x A100
- Time: ~24-48 hours
- Cost: **$31-$124**

**Scenario 7: 70B Full Fine-Tune**
- GPU: 8x H100 cluster
- Time: 48-100+ hours
- Cost: **$1,147-$2,390** (at $23.90/hr for cluster)

### Budget-Conscious Recommendations

1. **Start with QLoRA** - Reduces VRAM by 75%+
2. **Use reserved instances** - 50% discount on 36-month prepaid
3. **Checkpoint frequently** - Enable spot instance recovery
4. **Start small** - Validate on subset before full training

---

## Vultr Workflow

### 1. Account Setup

```bash
# Sign up at vultr.com
# Add payment method
# Enable Cloud GPU in account settings
```

### 2. Deploy GPU Instance

**Via Web Console:**
1. Products > Cloud GPU
2. Select region with availability
3. Choose GPU type (A100 recommended for training)
4. Select "NVIDIA NGC" marketplace application
5. Choose Ubuntu 22.04 base OS
6. Add SSH key
7. Deploy

**Instance Specs for Training:**
- A100 80GB: 12 vCPUs, 120GB RAM, 100GB NVMe
- H100: 28 vCPUs, 240GB RAM, 200GB NVMe

### 3. Initial Setup

```bash
# SSH into instance
ssh root@<your-ip>

# Verify GPU
nvidia-smi

# Update system
apt update && apt upgrade -y

# Install Miniconda (if not using NGC)
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
source ~/.bashrc
```

### 4. Environment Setup

```bash
# Create environment
conda create -n train python=3.11
conda activate train

# Install PyTorch with CUDA
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install training framework
pip install axolotl  # or unsloth, etc.

# Install additional dependencies
pip install transformers accelerate bitsandbytes peft
pip install deepspeed  # if using multi-GPU
pip install flash-attn --no-build-isolation
```

### 5. Data Preparation

```bash
# Create data directory
mkdir -p /root/data /root/output

# Upload dataset (from local machine)
scp dataset.json root@<your-ip>:/root/data/

# Or download from Hugging Face
python -c "from datasets import load_dataset; ds = load_dataset('your-dataset'); ds.save_to_disk('/root/data/dataset')"
```

### 6. Training Configuration

Create `config.yml`:

```yaml
base_model: meta-llama/Meta-Llama-3.1-8B
model_type: LlamaForCausalLM
tokenizer_type: AutoTokenizer

load_in_4bit: true
adapter: qlora

datasets:
  - path: /root/data/dataset.json
    type: alpaca

output_dir: /root/output

sequence_len: 4096
sample_packing: true

lora_r: 64
lora_alpha: 32
lora_dropout: 0.05
lora_target_linear: true

gradient_accumulation_steps: 4
micro_batch_size: 4
num_epochs: 3
learning_rate: 2e-4

warmup_ratio: 0.03
save_strategy: steps
save_steps: 500
logging_steps: 10

bf16: true
tf32: true
gradient_checkpointing: true
```

### 7. Run Training

```bash
# Single GPU
axolotl train config.yml

# Multi-GPU with DeepSpeed
accelerate launch --config_file ds_config.yaml axolotl train config.yml

# Background with logging
nohup axolotl train config.yml > training.log 2>&1 &
tail -f training.log
```

### 8. Monitor Training

```bash
# Watch GPU utilization
watch -n 1 nvidia-smi

# TensorBoard (optional)
pip install tensorboard
tensorboard --logdir /root/output --bind_all
# Access via http://<your-ip>:6006
```

### 9. Export and Download

```bash
# Merge LoRA adapter with base model
python -m axolotl.cli.merge_lora config.yml --lora_model_dir="/root/output"

# Compress for download
cd /root/output
tar -czvf model.tar.gz merged/

# Download (from local machine)
scp root@<your-ip>:/root/output/model.tar.gz ./
```

### 10. Cleanup

```bash
# Destroy instance when done (stop billing)
# Via web console: Server > Destroy

# Or keep for future use
# Via web console: Server > Halt (still charges for storage)
```

### Docker Alternative

```bash
# Use NGC PyTorch container
docker run --gpus all -it --rm \
  -v /root/data:/workspace/data \
  -v /root/output:/workspace/output \
  nvcr.io/nvidia/pytorch:24.01-py3

# Inside container
pip install axolotl transformers accelerate bitsandbytes peft
axolotl train /workspace/config.yml
```

---

## Cloud GPU Alternatives

### Provider Comparison

| Provider | H100/hr | A100 80GB/hr | A100 40GB/hr | RTX 4090/hr | Key Feature |
|----------|---------|--------------|--------------|-------------|-------------|
| **Vultr** | $2.99 | $1.29 | - | - | 32 global regions |
| **Lambda Labs** | $2.99 | $1.10 | - | - | ML-optimized, academic discount |
| **RunPod** | $1.99 | ~$1.00 | ~$0.80 | $0.34 | Per-second billing, templates |
| **Vast.ai** | $1.87 | $0.50 | - | $0.24-0.60 | P2P marketplace, cheapest |
| **AWS** | - | ~$4.10 | ~$3.10 | - | Enterprise, spot instances |

### When to Use Each

**Vultr:**
- Global deployment needs
- Need bare metal control
- Want consistent availability
- Production workloads

**Lambda Labs:**
- Academic research (50% discount)
- Pre-configured ML environment
- Need free egress (no download fees)
- Enterprise LLM training

**RunPod:**
- Fast iteration (200ms cold starts)
- Serverless deployment
- Container-native workflows
- Budget-conscious training

**Vast.ai:**
- Lowest cost priority
- Can handle variable reliability
- Training with checkpoints (resumable)
- Experimentation and research

### Cost Optimization Strategies

1. **Spot/Preemptible Instances**
   - Vast.ai: Built-in bidding
   - RunPod: Community cloud (cheaper, less reliable)
   - AWS: Spot instances (up to 90% savings)

2. **Reserved Capacity**
   - Vultr: 36-month prepaid (~50% off)
   - Lambda Labs: Reserved clusters

3. **Right-sizing**
   - Use QLoRA to reduce GPU requirements
   - Start on smaller GPUs, scale up only if needed

4. **Efficient Training**
   - Use Unsloth for 2-5x speedup (single GPU)
   - Enable gradient checkpointing
   - Use mixed precision (bf16/fp16)
   - Sample packing for shorter sequences

---

## Decision Framework

### When to Train Locally vs Cloud

**Stay Local (Apple Silicon 16GB):**
- 7B QLoRA fine-tuning
- Small datasets (<10k examples)
- Experimentation and hyperparameter search
- Cost: $0

**Go to Cloud:**
- Full fine-tuning of any size
- 14B+ models (even QLoRA)
- Large datasets (100k+ examples)
- Production training runs
- Multi-GPU requirements

### Recommended Path by Budget

**Tight Budget (<$50):**
1. Start with Vast.ai for experimentation
2. Use QLoRA exclusively
3. Checkpoint every 500 steps
4. Train in shorter sessions

**Moderate Budget ($50-200):**
1. Use Vultr A100 or Lambda Labs
2. QLoRA for 14B-32B models
3. LoRA for 7B models
4. Full fine-tune for 7B if needed

**Production Budget ($500+):**
1. Vultr or Lambda Labs reserved instances
2. Full fine-tuning up to 14B
3. Multi-GPU for 32B+
4. H100 clusters for 70B

### Quick Reference Card

```
7B Model:
  - QLoRA: Any GPU, ~$5, 2-4 hours
  - Full: 1x A100, ~$10, 6-8 hours

14B Model:
  - QLoRA: 1x A100/L40S, ~$15, 8-12 hours
  - Full: 2x A100, ~$45, 12-18 hours

32B Model:
  - QLoRA: 1x A100, ~$30, 18-24 hours
  - Full: 4x A100, ~$150+, 24-48 hours

70B Model:
  - QLoRA: 1-2x A100, ~$80, 24-48 hours
  - Full: 8x H100, ~$1500+, 48-100 hours
```

---

## References

### Vultr Documentation
- [Vultr Cloud GPU Overview](https://www.vultr.com/products/cloud-gpu/)
- [Vultr GPU Pricing](https://www.vultr.com/pricing/)
- [NVIDIA A100 on Vultr](https://www.vultr.com/products/cloud-gpu/nvidia-a100/)
- [Deploy PyTorch on Vultr GPU](https://docs.vultr.com/deploy-a-pytorch-workspace-on-a-vultr-cloud-gpu-server)
- [Deploy LLMs with OpenLLM](https://docs.vultr.com/how-to-deploy-large-language-models-on-vultr-cloud-gpu-using-openllm)

### Training Frameworks
- [Axolotl GitHub](https://github.com/axolotl-ai-cloud/axolotl)
- [Axolotl Documentation](https://docs.axolotl.ai/)
- [Unsloth Documentation](https://docs.unsloth.ai/)
- [DeepSpeed GitHub](https://github.com/microsoft/DeepSpeed)
- [FSDP vs DeepSpeed Comparison](https://huggingface.co/docs/accelerate/en/concept_guides/fsdp_and_deepspeed)

### VRAM Requirements
- [Modal: How Much VRAM for Fine-Tuning](https://modal.com/blog/how-much-vram-need-fine-tuning)
- [RunPod: LLM Fine-Tuning GPU Guide](https://www.runpod.io/blog/llm-fine-tuning-gpu-guide)
- [Unsloth Requirements](https://docs.unsloth.ai/get-started/beginner-start-here/unsloth-requirements)

### Framework Comparisons
- [Spheron: Axolotl vs Unsloth vs Torchtune](https://blog.spheron.network/comparing-llm-fine-tuning-frameworks-axolotl-unsloth-and-torchtune-in-2025)
- [Modal: Best Fine-Tuning Frameworks 2025](https://modal.com/blog/fine-tuning-llms)
- [Hyperbolic: Comparing Fine-Tuning Frameworks](https://www.hyperbolic.ai/blog/comparing-finetuning-frameworks)

### Alternative Providers
- [RunPod](https://www.runpod.io/)
- [Lambda Labs](https://lambdalabs.com/)
- [Vast.ai](https://vast.ai/)
- [GPU Price Comparison](https://getdeploying.com/gpus)

---

*Last updated: January 2026*
