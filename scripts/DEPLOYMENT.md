# End-to-End Deployment Automation Pipeline

Complete automation for deploying trained models from vast.ai training instances to LMStudio inference servers.

## Overview

The deployment pipeline handles the entire workflow:

```
vast.ai training → download LoRA → merge with base model → quantize to GGUF →
→ deploy to LMStudio → health check → evaluation
```

## Components

### 1. `deployment_config.yaml`
Central configuration file for the entire pipeline.

**Key sections:**
- `vast:` - vast.ai instance configuration and model locations
- `merge:` - Merging strategy (unsloth, batch size, etc.)
- `quantization:` - Quantization formats (Q4_K_M, Q5_K_M, Q8_0)
- `deployment:` - LMStudio configuration, ports, system prompts
- `health_check:` - Health check timeouts, test prompts, thresholds
- `notifications:` - Optional Slack/Discord notifications (disabled by default)
- `logging:` - Log files and retention policies
- `pipeline:` - Pipeline stages, error handling, cleanup

### 2. `download_from_vastai.py`
Download LoRA adapters from vast.ai instances.

**Features:**
- SSH connection to vast.ai instances via `vastai ssh-url`
- Rsync-based downloads with resume capability
- Automatic fallback to SCP if rsync unavailable
- SHA256 checksum verification
- Metadata tracking (instance ID, download time, file size)
- Optional zstd compression for storage
- Parallel downloads for multiple models

**Usage:**
```bash
# Download single model from instance
python3 download_from_vastai.py \
  --instance-id 12345678 \
  --model majora

# Download all models from config
python3 download_from_vastai.py --all-models --config deployment_config.yaml

# Resume with verification
python3 download_from_vastai.py \
  --instance-id 12345678 \
  --model majora \
  --resume \
  --verify

# Compress for storage
python3 download_from_vastai.py \
  --instance-id 12345678 \
  --model majora \
  --compress
```

**Output:**
- `models/backups/{model_name}_adapter_model.safetensors`
- `models/backups/{model_name}_adapter_model.safetensors.metadata.json`

### 3. `merge_and_quantize.py`
Merge LoRA adapters and quantize to GGUF.

**Features:**
- Load base model and LoRA adapter
- Optional unsloth-accelerated merging (faster, requires unsloth library)
- Standard transformers + peft fallback
- Multi-format quantization (Q4_K_M, Q5_K_M, Q8_0)
- CUDA support for faster processing
- Automatic cleanup of intermediate files
- Pipeline metadata tracking

**Usage:**
```bash
# Full pipeline: merge → convert → quantize
python3 merge_and_quantize.py \
  --adapter models/majora_adapter_model.safetensors \
  --output models/majora_v1 \
  --quantize q4_k_m,q5_k_m

# Merge only
python3 merge_and_quantize.py \
  --adapter models/majora_adapter_model.safetensors \
  --merge-only \
  --use-unsloth

# Convert to GGUF only
python3 merge_and_quantize.py \
  --adapter models/majora_merged \
  --convert-only

# Custom base model
python3 merge_and_quantize.py \
  --adapter models/adapter.safetensors \
  --base-model "meta-llama/Llama-2-7b-hf" \
  --output my_model
```

**Quantization formats:**
- `q4_k_m` - 4-bit, medium K (4.4 bits, recommended)
- `q4_k_s` - 4-bit, small K (lower memory)
- `q5_k_m` - 5-bit, medium K (higher quality)
- `q5_k_s` - 5-bit, small K
- `q8_0` - 8-bit (highest quality)
- `f16` - 16-bit float (no quantization)

**Output:**
- `models/{model_name}_v1-q4_k_m.gguf`
- `models/{model_name}_v1-q5_k_m.gguf`
- `models/{model_name}_v1/{pipeline_metadata.json}`

### 4. `health_check.py`
Comprehensive health checks for deployed models.

**Checks performed:**
- **File checks:** Existence, readability, GGUF format validation
- **API checks:** LMStudio endpoint reachability, response time
- **Model tests:** Response quality, token generation, latency
- **System resources:** Disk space, memory usage, GPU availability
- **Warnings:** Issues like high disk usage, memory pressure

**Usage:**
```bash
# Check all models
python3 health_check.py --all

# Specific model
python3 health_check.py --model majora

# Detailed diagnostics
python3 health_check.py --all --detailed

# Skip model response tests
python3 health_check.py --all --no-test

# JSON output for integration
python3 health_check.py --all --json | jq '.'

# Files only (quick check)
python3 health_check.py --files-only

# API only (endpoints)
python3 health_check.py --api-only
```

**Status levels:**
- `healthy` - All checks passed
- `degraded` - Some issues but still operational
- `unhealthy` - Critical failures

**Output:**
- Console: Colored status indicators (✓, ⚠, ✗)
- JSON: Structured results with metrics
- Logs: `{log_dir}/health_check.log`

### 5. `deploy_pipeline.sh`
Main orchestrator script that runs the entire pipeline.

**Features:**
- Sequential stage execution with error handling
- Resume from failed stage with `--from-checkpoint`
- Per-stage logging with summary reports
- Model version backups before deployment
- Automatic rollback capability
- Colored output with progress tracking
- Dry-run mode for testing

**Usage:**
```bash
# Deploy all models
./deploy_pipeline.sh --all-models

# Deploy single model
./deploy_pipeline.sh --model majora --instance-id 12345678

# Run specific stage
./deploy_pipeline.sh --stage quantize --model majora

# Skip certain stages
./deploy_pipeline.sh --all-models \
  --skip-download \
  --skip-evaluation

# Resume from failure
./deploy_pipeline.sh --from-checkpoint

# Rollback to previous version
./deploy_pipeline.sh --rollback majora

# Dry run (show what would happen)
./deploy_pipeline.sh --all-models --dry-run
```

**Pipeline stages:**
1. `download` - Download adapters from vast.ai
2. `merge` - Merge with base model
3. `quantize` - Create GGUF quantizations
4. `deploy` - Deploy to LMStudio
5. `health_check` - Verify deployment
6. `evaluate` - Run benchmarks

**Output:**
- `.logs/pipeline.log` - Complete pipeline logs
- `.logs/pipeline_summary.log` - Stage execution summary
- `.logs/{download,merge,quantize,deploy,health_check,evaluate}.log` - Per-stage logs

## Getting Started

### Prerequisites

```bash
# Python dependencies
pip install torch transformers peft pyyaml requests

# Optional: faster merging
pip install unsloth

# Optional: compression
pip install zstandard

# Optional: system monitoring
pip install psutil

# vastai CLI
pip install vastai
# Configure: vastai set api-key YOUR_KEY
```

### Step 1: Configure Deployment

Edit `deployment_config.yaml`:

```yaml
vast:
  models:
    majora:
      instance_id: 12345678  # From vast.ai dashboard
      output_path: "/workspace/output/majora_v1/adapter_model.safetensors"
      base_model: "Qwen/Qwen2.5-7B-Instruct"

deployment:
  lmstudio:
    home: ~/.lmstudio
    ports:
      majora: 5000
      # ... other models
```

### Step 2: Download Adapters

```bash
python3 download_from_vastai.py \
  --instance-id 12345678 \
  --model majora \
  --resume \
  --verify
```

### Step 3: Merge and Quantize

```bash
python3 merge_and_quantize.py \
  --adapter models/majora_adapter_model.safetensors \
  --output models/majora_v1 \
  --quantize q4_k_m,q5_k_m
```

### Step 4: Deploy to LMStudio

```bash
# Manual deployment (already in existing script)
bash deploy_to_lmstudio.sh

# Or via pipeline
./deploy_pipeline.sh --stage deploy
```

### Step 5: Verify Health

```bash
python3 health_check.py --all --detailed
```

## Running the Full Pipeline

### One-Command Deployment

```bash
./deploy_pipeline.sh --all-models
```

This runs all stages automatically:
1. Downloads from vast.ai
2. Merges adapters
3. Quantizes to multiple formats
4. Deploys to LMStudio
5. Runs health checks
6. Evaluates model quality

### Monitoring Progress

```bash
# Watch logs in real-time
tail -f .logs/pipeline.log

# Check stage-specific logs
tail -f .logs/quantize.log
tail -f .logs/health_check.log

# View summary
cat .logs/pipeline_summary.log
```

## Advanced Usage

### Parallel Model Deployment

Deploy multiple models simultaneously:

```bash
# Terminal 1: Majora
./deploy_pipeline.sh --model majora --instance-id 111111

# Terminal 2: Nayru (different instance)
./deploy_pipeline.sh --model nayru --instance-id 222222

# Terminal 3: Health check all
python3 health_check.py --all
```

### Custom Base Models

```bash
python3 merge_and_quantize.py \
  --adapter models/adapter.safetensors \
  --base-model "meta-llama/Llama-2-7b-hf" \
  --output models/llama2-custom
```

### Multiple Quantization Formats

```bash
python3 merge_and_quantize.py \
  --adapter models/adapter.safetensors \
  --quantize q4_k_m,q4_k_s,q5_k_m,q8_0
```

### Selective Stage Execution

```bash
# Skip download (use local adapter)
./deploy_pipeline.sh --model majora --skip-download

# Skip evaluation
./deploy_pipeline.sh --all-models --skip-evaluation

# Skip health check (faster)
./deploy_pipeline.sh --model majora --skip-health-check
```

### Recovery and Rollback

```bash
# If pipeline fails, resume from that point
./deploy_pipeline.sh --from-checkpoint

# Rollback to previous version
./deploy_pipeline.sh --rollback majora

# Backup before deployment
mkdir -p models/backups
cp models/majora*.gguf models/backups/
```

## Configuration Details

### Performance Tuning

```yaml
performance:
  parallel_downloads: 2    # Parallel downloads
  parallel_quantize: 1     # Serial quantization (uses full GPU)
  max_memory_gb: 32        # GPU memory limit
  gpu_memory_fraction: 0.9  # Use 90% of GPU
  mixed_precision: true    # fp16 operations
```

### Notifications Setup

Enable Slack notifications:

```yaml
notifications:
  enabled: true
  slack:
    enabled: true
    webhook_url: null  # Set SLACK_WEBHOOK_URL env var
    channel: "#afs-training"
```

```bash
export SLACK_WEBHOOK_URL="https://hooks.slack.com/..."
./deploy_pipeline.sh --all-models
```

### LMStudio Configuration

Each model has custom settings:

```yaml
deployment:
  models:
    majora:
      port: 5000
      context_size: 4096
      temperature: 0.7
      system_prompt: "You are an Oracle of Secrets expert..."
```

## Troubleshooting

### Download Issues

**Problem:** Connection timeout
```bash
# Check SSH connection
ssh $(vastai ssh-url <instance_id>) "ls -lh /workspace/output/"

# Increase timeout in config
health_check:
  timeout: 60
```

**Problem:** Checksum mismatch
```bash
# Re-download with force
python3 download_from_vastai.py \
  --instance-id 12345 \
  --model majora \
  --resume  # Will re-verify
```

### Merge Issues

**Problem:** Out of memory
```bash
# Use CPU merging (slower)
python3 merge_and_quantize.py \
  --adapter models/adapter.safetensors \
  --merge-only
  # Will use CPU if GPU OOM

# Or use smaller base model
--base-model "Qwen/Qwen2.5-3B-Instruct"
```

**Problem:** unsloth not available
```bash
# Falls back to standard merge automatically
# Or install unsloth:
pip install unsloth
```

### Quantization Issues

**Problem:** llama.cpp build fails
```bash
# Check CUDA availability
python3 -c "import torch; print(torch.cuda.is_available())"

# Build without CUDA
# Edit quantization flags in config:
llama_cpp_flags:
  cuda: false
  metal: false
```

### Health Check Failures

**Problem:** Models not responding
```bash
# Check if LMStudio is running
lsof -i :5000  # Majora port

# Restart LMStudio and wait
sleep 30
python3 health_check.py --all
```

**Problem:** Low disk space
```bash
# Check disk usage
df -h

# Clean old backups
rm models/backups/*.backup.*

# Compress models
python3 -c "
import zstandard as zstd
with open('model.gguf', 'rb') as f_in:
    data = f_in.read()
    with open('model.gguf.zst', 'wb') as f_out:
        f_out.write(zstd.ZstdCompressor().compress(data))
"
```

## Performance Benchmarks

Typical timings (with RTX 4090):

| Stage | Majora | Nayru | Veran |
|-------|--------|-------|-------|
| Download (5GB) | 3-5 min | 3-5 min | 3-5 min |
| Merge | 2-3 min | 2-3 min | 2-3 min |
| Convert to GGUF | 5-8 min | 5-8 min | 5-8 min |
| Quantize (all formats) | 15-20 min | 15-20 min | 15-20 min |
| Deploy | < 1 min | < 1 min | < 1 min |
| Health check | 1-2 min | 1-2 min | 1-2 min |
| **Total** | **~30 min** | **~30 min** | **~30 min** |

## File Structure

```
scripts/
├── deployment_config.yaml           # Central configuration
├── download_from_vastai.py          # Download adapters
├── merge_and_quantize.py            # Merge + quantize
├── health_check.py                  # Health verification
├── deploy_pipeline.sh               # Main orchestrator
├── deploy_to_lmstudio.sh           # Existing LMStudio deploy
└── DEPLOYMENT.md                    # This file

models/
├── majora_adapter_model.safetensors # Downloaded adapter
├── majora_v1-q4_k_m.gguf          # Quantized model
├── majora_v1-q5_k_m.gguf          # Alternative format
├── backups/                         # Version history
└── merged_hf/                       # Temporary merge output

.logs/
├── pipeline.log                     # Complete pipeline logs
├── pipeline_summary.log             # Stage summary
├── download.log                     # Stage-specific logs
├── merge.log
├── quantize.log
├── deploy.log
├── health_check.log
└── evaluate.log
```

## Integration with CI/CD

### GitHub Actions Example

```yaml
name: Deploy Models

on: workflow_dispatch

jobs:
  deploy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      - uses: actions/setup-python@v4
        with:
          python-version: "3.10"

      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install vastai

      - name: Configure vast.ai
        run: |
          vastai set api-key ${{ secrets.VAST_API_KEY }}

      - name: Deploy pipeline
        run: |
          ./scripts/deploy_pipeline.sh --all-models

      - name: Upload logs
        if: always()
        uses: actions/upload-artifact@v3
        with:
          name: deployment-logs
          path: .logs/
```

## Support and Debugging

### Enable Debug Logging

```bash
./deploy_pipeline.sh --log-level DEBUG --all-models
```

### Generate Diagnostic Report

```bash
python3 health_check.py --all --detailed --json > diagnostic_report.json
```

### Check System Requirements

```bash
python3 -c "
import torch, sys
print(f'Python: {sys.version}')
print(f'CUDA: {torch.cuda.is_available()}')
print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')
print(f'Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.0f}GB')
"
```

## See Also

- `deploy_to_lmstudio.sh` - Original LMStudio deployment script
- `compare_models.py` - Model evaluation and comparison
- `vastai_setup.py` - Training setup on vast.ai
