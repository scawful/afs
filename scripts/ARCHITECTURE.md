# Deployment Pipeline Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    DEPLOYMENT PIPELINE                           │
│                   (deploy_pipeline.sh)                           │
└─────────────────────────────────────────────────────────────────┘
        ↓
┌──────────────────────────────────────────────────────────────────┐
│  STAGE 1: DOWNLOAD (download_from_vastai.py)                     │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  vast.ai Instance                    Local Storage              │
│  ┌──────────────────┐               ┌─────────────────────┐     │
│  │ Training Results │               │ adapter.safetensors │     │
│  │  /workspace/     │──[SSH/rsync]→│ + metadata.json     │     │
│  │   output/*/      │   [resume]    │ + checksums         │     │
│  └──────────────────┘   [verify]    └─────────────────────┘     │
│                                                                   │
│  Features:                                                       │
│  • rsync with resume capability                                 │
│  • SHA256 checksum verification                                 │
│  • SCP fallback                                                 │
│  • Metadata tracking                                            │
│  • Optional compression (zstd)                                  │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
        ↓
┌──────────────────────────────────────────────────────────────────┐
│  STAGE 2: MERGE (merge_and_quantize.py - merge phase)           │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Base Model                    LoRA Adapter                      │
│  ┌──────────────────┐         ┌──────────────────┐               │
│  │ Qwen/Qwen2.5-7B  │         │ adapter.        │                │
│  │ (from huggingface)│ ──────→│ safetensors      │               │
│  └──────────────────┘         └──────────────────┘               │
│         ↓                              ↓                         │
│  ┌─────────────────────────────────────────────────┐             │
│  │ Merge Strategy (config selectable)              │             │
│  │ ┌──────────────────┐  ┌──────────────────────┐  │             │
│  │ │ unsloth          │  │ transformers + peft  │  │             │
│  │ │ (faster)         │  │ (standard)           │  │             │
│  │ └──────────────────┘  └──────────────────────┘  │             │
│  └─────────────────────────────────────────────────┘             │
│         ↓                                                         │
│  ┌─────────────────────────────────────────────────┐             │
│  │ HuggingFace Format (Merged Model)               │             │
│  │ /merged_hf/                                     │             │
│  │  ├── model.safetensors                          │             │
│  │  ├── config.json                                │             │
│  │  ├── tokenizer.json                             │             │
│  │  └── special_tokens_map.json                    │             │
│  └─────────────────────────────────────────────────┘             │
│                                                                   │
│  Merge Features:                                                 │
│  • Device auto-detection (CUDA/CPU)                             │
│  • Optional unsloth acceleration                                │
│  • 16-bit float precision                                        │
│  • Safe serialization                                           │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
        ↓
┌──────────────────────────────────────────────────────────────────┐
│  STAGE 3: CONVERT TO GGUF (merge_and_quantize.py - convert)     │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  HuggingFace Model         llama.cpp Script                      │
│  ┌──────────────────┐    ┌──────────────────────┐                │
│  │ /merged_hf/      │───→│ convert_hf_to_gguf.py│               │
│  │ (safetensors)    │    │                      │                │
│  └──────────────────┘    └──────────────────────┘                │
│         ↓                          ↓                             │
│  ┌─────────────────────────────────────────────────┐             │
│  │ F16 GGUF Format (Intermediate)                  │             │
│  │ • Full precision (16-bit float)                 │             │
│  │ • 10-15 GB file size                            │             │
│  │ • Maximum quality, minimum compression          │             │
│  └─────────────────────────────────────────────────┘             │
│                                                                   │
│  Convert Features:                                              │
│  • Uses llama.cpp for format conversion                         │
│  • Automatic dependency management                             │
│  • CUDA support for acceleration                               │
│  • Fallback to CPU conversion                                  │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
        ↓
┌──────────────────────────────────────────────────────────────────┐
│  STAGE 4: QUANTIZE (merge_and_quantize.py - quantize)           │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  F16 GGUF                 llama-quantize Binary                  │
│  ┌──────────────────┐    ┌──────────────────────┐                │
│  │ model.gguf       │───→│ Multiple Quantization│               │
│  │ (16-bit, 10GB)   │    │ Passes               │                │
│  └──────────────────┘    └──────────────────────┘                │
│         ↓
│         ├──→ Q4_K_M (4.4 bits) ──→ ┌─────────────────┐           │
│         │                           │ 3.5-4.0 GB      │           │
│         │                           │ [RECOMMENDED]   │           │
│         │                           └─────────────────┘           │
│         │
│         ├──→ Q5_K_M (5.3 bits) ──→ ┌─────────────────┐           │
│         │                           │ 4.5-5.0 GB      │           │
│         │                           │ [HIGH QUALITY]  │           │
│         │                           └─────────────────┘           │
│         │
│         └──→ Q8_0 (8 bits) ────────→ ┌─────────────────┐         │
│                                       │ 7.0-8.0 GB      │         │
│                                       │ [BEST QUALITY]  │         │
│                                       └─────────────────┘         │
│                                                                   │
│  Quantization Features:                                          │
│  • Parallel format generation                                   │
│  • CUDA-accelerated quantization                                │
│  • Bit-efficient compression                                     │
│  • Minimal quality loss (especially Q5/Q8)                      │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
        ↓
┌──────────────────────────────────────────────────────────────────┐
│  STAGE 5: DEPLOY (deploy_to_lmstudio.sh)                        │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Quantized Models          LMStudio Configuration                │
│  ┌──────────────────┐    ┌──────────────────────┐                │
│  │ model-q4_k_m.gguf│───→│ ~/.lmstudio/models/  │               │
│  │ model-q5_k_m.gguf│    │ (symlink/copy)       │                │
│  │ model-q8_0.gguf  │    └──────────────────────┘                │
│  └──────────────────┘             ↓                             │
│                      ┌──────────────────────────┐                │
│                      │ LMStudio Application     │                │
│                      │                          │                │
│                      │ Model Registry:          │                │
│                      │  majora: port 5000       │                │
│                      │  nayru:  port 5001       │                │
│                      │  veran:  port 5002       │                │
│                      │  ...                     │                │
│                      └──────────────────────────┘                │
│                                 ↓                               │
│                      ┌──────────────────────────┐                │
│                      │ API Server (Local)       │                │
│                      │ http://localhost:5000    │                │
│                      │ http://localhost:5001    │                │
│                      │ ...                      │                │
│                      └──────────────────────────┘                │
│                                                                   │
│  Deploy Features:                                               │
│  • Version backup (old models saved)                            │
│  • Symlink for space efficiency                                 │
│  • Per-model system prompts                                     │
│  • Port management                                              │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
        ↓
┌──────────────────────────────────────────────────────────────────┐
│  STAGE 5.5: HEALTH CHECK (health_check.py)                      │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  File Validation          API Verification       Model Testing   │
│  ┌───────────────────┐   ┌────────────────────┐ ┌──────────────┐ │
│  │ • GGUF format    │   │ • Port reachable   │ │ • Test prompt│ │
│  │ • File size      │   │ • Response time    │ │ • Quality    │ │
│  │ • Readability    │   │ • Model loaded     │ │ • Latency    │ │
│  │ • Checksum       │   │ • Error codes      │ │ • Tokens/sec │ │
│  └───────────────────┘   └────────────────────┘ └──────────────┘ │
│         ↓                         ↓                  ↓             │
│  ┌──────────────────────────────────────────────────────────────┐ │
│  │ Health Status Summary                                        │ │
│  │ ✓ healthy   - All checks passed                             │ │
│  │ ⚠ degraded  - Some issues but operational                  │ │
│  │ ✗ unhealthy - Critical failures                             │ │
│  └──────────────────────────────────────────────────────────────┘ │
│                                                                   │
│  System Checks:                                                 │
│  • Disk space availability                                      │
│  • Memory usage                                                 │
│  • GPU utilization (if available)                              │
│  • Resource warnings                                            │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
        ↓
┌──────────────────────────────────────────────────────────────────┐
│  STAGE 6: EVALUATE (compare_models.py)                          │
├──────────────────────────────────────────────────────────────────┤
│                                                                   │
│  Test Dataset           Model Queries          Metrics           │
│  ┌─────────────────┐   ┌──────────────┐   ┌──────────────────┐   │
│  │ eval_samples    │───→│ Multi-model  │───→│ • Latency        │   │
│  │ .jsonl          │   │ comparison   │   │ • Quality        │   │
│  └─────────────────┘   └──────────────┘   │ • Throughput     │   │
│                                             │ • Memory usage   │   │
│                                             └──────────────────┘   │
│                                                    ↓               │
│                                     ┌──────────────────────────┐   │
│                                     │ Evaluation Report        │   │
│                                     │ (JSON + Markdown)        │   │
│                                     └──────────────────────────┘   │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
        ↓
┌──────────────────────────────────────────────────────────────────┐
│  PIPELINE COMPLETE                                              │
│                                                                   │
│  Models ready for:                                              │
│  • Interactive chat via LMStudio UI                            │
│  • API calls (http://localhost:5000, etc.)                     │
│  • Integration with external applications                      │
│                                                                   │
└──────────────────────────────────────────────────────────────────┘
```

## Data Flow Summary

```
vast.ai instances
    ↓ [SSH/rsync]
local adapters
    ↓ [merge + quantize]
GGUF quantizations (Q4/Q5/Q8)
    ↓ [copy/symlink]
LMStudio models directory
    ↓ [load]
API servers (http://localhost:5000+)
    ↓ [health check + evaluate]
deployment report
```

## File Organization

```
/Users/scawful/src/lab/afs/
│
├── scripts/
│   ├── deployment_config.yaml          [Configuration]
│   ├── deploy_pipeline.sh              [Main orchestrator]
│   ├── download_from_vastai.py         [Download stage]
│   ├── merge_and_quantize.py           [Merge + quantize]
│   ├── health_check.py                 [Validation]
│   ├── deploy_to_lmstudio.sh           [Deployment]
│   ├── compare_models.py               [Evaluation]
│   ├── DEPLOYMENT.md                   [Full docs]
│   ├── QUICKSTART.md                   [Quick reference]
│   └── ARCHITECTURE.md                 [This file]
│
├── models/
│   ├── majora_adapter_model.safetensors
│   ├── majora_v1-q4_k_m.gguf
│   ├── majora_v1-q5_k_m.gguf
│   ├── majora_v1-q8_0.gguf
│   ├── nayru_adapter_model.safetensors
│   ├── nayru_v1-q4_k_m.gguf
│   │
│   ├── merged_hf/                      [Temporary]
│   │   ├── majora_merged/
│   │   └── nayru_merged/
│   │
│   └── backups/
│       ├── majora_v1-q4_k_m.gguf.backup.1705238400
│       └── majora_v1-q5_k_m.gguf.backup.1705238400
│
└── .logs/
    ├── pipeline.log
    ├── pipeline_summary.log
    ├── download.log
    ├── merge.log
    ├── quantize.log
    ├── deploy.log
    ├── health_check.log
    └── evaluate.log
```

## Configuration Hierarchy

```
deployment_config.yaml
│
├── vast               [vast.ai settings]
│   ├── instance IDs
│   ├── SSH key path
│   └── output paths
│
├── merge              [Merge strategy]
│   ├── use_unsloth
│   ├── save_merged_hf
│   └── batch_size
│
├── quantization       [Quantization config]
│   ├── formats (Q4/Q5/Q8)
│   ├── primary_format
│   └── llama_cpp_flags
│
├── deployment         [LMStudio config]
│   ├── lmstudio.home
│   ├── lmstudio.ports
│   ├── backup_dir
│   └── models[].config
│
├── health_check       [Health check config]
│   ├── timeout
│   ├── test_prompts
│   └── min_response_length
│
├── notifications      [Optional alerts]
│   ├── slack
│   ├── discord
│   └── email
│
├── logging            [Log configuration]
│   ├── log_dir
│   ├── level
│   └── retention_days
│
├── pipeline           [Pipeline behavior]
│   ├── stages
│   ├── continue_on_error
│   └── cleanup
│
├── evaluation         [Eval config]
│   ├── run_benchmarks
│   └── metrics
│
└── performance        [Tuning]
    ├── parallel_downloads
    ├── gpu_memory_fraction
    └── mixed_precision
```

## Error Handling & Recovery

```
Pipeline Execution
│
├─→ Stage Fails
│   ├─→ Log error with context
│   ├─→ Save checkpoint
│   └─→ Option 1: Retry
│       Option 2: Resume with --from-checkpoint
│       Option 3: Skip stage
│
├─→ Deployment Issue
│   └─→ Automatic backup of previous version
│       Manual rollback with --rollback
│
└─→ Health Check Failure
    ├─→ Detailed diagnostics
    ├─→ System resource check
    └─→ Suggested remediation
```

## Performance Characteristics

### Memory Usage (per stage)

| Stage | GPU Memory | CPU Memory | Notes |
|-------|-----------|-----------|-------|
| Download | Minimal | 100MB | Network I/O bound |
| Merge | 12-16GB | 2-4GB | Scales with model size |
| Convert | 2-4GB | 1-2GB | Sequential conversion |
| Quantize | 6-8GB | 1-2GB | Optimized kernel |
| Deploy | <1GB | <500MB | Just copying files |
| Health Check | 2-4GB | 500MB | Model inference |

### Time Complexity

| Stage | 7B Model | 13B Model | Notes |
|-------|----------|-----------|-------|
| Download (5GB) | 3-5m | 3-5m | Network dependent |
| Merge | 2-3m | 4-5m | Linear with params |
| Convert | 5-8m | 8-12m | Sequential |
| Quantize (3 fmt) | 15-20m | 25-35m | Parallel possible |
| Deploy | <1m | <1m | File operations |
| Health Check | 1-2m | 1-2m | Cold start included |

### Compression Ratios

| Format | Compression | Quality |
|--------|------------|---------|
| Q4_K_M | 75-80% | Good (recommended) |
| Q4_K_S | 75-80% | Good (lower memory) |
| Q5_K_M | 50-60% | Better |
| Q5_K_S | 50-60% | Better (lower memory) |
| Q8_0 | 30-40% | Excellent |
| F16 | 0% | Lossless |

## Integration Points

```
External Systems
│
├─ vast.ai          [Training infrastructure]
│  └─→ SSH access + vastai CLI
│
├─ HuggingFace Hub  [Base model + LoRA]
│  └─→ Direct download via transformers
│
├─ llama.cpp        [GGUF tools]
│  └─→ Auto-cloned + built
│
├─ LMStudio         [Inference]
│  └─→ File symlink + API calls
│
├─ Slack/Discord    [Notifications - optional]
│  └─→ Webhook integration
│
└─ Monitoring Tools [Integration]
    ├─ JSON reports
    ├─ Log files
    └─ Health check API
```

## Scaling Considerations

### Single Model
- All stages run sequentially
- ~30 minutes total
- Simple one-command deployment

### Multiple Models (Parallel)
```
Terminal 1: ./deploy_pipeline.sh --model majora --instance-id 111
Terminal 2: ./deploy_pipeline.sh --model nayru --instance-id 222
Terminal 3: ./deploy_pipeline.sh --model veran --instance-id 333
```
- Download parallelizes (bottleneck: bandwidth)
- Merge/quantize on single GPU (choose queue or distribute to multiple GPUs)
- Health check at end validates all

### Continuous Deployment
```
GitHub Actions / Jenkins / etc.
│
├─→ Trigger on training completion
├─→ Pull latest adapters
├─→ Run deployment pipeline
├─→ Log results
└─→ Notify team
```

See `DEPLOYMENT.md` CI/CD section for examples.
