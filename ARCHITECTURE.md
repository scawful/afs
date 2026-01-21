# AFS Training Architecture

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA INGESTION LAYER                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ToolBench (16K) ──┐                                                │
│  CodeSearchNet ────┤                                                │
│  Oracle Data ──────┤──▶ Converters ──▶ Quality Score ──▶ Merge     │
│  Synthetic Gen ────┤                                                │
│  Custom Datasets ──┘                                                │
│                                                                       │
└──────────────────────────────────┬──────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                       TRAINING ORCHESTRATION                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐           │
│  │ Majora v1    │   │ Veran v5     │   │ Din v2       │           │
│  │ (Quest)      │   │ (Logic)      │   │ (Optimize)   │           │
│  │ RTX 4090     │   │ GTX 1080     │   │ Titan Xp     │           │
│  │ $0.27/hr     │   │ $0.07/hr     │   │ $0.07/hr     │           │
│  └──────┬───────┘   └──────┬───────┘   └──────┬───────┘           │
│         │                   │                   │                    │
│  ┌──────▼───────┐   ┌──────▼───────┐                               │
│  │ Nayru v6     │   │ Farore v6    │                               │
│  │ (Assembly)   │   │ (Debug)      │                               │
│  │ RTX 3060     │   │ RTX 3060     │                               │
│  │ $0.06/hr     │   │ $0.07/hr     │                               │
│  └──────┬───────┘   └──────┬───────┘                               │
│         │                   │                                        │
│         └───────────┬───────┘                                        │
└─────────────────────┼────────────────────────────────────────────────┘
                      │
                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      POST-TRAINING PIPELINE                          │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  Download LoRA ──▶ Merge with Base ──▶ Quantize ──▶ Deploy         │
│                                                                       │
│  ┌───────────────────────────────────────────────────────────┐     │
│  │  Evaluation Suite                                          │     │
│  │  ├── Meta-Circular (models evaluate models)               │     │
│  │  ├── Screenshot Validation (visual comparison)            │     │
│  │  ├── Benchmark Suite (100+ questions, 9 categories)       │     │
│  │  └── Performance Metrics (speed, accuracy, quality)       │     │
│  └───────────────────────────────────────────────────────────┘     │
│                                                                       │
└──────────────────────────────────┬──────────────────────────────────┘
                                    │
                                    ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         DEPLOYMENT LAYER                             │
├─────────────────────────────────────────────────────────────────────┤
│                                                                       │
│  LMStudio API (ports 5000-5004)                                     │
│  ├── Majora v1 (port 5000) - Quest Specialist                      │
│  ├── Nayru (port 5001) - Assembly Expert                           │
│  ├── Veran v5 (port 5002) - Logic Specialist                       │
│  ├── Agahnim (port 5003) - General Purpose                         │
│  └── Hylia (port 5004) - Retrieval Specialist                      │
│                                                                       │
└─────────────────────────────────────────────────────────────────────┘
```

## Training Infrastructure Components

### 1. Logging System (`src/afs/logging_config.py` - 219 lines)
- JSON structured logging
- Automatic log rotation
- Performance tracking
- Contextual logging with run IDs

### 2. Testing Framework (`tests/test_training_pipeline.py` - 338 lines)
- Unit tests for all components
- Integration tests for full workflows
- Property-based testing
- Fixtures for sample data

### 3. CI/CD Pipeline (`.github/workflows/ci.yml`)
- Multi-version Python testing (3.10, 3.11, 3.12)
- Integration tests (non-GPU)
- Documentation builds (Sphinx)
- Model evaluation automation
- Deployment to Google Drive

### 4. Meta-Circular Evaluation (`scripts/meta_circular_eval.py` - 463 lines)
- Models evaluate other models
- Reliability-weighted scoring
- Feedback loops into training data
- Lambda calculus-inspired evaluation

### 5. Screenshot Validation (`scripts/screenshot_eval.py` - 410 lines)
- Visual regression testing
- Terminal output capture
- HTML comparison pages
- Similarity scoring

### 6. Model Comparison (`scripts/compare_models.py` - 569 lines)
- Multi-model benchmarking
- Per-category performance
- Interactive dashboards (Chart.js)
- Speed and accuracy metrics

## Data Flow

```
Raw Data Sources
      ↓
Converters (ToolBench, CodeSearchNet, etc.)
      ↓
Quality Scoring (0.0-1.0)
      ↓
Deduplication (by instruction text)
      ↓
Dataset Merging (60% expert, 25% general, 15% synthetic)
      ↓
Training on vast.ai (3 epochs, LoRA r=16)
      ↓
LoRA Adapter Download
      ↓
Merge with Base Model (Qwen2.5-Coder-7B)
      ↓
GGUF Quantization (Q4_K_M, Q5_K_M, Q8_0)
      ↓
LMStudio Deployment
      ↓
Evaluation Suite
      ↓
Production Use
```

## Model Specializations

| Model | Role | Training Data | Use Case |
|-------|------|---------------|----------|
| **Majora v1** | Quest Specialist | Oracle codebase (C#, Unity) + ToolBench | Codebase-specific questions, architecture |
| **Veran v5** | Logic Specialist | SNES hardware + synthetic | PPU, DMA, HDMA, Mode 7 |
| **Din v2** | Optimization | Optimization patterns + synthetic | Performance improvements |
| **Nayru v6** | Assembly Expert | 65816 ASM + CodeSearchNet | Code generation, assembly |
| **Farore v6** | Debug Specialist | Debugging examples + CodeSearchNet | Bug finding, fixes |

## Cost Analysis

### Training Costs (per session)
- Majora v1: $0.268/hr × 3hr = $0.80
- Veran v5: $0.068/hr × 3hr = $0.20
- Din v2: $0.068/hr × 3hr = $0.20
- Nayru v6: $0.060/hr × 3hr = $0.18
- Farore v6: $0.069/hr × 3hr = $0.21

**Total: $1.59 per training session (all 5 models)**

### Infrastructure Costs
- Development: $0 (local Mac + free vast.ai credits)
- CI/CD: $0 (GitHub Actions free tier)
- Storage: $0 (local + Google Drive)
- Deployment: $0 (LMStudio local)

**Total operational cost: ~$1.60 per training iteration**

## Performance Metrics

### Training Speed
- Majora v1 (223 samples): ~3 hours on RTX 4090
- Veran v5 (461 samples): ~3 hours on GTX 1080
- Din v2 (249 samples): ~3 hours on Titan Xp
- Nayru v6 (56 samples): ~2 hours on RTX 3060
- Farore v6 (39 samples): ~2 hours on RTX 3060

### Evaluation Coverage
- 9 categories of tests
- 100+ unique questions
- 4 output formats (MD, HTML, JSON, PNG)
- Meta-circular cross-validation
- Visual screenshot comparison

### Deployment Time
- Download LoRA: ~2 minutes per model
- Merge with base: ~5 minutes per model
- GGUF conversion: ~10 minutes per model
- LMStudio deployment: ~2 minutes per model

**Total deployment: ~20 minutes per model (100 minutes for all 5)**

## Technical Stack

- **Training:** PyTorch, Unsloth, LoRA
- **Models:** Qwen2.5-Coder-7B base
- **Quantization:** llama.cpp GGUF
- **Deployment:** LMStudio
- **Testing:** pytest, hypothesis
- **CI/CD:** GitHub Actions
- **Docs:** MkDocs, Sphinx
- **Monitoring:** Custom scripts + vast.ai CLI

## Session: 2026-01-14

**Status:** 5 models training in parallel
**Cost:** $0.533/hr
**Completion:** ~3 hours (09:08 UTC)
**Background agents:** 3 running (ToolBench DL, docs, deployment)
**Context usage:** 82K/200K tokens (41%)
