# Zelda Expert Models (Triforce MoE)

Fine-tuned models for 65816 assembly tasks, named after Oracle of Ages/Seasons characters.

## Mac Deployment (Ollama)

Current models deployed on Mac via Ollama:

| Model | Version | Size | Purpose |
|-------|---------|------|---------|
| din-v2 | v2 | 4.7 GB | Optimization expert - reduces cycle counts, improves efficiency |
| nayru-v5 | v5 | 4.8 GB | Code generation - writes 65816 assembly from descriptions |
| veran-v1 | v1 | 4.7 GB | Code analysis - explains routines, documents functionality |
| farore-v1 | v1 | 4.7 GB | Debugging expert - identifies bugs, suggests fixes |

### Usage

```bash
ollama run din-v2 "Optimize this loop: ..."
ollama run nayru-v5 "Write a DMA transfer routine"
ollama run veran-v1 "Explain this code: ..."
ollama run farore-v1 "Debug this routine that crashes: ..."
```

## Windows Archive (Medical-Mechanica)

Full version history available on Windows storage (D: drive):

**Mount:** `~/Mounts/mm-d/` (via sshfs to medical-mechanica)

**Path:** `~/Mounts/mm-d/Ollama/models/`

### Available Versions

| Model | Versions | Location |
|-------|----------|----------|
| Din | v1, v2, v3, v3-fewshot | `manifests/registry.ollama.ai/library/din-*/latest` |
| Nayru | v4, v5 | `manifests/registry.ollama.ai/library/nayru-*/latest` |
| Farore | v1 | `manifests/registry.ollama.ai/library/farore-*/latest` |

### Version Notes

- **Din v1-v3:** Progressive improvements in optimization quality
- **Din v3-fewshot:** Trained with few-shot examples
- **Nayru v4-v5:** v5 has improved code generation coherence
- **Veran:** Only v1 trained so far (from veran-lora-fused-v2 adapters)

## Local Model Files

GGUF files for Ollama Modelfile deployment:

```
~/src/lab/afs/models/
├── din-fused/
│   └── din-q4_k_m.gguf        # Q4_K_M quantized (used by din-v2)
├── veran-fused/
│   └── veran-q4_k_m.gguf      # Q4_K_M quantized (used by veran-v1)
└── nayru/
    └── (adapters only - nayru-v5 uses Windows-synced blobs)
```

**Note:** Nayru v5 was synced from Windows Ollama blobs, not converted from local adapters.

## Deployment Scripts

Modelfiles for creating Ollama models from GGUF:

```
~/src/lab/afs/scripts/
├── Modelfile.din              # din-v2 deployment config
└── Modelfile.veran            # veran-v1 deployment config
```

### Deploy from GGUF

```bash
cd ~/src/lab/afs/scripts
ollama create din-v2 -f Modelfile.din
ollama create veran-v1 -f Modelfile.veran
```

## Syncing Models from Windows

To sync additional versions from Windows:

1. Mount Windows storage:
   ```bash
   sshfs medical-mechanica:/d/ ~/Mounts/mm-d/
   ```

2. Find the model manifest:
   ```bash
   ls ~/Mounts/mm-d/Ollama/models/manifests/registry.ollama.ai/library/
   ```

3. Copy blobs referenced in manifest to Mac Ollama:
   ```bash
   # Read manifest to get blob SHAs
   cat ~/Mounts/mm-d/Ollama/models/manifests/registry.ollama.ai/library/<model>/latest

   # Copy each blob (model, adapter, system, params, config)
   cp ~/Mounts/mm-d/Ollama/models/blobs/sha256-<hash> ~/.ollama/models/blobs/
   ```

4. Create manifest on Mac:
   ```bash
   mkdir -p ~/.ollama/models/manifests/registry.ollama.ai/library/<model>
   # Copy or recreate manifest file
   ```

5. Verify:
   ```bash
   ollama list
   ```

## Training Source

All models fine-tuned using MLX LoRA on Qwen2.5 base:

| Model | Base Model | Training |
|-------|------------|----------|
| Din | Qwen2.5-7B-Instruct | din-lora-adapters-v2 |
| Nayru | Qwen2.5-Coder-7B | (adapters on Windows) |
| Veran | Qwen2.5-7B-Instruct | veran-lora-fused-v2 |
| Farore | Qwen2.5-7B-Instruct | (adapters on Windows) |

## Conversion Pipeline

For MLX models with pre-quantization:

```bash
# 1. Dequantize MLX → bf16
python3 -m mlx_lm convert --hf-path <fused-model> --mlx-path <output> -d --dtype bfloat16

# 2. Convert to GGUF f16
python3 llama.cpp/convert_hf_to_gguf.py <bf16-model> --outtype f16 --outfile model-f16.gguf

# 3. Quantize to Q4_K_M
./llama.cpp/build/bin/llama-quantize model-f16.gguf model-q4_k_m.gguf Q4_K_M

# 4. Clean up intermediate files immediately to save storage
rm -rf <bf16-model> model-f16.gguf
```

**Important:** Delete intermediate files immediately after conversion to save storage.
