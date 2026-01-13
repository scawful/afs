# GGUF Conversion Guide

Converting trained LoRA adapters to GGUF format for LMStudio deployment.

## Prerequisites

- llama.cpp (available at `~/src/third_party/llama.cpp/`)
- Python with `peft`, `transformers`, `accelerate`
- Sufficient disk space (~30GB for 7B model conversion)

## Option 1: Local Conversion (Mac)

For smaller models or if you have enough RAM:

```bash
cd ~/src/lab/afs/scripts

# Merge and convert Agahnim v2
python3 merge_and_convert.py \
  --adapter ~/src/lab/afs/models/agahnim-v2-lora/final \
  --base-model Qwen/Qwen2.5-Coder-7B-Instruct \
  --output agahnim-7b-v2 \
  --quant q8_0
```

## Option 2: Remote Conversion (vast.ai)

Better for 7B models - uses GPU instance:

### Step 1: Upload LoRA to Instance

```bash
# Upload LoRA adapter
scp -P 28668 -r ~/src/lab/afs/models/nayru-v9-lora/final root@ssh9.vast.ai:/workspace/nayru-v9-lora/
```

### Step 2: Run Remote Merge

```bash
ssh -p 28668 root@ssh9.vast.ai

# On remote instance:
cd /workspace

# Install merge requirements
pip install peft transformers accelerate

# Merge LoRA with base model
python3 << 'EOF'
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-Coder-7B-Instruct",
    torch_dtype=torch.float16,
    device_map="auto"
)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, "/workspace/nayru-v9-lora")

print("Merging weights...")
model = model.merge_and_unload()

print("Saving merged model...")
model.save_pretrained("/workspace/nayru-v9-merged", safe_serialization=True)
tokenizer.save_pretrained("/workspace/nayru-v9-merged")

print("Done!")
EOF
```

### Step 3: Convert to GGUF

```bash
# Clone llama.cpp if not present
git clone https://github.com/ggerganov/llama.cpp.git
pip install -r llama.cpp/requirements.txt

# Convert to GGUF
python llama.cpp/convert_hf_to_gguf.py \
  /workspace/nayru-v9-merged \
  --outfile /workspace/nayru-v9-f16.gguf \
  --outtype f16

# Quantize (Q4_K_M for best speed/quality)
cd llama.cpp && make -j llama-quantize && cd ..
./llama.cpp/llama-quantize \
  /workspace/nayru-v9-f16.gguf \
  /workspace/nayru-v9-Q4_K_M.gguf \
  Q4_K_M
```

### Step 4: Download GGUF

**Option A: Download to Mac**
```bash
scp -P 28668 root@ssh9.vast.ai:/workspace/nayru-v9-Q4_K_M.gguf ~/src/lab/afs/models/gguf/
```

**Option B: Download directly to Windows (Recommended)**

Since LMStudio runs on Windows (medical-mechanica), download directly there:

```powershell
# PowerShell or Git Bash on Windows
scp -P 28668 root@ssh9.vast.ai:/workspace/nayru-v9-Q4_K_M.gguf D:\LMStudio\models\
```

See [Windows SSH Setup](#windows-ssh-setup) below for first-time configuration.

## Quantization Options

| Type | Size (7B) | Quality | Speed |
|------|-----------|---------|-------|
| F16 | ~14GB | Best | Slow |
| Q8_0 | ~7.5GB | Excellent | Good |
| Q6_K | ~5.5GB | Very Good | Better |
| Q5_K_M | ~4.8GB | Good | Better |
| Q4_K_M | ~4.1GB | Good | Fast |
| Q4_0 | ~3.8GB | Acceptable | Fastest |

**Recommendation:** Q4_K_M for deployment, Q8_0 for evaluation.

## LMStudio Deployment

1. Copy GGUF to LMStudio models folder:
   ```bash
   cp nayru-v9-Q4_K_M.gguf ~/Library/Application\ Support/LMStudio/models/
   ```

2. Open LMStudio and load the model

3. Update AFS orchestrator to use `nayru-lm` agent

## Batch Conversion Script

For converting all trained models:

```bash
#!/bin/bash
MODELS="agahnim-v2 hylia-v2 nayru-v9 router-v2"

for model in $MODELS; do
  echo "Converting $model..."
  python3 merge_and_convert.py \
    --adapter ~/src/lab/afs/models/${model}-lora/final \
    --output ~/src/lab/afs/models/gguf/${model} \
    --quant q4_k_m
done
```

## Troubleshooting

### Out of Memory

- Use remote conversion on vast.ai instance
- Try smaller quantization (Q4_K_M instead of Q8_0)
- Close other applications

### Slow Conversion

- Use GPU-enabled llama.cpp build
- Consider using F16 output type (faster, but larger)

### Model Not Loading in LMStudio

- Check GGUF file integrity
- Verify model architecture is supported
- Try re-quantizing with different options

## Windows SSH Setup

First-time setup for downloading directly to Windows (medical-mechanica).

### Option A: WinSCP (Recommended GUI)

WinSCP is already installed with a saved session for vast.ai.

1. Open WinSCP from Start Menu
2. Select saved session **"vast-ai-hylia"**
3. Click **Login**
4. Navigate to `/workspace/gguf/`
5. Drag & drop GGUFs to `C:\Users\starw\.lmstudio\models\`

**To create new sessions for other vast.ai instances:**
- Protocol: SFTP
- Host: `sshX.vast.ai` (check vast.ai dashboard)
- Port: `XXXXX` (from vast.ai)
- User: `root`
- Advanced → SSH → Authentication → Private key: `C:\Users\starw\.ssh\id_ed25519`

### Option B: Command Line (OpenSSH)

### Step 1: Install OpenSSH (if needed)

Windows 10/11 includes OpenSSH. Verify in PowerShell:
```powershell
ssh -V
```

If not installed:
```powershell
# Run PowerShell as Admin
Add-WindowsCapability -Online -Name OpenSSH.Client~~~~0.0.1.0
```

Or use **Git Bash** (comes with Git for Windows).

### Step 2: Copy SSH Key from Mac

On Mac, copy your vast.ai SSH key to Windows:
```bash
# From Mac - copy key to Windows via network share
cp ~/.ssh/id_ed25519 /Volumes/medical-mechanica/Users/<username>/.ssh/
cp ~/.ssh/id_ed25519.pub /Volumes/medical-mechanica/Users/<username>/.ssh/

# Or use scp if network share not mounted
scp ~/.ssh/id_ed25519* medical-mechanica:/Users/<username>/.ssh/
```

Alternatively, on Windows, generate a new key and add to vast.ai:
```powershell
ssh-keygen -t ed25519 -C "medical-mechanica"
# Then add ~/.ssh/id_ed25519.pub to vast.ai account settings
```

### Step 3: Test Connection

```powershell
# Test vast.ai connection from Windows
ssh -p 28668 root@ssh9.vast.ai "echo 'Connection successful'"
```

### Step 4: Download Models

```powershell
# Download single model to LMStudio dir
scp -P 28668 root@ssh9.vast.ai:/workspace/gguf/nayru-v9-q8_0.gguf C:\Users\starw\.lmstudio\models\

# Download all GGUFs
scp -P 28668 root@ssh9.vast.ai:/workspace/gguf/*.gguf C:\Users\starw\.lmstudio\models\
```

### LMStudio Model Path

LMStudio model directory on Windows (medical-mechanica):
```
C:\Users\starw\.lmstudio\models\
```

176GB free on Windows - plenty of space for all models.

### Troubleshooting

**Permission denied (publickey)**
- Ensure SSH key is in `C:\Users\<username>\.ssh\`
- Check key permissions: `icacls id_ed25519` (should be user-only access)
- Try: `ssh -i C:\Users\<username>\.ssh\id_ed25519 -p 28668 root@ssh9.vast.ai`

**Host key verification failed**
- Add `-o StrictHostKeyChecking=no` for first connection
- Or manually accept the host key when prompted

**Slow downloads**
- vast.ai bandwidth varies by instance
- Try one download at a time for better throughput
- Use `rsync` for resumable downloads (requires WSL or Git Bash)
