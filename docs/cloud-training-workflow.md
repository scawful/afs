# Cloud GPU Training Workflow for LoRA Fine-Tuning

A practical guide for training LoRA adapters on Vast.ai cloud GPUs, based on the Veran SNES hardware training experience.

## Overview

**Use Case**: Fine-tuning 7B+ models when local resources are insufficient or too slow.

**What We Trained**:
- Model: Qwen2.5-Coder-7B-Instruct
- Task: SNES hardware register explanations
- Data: 90 examples of assembly code with hardware documentation
- Time: ~3.5 minutes on RTX 5090
- Cost: ~$0.05

## Prerequisites

### API Keys
```bash
# In ~/.secrets
export VAST_API_KEY="your_vast_ai_key"
```

### Local Tools
```bash
# Install vastai CLI
pip install vastai

# Configure
vastai set api-key $VAST_API_KEY
```

### Training Data
Prepare JSONL with instruction/output pairs:
```json
{"instruction": "Explain this 65816 code:\nLDA #$80\nSTA $2100", "output": "Enables force blank..."}
```

## Step-by-Step Workflow

### 1. Search for GPUs

```bash
# Find available GPUs
vastai search offers 'gpu_name=RTX_5090 rentable=true disk_space>=50' --order 'dph_total'

# Alternative options
vastai search offers 'gpu_name=RTX_4090 rentable=true' --order 'dph_total'  # Cheaper
vastai search offers 'gpu_name=A100 rentable=true' --order 'dph_total'      # More VRAM
```

**GPU Selection Guide**:
| GPU | VRAM | $/hr | Best For |
|-----|------|------|----------|
| RTX 3090 | 24GB | $0.10-0.20 | Budget 7B QLoRA |
| RTX 4090 | 24GB | $0.25-0.35 | Fast 7B training |
| RTX 5090 | 32GB | $0.28-0.40 | 7B-14B training |
| A100 | 80GB | $0.50-1.50 | 14B+ full precision |

### 2. Create Instance

```bash
# Create with PyTorch image
INSTANCE_ID=$(vastai create instance <OFFER_ID> \
  --image pytorch/pytorch:2.5.1-cuda12.4-cudnn9-devel \
  --disk 50 \
  --ssh \
  --direct 2>&1 | grep -oP "new_contract': \K\d+")

echo "Instance ID: $INSTANCE_ID"
```

### 3. Wait for Instance & Add SSH Key

```bash
# Add your SSH key
vastai create ssh-key "$(cat ~/.ssh/id_rsa.pub)"

# Wait for running status
while true; do
  STATUS=$(vastai show instance $INSTANCE_ID | awk 'NR==2 {print $3}')
  echo "Status: $STATUS"
  [[ "$STATUS" == "running" ]] && break
  sleep 10
done

# Attach SSH key to instance
vastai attach ssh $INSTANCE_ID "$(cat ~/.ssh/id_rsa.pub)"

# Get SSH URL
SSH_URL=$(vastai ssh-url $INSTANCE_ID)
echo "SSH: $SSH_URL"
```

### 4. Connect & Setup Environment

```bash
# Parse SSH details (format: ssh://root@IP:PORT)
SSH_HOST=$(echo $SSH_URL | sed 's|ssh://root@||' | cut -d: -f1)
SSH_PORT=$(echo $SSH_URL | sed 's|ssh://root@||' | cut -d: -f2)

# Test connection
ssh -p $SSH_PORT root@$SSH_HOST "nvidia-smi --query-gpu=name,memory.total --format=csv"

# Install packages
ssh -p $SSH_PORT root@$SSH_HOST "pip install -q peft transformers datasets accelerate trl bitsandbytes"
```

### 5. Upload Training Data

```bash
# Create directory
ssh -p $SSH_PORT root@$SSH_HOST "mkdir -p /workspace/training/data"

# Upload data
scp -P $SSH_PORT your_training_data.jsonl root@$SSH_HOST:/workspace/training/data/
```

### 6. Create Training Script

```python
# train.py - Upload this to /workspace/training/
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM, 
    AutoTokenizer, 
    Trainer, 
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Configuration
MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"  # Or your base model
DATA_PATH = "/workspace/training/data/training.jsonl"
OUTPUT_DIR = "/workspace/training/output"
SYSTEM_PROMPT = "Your system prompt here"

# Load data
examples = []
with open(DATA_PATH) as f:
    for line in f:
        ex = json.loads(line)
        text = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{ex['instruction']}<|im_end|>\n<|im_start|>assistant\n{ex['output']}<|im_end|>"
        examples.append({"text": text})

# Load model with 4-bit quantization
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
model = prepare_model_for_kbit_training(model)

# LoRA config
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

# Tokenize
def tokenize(example):
    return tokenizer(example["text"], truncation=True, max_length=512, padding="max_length")

dataset = Dataset.from_list(examples)
tokenized = dataset.map(tokenize, remove_columns=["text"])

# Training
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_steps=20,
    logging_steps=10,
    save_steps=50,
    bf16=True,
    optim="paged_adamw_8bit",
    report_to="none",
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    train_dataset=tokenized,
    args=training_args,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

trainer.train()
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
```

### 7. Run Training

```bash
ssh -p $SSH_PORT root@$SSH_HOST "cd /workspace/training && python train.py"
```

### 8. Download Results

```bash
# Download adapters
mkdir -p models/cloud-adapters
scp -P $SSH_PORT root@$SSH_HOST:/workspace/training/output/adapter_config.json models/cloud-adapters/
scp -P $SSH_PORT root@$SSH_HOST:/workspace/training/output/adapter_model.safetensors models/cloud-adapters/
```

### 9. Destroy Instance

```bash
vastai destroy instance $INSTANCE_ID
```

## Local Inference

### Windows (CUDA)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import torch

# Load with 4-bit quantization
bnb_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct", 
    quantization_config=bnb_config, device_map="auto")
model = PeftModel.from_pretrained(model, "models/cloud-adapters")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-7B-Instruct")

# Generate
inputs = tokenizer("Your prompt", return_tensors="pt").to("cuda")
outputs = model.generate(**inputs, max_new_tokens=200)
print(tokenizer.decode(outputs[0]))
```

### Mac (MLX) - Requires Conversion

To use on Mac, you need to:
1. Merge LoRA adapters into base model
2. Convert merged model to MLX format

```bash
# Merge adapters (on Windows/cloud)
python -c "
from peft import PeftModel
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-Coder-7B-Instruct')
model = PeftModel.from_pretrained(model, 'models/cloud-adapters')
model = model.merge_and_unload()
model.save_pretrained('models/merged-model')
"

# Convert to MLX (on Mac)
python -m mlx_lm.convert --hf-path models/merged-model --mlx-path models/mlx-model -q
```

### Ollama - Convert to GGUF

```bash
# Install llama.cpp
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && make

# Convert merged model to GGUF
python convert_hf_to_gguf.py ../models/merged-model --outfile veran.gguf --outtype q4_k_m

# Create Modelfile
cat << 'MODELFILE' > Modelfile
FROM veran.gguf
SYSTEM "You are Veran, a 65816 assembly expert..."
MODELFILE

# Import to Ollama
ollama create veran -f Modelfile
```

## Cost Optimization Tips

1. **Use spot instances**: Vast.ai marketplace has variable pricing
2. **Clear GPU memory**: Kill previous processes before training
3. **Use 4-bit quantization**: Reduces memory 4x, enables larger models
4. **Checkpoint frequently**: Resume if interrupted
5. **Right-size your GPU**: Don't overpay for unused VRAM

## Troubleshooting

### OOM Errors
- Reduce batch size to 1
- Use 4-bit quantization
- Reduce max_seq_length
- Use gradient checkpointing
- Target fewer LoRA modules (just q_proj, v_proj)

### SSH Connection Issues
- Wait longer for instance startup
- Re-attach SSH key after instance starts
- Check SSH key format (must start with `ssh-rsa` or `ssh-ed25519`)

### Training Not Learning
- Verify data format matches model's chat template
- Check learning rate (2e-4 is good for LoRA)
- Ensure enough epochs (5+ for small datasets)
- Put important examples FIRST in training data

## Evaluation Results (Veran Cloud v1)

After training on 90 SNES hardware examples:

| Category | Score | Tests |
|----------|-------|-------|
| Basic 65816 | 67% | 3 |
| SNES Hardware | 30% | 5 |
| **Overall** | **44%** | 8 |

**Training Metrics:**
- Initial loss: 3.28
- Final loss: 0.67
- Loss reduction: 79%

**Known Issues:**
- Model confuses register names (e.g., $2100 called "VMAIN" instead of "INIDISP")
- Need more training data with explicit register name mappings

## File Structure

```
models/
├── veran-cloud-adapters/     # Downloaded from cloud
│   ├── adapter_config.json
│   └── adapter_model.safetensors
├── merged-model/             # After merging (optional)
└── mlx-model/                # MLX converted (Mac)

scripts/
├── run_veran_cloud.py        # Local inference
└── eval_veran_cloud.py       # Evaluation
```

---

*Last updated: January 2026*
*Based on Veran SNES hardware training on Vast.ai RTX 5090*
