# Deployment Guide

Complete guide for converting trained LoRA models to GGUF format and deploying with LMStudio.

## Quick Start

```bash
# 1. Convert trained model to GGUF
python3 scripts/convert_to_gguf.py \
  --model ~/models/adapters/afs/majora-v1-lora \
  --output ~/models/gguf/majora.gguf

# 2. Deploy to LMStudio
./scripts/deploy_to_lmstudio.sh

# 3. Start LMStudio and load model
# In LMStudio UI: Models → Load majora.gguf

# 4. Start API server
# In LMStudio UI: Local Server → Run

# 5. Query the model
curl http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What does LDA do?", "temperature": 0.7}'
```

## GGUF Conversion

Convert LoRA adapters to GGUF format for LMStudio.

### Prerequisites

```bash
# Install GGML tools
pip install llama-cpp-python
pip install ggml

# Or build from source
git clone https://github.com/ggerganov/llama.cpp
cd llama.cpp && make
```

### Convert Single Model

```bash
python3 scripts/convert_to_gguf.py \
  --model ~/models/adapters/afs/majora-v1-lora \
  --output ~/models/gguf/majora.gguf \
  --base-model unsloth/Qwen2.5-Coder-7B-Instruct \
  --quantization Q4_K_M
```

Options:
- `--quantization`: Q4_K_M (recommended), Q5_K_M, Q6_K, F16
- `--device`: cuda, cpu (auto-detect if not specified)
- `--dtype`: float32, float16, bfloat16
- `--chunk-size`: Default 32 (increase for slower/more complete)

### Convert Multiple Models

```bash
# Convert all LoRA adapters in directory
python3 scripts/convert_to_gguf.py \
  --input-dir models/ \
  --output-dir models/ \
  --base-model unsloth/Qwen2.5-Coder-7B-Instruct

# This creates:
# ~/models/gguf/majora-v1-lora.gguf
# ~/models/gguf/nayru-asm-expert-lora.gguf
# ~/models/gguf/veran-v5-logic-lora.gguf
# etc.
```

### Quantization Levels

| Level | Size | Speed | Quality | Recommended For |
|-------|------|-------|---------|-----------------|
| Q4_K_M | 4-5GB | Fast | Good | General use |
| Q5_K_M | 6-7GB | Medium | Very Good | Better quality needed |
| Q6_K | 8-9GB | Slower | Excellent | Maximum quality |
| F16 | 13-15GB | Slow | Perfect | GPU only |

For most use cases, **Q4_K_M** (4-bit quantization) provides good balance.

### Programmatic Conversion

```python
from afs.deployment import convert_lora_to_gguf

# Single model
convert_lora_to_gguf(
    lora_path="~/models/adapters/afs/majora-v1-lora",
    output_path="~/models/gguf/majora.gguf",
    base_model="unsloth/Qwen2.5-Coder-7B-Instruct",
    quantization="Q4_K_M",
    device="cuda"
)

# Batch conversion
from pathlib import Path

lora_dir = Path("models")
for lora_model in lora_dir.glob("*-lora"):
    convert_lora_to_gguf(
        lora_path=str(lora_model),
        output_path=str(lora_model).replace("-lora", "") + ".gguf",
        base_model="unsloth/Qwen2.5-Coder-7B-Instruct",
        quantization="Q4_K_M"
    )
    print(f"Converted {lora_model}")
```

## LMStudio Setup

LMStudio provides easy local model serving.

### Installation

1. Download from https://lmstudio.ai/
2. Install and launch
3. Models will be downloaded to:
   - macOS: `~/Library/Application Support/LM Studio/models/`
   - Linux: `~/.cache/lm-studio/models/`
   - Windows: `%AppData%\LM Studio\models\`

### Load Model

1. Click "Chat" or "Local Server"
2. Find your GGUF file
3. Load model
4. Wait for initialization

### Start Server

```bash
# Using LMStudio CLI (if available)
lmstudio start-server majora.gguf

# Or use UI:
# 1. Open Local Server tab
# 2. Click "Run" button
# 3. Note the port (default: 8000)
```

### Deploy Multiple Models

```bash
# Script to link models to LMStudio directory
./scripts/deploy_to_lmstudio.sh
```

This script:
1. Finds GGUF model files
2. Links them to LMStudio directory
3. Creates configuration files
4. Generates test scripts
5. Documents setup

Output:
```
LMStudio deployment complete:
  Majora:  /path/to/lmstudio/models/majora.gguf
  Nayru:   /path/to/lmstudio/models/nayru.gguf
  Veran:   /path/to/lmstudio/models/veran.gguf

Test with:
  curl http://localhost:8000/api/chat -H "Content-Type: application/json" \
    -d '{"prompt": "test"}'
```

## Network Inference

Run LMStudio servers on different machines/ports.

### Setup Multiple Servers

```bash
# Start models on different ports
# Terminal 1: Majora (port 5000)
lmstudio start-server ~/models/gguf/majora.gguf --port 5000

# Terminal 2: Nayru (port 5001)
lmstudio start-server ~/models/gguf/nayru.gguf --port 5001

# Terminal 3: Veran (port 5002)
lmstudio start-server ~/models/gguf/veran.gguf --port 5002
```

Or use LMStudio UI to manually start servers on different ports.

### Network Configuration

For remote access, expose network ports:

```bash
# LMStudio config (~/.config/lm-studio/config.json)
{
  "server": {
    "host": "0.0.0.0",  # Listen on all interfaces
    "port": 8000,
    "cors": true
  }
}
```

**Warning:** This exposes your model to network attacks. Use firewall or VPN for security.

### Query Remote Models

```python
import requests

# Query Majora on remote machine
response = requests.post(
    "http://192.168.1.100:5000/api/chat",
    json={"prompt": "Write Python code to reverse a string"}
)
print(response.json()["choices"][0]["text"])
```

## API Integration

Query models from Python or other languages.

### Python Client

```python
from afs.deployment import LMStudioClient

# Single model
client = LMStudioClient(port=8000)

response = client.chat(
    prompt="What does LDA do in 65816 assembly?",
    temperature=0.7,
    max_tokens=500
)
print(response)
```

### REST API

Query using HTTP requests:

```bash
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "Question here",
    "temperature": 0.7,
    "max_tokens": 500,
    "top_p": 0.9,
    "frequency_penalty": 0.0
}'
```

Response:
```json
{
  "choices": [
    {
      "text": "Answer here"
    }
  ],
  "model": "majora.gguf",
  "usage": {
    "prompt_tokens": 15,
    "completion_tokens": 80
  }
}
```

### Parameters

| Parameter | Type | Default | Notes |
|-----------|------|---------|-------|
| prompt | string | Required | Question or task |
| temperature | float | 0.7 | 0.0-2.0 (higher = more creative) |
| max_tokens | int | 128 | Maximum response length |
| top_p | float | 0.95 | Nucleus sampling |
| frequency_penalty | float | 0.0 | Penalize repetition |
| presence_penalty | float | 0.0 | Penalize new tokens |

## Load Balancing

Run multiple model replicas for throughput.

### Round-Robin

```python
from itertools import cycle

servers = [
    "http://localhost:5000",
    "http://localhost:5001",
    "http://localhost:5002"
]

server_pool = cycle(servers)

def query_model(prompt):
    server = next(server_pool)
    # Query server...
```

### Queue-Based

```python
import queue
import threading

# Request queue
request_queue = queue.Queue()
response_dict = {}

def worker(server_url):
    while True:
        request_id, prompt = request_queue.get()
        response = requests.post(f"{server_url}/api/chat", json={"prompt": prompt})
        response_dict[request_id] = response.json()

# Start workers
for port in [5000, 5001, 5002]:
    t = threading.Thread(target=worker, args=(f"http://localhost:{port}",))
    t.daemon = True
    t.start()

# Submit requests
request_queue.put((1, "Question 1"))
request_queue.put((2, "Question 2"))
```

## Production Deployment

### Docker Container

```dockerfile
FROM nvidia/cuda:12.0-runtime

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    git

WORKDIR /app

# Install LMStudio
RUN pip install llama-cpp-python

# Copy models
COPY models/ /app/models/

# Copy startup script
COPY scripts/start_server.sh /app/

EXPOSE 8000

CMD ["./start_server.sh"]
```

Build and run:
```bash
docker build -t afs-models .
docker run --gpus all -p 8000:8000 afs-models
```

### Systemd Service

```ini
# /etc/systemd/system/afs-model.service
[Unit]
Description=AFS Model Server
After=network.target

[Service]
Type=simple
User=afs
WorkingDirectory=/home/afs/src/lab/afs
ExecStart=/home/afs/src/lab/afs/scripts/start_server.sh
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

Enable and start:
```bash
sudo systemctl enable afs-model
sudo systemctl start afs-model
sudo systemctl status afs-model
```

### Monitoring

```python
import psutil
import requests
from datetime import datetime

def monitor_server(port, name):
    """Monitor model server health and performance."""

    while True:
        try:
            # Health check
            response = requests.post(
                f"http://localhost:{port}/api/chat",
                json={"prompt": "test"},
                timeout=5
            )
            status = "OK" if response.status_code == 200 else "ERROR"
        except:
            status = "OFFLINE"

        # Resource usage
        process = psutil.Process()
        memory = process.memory_info().rss / 1024**3  # GB

        print(f"[{datetime.now()}] {name}: {status}, Memory: {memory:.1f}GB")

        time.sleep(60)  # Check every minute
```

## Troubleshooting

### Model Loading Issues

**Problem**: GGUF file too large for memory

**Solutions**:
1. Use smaller quantization (Q3_K instead of F16)
2. Use CPU inference (slower but uses less VRAM)
3. Use smaller base model (3B instead of 7B)

**Problem**: Conversion fails with OOM

**Solutions**:
1. Increase available VRAM
2. Use CPU for conversion: `--device cpu`
3. Use lower quantization
4. Close other applications

### Performance Issues

**Problem**: Slow inference (>10s per token)

**Solutions**:
1. Use GPU: `--device cuda`
2. Use lower quantization (Q4_K_M instead of Q6_K)
3. Check GPU memory with `nvidia-smi`
4. Reduce max_tokens
5. Use smaller model

**Problem**: Server crashes during inference

**Solutions**:
1. Check memory: `nvidia-smi`
2. Reduce batch size or max_tokens
3. Use CPU fallback
4. Check GPU stability: `nvidia-smi -i 0 -l 1` for monitoring

### Connection Issues

**Problem**: "Connection refused" when querying model

**Solutions**:
1. Check LMStudio is running: `ps aux | grep lmstudio`
2. Verify port: `lsof -i :8000`
3. Check firewall: `sudo ufw status`
4. Restart server: `killall -9 server_executable && lmstudio start-server`

**Problem**: Intermittent connection failures

**Solutions**:
1. Implement retry logic with exponential backoff
2. Use connection pooling
3. Check network stability
4. Monitor server logs for crashes

### Quality Issues

**Problem**: Responses are incoherent or poor quality

**Solutions**:
1. Check quantization level (try Q5_K_M or Q6_K)
2. Verify it's the right model
3. Try different temperature settings
4. Check training data quality

## Performance Tuning

### Inference Speed

Approximate tokens per second (with RTX 4090):

| Model | Quantization | VRAM | Speed |
|-------|--------------|------|-------|
| 7B | Q4_K_M | 6GB | 30 tok/s |
| 7B | Q5_K_M | 7GB | 25 tok/s |
| 7B | Q6_K | 9GB | 20 tok/s |
| 7B | F16 | 14GB | 15 tok/s |
| 3B | Q4_K_M | 3GB | 60 tok/s |

### Memory Optimization

```python
# Load model with memory mapping
# In LMStudio config:
{
  "inference": {
    "use_mmap": true,
    "use_mlock": false,  # Set true to lock in RAM
    "threads": 8,
    "batch_size": 256
  }
}
```

### Batch Processing

```python
# Process multiple prompts efficiently
prompts = [
    "Question 1",
    "Question 2",
    "Question 3"
]

# Sequential is slow
for prompt in prompts:
    response = client.chat(prompt)  # ~2-3s each

# Parallel is faster
import concurrent.futures
with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
    results = list(executor.map(client.chat, prompts))
```

## Backup & Recovery

### Backup Models

```bash
# Backup to Google Drive
python3 scripts/gdrive_backup.py \
  --models ~/models/gguf/**/*.gguf \
  --output ~/Google\ Drive/AFS_Backups/models/

# Backup to S3
aws s3 sync models/ s3://afs-backup/models/
```

### Recovery

```bash
# From Google Drive
cp ~/Google\ Drive/AFS_Backups/models/*.gguf ~/models/gguf/

# From S3
aws s3 sync s3://afs-backup/models/ models/
```

## File Locations

```
~/src/lab/afs/
├── models/
│   ├── *.jsonl                  # Training data
│   ├── *.Modelfile              # Modelfiles
│   └── ...
│
└── scripts/
    ├── convert_to_gguf.py       # Conversion script
    ├── deploy_to_lmstudio.sh    # Deployment script
    └── start_server.sh          # Server startup

~/.config/lm-studio/
├── config.json                  # LMStudio configuration
└── models/                      # Symbolic links to GGUF files
    ├── majora.gguf → ~/models/gguf/majora.gguf
    └── ...

Recommended paths:
~/models/gguf/                   # Store GGUF files here
~/backups/models/                # Backup location
```

## See Also

- [Training Guide](training.md) - How to train models
- [Evaluation Guide](evaluation.md) - Evaluating models
- [API Reference](api.md) - Python and REST APIs

---

**Last Updated:** January 2026
