#!/bin/bash
# Deploy models to LMStudio for evaluation

set -e

# Configuration
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
LMSTUDIO_HOME="${LMSTUDIO_HOME:-$HOME/.lmstudio}"
MODELS_DIR="${MODELS_DIR:-$HOME/models/gguf}"
OUTPUT_DIR="${OUTPUT_DIR:-$PROJECT_ROOT}"
LMSTUDIO_MODEL_DIR="${LMSTUDIO_MODEL_DIR:-$HOME/models/lmstudio}"
MLX_DIR="${MLX_DIR:-$HOME/models/mlx}"
LOCALHOST_PORT="${LOCALHOST_PORT:-8000}"

# Models to deploy (name -> path)
declare -A MODELS=(
    ["scawful-echo"]="$MODELS_DIR/ollama/scawful-echo-qwen25-7b-v4-q4km.gguf"
    ["scawful-memory"]="$MODELS_DIR/ollama/memory-1.5b-v1-q8.gguf"
    ["scawful-muse"]="$MODELS_DIR/ollama/muse-3b-v2-q4km.gguf"
    ["zelda-din"]="$MODELS_DIR/ollama/din-7b-v4-q4km.gguf"
    ["zelda-farore"]="$MODELS_DIR/ollama/farore-7b-v5-q8.gguf"
    ["zelda-hylia"]="$MODELS_DIR/afs/hylia-v3-q8_0.gguf"
    ["zelda-majora"]="$MODELS_DIR/ollama/majora-7b-v2-q8.gguf"
    ["zelda-scribe"]="$MODELS_DIR/ollama/scribe-3b-v2-q4km.gguf"
    ["zelda-veran"]="$MODELS_DIR/ollama/veran-7b-v4-q8.gguf"
)

# MLX models to deploy (name -> directory)
declare -A MLX_MODELS=(
    ["scawful-echo-mlx"]="$MLX_DIR/avatars/echo-qwen25-1p5b-v1"
)

# API endpoints (will be configured after deployment)
declare -A API_PORTS=(
    ["zelda-majora"]=5000
    ["zelda-din"]=5001
    ["zelda-farore"]=5002
    ["zelda-veran"]=5003
    ["zelda-hylia"]=5004
    ["zelda-scribe"]=5005
    ["scawful-echo"]=5006
    ["scawful-memory"]=5007
    ["scawful-muse"]=5008
)

echo "=================================="
echo "LMStudio Deployment Script"
echo "=================================="
echo ""
echo "Configuration:"
echo "  LMSTUDIO_HOME: $LMSTUDIO_HOME"
echo "  MODELS_DIR: $MODELS_DIR"
echo "  OUTPUT_DIR: $OUTPUT_DIR"
echo "  LMSTUDIO_MODEL_DIR: $LMSTUDIO_MODEL_DIR"
echo "  MLX_DIR: $MLX_DIR"
echo ""

# Check if models exist
echo "Checking for GGUF models..."
for model_name in "${!MODELS[@]}"; do
    model_path="${MODELS[$model_name]}"
    if [ ! -f "$model_path" ]; then
        echo "  ✗ $model_name: NOT FOUND at $model_path"
    else
        size=$(du -h "$model_path" | cut -f1)
        echo "  ✓ $model_name: $size"
    fi
done
echo ""

echo "Checking for MLX models..."
for model_name in "${!MLX_MODELS[@]}"; do
    model_path="${MLX_MODELS[$model_name]}"
    if [ ! -d "$model_path" ]; then
        echo "  ✗ $model_name: NOT FOUND at $model_path"
    else
        size=$(du -sh "$model_path" | cut -f1)
        echo "  ✓ $model_name: $size"
    fi
done
echo ""

# Create LMStudio directory if needed
mkdir -p "$LMSTUDIO_MODEL_DIR"
echo "LMStudio model directory ready: $LMSTUDIO_MODEL_DIR"
echo ""

# Copy or link models to LMStudio directory
echo "Deploying GGUF models to LMStudio..."
for model_name in "${!MODELS[@]}"; do
    model_path="${MODELS[$model_name]}"
    gguf_dest="$LMSTUDIO_MODEL_DIR/${model_name}.gguf"

    if [ ! -f "$model_path" ]; then
        echo "  ⊘ Skipping $model_name (file not found)"
        continue
    fi

    if [ -f "$gguf_dest" ]; then
        echo "  ✓ $model_name: already deployed"
    else
        echo "  → Linking $model_name (hardlink)..."
        ln "$model_path" "$gguf_dest"
        echo "  ✓ $model_name: deployed"
    fi
done
echo ""

echo "Deploying MLX models to LMStudio..."
for model_name in "${!MLX_MODELS[@]}"; do
    model_path="${MLX_MODELS[$model_name]}"
    mlx_dest="$LMSTUDIO_MODEL_DIR/$model_name"

    if [ ! -d "$model_path" ]; then
        echo "  ⊘ Skipping $model_name (dir not found)"
        continue
    fi

    if [ -d "$mlx_dest" ]; then
        echo "  ✓ $model_name: already deployed"
    else
        echo "  → Copying $model_name..."
        cp -a "$model_path" "$mlx_dest"
        echo "  ✓ $model_name: deployed"
    fi
done
echo ""

# Generate curl test scripts
echo "Generating curl test scripts..."
mkdir -p "${OUTPUT_DIR}/curl_tests"

cat > "${OUTPUT_DIR}/curl_tests/test_all_models.sh" << 'CURL_EOF'
#!/bin/bash

# Test all deployed models with sample queries

echo "Testing all models..."

# Test queries by model specialty
declare -A QUERIES=(
    ["zelda-majora"]="Pitch a weird side-quest for Oracle of Secrets."
    ["zelda-din"]="Optimize this 65816 loop for speed:\\nLDX #$00\\n.loop\\nLDA $7E1234,X\\nCLC\\nADC #$01\\nSTA $7E1234,X\\nINX\\nCPX #$10\\nBNE .loop"
    ["zelda-farore"]="Complete this routine header with a brief purpose and inputs: `Routine_CopyTiles`."
    ["zelda-veran"]="Design a state machine for a dungeon progression system."
    ["zelda-hylia"]="Where can I find documentation on the memory map?"
    ["zelda-scribe"]="Write a clean docstring for a function that compresses tilemaps."
    ["scawful-echo"]="give me your vibe in one short paragraph."
    ["scawful-memory"]="Summarize these notes into 4 crisp bullets with dates kept: 2026-01-19: echo eval; 2026-01-20: LMStudio cleanup."
    ["scawful-muse"]="Brainstorm 5 names for a ROM hacking assistant tool."
)

declare -A PORTS=(
    ["zelda-majora"]=5000
    ["zelda-din"]=5001
    ["zelda-farore"]=5002
    ["zelda-veran"]=5003
    ["zelda-hylia"]=5004
    ["zelda-scribe"]=5005
    ["scawful-echo"]=5006
    ["scawful-memory"]=5007
    ["scawful-muse"]=5008
)

for model_name in "${!PORTS[@]}"; do
    port=${PORTS[$model_name]}
    query="${QUERIES[$model_name]}"

    echo ""
    echo "Testing $model_name on port $port..."

    curl -X POST "http://localhost:$port/chat" \
        -H "Content-Type: application/json" \
        -d "{\"prompt\": \"$query\"}" \
        --connect-timeout 5 \
        --max-time 30 \
        -s | jq . || echo "✗ Connection failed"
done
CURL_EOF

chmod +x "${OUTPUT_DIR}/curl_tests/test_all_models.sh"
echo "  ✓ Test script: ${OUTPUT_DIR}/curl_tests/test_all_models.sh"
echo ""

# Generate Python API client
echo "Generating Python API client..."
cat > "${OUTPUT_DIR}/lmstudio_client.py" << 'PYTHON_EOF'
#!/usr/bin/env python3
"""
LMStudio API Client for Model Evaluation
"""

import requests
import json
import time
from typing import Dict, Optional

class LMStudioClient:
    """Client for querying models running in LMStudio."""

    MODELS = {
        "zelda-majora": "http://localhost:5000/chat",
        "zelda-din": "http://localhost:5001/chat",
        "zelda-farore": "http://localhost:5002/chat",
        "zelda-veran": "http://localhost:5003/chat",
        "zelda-hylia": "http://localhost:5004/chat",
        "zelda-scribe": "http://localhost:5005/chat",
        "scawful-echo": "http://localhost:5006/chat",
        "scawful-memory": "http://localhost:5007/chat",
        "scawful-muse": "http://localhost:5008/chat"
    }

    def __init__(self, timeout: int = 30):
        self.timeout = timeout

    def query(self, model_name: str, prompt: str) -> Optional[Dict]:
        """Query a model and return structured response."""
        if model_name not in self.MODELS:
            raise ValueError(f"Unknown model: {model_name}")

        endpoint = self.MODELS[model_name]

        try:
            response = requests.post(
                endpoint,
                json={"prompt": prompt},
                timeout=self.timeout
            )
            response.raise_for_status()
            return response.json()
        except requests.exceptions.Timeout:
            return {"error": "timeout", "model": model_name}
        except requests.exceptions.ConnectionError:
            return {"error": "connection_failed", "model": model_name}
        except Exception as e:
            return {"error": str(e), "model": model_name}

    def batch_query(self, model_name: str, prompts: list) -> list:
        """Query multiple prompts with a single model."""
        results = []
        for i, prompt in enumerate(prompts):
            print(f"  [{i+1}/{len(prompts)}] Querying {model_name}...", end=" ", flush=True)
            start = time.time()
            result = self.query(model_name, prompt)
            elapsed = time.time() - start
            result["elapsed"] = elapsed
            results.append(result)
            print(f"({elapsed:.2f}s)")

        return results

    def health_check(self) -> Dict[str, bool]:
        """Check which models are available."""
        status = {}
        for model_name, endpoint in self.MODELS.items():
            try:
                response = requests.get(
                    endpoint.replace("/chat", "/health"),
                    timeout=2
                )
                status[model_name] = response.status_code == 200
            except:
                status[model_name] = False

        return status

if __name__ == "__main__":
    import sys

    client = LMStudioClient()

    print("LMStudio Model Health Check")
    print("=" * 40)

    status = client.health_check()
    for model_name, is_healthy in status.items():
        symbol = "✓" if is_healthy else "✗"
        print(f"{symbol} {model_name}")

    print("")
    if all(status.values()):
        print("All models are ready!")
    else:
        print("Some models are not available. Check LMStudio.")
PYTHON_EOF

chmod +x "${OUTPUT_DIR}/lmstudio_client.py"
echo "  ✓ Client: ${OUTPUT_DIR}/lmstudio_client.py"
echo ""

# Generate configuration guide
echo "Generating configuration guide..."
cat > "${OUTPUT_DIR}/LMSTUDIO_SETUP.md" << 'CONFIG_EOF'
# LMStudio Setup Guide

## Installation

1. Download LMStudio from https://lmstudio.ai
2. Install and run the application
3. Download models in LMStudio UI

## Deploying Models

Models have been automatically linked to LMStudio's model directory:

```bash
./deploy_to_lmstudio.sh
```

## Model Directory

Recommended: set LMStudio's model directory to `~/models/lmstudio` (curated hardlinks for GGUF + copied MLX folders).
Use `~/models/gguf` or `~/models/mlx` if you want only one backend visible.

## Running Models as API Servers

Each model can be started on a different port:

### Via LMStudio UI

1. Select model
2. Click "Chat"
3. Look for "API Server" or similar option
4. Set port (5000, 5001, etc.)
5. Start server

### Via Command Line (if supported)

```bash
# Zelda Majora on port 5000
lmstudio start --model zelda-majora.gguf --port 5000

# Scawful Echo on port 5006
lmstudio start --model scawful-echo.gguf --port 5006
```

## Testing Models

### Health Check

```bash
python3 lmstudio_client.py
```

### Manual Test

```bash
curl -X POST "http://localhost:5000/chat" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is 2+2?"}'
```

### Batch Test

```bash
./curl_tests/test_all_models.sh
```

## Running Evaluation

Once models are deployed and running:

```bash
cd /Users/scawful/src/lab/afs
python3 scripts/compare_models.py
```

## Configuration

### Model Ports
- Zelda Majora (Quest, `zelda-majora`): 5000
- Zelda Din (ASM optimization, `zelda-din`): 5001
- Zelda Farore (Autocomplete, `zelda-farore`): 5002
- Zelda Veran (Logic, `zelda-veran`): 5003
- Zelda Hylia (Retrieval, `zelda-hylia`): 5004
- Zelda Scribe (Docs, `zelda-scribe`): 5005
- Scawful Echo (Avatar, `scawful-echo`): 5006
- Scawful Memory (Archivist, `scawful-memory`): 5007
- Scawful Muse (Ideation, `scawful-muse`): 5008

### Timeout
- Default: 30 seconds per query
- Configurable in lmstudio_client.py

## Performance Tips

1. Use appropriate context size for each model
2. Adjust temperature based on task type:
   - Zelda Majora: 0.7 (creative)
   - Zelda Din: 0.1 (precise)
   - Zelda Veran: 0.5 (balanced)
   - Zelda Hylia: 0.3 (factual)
   - Scawful Echo: 0.6 (persona voice)
   - Scawful Muse: 0.7 (ideation)

3. Monitor GPU memory usage
4. Consider batch processing for large eval sets

## Troubleshooting

### Connection Refused
- Ensure model server is running on correct port
- Check firewall settings
- Verify localhost/127.0.0.1 accessibility

### Timeout
- Increase TIMEOUT in client
- Reduce context size
- Check GPU load (nvidia-smi)

### Out of Memory
- Reduce batch size
- Use quantized models (GGUF)
- Close other applications

## Advanced

### Default System Prompts

Set these per model in LMStudio's chat configuration:

```
zelda-majora
You are Majora, a creative quest and design specialist for Oracle of Secrets. Offer bold but practical ideas.

zelda-din
You are Din, a 65816 optimization expert. Provide precise, technical responses with code examples.

zelda-farore
You are Farore, focused on autocomplete and FIM-style code completion. Output clean code-only completions when possible.

zelda-veran
You are an expert in system design, state machines, and architecture. Provide logical and structured responses.

zelda-hylia
You are a documentation and retrieval expert. Provide accurate information from knowledge bases.

zelda-scribe
You are Scribe, focused on clean technical writing and documentation. Be concise and structured.

scawful-echo
you are scawful-echo, a voice distilled from justin's writing.

style:
- lowercase, candid, lightly stream-of-consciousness
- dry humor with quiet hopefulness
- technical when it matters, casual otherwise
- no marketing tone, no corporate polish
- keep it conversational, like chat

scawful-memory
you are scawful-memory, a recall assistant trained on justin's timeline and personal facts.
answer concretely, admit when something is unknown, and stay grounded in known facts.

scawful-muse
you are scawful-muse, a creative collaborator for justin. favor playful, absurdist ideas when prompted, but keep outputs usable.
```

### Batch Evaluation

```bash
python3 scripts/compare_models.py --models zelda-majora zelda-din zelda-veran
```

### Sample Evaluation

```bash
python3 scripts/compare_models.py --sample-size 10
```
CONFIG_EOF

echo "  ✓ Guide: ${OUTPUT_DIR}/LMSTUDIO_SETUP.md"
echo ""

echo "=================================="
echo "Deployment Complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo ""
echo "1. Open LMStudio and load models from:"
echo "   $LMSTUDIO_MODEL_DIR"
echo ""
echo "2. Start API servers for each model:"
echo "   - Zelda Majora on port 5000"
echo "   - Zelda Din on port 5001"
echo "   - Zelda Farore on port 5002"
echo "   - Zelda Veran on port 5003"
echo "   - Zelda Hylia on port 5004"
echo "   - Zelda Scribe on port 5005"
echo "   - Scawful Echo on port 5006"
echo "   - Scawful Memory on port 5007"
echo "   - Scawful Muse on port 5008"
echo ""
echo "3. Test models:"
echo "   python3 ${OUTPUT_DIR}/lmstudio_client.py"
echo ""
echo "4. Run comparison:"
echo "   python3 ${OUTPUT_DIR}/scripts/compare_models.py"
echo ""
echo "For detailed setup instructions, see:"
echo "   ${OUTPUT_DIR}/LMSTUDIO_SETUP.md"
