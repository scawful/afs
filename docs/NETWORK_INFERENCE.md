# Network Inference Setup Guide

Configuring LMStudio on Windows as a network inference server accessible from Mac.

## Current Status (2026-01-11)

**Models Deployed:**
| Model | File | Size | Location |
|-------|------|------|----------|
| Nayru v9 | nayru-v9-q8_0.gguf | 8.1GB | Mac + Windows |
| Hylia v2 | hylia-v2-q8_0.gguf | 8.1GB | Mac + Windows |
| Agahnim v2 | agahnim-v2-q8_0.gguf | 8.1GB | Mac + Windows |
| Router v2 | router-v2-q8_0.gguf | 3.3GB | Mac + Windows |

**Windows (medical-mechanica):** `C:\Users\starw\.lmstudio\models\`
**Mac:** `~/models/gguf/`

## Prerequisites

- Windows machine (medical-mechanica) with LMStudio installed
- Mac (your workstation) on the same local network
- AFS orchestrator updated with remote support

## Option 1: Direct AFS Integration (Command Line)

### Windows Setup (LMStudio Server)

1. Open LMStudio on Windows
2. Load your model (e.g., nayru-7b-v9, majora-7b-v2)
3. Go to "Local Server" tab (left sidebar)
4. Configure server settings:
   - **Port**: 1234 (default)
   - **CORS**: Enable cross-origin requests
   - Click **"Start Server"**

5. Allow Windows Firewall:
   - Windows Security → Firewall → Allow an app
   - Add LMStudio or allow port 1234

### Mac Usage

```bash
# Test connectivity first
curl http://medical-mechanica:1234/v1/models

# Use orchestrator with remote backend
python3 ~/src/lab/afs/tools/orchestrator.py \
  --agent majora \
  --backend lmstudio-remote \
  --prompt "Describe the Time System implementation"

# Or use any -lm agent with remote
python3 ~/src/lab/afs/tools/orchestrator.py \
  --agent nayru-lm \
  --backend lmstudio-remote \
  --prompt "Generate a sprite state machine"
```

### Verify Configuration

The orchestrator uses these backend settings:

```python
BACKENDS = {
    "lmstudio": {
        "host": "http://localhost:1234",
        "endpoint": "/v1/chat/completions",
    },
    "lmstudio-remote": {
        "host": "http://medical-mechanica:1234",
        "endpoint": "/v1/chat/completions",
    },
}
```

If your Windows machine has a different hostname/IP, update `orchestrator.py`.

## Option 2: OpenWebUI (Web Chat Interface)

OpenWebUI provides a ChatGPT-like web interface for local models.

### Install OpenWebUI (Mac or Docker)

**Option A: Docker (Recommended)**
```bash
docker run -d -p 3000:8080 \
  --add-host=host.docker.internal:host-gateway \
  -v open-webui:/app/backend/data \
  --name open-webui \
  ghcr.io/open-webui/open-webui:main
```

**Option B: pip install**
```bash
pip install open-webui
open-webui serve --port 3000
```

### Configure OpenWebUI to Use LMStudio

1. Open http://localhost:3000 in browser
2. Create admin account (first run)
3. Go to **Settings → Admin → Connections**
4. Add OpenAI-compatible endpoint:
   - **URL**: `http://medical-mechanica:1234/v1`
   - **API Key**: `not-needed` (LMStudio doesn't require auth)
5. Click "Verify" then "Save"

### Using OpenWebUI

1. Start new chat
2. Select model dropdown → Choose your LMStudio model
3. Chat normally - requests go to Windows machine

### Inject System Prompts

For AFS-style specialized agents in OpenWebUI:

1. Go to **Workspace → Models**
2. Click **"Create a Model"**
3. Configure:
   - **Name**: `Nayru (65816 Expert)`
   - **Base Model**: Select your GGUF
   - **System Prompt**: Copy from orchestrator.py AGENTS config

Example system prompt for Nayru:
```
You are Nayru, the Goddess of Wisdom and 65816 code generation specialist.
You create elegant, correct assembly code with clear structure for Zelda 3.
Generate sprites, bosses, items, and state machines following ALTTP conventions.
```

## Option 3: Continue.dev (VS Code Integration)

For VS Code users wanting inline code assistance:

1. Install Continue extension in VS Code
2. Edit `~/.continue/config.json`:

```json
{
  "models": [
    {
      "title": "Nayru (LMStudio Remote)",
      "provider": "openai",
      "model": "gguf/ollama/nayru-7b-v9.gguf",
      "apiBase": "http://medical-mechanica:1234/v1",
      "apiKey": "not-needed"
    }
  ]
}
```

3. Use `Cmd+L` to chat with model in VS Code

## Troubleshooting

### Connection Refused

```bash
# Test from Mac
ping medical-mechanica
curl http://medical-mechanica:1234/v1/models
```

- Check Windows Firewall allows port 1234
- Verify LMStudio server is running
- Try IP address instead of hostname

### Model Not Found

LMStudio serves the currently loaded model. Ensure:
1. Model is loaded in LMStudio UI
2. Model name matches what you're requesting

List available models:
```bash
curl http://medical-mechanica:1234/v1/models | jq
```

### Slow Response

- Check network bandwidth between machines
- Consider using Q4_K_M quantization for faster inference
- LMStudio on RTX 3090/4090 should give good speeds

## Performance Tips

1. **Batch requests**: The orchestrator can queue multiple prompts
2. **Use smaller quants**: Q4_K_M is often good enough for most tasks
3. **Keep model loaded**: Avoid model swapping if possible
4. **GPU offload**: Ensure LMStudio is using GPU (check nvidia-smi)

## Security Notes

- LMStudio server has no authentication
- Only expose on trusted local network
- Don't port-forward to internet without auth layer
