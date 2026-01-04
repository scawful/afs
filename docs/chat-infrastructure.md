# AFS Chat Infrastructure

Modern chat interface for testing Zelda-trained assembly models (Din, Nayru, Farore, Veran) and Scribe models.

## Quick Start

```bash
# Simple mode - Open WebUI with Windows GPU tunnel preferred
./scripts/chat-service.sh start simple

# Access at http://localhost:3000
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     Open WebUI (Port 3000)                   │
│              Modern chat interface with markdown,            │
│              code highlighting, model switching              │
└─────────────────────┬────────────────────────────────────────┘
                      │ OpenAI-compatible API
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                  AFS Gateway (Port 8000)                     │
│         - Model persona injection (din/nayru/etc)            │
│         - MoE routing based on intent                        │
│         - Backend failover                                   │
└─────────────────────┬────────────────────────────────────────┘
                      │
        ┌─────────────┼─────────────┐
        ▼             ▼             ▼
┌───────────┐  ┌───────────┐  ┌───────────┐
│  Local    │  │  Windows  │  │  vast.ai  │
│  Ollama   │  │  GPU      │  │  on-demand│
│  :11434   │  │  :11435   │  │  :11436   │
└───────────┘  └───────────┘  └───────────┘
```

## Deployment Options

### 1. Simple Mode (Recommended for Testing)

Connects Open WebUI directly to Ollama. Prefers the Windows GPU tunnel
on port 11435, with a fallback to local Ollama on 11434.

```bash
./scripts/chat-service.sh start simple
# legacy wrapper
./scripts/chat-start.sh simple
# or
docker compose -f docker/docker-compose.simple.yml up -d
```

**Requirements:**
- Docker
- Windows GPU tunnel (`localhost:11435`) or local Ollama (`ollama serve`)
- Models loaded on whichever backend is used
- Optional: LiteLLM container (auto-started when Gemini/Anthropic keys are present)

### 2. Full Mode (With Gateway)

Includes AFS Gateway for persona injection, MoE routing, and backend switching.

```bash
./scripts/chat-service.sh start full
# legacy wrapper
./scripts/chat-start.sh full
# or
docker compose -f docker/docker-compose.yml up -d
```

**Endpoints:**
- Open WebUI: http://localhost:3000
- Gateway API: http://localhost:8000
- Health: http://localhost:8000/health

### 3. Development Mode

Run gateway locally with auto-reload for development.

```bash
./scripts/chat-start.sh gateway
# or
afs gateway serve --reload
```

## Configuration and Secrets

Open WebUI reads two env files in simple mode:

- `~/.config/afs/openwebui.env` (non-secret defaults)
- `~/.config/afs/openwebui.secrets.env` (generated at start)

`scripts/chat-service.sh` generates the secrets file on each start by loading
keys from:

- `~/.config/afs/secrets.env`
- `~/.secrets`
- Or the file pointed to by `AFS_SECRETS_FILE`

Supported keys:

- `OPENAI_API_KEYS` and `OPENAI_API_BASE_URLS` (semicolon-separated; order must match)
- `OPENAI_API_KEY` and `OPENROUTER_API_KEY` (auto-combined into `OPENAI_API_KEYS`)
- `ANTHROPIC_API_KEY` (or `CLAUDE_API_KEY` as a fallback)
- `GEMINI_API_KEY` and `GEMINI_API_BASE_URL` (image endpoints only)

Note: this Open WebUI image does not include direct Gemini or Anthropic chat
providers. Use OpenRouter (or another OpenAI-compatible proxy) for those models.

### Cloud Providers via LiteLLM

When `ANTHROPIC_API_KEY` or `GEMINI_API_KEY` is present, `chat-service.sh` starts
LiteLLM (OpenAI-compatible proxy) and appends its base URL to
`OPENAI_API_BASE_URLS` so Open WebUI can list those models.

Generated files:

- `~/.config/afs/litellm.env` (secrets for LiteLLM)
- `~/.config/afs/litellm.yaml` (model list template)

LiteLLM runs only when at least one of `ANTHROPIC_API_KEY`, `GEMINI_API_KEY`,
`LITELLM_MASTER_KEY`, or `LITELLM_API_KEY` is set.

**Unknown / needs verification:** the default model IDs in `litellm.yaml` are
best-guess placeholders. Verify with LiteLLM docs and adjust as needed.

Example `~/.config/afs/secrets.env`:

```bash
export OPENAI_API_KEY="sk-..."
export OPENROUTER_API_KEY="sk-or-..."
# Optional override if you want to control the base URL list explicitly
export OPENAI_API_BASE_URLS="https://api.openai.com/v1;https://openrouter.ai/api/v1"
```

Restart to apply:

```bash
./scripts/chat-service.sh restart simple
```

## Compose Overrides

`chat-service.sh` can target alternate compose files or override the URL it prints:

- `AFS_CHAT_COMPOSE_FILE`: replace the base compose file (default: `docker-compose.simple.yml`)
- `AFS_CHAT_COMPOSE_OVERRIDE`: colon-separated list of extra compose files to apply
- `AFS_CHAT_URL`: override the URL shown in status output

Example (halext-nj):

```bash
AFS_CHAT_COMPOSE_FILE=docker-compose.halext-nj.yml \
AFS_CHAT_URL=https://chat.halext.org \
./scripts/chat-service.sh start simple
```

## Model Personas

The gateway exposes these persona models:

| Model | Expert | Description |
|-------|--------|-------------|
| `din` | din-v2 | Optimization specialist - faster, smaller code |
| `nayru` | nayru-v5 | Code generation - correct, elegant assembly |
| `farore` | farore-v1 | Debugging - find and fix bugs |
| `veran` | fallback | SNES hardware specialist |
| `scribe` | fallback | Documentation and explanations |

Each persona has a themed system prompt and routes to the appropriate fine-tuned model.

## CLI Commands

```bash
# Gateway management
afs gateway serve           # Start API server
afs gateway health          # Check backend health
afs gateway backends        # List backends
afs gateway chat "query"    # Quick chat test
afs gateway docker up       # Start Docker containers

# vast.ai on-demand GPU
afs vastai up               # Provision GPU instance
afs vastai status           # Check instance status
afs vastai down             # Teardown instance
afs vastai tunnel           # Set up SSH tunnel
```

## Service Manager Notes

The `afs services` CLI uses `scripts/chat-service.sh` for Open WebUI so secrets
are synced before startup:

```bash
afs services restart openwebui
```

## Backend Configuration

Backends are configured in `~/.config/afs/backends.json`:

```json
{
  "backends": [
    {
      "name": "local",
      "type": "local",
      "host": "localhost",
      "port": 11434,
      "enabled": true,
      "priority": 1
    },
    {
      "name": "windows",
      "type": "windows",
      "host": "localhost",
      "port": 11435,
      "ssh_host": "medical-mechanica",
      "enabled": true,
      "priority": 2
    },
    {
      "name": "vastai",
      "type": "vastai",
      "host": "localhost",
      "port": 11436,
      "enabled": false,
      "priority": 0
    }
  ]
}
```

Higher priority backends are preferred. The gateway automatically fails over to available backends.

## Windows GPU Setup

For the Windows backend (`medical-mechanica`):

1. Install Ollama on Windows
2. Start Ollama: `ollama serve`
3. Load models: `ollama pull qwen2.5-coder:7b`
4. Set up a one-off SSH tunnel:
   ```bash
   ssh -L 11435:127.0.0.1:11434 medical-mechanica
   ```
5. For a persistent tunnel on macOS, use launchd:
   - LaunchAgent: `~/Library/LaunchAgents/com.afs.openwebui-ollama-tunnel.plist`
   - Kickstart: `launchctl kickstart -k gui/$(id -u)/com.afs.openwebui-ollama-tunnel`

The gateway and Open WebUI will use the Windows backend when available.

## Model Location Tags

Open WebUI is configured to tag Ollama models by host:

- `win.*` with tags `windows`, `gpu`
- `mac.*` with tags `mac`, `local`

This is stored in the Open WebUI config database so models are clearly labeled
by source. If you change `OLLAMA_BASE_URLS`, update the tags accordingly.

## vast.ai On-Demand

For heavy inference or when local resources are insufficient:

```bash
# Provision instance
afs vastai up --gpu RTX_4090

# Check status
afs vastai status

# Set up tunnel
afs vastai tunnel --port 11436

# When done, tear down (saves money!)
afs vastai down
```

**Note:** Requires `vastai` CLI installed (`pip install vastai`).

## API Reference

### OpenAI-Compatible Endpoints

```bash
# List models
curl http://localhost:8000/v1/models

# Chat completion
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "din",
    "messages": [{"role": "user", "content": "Optimize this: LDA #$00 STA $7E0000"}],
    "stream": false
  }'

# Streaming
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nayru",
    "messages": [{"role": "user", "content": "Write a DMA transfer routine"}],
    "stream": true
  }'
```

### AFS-Specific Endpoints

```bash
# Health check with backend status
curl http://localhost:8000/health

# List backends
curl http://localhost:8000/backends

# Activate specific backend
curl -X POST http://localhost:8000/backends/windows/activate

# List MoE experts
curl http://localhost:8000/moe/experts
```

## Private Deployment

See `docs/openwebui-private-deploy.md` for the halext-nj deployment sketch.

## Troubleshooting

### Open WebUI can't connect
- Check that Ollama is running: `curl localhost:11434/api/tags`
- Check the Windows tunnel: `curl localhost:11435/api/tags`
- Check Docker network: `docker logs afs-chat`

### Gateway not routing correctly
- Check health: `afs gateway health`
- Verify backends: `afs gateway backends`

### Models not available
- Load models in Ollama: `ollama pull din-v2:latest`
- Check available: `ollama list`
