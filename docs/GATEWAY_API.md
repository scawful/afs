# AFS Gateway API Reference

The AFS Gateway provides an OpenAI-compatible API for routing requests to specialized 65816 assembly expert models. It supports multiple backends (local, Windows, vast.ai) with automatic failover and Mixture-of-Experts (MoE) routing.

## Overview

- **FastAPI server** with full OpenAI API compatibility
- **MoE routing** for automatic expert selection based on query intent
- **Multi-backend support** with priority-based failover
- **Persona system** for specialized assembly expertise
- **Streaming support** with Server-Sent Events (SSE)

## Endpoints

### Chat Completions

```
POST /v1/chat/completions
```

OpenAI-compatible chat completions endpoint. Supports both streaming and non-streaming responses.

**Request Body:**

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `model` | string | required | Model or persona name (e.g., `din`, `nayru`, `farore`) |
| `messages` | array | required | Array of message objects with `role` and `content` |
| `temperature` | float | 0.7 | Sampling temperature (0.0-1.0) |
| `top_p` | float | 0.9 | Nucleus sampling parameter |
| `max_tokens` | int | null | Maximum tokens to generate |
| `stream` | bool | false | Enable streaming response |
| `stop` | array/string | null | Stop sequences |

**Example Request:**

```json
{
  "model": "din",
  "messages": [
    {"role": "user", "content": "Optimize this loop for the 65816: LDA $00 : STA $02 : LDA $01 : STA $03"}
  ],
  "temperature": 0.3,
  "stream": false
}
```

**Example Response:**

```json
{
  "id": "chatcmpl-a1b2c3d4",
  "object": "chat.completion",
  "created": 1704288000,
  "model": "din",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "REP #$20 : LDA $00 : STA $02 : SEP #$20\n; 16-bit move saves 2 cycles"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 42,
    "completion_tokens": 28,
    "total_tokens": 70
  }
}
```

### Streaming Response Format

When `stream: true`, the response uses Server-Sent Events (SSE):

```
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1704288000,"model":"din","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1704288000,"model":"din","choices":[{"index":0,"delta":{"content":"REP"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1704288000,"model":"din","choices":[{"index":0,"delta":{"content":" #$20"},"finish_reason":null}]}

data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1704288000,"model":"din","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

### List Models

```
GET /v1/models
```

Returns available models including personas and backend-specific models.

**Example Response:**

```json
{
  "object": "list",
  "data": [
    {
      "id": "din",
      "object": "model",
      "created": 1704288000,
      "owned_by": "afs",
      "expert_name": "din",
      "description": "You are Din, the Goddess of Power and 65816 assembly optimization...",
      "intent": "optimization",
      "backend": "local"
    },
    {
      "id": "nayru",
      "object": "model",
      "created": 1704288000,
      "owned_by": "afs",
      "expert_name": "nayru",
      "intent": "generation",
      "backend": "local"
    },
    {
      "id": "din-v2:latest",
      "object": "model",
      "created": 1704288000,
      "owned_by": "afs",
      "backend": "local"
    }
  ]
}
```

### Health Check

```
GET /health
```

Returns gateway health status and backend availability.

**Example Response:**

```json
{
  "status": "healthy",
  "backends": {
    "local": {
      "healthy": true,
      "error": null,
      "models": ["din-v2:latest", "nayru-v5:latest", "farore-v1:latest", "qwen2.5-coder:7b"]
    },
    "windows": {
      "healthy": true,
      "error": null,
      "models": ["llama3.1:70b", "codellama:34b"]
    },
    "vastai": {
      "healthy": false,
      "error": "Connection refused",
      "models": []
    }
  },
  "active_backend": "windows",
  "version": "0.1.0"
}
```

### List Backends

```
GET /backends
```

Returns all configured backends with their status.

**Example Response:**

```json
{
  "backends": [
    {
      "name": "local",
      "type": "local",
      "enabled": true,
      "priority": 1,
      "healthy": true
    },
    {
      "name": "windows",
      "type": "windows",
      "enabled": true,
      "priority": 2,
      "healthy": true
    },
    {
      "name": "vastai",
      "type": "vastai",
      "enabled": false,
      "priority": 0,
      "healthy": false
    }
  ],
  "active": "windows"
}
```

### Activate Backend

```
POST /backends/{name}/activate
```

Manually switch to a specific backend.

**Example:**

```bash
curl -X POST http://localhost:8000/backends/local/activate
```

**Response:**

```json
{"status": "ok", "active": "local"}
```

### Provision vast.ai Instance

```
POST /backends/vastai/provision?gpu_type=RTX_4090
```

Provision an on-demand vast.ai GPU instance.

**Query Parameters:**

| Parameter | Default | Description |
|-----------|---------|-------------|
| `gpu_type` | RTX_4090 | GPU type to provision |

**Response:**

```json
{"status": "provisioned", "gpu_type": "RTX_4090"}
```

### Teardown vast.ai Instance

```
POST /backends/vastai/teardown
```

Terminate the active vast.ai instance.

**Response:**

```json
{"status": "terminated"}
```

### List MoE Experts

```
GET /moe/experts
```

Returns configured Mixture-of-Experts routing information.

**Example Response:**

```json
{
  "experts": [
    {
      "name": "din",
      "model_id": "din-v2:latest",
      "intent": "optimization",
      "description": "65816 assembly optimization specialist"
    },
    {
      "name": "nayru",
      "model_id": "nayru-v5:latest",
      "intent": "generation",
      "description": "Code generation specialist"
    },
    {
      "name": "farore",
      "model_id": "farore-v1:latest",
      "intent": "debugging",
      "description": "Debugging and error analysis specialist"
    }
  ]
}
```

## Personas

Personas are virtual models that inject specialized system prompts and route to appropriate expert models.

| Persona | Intent | Model | Temperature | Description |
|---------|--------|-------|-------------|-------------|
| `din` | optimization | din-v2:latest | 0.3 | Goddess of Power - 65816 optimization specialist. Focus: cycle counting, register optimization, 16-bit operations. |
| `nayru` | generation | nayru-v5:latest | 0.5 | Goddess of Wisdom - Code generation specialist. Focus: correctness, readability, clean subroutine design. |
| `farore` | debugging | farore-v1:latest | 0.4 | Goddess of Courage - Debugging specialist. Focus: register state analysis, stack issues, branch logic. |
| `veran` | analysis | qwen2.5-coder:7b | 0.5 | Sorceress of Shadows - SNES hardware specialist. Focus: PPU, APU, DMA, HDMA, Mode 7. |
| `scribe` | general | qwen2.5-coder:7b | 0.6 | Royal Scribe - Documentation specialist. Focus: instruction docs, code comments, tutorials. |

### Using Personas

Request with a persona name as the model:

```json
{
  "model": "veran",
  "messages": [
    {"role": "user", "content": "Explain Mode 7 rotation registers"}
  ]
}
```

The gateway will:
1. Inject the persona's system prompt
2. Apply persona-specific temperature/top_p settings
3. Route to the appropriate backend model

## Backend Configuration

### Default Backends

| Name | Type | Host:Port | Priority | Description |
|------|------|-----------|----------|-------------|
| `local` | LOCAL | localhost:11434 | 1 | Local Ollama instance |
| `windows` | WINDOWS | localhost:11435 | 2 | Windows machine via SSH tunnel |
| `vastai` | VASTAI | localhost:11436 | 0 | On-demand vast.ai GPU (disabled by default) |

### Priority and Failover

- Higher priority backends are preferred when healthy
- Gateway automatically selects the highest-priority healthy backend
- If the active backend fails, requests fail over to the next available
- vast.ai is on-demand only (priority 0, disabled by default)

### Backend Properties

```python
BackendConfig(
    name="windows",
    type=BackendType.WINDOWS,
    host="localhost",
    port=11435,
    enabled=True,
    priority=2,
    ssh_host="halext-nj",  # SSH tunnel endpoint
    ssh_port=22,
)
```

### SSH Tunnel Setup (Windows Backend)

The Windows backend expects an SSH tunnel forwarding port 11435 to the Windows machine's Ollama:

```bash
ssh -L 11435:localhost:11434 halext-nj -N
```

## Running the Gateway

### CLI Commands

```bash
# Start the gateway server
afs gateway serve
afs gateway serve --host 0.0.0.0 --port 8000
afs gateway serve --reload  # Development mode with auto-reload

# Check backend health
afs gateway health

# List/manage backends
afs gateway backends
afs gateway backends --activate local

# Quick chat test
afs gateway chat "Optimize LDA $00 : STA $02"
afs gateway chat -m nayru "Generate a DMA transfer routine"
afs gateway chat -m din --stream "Unroll this loop"

# Docker management
afs gateway docker up
afs gateway docker down
afs gateway docker logs
afs gateway docker simple-up  # Minimal deployment
```

### vast.ai Management

```bash
# Provision GPU instance
afs vastai up
afs vastai up --gpu RTX_4090 --disk 50

# Check status
afs vastai status

# Set up SSH tunnel
afs vastai tunnel --port 11436

# Teardown instance
afs vastai down
```

### Direct Python

```bash
# Using uvicorn directly
uvicorn afs.gateway.server:app --host 0.0.0.0 --port 8000

# With reload for development
uvicorn afs.gateway.server:app --host 0.0.0.0 --port 8000 --reload

# Using the module
python -m afs.gateway.server
```

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AFS_GATEWAY_HOST` | 0.0.0.0 | Server bind address |
| `AFS_GATEWAY_PORT` | 8000 | Server port |
| `AFS_BACKEND_CONFIG` | ~/.config/afs/backends.json | Backend configuration file |
| `VASTAI_API_KEY` | - | vast.ai API key for provisioning |

### Docker Deployment

Using the provided docker-compose files:

```bash
# Full deployment
docker compose -f docker/docker-compose.yml up -d

# Simple deployment (gateway only)
docker compose -f docker/docker-compose.simple.yml up -d
```

## Error Responses

All errors follow OpenAI API error format:

```json
{
  "error": {
    "message": "No backend available",
    "type": "service_unavailable",
    "code": 503
  }
}
```

### Common Error Codes

| Code | Meaning |
|------|---------|
| 400 | Bad request (invalid parameters) |
| 500 | Internal server error |
| 503 | No healthy backend available |

### Stream Error Format

Errors during streaming are embedded in the stream:

```
data: {"id":"chatcmpl-abc123","object":"chat.completion.chunk","created":1704288000,"model":"din","choices":[{"index":0,"delta":{"content":"\n\n[Error: Connection timeout]"},"finish_reason":"error"}]}

data: [DONE]
```

## Integration Examples

### Open WebUI

The gateway is designed for use with Open WebUI. Configure Open WebUI with:

- **API Base URL**: `http://localhost:8000/v1`
- **API Key**: Not required (or any placeholder)

### curl

```bash
# Non-streaming
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "din",
    "messages": [{"role": "user", "content": "Optimize: LDA $00"}]
  }'

# Streaming
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "din",
    "messages": [{"role": "user", "content": "Optimize: LDA $00"}],
    "stream": true
  }'
```

### Python (requests)

```python
import requests

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "din",
        "messages": [{"role": "user", "content": "Optimize: LDA $00"}],
    }
)

print(response.json()["choices"][0]["message"]["content"])
```

### Python (openai library)

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="not-needed"  # Gateway doesn't require auth
)

response = client.chat.completions.create(
    model="din",
    messages=[{"role": "user", "content": "Optimize: LDA $00"}],
)

print(response.choices[0].message.content)
```

## Architecture

```
                    +-------------------+
                    |   Open WebUI      |
                    |   or other client |
                    +--------+----------+
                             |
                             v
+----------------------------+---------------------------+
|                     AFS Gateway                        |
|  +-------------+  +---------------+  +-------------+   |
|  |  FastAPI    |  |  MoE Router   |  |  Personas   |   |
|  |  Server     |  |  (intent      |  |  (system    |   |
|  |             |  |   classifier) |  |   prompts)  |   |
|  +------+------+  +-------+-------+  +------+------+   |
|         |                 |                 |          |
|         +--------+--------+-----------------+          |
|                  |                                     |
|         +--------v--------+                            |
|         | Backend Manager |                            |
|         | (health, select)|                            |
|         +--------+--------+                            |
+------------------|-------------------------------------+
                   |
     +-------------+-------------+
     |             |             |
     v             v             v
+--------+   +---------+   +----------+
| LOCAL  |   | WINDOWS |   | VASTAI   |
| :11434 |   | :11435  |   | :11436   |
| Ollama |   | (SSH)   |   | (on-     |
|        |   | Ollama  |   |  demand) |
+--------+   +---------+   +----------+
```
