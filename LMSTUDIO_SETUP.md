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
