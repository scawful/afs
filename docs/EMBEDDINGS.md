# AFS Embeddings

AFS includes a file-based embedding index for semantic search over knowledge
documents. It supports multiple embedding providers and a keyword-only fallback.

## Quick Start

```bash
# Index markdown knowledge docs with Gemini vectors
afs embeddings index \
  --knowledge-path ~/.context/knowledge \
  --provider gemini \
  --include "*.md"

# Search
afs embeddings search \
  --knowledge-path ~/.context/knowledge \
  --provider gemini \
  "how to debug a sprite"
```

## Providers

| Provider | Flag | Default Model | Requires |
|----------|------|---------------|----------|
| None (keyword) | `--provider none` | — | Nothing |
| Gemini | `--provider gemini` | `gemini-embedding-001` | `GEMINI_API_KEY` + `google-genai` or `httpx` |
| Ollama | `--provider ollama` | `nomic-embed-text` | Local Ollama server |
| OpenAI | `--provider openai` | `text-embedding-3-small` | `OPENAI_API_KEY` + `httpx` |
| HuggingFace | `--provider hf` | `nomic-embed-text` | `torch` + `transformers` |

Install optional dependencies:

```bash
pip install -e ".[gemini]"      # google-genai
pip install -e ".[embeddings]"  # httpx (for OpenAI/Gemini HTTP fallback)
pip install -e ".[training]"    # torch + transformers (for HF provider)
```

## Indexing

```bash
afs embeddings index \
  --knowledge-path <root> \
  --provider <provider> \
  --include "*.md" \
  [--model <model>] \
  [--max-files 10000] \
  [--preview-chars 1000] \
  [--embed-chars 2000] \
  [--max-bytes 2000000] \
  [--include-hidden]
```

The index is stored at `<knowledge-path>/embedding_index.json` with per-document
embedding files in `<knowledge-path>/embeddings/`.

Each indexed document stores:
- `id`: unique document identifier (`<root>::<relative_path>`)
- `source_path`: absolute path to the source file
- `text_preview`: first 1000 characters (for display)
- `search_text`: first 2000 characters (for keyword matching)
- `embedding`: vector from the configured provider (empty if `--provider none`)
- `size_bytes`: file size
- `modified_at`: last modification timestamp

### Re-indexing

Re-running `index` overwrites the existing index. This is safe — the index is
derived data and can always be rebuilt.

## Searching

```bash
afs embeddings search \
  --knowledge-path <root> \
  --provider <provider> \
  "<query>" \
  [--top-k 5] \
  [--min-score 0.3] \
  [--preview] \
  [--json]
```

When a vector provider is configured, search computes cosine similarity between
the query embedding and all indexed document embeddings. Without a provider
(`--provider none`), it falls back to keyword scoring against the indexed
`search_text` and document ID.

### Asymmetric Retrieval (Gemini)

Gemini's embedding API supports different task types for documents vs queries.
AFS handles this automatically:

- **Indexing**: uses `RETRIEVAL_DOCUMENT` task type
- **Searching**: uses `RETRIEVAL_QUERY` task type

This improves retrieval quality because the model generates different embedding
representations optimized for each role. Override with `--gemini-task-type`.

## Evaluation

```bash
afs embeddings eval \
  --knowledge-path <root> \
  --provider <provider> \
  --query-file eval_cases.jsonl \
  [--top-k 5] \
  [--min-score 0.3] \
  [--match any|doc_id|path] \
  [--details] \
  [--json]
```

Eval cases are JSONL with one case per line:

```json
{"query": "release checklist", "expected": ["project/release-checklist.md"]}
{"query": "api authentication", "expected_doc_id": "architecture/auth.md"}
{"query": "train model", "expected_path_contains": "models/workflows"}
```

Reports hit rate, MRR (Mean Reciprocal Rank), and average hit score.

## Gemini Integration

The embedding system is part of the broader Gemini integration:

```bash
# One-time setup — MCP registration in ~/.gemini/settings.json
afs gemini setup

# Check everything: API key, SDK, settings, MCP, index, ping
afs gemini status

# Generate context from knowledge base
afs gemini context "sprite development"
afs gemini context "sprite development" --include-content --json
afs gemini context   # dump full knowledge INDEX.md
```

### Available Gemini Embedding Models

| Model | Dimensions | Notes |
|-------|-----------|-------|
| `gemini-embedding-001` | 3072 | Default, production-ready |
| `gemini-embedding-2-preview` | 3072 | Preview, may have improved quality |

### Environment Variables

| Variable | Purpose |
|----------|---------|
| `GEMINI_API_KEY` | Primary API key for Gemini |
| `GOOGLE_API_KEY` | Fallback API key |

### SDK vs HTTP Fallback

The Gemini provider tries `google-genai` SDK first. If the SDK is not installed,
it falls back to a plain HTTP request via `httpx` or `requests`. This means you
can use Gemini embeddings with just `httpx` installed — no need for the full
Google SDK.

## Provider-Specific Options

### Gemini

| Flag | Default | Description |
|------|---------|-------------|
| `--gemini-api-key` | `$GEMINI_API_KEY` | API key override |
| `--gemini-task-type` | `RETRIEVAL_DOCUMENT` | Task type (auto-switches for search) |

### Ollama

| Flag | Default | Description |
|------|---------|-------------|
| `--host` | `http://localhost:11435` | Ollama server URL |

### OpenAI

| Flag | Default | Description |
|------|---------|-------------|
| `--openai-base-url` | `https://api.openai.com/v1` | API base URL |
| `--openai-api-key` | `$OPENAI_API_KEY` | API key override |

### HuggingFace

| Flag | Default | Description |
|------|---------|-------------|
| `--hf-device` | `auto` | Device: auto, cpu, cuda, mps |
| `--hf-max-tokens` | `512` | Max token length |
| `--hf-pooling` | `mean` | Pooling: mean, cls |
| `--hf-no-normalize` | false | Disable L2 normalization |

## Architecture

The embedding system uses a registry-based provider architecture:

```python
from afs.embeddings import create_embed_fn

# Create an embedding function
embed = create_embed_fn("gemini", model="gemini-embedding-001")

# Use it
vector = embed("How to write an ASM hook")
print(f"Dimensions: {len(vector)}")  # 3072
```

Custom providers can be registered:

```python
from afs.embeddings import register_embedding_backend

def my_embed_factory(model, **kwargs):
    def embed(text):
        # your embedding logic
        return [0.0] * 768
    return embed

register_embedding_backend("custom", my_embed_factory)
```

Registered backends: `ollama`, `hf`, `openai`, `gemini`.
