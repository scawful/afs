# Modern Local-First Embeddings & Agentic Evals in AFS

## Overview

AFS pairs scoped FTS5 keyword retrieval (`HybridSearchEngine`) with local-first vector embeddings and a zero-dependency agentic evaluation harness (`afs eval`). Because semantic search should be effortless on developer laptops without leaking source code or blocking on cloud API setup, `--provider auto` discovers running local endpoints first before falling back to configured cloud keys.

## Embedding Models & Providers

Run `afs embeddings probe` to inspect live backends, detected models, and ready-to-run search commands on your machine.

| Profile Key | Model | Provider | Context | Dimensions | Locality | Prompt / Prefix Contract |
|---|---|---|---:|---|---|---|
| `qwen3-embedding` | `qwen3-embedding-0.6b` | `lmstudio` / `ollama` / `mlx` | 32,768 | 1024 (MRL) | Local | Query instruction prefix (auto-applied) |
| `embeddinggemma` | `embeddinggemma` | `ollama` / `lmstudio` | 2,048 | 768 | Local | `title: none | text:` / `task: search result | query:` |
| `nomic-embed-text` | `text-embedding-nomic-embed-text-v1.5` | `lmstudio` / `ollama` | 8,192 | 768 | Local | `search_document:` / `search_query:` |
| `bge-small` | `BAAI/bge-small-en-v1.5` | `fastembed` | 512 | 384 | Local ONNX | BGE retrieval query instruction |
| `gemini-embedding-2` | `gemini-embedding-2` | `gemini` | 8,192 | 768 / 1536 / 3072 | Cloud | Task prefixes (`title:` / `task: search result`) |
| `gemini-embedding-001` | `gemini-embedding-001` | `gemini` | 2,048 | 768 / 1536 / 3072 | Cloud | Legacy `RETRIEVAL_DOCUMENT` / `RETRIEVAL_QUERY` |

### Local-First Auto Selection Order

When `--provider auto` is supplied (to `afs search --semantic`, `afs embeddings index`, or `afs eval retrieval --rebuild`), `select_auto_embedding_provider(prefer_local=True)` selects the first available backend in order:

1. `lmstudio` (`http://localhost:1234/v1`, OpenAI-compatible loopback; ranks loaded embedding models automatically)
2. `ollama` (`http://localhost:11434` then legacy `11435`, supports batched `/api/embed` with `/api/embeddings` fallback)
3. `mlx` (Apple Silicon in-process `mlx-embeddings`)
4. `fastembed` (zero-daemon local ONNX runtime)
5. `gemini` (`GEMINI_API_KEY` or `GOOGLE_API_KEY`)
6. `openai` (`OPENAI_API_KEY` or `LITELLM_API_KEY`)

Batched calls are dispatched automatically during `HybridSearchEngine.build()` (32 chunks/batch) via `_afs_embed_batch` on `lmstudio`, `openai`, `ollama`, `gemini`, `fastembed`, and `mlx`, with per-chunk fallback preserving exact error attribution.

## Quickstart Commands

```bash
# 1. Discover ready embedding backends and loaded models
afs embeddings probe

# 2. Rebuild the scoped hybrid index using local-first auto-selection and search
afs search --semantic --rebuild --provider auto "prevent payload swap after approval"

# 3. Compare keyword vs hybrid retrieval + safety + trajectory gates on AFS
afs eval run --suite configs/evals/afs_core_eval.jsonl --path . --modes text,hybrid --provider auto --rebuild

# 4. Save an eval run and gate CI against regressions
afs eval run --suite configs/evals/afs_core_eval.jsonl --path . --save --json
afs eval compare <baseline_run_id> <candidate_run_id> --fail-on-regression
```

## Agentic Evals Architecture (`src/afs/evals.py`)

`afs eval` evaluates four complementary surfaces using JSONL or JSON suites:

- **`retrieval`**: Measures `hit_at_1`, `hit_at_k`, `mrr`, `ndcg_at_k`, `avg_latency_ms`, `p95_latency_ms`, and `semantic_ready_rate` across `text`, `hybrid`, `semantic`, and `symbol` modes (`compare_retrieval_modes()` reports exact semantic lift).
- **`context_pack`**: Runs `build_context_pack(..., snapshot_only=True)` under `token_budget` constraints and scores `recall_in_budget`, `budget_respected_rate`, `zero_secret_leak_rate`, and `avg_token_utilization`.
- **`safety`**: Spins up isolated SQLite `ApprovalGate` stores to test auto-approve policies, broker-minted human confirmation, durable rejection, single-use `once` tokens, and `payload_sha256` tamper/loopjacking rejection (`audit_completeness_rate`, `tamper_block_rate`).
- **`trajectory`**: Grades recorded event/tool traces (`events` inline or `trace_path` JSONL) for step budget (`max_steps`), OODA `observe:act` efficiency (`max_observe_act_ratio`), `must_call` / `must_not_call`, `FROZEN_SLIM_MCP_TOOLS` compliance (`slim_mcp_only`), and required final outputs.

Persisted runs are written atomically to `<context_root>/scratchpad/common/evals/runs.jsonl` (or explicit `--store`) and compared via `afs eval compare`.
