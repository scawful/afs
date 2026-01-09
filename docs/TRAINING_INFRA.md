# Training Infrastructure

AFS training tooling focuses on producing portable JSONL datasets and evaluation
artifacts that can move between machines and toolchains.

## Output locations (gitignored)

By convention, write generated data and artifacts into gitignored directories:
- `distillation_data/`
- `generated_data/`
- `training_data/`
- `benchmark_results/`
- `models/`
- `data/`
- `benchmarks/`

Use `--output` flags to redirect anywhere else, including external volumes.

## Data export sources

These commands export `TrainingSample` JSONL with optional redaction:
- `afs training memory-export --output training_data/memory.jsonl`
- `afs training history-export --output training_data/history.jsonl`
- `afs training antigravity-export --output training_data/antigravity.jsonl`
- `afs training gemini-export --output training_data/gemini.jsonl`
- `afs training claude-export --output training_data/claude.jsonl`
- `afs training codex-export --output training_data/codex.jsonl`

To import Codex logs into AFS history first:
`afs training codex-history-import --history-root ~/.context/history`

## Preparation and curation

Split, score, and rebalance datasets:
```bash
afs training prepare --input training_data/memory.jsonl --output training_data/splits
afs scoring score --input training_data/memory.jsonl --output training_data/memory_scored.jsonl
afs training rebalance --input training_data/memory_scored.jsonl --output training_data/memory_rebalanced.jsonl
```

Run the full pipeline when you want scoring, augmentation, dedupe, and splits:
```bash
afs pipeline run --input training_data/memory.jsonl --output training_data/pipeline
afs pipeline status --dir training_data/pipeline
```

Use the discriminator tools for filtering:
```bash
afs discriminator data --sources src/ --output training_data/discriminator.jsonl
afs discriminator train --input training_data/discriminator.jsonl --output models/electra
afs discriminator filter --model models/electra --input training_data/memory.jsonl --output training_data/memory_filtered.jsonl
```

## Format conversion

Convert JSONL to a training framework format:
```bash
afs training convert --input training_data/memory_scored.jsonl --format alpaca
```

Default CLI formats include `alpaca`, `sharegpt`, and `openai`. Additional
converters can be registered via plugins; see `src/afs/training/converters/__init__.py`.

## Distillation

Generate, resume, and export distillation runs:
```bash
afs distill generate --count 1000 --output distillation_data
afs distill status --checkpoint distillation_data/checkpoint.jsonl
afs distill export --checkpoint distillation_data/checkpoint.jsonl --output training_data/distilled.jsonl
afs distill teachers
```

## Evaluation and benchmarks

Embedding evaluation and benchmarks are separate entry points:
```bash
afs embeddings index --project afs --source ~/src
afs embeddings eval --project afs --query-file examples/embedding_eval.jsonl --json
afs benchmark run --datasets benchmarks --model my-model
afs benchmark leaderboard
```

## Agent setup (Gemini)

Use the venv helper to keep agent runs consistent across macOS and Linux:
```bash
./scripts/afs-venv
```

For non-interactive agents, prefer `scripts/afs` (it respects `.venv` and
`AFS_PYTHON`). If needed, override with:
- `AFS_PYTHON=/path/to/python3`
- `AFS_VENV=/path/to/.venv`
- `AFS_VENV_EXTRAS=extra1,extra2` (extras defined in the project metadata)

## Portability notes

- Training data is JSONL-first, so exports and conversions remain usable even
  when training happens in a separate toolchain.
- Plugin modules are plain Python; keep plugins pure-Python to avoid compilation
  requirements on new machines.
