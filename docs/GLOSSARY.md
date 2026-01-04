# AFS Glossary

> **See also:** `~/src/docs/GLOSSARY.md` for workspace-wide terms and conventions.

## Core Concepts

### AFS (Agentic File System)
The framework for AI agent infrastructure. Manages context directories, orchestration, and provides domain-specific capabilities for ALTTP/65816 assembly tasks.

### Context Root
A `.context/` directory that holds agent state for a project. Contains subdirectories like `scratchpad/`, `memory/`, and `knowledge/`.

### Scratchpad
Agent-writable directory within a context root. Used for working memory, current state, and temporary files.

### Mount Point
A context root linked to a project directory. Can be local (in-project) or external (shared across projects).

---

## Training Concepts

### Decoder Model
An LLM that generates text (Qwen, DeepSeek, GPT). Fine-tuned with instruction-response pairs to generate assembly code.

### Encoder Model
A model that creates embeddings/representations (BERT, ELECTRA). Pre-trained with MLM to understand assembly code structure.

### MLM (Masked Language Modeling)
Pre-training objective where random tokens are masked and the model predicts them. Used for encoder pre-training.

### Instruction Fine-tuning
Training a decoder model on instruction-response pairs. The model learns to follow instructions and generate appropriate outputs.

### LoRA (Low-Rank Adaptation)
Efficient fine-tuning method that trains small adapter layers instead of full model weights.

---

## Data Concepts

### Training Sample
A single example for model training. For decoders: `{instruction, input, output}`. For encoders: just text.

### CoT (Chain of Thought)
Explicit reasoning steps included in training data. Helps models learn to "think through" problems.

### Augmentation
Expanding training data by creating variations (paraphrasing instructions, adding noise, etc.).

### Converter
Transforms training data into format required by specific frameworks (MLX, Alpaca, ChatML, etc.).

---

## Tokenizer Concepts

### Tokenizer
Converts text to token IDs for model input. Different from LLM tokenizers, the ASM tokenizer preserves assembly semantics.

### Vocabulary
The set of tokens a tokenizer knows. Maps token strings to integer IDs.

### Pre-tokenizer
First pass that splits text into semantic units before vocabulary lookup.

### Special Tokens
Reserved tokens with special meaning: `[PAD]`, `[UNK]`, `[CLS]`, `[SEP]`, `[MASK]`.

### Semantic Tokenization
Preserving meaningful units as single tokens. E.g., `LDA $7F00,X` → `["LDA", "$7F00", ",X"]` not `["L", "DA", " ", "$", "7", ...]`.

---

## Assembly Concepts

### 65816
The 16-bit CPU used in the SNES. Superset of the 6502 with 24-bit addressing.

### Opcode
A CPU instruction mnemonic like `LDA`, `STA`, `JSR`, `BNE`.

### Addressing Mode
How an instruction accesses data: immediate (`#$FF`), absolute (`$1234`), indexed (`$1234,X`), indirect (`($12)`), etc.

### asar
The assembler used for SNES ROM hacking. Can validate assembly syntax.

### ALTTP
"A Link to the Past" - The Legend of Zelda game for SNES. Primary domain for AFS assembly capabilities.

---

## Framework Terms

### MLX
Apple's machine learning framework for Apple Silicon. One of the supported training backends.

### Unsloth
Fast fine-tuning library for LLMs. Another supported training backend.

### HuggingFace
ML library ecosystem. The tokenizer and trainer follow HF conventions for compatibility.

### ELECTRA
Encoder model architecture that learns by detecting replaced tokens. Used for quality discrimination.

---

## AFS Module Map

| Module | Type | Purpose |
|--------|------|---------|
| `manager` | Core | Context directory management |
| `config` | Core | Configuration loading |
| `discovery` | Core | Find context roots across projects |
| `orchestration` | Core | Route tasks to agents |
| `plugins` | Core | Plugin discovery and loading |
| `services` | Core | Background daemon definitions |
| `schema` | Core | Policy and config schemas |
| `generators` | Domain | Training data generation |
| `training` | Domain | Model training utilities |
| `tokenizer` | Domain | 65816 assembly tokenizer |
| `discriminator` | Domain | Quality filtering models |
| `knowledge` | Domain | ALTTP/SNES reference data |

---

## Training Pipeline Terms

| Term | Description |
|------|-------------|
| `BaseGenerator` | Abstract class for training data generators |
| `TrainingSample` | Dataclass for a single training example |
| `ASMTokenizer` | Custom tokenizer for 65816 assembly |
| `ASMTrainer` | Trains encoder models with ASM tokenizer |
| `ModelRegistry` | Tracks experiments and enables A/B testing |
| `DatasetSplitter` | Splits data into train/val/test with stratification |

---

## File Conventions

| File | Purpose |
|------|---------|
| `*.jsonl` | Training data (one JSON object per line) |
| `vocab.json` | Tokenizer vocabulary |
| `tokenizer_config.json` | Tokenizer settings |
| `training_config.json` | Training hyperparameters |
| `model_registry.json` | Experiment tracking database |
