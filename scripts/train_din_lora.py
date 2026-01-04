#!/usr/bin/env python3
"""
LoRA fine-tuning for din (65816 optimization expert) using MLX.

Usage:
    python scripts/train_din_lora.py --prepare    # Prepare training data
    python scripts/train_din_lora.py --train      # Run LoRA training
    python scripts/train_din_lora.py --fuse       # Fuse adapters into model
    python scripts/train_din_lora.py --export     # Export to Ollama GGUF
    python scripts/train_din_lora.py --all        # Run full pipeline
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
TRAIN_DATA = MODELS_DIR / "din_combined_training.jsonl"  # v2: 154 examples (50 original + 104 curated)
MLX_DATA_DIR = MODELS_DIR / "din-lora-data-v2"
MLX_MODEL_DIR = MODELS_DIR / "din-lora-model"
ADAPTERS_DIR = MODELS_DIR / "din-lora-adapters-v2"
FUSED_DIR = MODELS_DIR / "din-lora-fused-v2"

# Base models - choose based on available memory
# 16GB RAM: Use 3B or 7B with 4-bit quantization
# 32GB+ RAM: Can use 14B or larger
BASE_MODELS = {
    "3b": "Qwen/Qwen2.5-Coder-3B-Instruct",   # ~2GB, safest for 16GB
    "7b": "Qwen/Qwen2.5-Coder-7B-Instruct",   # ~4GB, works on 16GB
    "14b": "Qwen/Qwen2.5-Coder-14B-Instruct", # ~8GB, needs 32GB+
}
BASE_MODEL = BASE_MODELS["7b"]  # Default to 7B

# Training hyperparameters (tuned for 16GB unified memory)
LORA_CONFIG = {
    "lora_layers": 8,        # Fewer layers = less memory (8 is good balance)
    "lora_rank": 4,          # Lower rank = less memory (4 is minimum useful)
    "batch_size": 1,         # Batch size 1 for 16GB RAM
    "iters": 750,            # v2: 750 iters for expanded 154-example dataset
    "learning_rate": 2e-5,   # Slightly higher LR for fewer iters
    "warmup": 50,            # Warmup steps
    "grad_checkpoint": True, # Enable gradient checkpointing for memory
}

# Memory estimates:
# - Qwen2.5-Coder-7B 4-bit: ~4GB
# - LoRA adapters (rank 4, 8 layers): ~50MB
# - Optimizer states: ~200MB
# - Activations (batch 1): ~2-4GB
# Total: ~8-10GB, leaving headroom for system


def prepare_data():
    """Convert training data to MLX chat format."""
    print("Preparing training data...")

    MLX_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # MLX expects conversation format for instruction tuning
    train_examples = []
    valid_examples = []

    with open(TRAIN_DATA) as f:
        examples = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(examples)} examples from {TRAIN_DATA}")

    # Split 90/10 train/valid
    split_idx = int(len(examples) * 0.9)

    for i, ex in enumerate(examples):
        # Convert to chat format
        conversation = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are Din, a 65816 assembly optimization expert. Output ONLY optimized assembly code, no explanations."
                },
                {
                    "role": "user",
                    "content": ex["instruction"]
                },
                {
                    "role": "assistant",
                    "content": ex["output"]
                }
            ]
        }

        if i < split_idx:
            train_examples.append(conversation)
        else:
            valid_examples.append(conversation)

    # Also add general training data if it exists
    general_train = MODELS_DIR / "train_validated_cleaned.jsonl"
    if general_train.exists():
        with open(general_train) as f:
            for line in f:
                if not line.strip():
                    continue
                ex = json.loads(line)
                # Filter for optimization-related examples
                instruction = ex.get("instruction", "").lower()
                if any(kw in instruction for kw in ["optim", "faster", "smaller", "improve", "reduce"]):
                    conversation = {
                        "messages": [
                            {"role": "system", "content": "You are Din, a 65816 assembly optimization expert. Output ONLY optimized assembly code, no explanations."},
                            {"role": "user", "content": f"Optimize this 65816 code:\n{ex.get('output', '')}"},
                            {"role": "assistant", "content": ex.get("output", "")}  # Self-optimization (identity)
                        ]
                    }
                    train_examples.append(conversation)

    # Write training data
    train_file = MLX_DATA_DIR / "train.jsonl"
    valid_file = MLX_DATA_DIR / "valid.jsonl"

    with open(train_file, "w") as f:
        for ex in train_examples:
            f.write(json.dumps(ex) + "\n")

    with open(valid_file, "w") as f:
        for ex in valid_examples:
            f.write(json.dumps(ex) + "\n")

    print(f"Wrote {len(train_examples)} training examples to {train_file}")
    print(f"Wrote {len(valid_examples)} validation examples to {valid_file}")


def download_model():
    """Download base model if not present."""
    if MLX_MODEL_DIR.exists():
        print(f"Model already exists at {MLX_MODEL_DIR}")
        return

    print(f"Downloading {BASE_MODEL}...")

    # Use mlx_lm to download and convert
    cmd = [
        sys.executable, "-m", "mlx_lm.convert",
        "--hf-path", BASE_MODEL,
        "--mlx-path", str(MLX_MODEL_DIR),
        "-q",  # Quantize to 4-bit for memory efficiency
    ]

    subprocess.run(cmd, check=True)
    print(f"Model downloaded to {MLX_MODEL_DIR}")


def train_lora():
    """Run LoRA fine-tuning."""
    print("Starting LoRA training...")

    ADAPTERS_DIR.mkdir(parents=True, exist_ok=True)

    # Ensure model is downloaded
    download_model()

    # Create config file for LoRA-specific settings
    config_path = MLX_DATA_DIR / "lora_config.yaml"
    config_content = f"""# LoRA fine-tuning config for din v2
lora_parameters:
  rank: {LORA_CONFIG["lora_rank"]}
  alpha: {LORA_CONFIG["lora_rank"] * 2}
  dropout: 0.0
  scale: {LORA_CONFIG["lora_rank"] * 2 / LORA_CONFIG["lora_rank"]}
"""
    with open(config_path, "w") as f:
        f.write(config_content)

    cmd = [
        sys.executable, "-m", "mlx_lm", "lora",
        "--model", str(MLX_MODEL_DIR),
        "--train",
        "--data", str(MLX_DATA_DIR),
        "--adapter-path", str(ADAPTERS_DIR),
        "--batch-size", str(LORA_CONFIG["batch_size"]),
        "--num-layers", str(LORA_CONFIG["lora_layers"]),
        "--iters", str(LORA_CONFIG["iters"]),
        "--learning-rate", str(LORA_CONFIG["learning_rate"]),
        "--steps-per-report", "10",
        "--steps-per-eval", "200",
        "--save-every", "100",
        "--test-batches", "5",
        "--grad-checkpoint",  # Memory optimization
    ]

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    print(f"LoRA adapters saved to {ADAPTERS_DIR}")


def fuse_model():
    """Fuse LoRA adapters into base model."""
    print("Fusing LoRA adapters...")

    cmd = [
        sys.executable, "-m", "mlx_lm.fuse",
        "--model", str(MLX_MODEL_DIR),
        "--adapter-path", str(ADAPTERS_DIR),
        "--save-path", str(FUSED_DIR),
    ]

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    print(f"Fused model saved to {FUSED_DIR}")


def export_to_gguf():
    """Export fused model to GGUF format for Ollama."""
    print("Exporting to GGUF...")

    gguf_path = MODELS_DIR / "din-v4.gguf"

    # MLX models can be converted to GGUF using llama.cpp's convert script
    # or we can use the mlx_lm.convert with --to-gguf flag if available

    # For now, provide instructions
    print("\n" + "="*60)
    print("To export to GGUF for Ollama:")
    print("="*60)
    print(f"""
1. Install llama.cpp:
   brew install llama.cpp

2. Convert MLX model to GGUF:
   python -m mlx_lm.convert \\
       --mlx-path {FUSED_DIR} \\
       --hf-path {FUSED_DIR}/hf_export \\
       --to-hf

3. Then use llama.cpp to create GGUF:
   cd /path/to/llama.cpp
   python convert_hf_to_gguf.py {FUSED_DIR}/hf_export \\
       --outfile {gguf_path} \\
       --outtype q4_k_m

4. Create Ollama model:
   cat > {MODELS_DIR}/din-v4.Modelfile << 'EOF'
FROM {gguf_path}
SYSTEM "You are Din, a 65816 assembly optimization expert. Output ONLY optimized assembly code."
PARAMETER temperature 0.3
PARAMETER top_p 0.9
EOF

   OLLAMA_HOST=http://localhost:11435 ollama create din-v4 -f {MODELS_DIR}/din-v4.Modelfile
""")

    # Alternative: Use the fused model directly with MLX
    print("\nAlternatively, use the fused model directly with MLX:")
    print(f"  python -m mlx_lm.generate --model {FUSED_DIR} --prompt 'Optimize this...'")


def test_model():
    """Test the trained model."""
    print("Testing trained model...")

    test_prompt = "Optimize this 65816 code:\nLDA #$00\nSTA $10\nLDA #$00\nSTA $11"

    cmd = [
        sys.executable, "-m", "mlx_lm.generate",
        "--model", str(FUSED_DIR if FUSED_DIR.exists() else MLX_MODEL_DIR),
        "--adapter-path", str(ADAPTERS_DIR) if ADAPTERS_DIR.exists() else "",
        "--prompt", test_prompt,
        "--max-tokens", "100",
    ]

    # Remove empty adapter path if no adapters
    cmd = [c for c in cmd if c]

    print(f"Prompt: {test_prompt}")
    print("Response:")
    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for din")
    parser.add_argument("--prepare", action="store_true", help="Prepare training data")
    parser.add_argument("--train", action="store_true", help="Run LoRA training")
    parser.add_argument("--fuse", action="store_true", help="Fuse adapters into model")
    parser.add_argument("--export", action="store_true", help="Export to Ollama GGUF")
    parser.add_argument("--test", action="store_true", help="Test the model")
    parser.add_argument("--all", action="store_true", help="Run full pipeline")

    args = parser.parse_args()

    if args.all:
        prepare_data()
        train_lora()
        fuse_model()
        export_to_gguf()
        return

    if args.prepare:
        prepare_data()

    if args.train:
        train_lora()

    if args.fuse:
        fuse_model()

    if args.export:
        export_to_gguf()

    if args.test:
        test_model()

    if not any([args.prepare, args.train, args.fuse, args.export, args.test, args.all]):
        parser.print_help()


if __name__ == "__main__":
    main()
