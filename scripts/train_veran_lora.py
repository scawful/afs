#!/usr/bin/env python3
"""
LoRA fine-tuning for Veran (65816 code explanation expert) using MLX.

Veran is the inverse of Nayru - given assembly code, explain what it does.

Usage:
    python scripts/train_veran_lora.py --prepare    # Prepare training data
    python scripts/train_veran_lora.py --train      # Run LoRA training
    python scripts/train_veran_lora.py --fuse       # Fuse adapters into model
    python scripts/train_veran_lora.py --test       # Test the model
    python scripts/train_veran_lora.py --all        # Run full pipeline
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
TRAIN_DATA = MODELS_DIR / "veran_combined_v2.jsonl"
MLX_DATA_DIR = MODELS_DIR / "veran-lora-data"
MLX_MODEL_DIR = MODELS_DIR / "din-lora-model"  # Reuse din's base model
ADAPTERS_DIR = MODELS_DIR / "veran-lora-adapters"
FUSED_DIR = MODELS_DIR / "veran-lora-fused"

# Base model (same as din - already downloaded)
BASE_MODELS = {
    "3b": "Qwen/Qwen2.5-Coder-3B-Instruct",
    "7b": "Qwen/Qwen2.5-Coder-7B-Instruct",
    "14b": "Qwen/Qwen2.5-Coder-14B-Instruct",
}
BASE_MODEL = BASE_MODELS["7b"]

# System prompt for Veran
VERAN_SYSTEM_PROMPT = """You are Veran, a 65816 assembly code explanation expert. Given assembly code, explain what it does clearly and concisely.

For each code block:
1. State the PURPOSE (what does it accomplish?)
2. Walk through KEY STEPS (how does it work?)
3. Identify PATTERNS (common idioms if applicable)
4. Note ASSUMPTIONS (register modes, memory state)

Be concise. Focus on understanding, not exhaustive detail."""

# Training hyperparameters
LORA_CONFIG = {
    "lora_layers": 8,
    "lora_rank": 8,          # Slightly higher for explanation tasks
    "batch_size": 1,
    "iters": 750,            # More iters for expanded dataset (250 examples)
    "learning_rate": 2e-5,
    "warmup": 50,
    "grad_checkpoint": True,
}


def prepare_data():
    """Convert training data to MLX chat format."""
    print("Preparing Veran training data...")

    MLX_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if not TRAIN_DATA.exists():
        print(f"Error: Training data not found at {TRAIN_DATA}")
        print("Run: python scripts/generate_veran_data.py --all")
        return

    train_examples = []
    valid_examples = []

    with open(TRAIN_DATA) as f:
        examples = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(examples)} examples from {TRAIN_DATA}")

    # Split 90/10 train/valid
    split_idx = int(len(examples) * 0.9)

    for i, ex in enumerate(examples):
        conversation = {
            "messages": [
                {
                    "role": "system",
                    "content": VERAN_SYSTEM_PROMPT
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

    cmd = [
        sys.executable, "-m", "mlx_lm.convert",
        "--hf-path", BASE_MODEL,
        "--mlx-path", str(MLX_MODEL_DIR),
        "-q",
    ]

    subprocess.run(cmd, check=True)
    print(f"Model downloaded to {MLX_MODEL_DIR}")


def train_lora():
    """Run LoRA fine-tuning."""
    print("Starting Veran LoRA training...")

    ADAPTERS_DIR.mkdir(parents=True, exist_ok=True)

    # Ensure model is downloaded
    download_model()

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
        "--steps-per-eval", "100",
        "--save-every", "100",
        "--test-batches", "5",
        "--grad-checkpoint",
    ]

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    print(f"LoRA adapters saved to {ADAPTERS_DIR}")


def fuse_model():
    """Fuse LoRA adapters into base model."""
    print("Fusing Veran LoRA adapters...")

    cmd = [
        sys.executable, "-m", "mlx_lm.fuse",
        "--model", str(MLX_MODEL_DIR),
        "--adapter-path", str(ADAPTERS_DIR),
        "--save-path", str(FUSED_DIR),
    ]

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    print(f"Fused model saved to {FUSED_DIR}")


def test_model():
    """Test the trained model with explanation prompts."""
    print("Testing Veran model...")

    test_cases = [
        "Explain this 65816 code:\nSTZ $10\nSTZ $11\nSTZ $12",
        "Explain this 65816 code:\nLDA $10\nASL A\nTAX\nLDA.l Table,X",
        "Explain this 65816 code:\nREP #$20\nLDA $00\nSTA $10\nSEP #$20",
    ]

    # Use Python API for proper chat format
    test_script = f'''
import mlx.core as mx
from mlx_lm import load, generate

model_path = "{FUSED_DIR if FUSED_DIR.exists() else MLX_MODEL_DIR}"
print(f"Loading model from {{model_path}}...")
model, tokenizer = load(model_path)

system_prompt = """{VERAN_SYSTEM_PROMPT}"""

test_cases = {test_cases}

for i, test in enumerate(test_cases):
    print(f"\\n=== Test {{i+1}} ===")
    print(f"Input: {{test[:50]}}...")

    messages = [
        {{"role": "system", "content": system_prompt}},
        {{"role": "user", "content": test}}
    ]

    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    response = generate(model, tokenizer, prompt=prompt, max_tokens=200, verbose=False)
    print(f"Output: {{response}}")
'''

    cmd = [sys.executable, "-c", test_script]
    subprocess.run(cmd)


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning for Veran")
    parser.add_argument("--prepare", action="store_true", help="Prepare training data")
    parser.add_argument("--train", action="store_true", help="Run LoRA training")
    parser.add_argument("--fuse", action="store_true", help="Fuse adapters into model")
    parser.add_argument("--test", action="store_true", help="Test the model")
    parser.add_argument("--all", action="store_true", help="Run full pipeline")

    args = parser.parse_args()

    if args.all:
        prepare_data()
        train_lora()
        fuse_model()
        test_model()
        return

    if args.prepare:
        prepare_data()

    if args.train:
        train_lora()

    if args.fuse:
        fuse_model()

    if args.test:
        test_model()

    if not any([args.prepare, args.train, args.fuse, args.test, args.all]):
        parser.print_help()


if __name__ == "__main__":
    main()
