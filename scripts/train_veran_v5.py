#!/usr/bin/env python3
"""Train Veran v5 with rehearsal buffer to prevent catastrophic forgetting.

Veran is the SNES hardware expert specializing in PPU, DMA, HDMA, Mode 7, and timing.

This script:
1. Loads new v5 training data
2. Merges with rehearsal buffer from v1-v4
3. Trains with proper LoRA settings to preserve base knowledge
4. Evaluates on v1-v4 capability tests to verify no regression

Prerequisites:
    - models/veran_v5_new.jsonl (new training data for v5)
    - ~/.context/training/rehearsal/veran_v4.jsonl (rehearsal buffer from previous versions)
    - Base model: Qwen2.5-7B-Instruct

Usage:
    # First, build rehearsal buffer from v1-v4 data:
    python3 scripts/build_rehearsal_buffer.py \\
        --input models/veran_combined_v2.jsonl \\
        --input models/veran_snes_hardware_v2.jsonl \\
        --input models/veran_register_emphasis.jsonl \\
        --output ~/.context/training/rehearsal/veran_v4.jsonl \\
        --version v2 --version v3 --version v4 \\
        --quality-threshold 0.7 \\
        --top-ratio 0.3

    # Then train Veran v5 with rehearsal:
    python3 scripts/train_veran_v5.py \\
        --new-data models/veran_v5_new.jsonl \\
        --rehearsal ~/.context/training/rehearsal/veran_v4.jsonl \\
        --output /workspace/output/veran-v5-lora

    # On vast.ai with A100:
    vast.ai create instance --image unsloth \\
        --command "cd /workspace && python3 scripts/train_veran_v5.py --new-data models/veran_v5_new.jsonl --rehearsal ~/.context/training/rehearsal/veran_v4.jsonl"
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from datetime import datetime

# Training configuration for Veran v5
SYSTEM_PROMPT = """You are Veran, an expert on Super Nintendo (SNES) hardware.

Your expertise:
- Picture Processing Unit (PPU): registers, modes, sprites, backgrounds
- Direct Memory Access (DMA): channel setup, timing, transfers
- Horizontal Direct Memory Access (HDMA): gradients, effects, table setup
- Mode 7: rotation, scaling, affine transformations
- Timing and synchronization: VBlank, HBlank, scanlines
- Hardware registers: $2100-$21FF address range

When answering:
1. Reference specific hardware registers (e.g., $2105 for BG mode)
2. Explain timing constraints and DMA safety windows
3. Provide concrete code examples in 65816 assembly
4. Warn about common pitfalls (forced blank timing, HDMA conflicts, etc.)
5. Consider performance implications"""

# LoRA configuration optimized for knowledge preservation
LORA_CONFIG = {
    "r": 16,              # LoRA rank (moderate)
    "alpha": 32,          # Scaling factor
    "dropout": 0.05,      # Light dropout to prevent overfitting
    "target_modules": [   # Apply to attention and FFN
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj",
    ],
    "bias": "none",
    "task_type": "CAUSAL_LM",
}

# Training hyperparameters
TRAINING_CONFIG = {
    "base_model": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
    "epochs": 3,
    "batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,  # Lower LR to preserve knowledge
    "warmup_steps": 100,
    "max_seq_length": 2048,
    "weight_decay": 0.01,
    "lr_scheduler_type": "cosine",
    "logging_steps": 10,
    "save_steps": 500,
    "eval_steps": 500,
}


def load_samples(path: Path) -> list[dict]:
    """Load training samples from JSONL."""
    samples = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line))
    return samples


def prepare_training_data(
    new_data_path: Path,
    rehearsal_path: Path,
    rehearsal_ratio: float = 0.3,
    output_path: Path | None = None
) -> Path:
    """Prepare training data by merging new data with rehearsal buffer.

    Args:
        new_data_path: Path to new v5 training data
        rehearsal_path: Path to rehearsal buffer
        rehearsal_ratio: Proportion of rehearsal samples (default: 0.3 = 30%)
        output_path: Optional output path (default: temp file)

    Returns:
        Path to merged training data
    """
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

    from afs.training.rehearsal import load_rehearsal_buffer

    print("=" * 60)
    print("Preparing Training Data")
    print("=" * 60)

    # Load new data
    print(f"\nLoading new data from {new_data_path.name}...")
    new_samples = load_samples(new_data_path)
    print(f"  Loaded {len(new_samples)} new samples")

    # Load rehearsal buffer
    print(f"\nLoading rehearsal buffer from {rehearsal_path.name}...")
    buffer = load_rehearsal_buffer(rehearsal_path)
    print(f"  Loaded {len(buffer.samples)} rehearsal samples")

    # Convert TrainingSample to dict for new samples
    from afs.generators.base import TrainingSample
    new_training_samples = [TrainingSample(**s) if isinstance(s, dict) else s for s in new_samples]

    # Merge
    print(f"\nMerging with {rehearsal_ratio:.0%} rehearsal ratio...")
    merged = buffer.merge_with_new_data(
        new_training_samples,
        rehearsal_ratio=rehearsal_ratio,
        shuffle=True,
        seed=42
    )

    rehearsal_count = len(merged) - len(new_samples)
    print(f"  New: {len(new_samples)} samples")
    print(f"  Rehearsal: {rehearsal_count} samples")
    print(f"  Total: {len(merged)} samples")

    # Save merged data
    if output_path is None:
        output_path = Path("models/veran_v5_with_rehearsal.jsonl")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for sample in merged:
            f.write(json.dumps(sample.to_dict()) + '\n')

    print(f"\nSaved merged data to {output_path}")

    return output_path


def train_model(
    training_data_path: Path,
    output_dir: Path,
    config: dict | None = None
) -> None:
    """Train Veran v5 model with Unsloth.

    Args:
        training_data_path: Path to merged training data
        output_dir: Output directory for model
        config: Optional training config (uses defaults if None)
    """
    try:
        from unsloth import FastLanguageModel
        from transformers import TrainingArguments
        from trl import SFTTrainer
    except ImportError:
        print("\nError: Unsloth not installed. This script should be run on a GPU instance with Unsloth.")
        print("Install with: pip install unsloth[cu121] transformers trl")
        return

    config = config or TRAINING_CONFIG

    print("\n" + "=" * 60)
    print("Training Veran v5")
    print("=" * 60)

    # Load base model
    print(f"\nLoading base model: {config['base_model']}...")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=config['base_model'],
        max_seq_length=config['max_seq_length'],
        dtype=None,  # Auto-detect
        load_in_4bit=True,
    )

    # Add LoRA adapters
    print("\nAdding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        **LORA_CONFIG,
    )

    # Load training data
    print(f"\nLoading training data from {training_data_path.name}...")
    samples = load_samples(training_data_path)
    print(f"  Loaded {len(samples)} samples")

    # Format for training (ShareGPT format)
    def format_sample(sample):
        return {
            "conversations": [
                {"from": "system", "value": SYSTEM_PROMPT},
                {"from": "human", "value": sample["instruction"]},
                {"from": "gpt", "value": sample["output"]},
            ]
        }

    formatted_data = [format_sample(s) for s in samples]

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=config['epochs'],
        per_device_train_batch_size=config['batch_size'],
        gradient_accumulation_steps=config['gradient_accumulation_steps'],
        learning_rate=config['learning_rate'],
        warmup_steps=config['warmup_steps'],
        logging_steps=config['logging_steps'],
        save_steps=config['save_steps'],
        eval_steps=config['eval_steps'],
        lr_scheduler_type=config['lr_scheduler_type'],
        weight_decay=config['weight_decay'],
        fp16=True,
        optim="adamw_8bit",
        save_total_limit=3,
        load_best_model_at_end=True,
        report_to="none",  # Disable wandb
    )

    # Create trainer
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=formatted_data,
        max_seq_length=config['max_seq_length'],
        args=training_args,
        packing=False,  # Don't pack samples
    )

    # Train
    print("\nStarting training...")
    print(f"  Epochs: {config['epochs']}")
    print(f"  Batch size: {config['batch_size']}")
    print(f"  Learning rate: {config['learning_rate']}")
    print(f"  Max sequence length: {config['max_seq_length']}")

    trainer.train()

    # Save model
    print(f"\nSaving model to {output_dir}...")
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

    print("\n" + "=" * 60)
    print("Training Complete")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description="Train Veran v5 with rehearsal buffer.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "--new-data",
        type=Path,
        required=True,
        help="New training data for v5 (JSONL)",
    )
    parser.add_argument(
        "--rehearsal",
        type=Path,
        default=Path("~/.context/training/rehearsal/veran_v4.jsonl"),
        help="Rehearsal buffer from v1-v4 (default: ~/.context/training/rehearsal/veran_v4.jsonl)",
    )
    parser.add_argument(
        "--rehearsal-ratio",
        type=float,
        default=0.3,
        help="Proportion of rehearsal samples (default: 0.3 = 30%%)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/workspace/output/veran-v5-lora"),
        help="Output directory for trained model",
    )
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only prepare training data, don't train",
    )

    args = parser.parse_args()

    # Expand paths
    new_data_path = args.new_data.expanduser().resolve()
    rehearsal_path = args.rehearsal.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()

    # Validate inputs
    if not new_data_path.exists():
        print(f"Error: New data file not found: {new_data_path}")
        return 1

    if not rehearsal_path.exists():
        print(f"Error: Rehearsal buffer not found: {rehearsal_path}")
        print("\nFirst build the rehearsal buffer with:")
        print("  python3 scripts/build_rehearsal_buffer.py --input models/veran_*.jsonl --output ~/.context/training/rehearsal/veran_v4.jsonl")
        return 1

    # Prepare training data
    training_data_path = prepare_training_data(
        new_data_path,
        rehearsal_path,
        rehearsal_ratio=args.rehearsal_ratio,
    )

    if args.prepare_only:
        print(f"\nPrepared training data saved to: {training_data_path}")
        print("Run without --prepare-only to train the model.")
        return 0

    # Train model
    train_model(training_data_path, output_dir)

    print(f"\nModel saved to: {output_dir}")
    print("\nNext steps:")
    print("1. Evaluate on v1-v4 capability tests: python3 scripts/evaluate_veran.py --model veran-v5-lora")
    print("2. Convert to GGUF: python3 scripts/merge_and_convert.py --base Qwen2.5-7B --lora {output_dir} --output veran-v5-Q8_0.gguf")
    print("3. Deploy to LMStudio: Copy GGUF to D:\\models\\gguf\\afs\\")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
