#!/usr/bin/env python3
"""Train Majora v1 - Oracle of Secrets codebase expert.

Majora specializes in understanding the Oracle of Secrets SNES ROM hack codebase:
- 65816 assembly code patterns
- Memory layout (WRAM/SRAM variables)
- Quest flow and progression
- Sprite system and behaviors
- Architecture and design patterns

Base model: Qwen2.5-Coder-7B-Instruct (code-specialized model)
Training data mix:
- 70% Oracle codebase samples (docs, assembly, memory maps, quests)
- 20% ToolBench tool use samples
- 10% CodeSearchNet code understanding samples

Prerequisites:
    - models/majora_v1_training.jsonl (mixed training data)
    - Base model: Qwen2.5-Coder-7B-Instruct
    - vast.ai GPU instance with Unsloth

Usage:
    # Local preparation (Mac):
    python3 scripts/train_majora_v1.py --prepare-only \
        --oracle ~/.context/training/oracle/majora_v1_processed/train.jsonl \
        --toolbench ~/.context/training/toolbench/processed/train.jsonl \
        --codesearchnet ~/.context/training/datasets/CodeSearchNet/assembly.jsonl \
        --output models/majora_v1_training.jsonl

    # On vast.ai with A100:
    python3 scripts/train_majora_v1.py \
        --data models/majora_v1_training.jsonl \
        --output /workspace/output/majora-v1-lora \
        --epochs 3

    # Convert to GGUF after training:
    python3 scripts/merge_and_convert.py \
        --base Qwen2.5-Coder-7B \
        --lora /workspace/output/majora-v1-lora \
        --output majora-v1-Q8_0.gguf
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from datetime import datetime

# Training configuration for Majora v1
SYSTEM_PROMPT = """You are Majora, an expert on the Oracle of Secrets SNES ROM hack codebase.

Your expertise:
- 65816 assembly code: Reading, understanding, and explaining SNES assembly
- Memory layout: WRAM ($7E0000+) and SRAM ($700000+) variable organization
- Quest system: Progression flags, event triggers, and quest flow
- Sprite system: Enemy behaviors, animations, and interactions
- Architecture: System organization, module interactions, and design patterns
- Documentation: Understanding and referencing Oracle's extensive docs

When answering:
1. Reference specific files and line numbers when possible
2. Explain assembly patterns in clear language
3. Link to relevant documentation sections
4. Consider memory constraints and SNES hardware limits
5. Provide concrete code examples when helpful
6. Explain the "why" behind design decisions

You have deep knowledge of:
- Core/ : Main game engine and systems
- Sprites/ : Enemy and NPC implementations
- Items/ : Item system and behaviors
- Masks/ : Mask transformation mechanics
- Menu/ : Menu systems and UI
- Overworld/ : Overworld logic and transitions
- Dungeons/ : Dungeon room logic and puzzles
- Docs/ : Comprehensive documentation and guides"""

# LoRA configuration for code understanding
LORA_CONFIG = {
    "r": 16,              # LoRA rank
    "alpha": 32,          # Scaling factor
    "dropout": 0.05,      # Light dropout
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
    "base_model": "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit",  # Code-specialized base
    "epochs": 3,
    "batch_size": 4,
    "gradient_accumulation_steps": 4,
    "learning_rate": 2e-4,
    "warmup_steps": 100,
    "max_seq_length": 4096,  # Longer context for code
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


def prepare_mixed_dataset(
    oracle_path: Path,
    toolbench_path: Path | None = None,
    codesearchnet_path: Path | None = None,
    oracle_ratio: float = 0.7,
    toolbench_ratio: float = 0.2,
    codesearchnet_ratio: float = 0.1,
    output_path: Path | None = None,
    seed: int = 42
) -> Path:
    """Prepare mixed training dataset.

    Args:
        oracle_path: Path to Oracle training data
        toolbench_path: Path to ToolBench data (optional)
        codesearchnet_path: Path to CodeSearchNet data (optional)
        oracle_ratio: Proportion of Oracle samples (default 0.7)
        toolbench_ratio: Proportion of ToolBench samples (default 0.2)
        codesearchnet_ratio: Proportion of CodeSearchNet samples (default 0.1)
        output_path: Output path for mixed data
        seed: Random seed for reproducibility

    Returns:
        Path to mixed training data
    """
    print("=" * 60)
    print("Preparing Mixed Training Dataset")
    print("=" * 60)

    # Load Oracle data (required)
    print(f"\nLoading Oracle data from {oracle_path.name}...")
    oracle_samples = load_samples(oracle_path)
    print(f"  Loaded {len(oracle_samples)} Oracle samples")

    # Calculate target counts
    total_oracle = len(oracle_samples)
    target_toolbench = int(total_oracle * (toolbench_ratio / oracle_ratio))
    target_codesearchnet = int(total_oracle * (codesearchnet_ratio / oracle_ratio))

    # Load ToolBench data (optional)
    toolbench_samples = []
    if toolbench_path and toolbench_path.exists():
        print(f"\nLoading ToolBench data from {toolbench_path.name}...")
        all_toolbench = load_samples(toolbench_path)
        print(f"  Loaded {len(all_toolbench)} ToolBench samples")

        # Sample to target count
        if len(all_toolbench) > target_toolbench:
            random.seed(seed)
            toolbench_samples = random.sample(all_toolbench, target_toolbench)
            print(f"  Sampled {len(toolbench_samples)} samples")
        else:
            toolbench_samples = all_toolbench
            print(f"  Using all {len(toolbench_samples)} samples")

    # Load CodeSearchNet data (optional)
    codesearchnet_samples = []
    if codesearchnet_path and codesearchnet_path.exists():
        print(f"\nLoading CodeSearchNet data from {codesearchnet_path.name}...")
        all_codesearchnet = load_samples(codesearchnet_path)
        print(f"  Loaded {len(all_codesearchnet)} CodeSearchNet samples")

        # Sample to target count
        if len(all_codesearchnet) > target_codesearchnet:
            random.seed(seed)
            codesearchnet_samples = random.sample(all_codesearchnet, target_codesearchnet)
            print(f"  Sampled {len(codesearchnet_samples)} samples")
        else:
            codesearchnet_samples = all_codesearchnet
            print(f"  Using all {len(codesearchnet_samples)} samples")

    # Merge all samples
    print(f"\nMerging datasets...")
    mixed = oracle_samples + toolbench_samples + codesearchnet_samples

    # Shuffle
    random.seed(seed)
    random.shuffle(mixed)

    print(f"  Oracle: {len(oracle_samples)} samples ({len(oracle_samples)/len(mixed)*100:.1f}%)")
    print(f"  ToolBench: {len(toolbench_samples)} samples ({len(toolbench_samples)/len(mixed)*100:.1f}%)")
    print(f"  CodeSearchNet: {len(codesearchnet_samples)} samples ({len(codesearchnet_samples)/len(mixed)*100:.1f}%)")
    print(f"  Total: {len(mixed)} samples")

    # Save mixed data
    if output_path is None:
        output_path = Path("models/majora_v1_training.jsonl")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        for sample in mixed:
            f.write(json.dumps(sample) + '\n')

    print(f"\nSaved mixed dataset to {output_path}")

    return output_path


def train_model(
    training_data_path: Path,
    output_dir: Path,
    config: dict | None = None
) -> None:
    """Train Majora v1 model with Unsloth.

    Args:
        training_data_path: Path to training data
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
    print("Training Majora v1")
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
        description="Train Majora v1 - Oracle of Secrets codebase expert.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Data preparation args
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only prepare training data, don't train",
    )
    parser.add_argument(
        "--oracle",
        type=Path,
        help="Oracle training data path",
    )
    parser.add_argument(
        "--toolbench",
        type=Path,
        help="ToolBench training data path (optional)",
    )
    parser.add_argument(
        "--codesearchnet",
        type=Path,
        help="CodeSearchNet training data path (optional)",
    )
    parser.add_argument(
        "--oracle-ratio",
        type=float,
        default=0.7,
        help="Oracle data ratio (default: 0.7 = 70%%)",
    )
    parser.add_argument(
        "--toolbench-ratio",
        type=float,
        default=0.2,
        help="ToolBench data ratio (default: 0.2 = 20%%)",
    )
    parser.add_argument(
        "--codesearchnet-ratio",
        type=float,
        default=0.1,
        help="CodeSearchNet data ratio (default: 0.1 = 10%%)",
    )

    # Training args
    parser.add_argument(
        "--data",
        type=Path,
        help="Training data path (for training mode)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("/workspace/output/majora-v1-lora"),
        help="Output directory for trained model",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=3,
        help="Number of training epochs (default: 3)",
    )

    args = parser.parse_args()

    # Prepare mode
    if args.prepare_only:
        if not args.oracle:
            print("Error: --oracle required in prepare mode")
            return 1

        oracle_path = args.oracle.expanduser().resolve()
        toolbench_path = args.toolbench.expanduser().resolve() if args.toolbench else None
        codesearchnet_path = args.codesearchnet.expanduser().resolve() if args.codesearchnet else None
        output_path = args.output.expanduser().resolve()

        # Validate inputs
        if not oracle_path.exists():
            print(f"Error: Oracle data not found: {oracle_path}")
            return 1

        # Prepare dataset
        training_data_path = prepare_mixed_dataset(
            oracle_path=oracle_path,
            toolbench_path=toolbench_path,
            codesearchnet_path=codesearchnet_path,
            oracle_ratio=args.oracle_ratio,
            toolbench_ratio=args.toolbench_ratio,
            codesearchnet_ratio=args.codesearchnet_ratio,
            output_path=output_path,
        )

        print(f"\nPrepared training data saved to: {training_data_path}")
        print("Run without --prepare-only to train the model.")
        return 0

    # Train mode
    if not args.data:
        print("Error: --data required in training mode")
        return 1

    training_data_path = args.data.expanduser().resolve()
    output_dir = args.output.expanduser().resolve()

    if not training_data_path.exists():
        print(f"Error: Training data not found: {training_data_path}")
        return 1

    # Override epochs if specified
    config = TRAINING_CONFIG.copy()
    if args.epochs:
        config['epochs'] = args.epochs

    # Train model
    train_model(training_data_path, output_dir, config)

    print(f"\nModel saved to: {output_dir}")
    print("\nNext steps:")
    print("1. Evaluate on Oracle codebase questions")
    print("2. Convert to GGUF: python3 scripts/merge_and_convert.py --base Qwen2.5-Coder-7B --lora {output_dir} --output majora-v1-Q8_0.gguf")
    print("3. Deploy to LMStudio: Copy GGUF to D:\\\\models\\\\gguf\\\\afs\\\\")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
