#!/usr/bin/env python3
"""Example: Train an ASM encoder for AFS domain capabilities.

Part of the AFS (Agentic File System) framework. This script demonstrates
how to train a BERT-style masked language model on 65816 assembly code
using the custom ASM tokenizer.

The resulting encoder can be used by AFS agents for:
- Assembly code understanding and semantic search
- Pre-training for downstream ALTTP/SNES tasks
- Embedding generation for code similarity

Usage:
    python examples/train_asm_encoder.py --data data/alttp_routines.jsonl

Or with a pre-trained tokenizer:
    python examples/train_asm_encoder.py \
        --data data/alttp_routines.jsonl \
        --tokenizer models/asm-tokenizer
"""

import argparse
import json
from pathlib import Path
import sys

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from afs.tokenizer import ASMTokenizer
from afs.training import ASMTrainer, ASMTrainerConfig


def load_training_data(path: Path) -> list[str]:
    """Load training data from JSONL or text file."""
    texts = []

    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            # Try JSONL format
            if line.startswith("{"):
                try:
                    data = json.loads(line)
                    # Support various field names
                    for field in ["text", "code", "input", "routine", "asm"]:
                        if field in data:
                            texts.append(data[field])
                            break
                    continue
                except json.JSONDecodeError:
                    pass

            # Plain text (assume each line is a sample)
            texts.append(line)

    return texts


def main():
    parser = argparse.ArgumentParser(description="Train ASM encoder model")
    parser.add_argument(
        "--data",
        type=Path,
        required=True,
        help="Path to training data (JSONL or text)",
    )
    parser.add_argument(
        "--val-data",
        type=Path,
        default=None,
        help="Optional validation data",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=None,
        help="Path to pre-trained tokenizer (creates new if not provided)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./asm-encoder-model"),
        help="Output directory for model",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Training batch size",
    )
    parser.add_argument(
        "--hidden-size",
        type=int,
        default=256,
        help="Model hidden dimension",
    )
    parser.add_argument(
        "--num-layers",
        type=int,
        default=4,
        help="Number of transformer layers",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=5e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum sequence length",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda", "mps"],
        help="Training device",
    )

    args = parser.parse_args()

    # Load or create tokenizer
    if args.tokenizer and args.tokenizer.exists():
        print(f"Loading tokenizer from {args.tokenizer}")
        tokenizer = ASMTokenizer.load(args.tokenizer)
    else:
        print("Creating new tokenizer...")
        tokenizer = ASMTokenizer(max_length=args.max_length)

        # Train on corpus if creating new
        print("Training tokenizer on corpus...")
        texts = load_training_data(args.data)
        added = tokenizer.train_on_corpus(texts, min_frequency=2)
        print(f"Added {added} tokens from corpus. Vocab size: {len(tokenizer)}")

        # Save tokenizer
        tokenizer_path = args.output / "tokenizer"
        tokenizer.save(tokenizer_path)
        print(f"Saved tokenizer to {tokenizer_path}")

    print(f"Tokenizer vocab size: {len(tokenizer)}")

    # Load training data
    print(f"\nLoading training data from {args.data}")
    train_texts = load_training_data(args.data)
    print(f"Loaded {len(train_texts)} training samples")

    # Sample statistics
    if train_texts:
        sample = train_texts[0]
        tokens = tokenizer.tokenize(sample)
        encoded = tokenizer.encode(sample)
        print(f"\nSample encoding:")
        print(f"  Text: {sample[:80]}...")
        print(f"  Tokens: {tokens[:10]}...")
        print(f"  Token count: {len(tokens)}")
        print(f"  Encoded length: {len(encoded['input_ids'])}")

    # Load validation data
    val_texts = None
    if args.val_data:
        print(f"\nLoading validation data from {args.val_data}")
        val_texts = load_training_data(args.val_data)
        print(f"Loaded {len(val_texts)} validation samples")

    # Configure training
    config = ASMTrainerConfig(
        output_dir=args.output,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        num_heads=max(1, args.hidden_size // 64),  # Scale heads with hidden size
        intermediate_size=args.hidden_size * 4,
        max_position_embeddings=args.max_length,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        device=args.device,
    )

    print(f"\nTraining configuration:")
    print(f"  Hidden size: {config.hidden_size}")
    print(f"  Layers: {config.num_layers}")
    print(f"  Heads: {config.num_heads}")
    print(f"  Max length: {config.max_position_embeddings}")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Device: {config.device}")

    # Train
    print(f"\nStarting training...")
    trainer = ASMTrainer(tokenizer=tokenizer, config=config)
    metrics = trainer.train(train_texts, val_texts)

    print(f"\nTraining complete!")
    print(f"  Final loss: {metrics.get('final_loss', 'N/A')}")
    print(f"  Best val loss: {metrics.get('best_val_loss', 'N/A')}")
    print(f"  Total steps: {metrics.get('global_steps', 'N/A')}")
    print(f"\nModel saved to {args.output}")


if __name__ == "__main__":
    main()
