#!/usr/bin/env python3
"""
Train Din v2 - Code optimization and performance expert.

Uses Qwen2.5-Coder-7B fine-tuned on optimization patterns:
- Performance optimization techniques
- Memory layout optimization
- Cycle-efficient code patterns
- Hardware-specific optimizations (SNES PPU, DMA)
- Compiler optimization strategies

Usage:
    python train_din.py --data din_v2_enhanced.jsonl --output din-v2-lora
"""

import json
import argparse
from pathlib import Path

SYSTEM_PROMPT = """You are Din, a code optimization and performance expert.

Your expertise:
- Performance profiling and bottleneck identification
- Memory access patterns and cache optimization
- Algorithmic complexity analysis
- Hardware-specific optimizations (SNES, modern CPUs)
- Compiler optimizations and code generation
- Cycle-accurate timing analysis

When optimizing code:
1. Identify performance bottlenecks first
2. Consider memory access patterns
3. Optimize hot paths aggressively
4. Preserve correctness and readability
5. Measure and validate improvements"""

def load_training_data(path: str) -> list:
    """Load JSONL training data."""
    examples = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples

def main():
    parser = argparse.ArgumentParser(description="Train Din v2 model")
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output", type=str, default="/workspace/output/din-v2-lora")
    parser.add_argument("--base-model", type=str, default="unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-4)
    args = parser.parse_args()

    print(f"Loading training data from {args.data}...")
    examples = load_training_data(args.data)
    print(f"Loaded {len(examples)} examples")

    # Training requires Unsloth
    from unsloth import FastLanguageModel
    from datasets import Dataset
    from trl import SFTTrainer
    from transformers import TrainingArguments

    print(f"Loading base model: {args.base_model}")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=args.base_model,
        max_seq_length=2048,
        dtype=None,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    # Format data for training
    def format_example(ex):
        instruction = ex.get("instruction", "")
        output = ex.get("output", "")
        return {
            "text": tokenizer.apply_chat_template([
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": instruction},
                {"role": "assistant", "content": output}
            ], tokenize=False)
        }

    dataset = Dataset.from_list([format_example(ex) for ex in examples])

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        args=TrainingArguments(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=4,
            warmup_steps=10,
            num_train_epochs=args.epochs,
            learning_rate=args.lr,
            fp16=True,
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            seed=42,
            output_dir=args.output,
            save_strategy="epoch",
        ),
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving model to {args.output}")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)
    print("Training complete!")

if __name__ == "__main__":
    main()
