#!/usr/bin/env python3
"""
Train Agahnim v2 - asar build and integration expert.

Uses Qwen2.5-Coder-7B fine-tuned on build/integration tasks:
- asar directives (org, pushpc/pullpc, incsrc, assert)
- Namespace management and exports
- Hook patterns for vanilla code modification
- Bank allocation and ROM layout
- Include order and dependency management

Usage:
    python train_agahnim.py --data agahnim_v2_training.jsonl --output agahnim-v2-lora
"""

import json
import argparse
from pathlib import Path

SYSTEM_PROMPT = """You are Agahnim, a 65816 build and integration expert for asar assembler.

Your expertise:
- asar directives (org, pushpc/pullpc, incsrc, assert)
- Namespace management and exports
- Hook patterns for vanilla code modification
- Bank allocation and ROM layout
- Include order and dependency management

When providing solutions:
1. Use proper pushpc/pullpc for patches
2. Include namespace blocks when needed
3. Add boundary assertions for safety
4. Explain include order requirements"""

def load_training_data(path: str) -> list:
    """Load JSONL training data."""
    examples = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples

def main():
    parser = argparse.ArgumentParser(description="Train Agahnim v2 model")
    parser.add_argument("--data", type=str, default="/workspace/data/agahnim_v2_training.jsonl")
    parser.add_argument("--output", type=str, default="/workspace/output/agahnim-v2-lora")
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

    def format_prompt(example):
        msgs = example.get("messages", [])
        text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)
        return {"text": text}

    dataset = Dataset.from_list(examples)
    dataset = dataset.map(format_prompt)

    # Calculate max_steps from epochs
    steps_per_epoch = len(examples) // (args.batch_size * 4)  # 4 = gradient accumulation
    max_steps = steps_per_epoch * args.epochs
    print(f"Training for {args.epochs} epochs = {max_steps} steps")

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        args=TrainingArguments(
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=4,
            warmup_steps=50,
            max_steps=max_steps,
            learning_rate=args.lr,
            fp16=False,
            bf16=True,
            logging_steps=10,
            optim="adamw_8bit",
            weight_decay=0.01,
            lr_scheduler_type="cosine",
            output_dir=args.output,
            save_strategy="steps",
            save_steps=max_steps,  # Save at end
        ),
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving adapter to {args.output}")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    # Also save merged model for GGUF conversion
    print("Merging adapter with base model...")
    model.save_pretrained_merged(
        f"{args.output}-merged",
        tokenizer,
        save_method="merged_16bit",
    )

    print("Training complete!")
    print(f"Adapter saved to: {args.output}")
    print(f"Merged model saved to: {args.output}-merged")
    print("\nNext steps:")
    print("1. Convert to GGUF: python llama.cpp/convert_hf_to_gguf.py ...")
    print("2. Quantize: ./llama.cpp/llama-quantize ...")
    print("3. Deploy to LMStudio")

if __name__ == "__main__":
    main()
