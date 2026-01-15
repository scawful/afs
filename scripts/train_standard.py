#!/usr/bin/env python3
"""
Standard LoRA training script without Unsloth.
Works with any model and uses standard transformers + PEFT.
"""

import json
import argparse
from pathlib import Path
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from datasets import Dataset

def load_training_data(path: str) -> list:
    """Load JSONL training data."""
    examples = []
    with open(path, 'r') as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line))
    return examples

def format_chat_template(example, tokenizer):
    """Format example into chat template."""
    messages = [
        {"role": "system", "content": example.get("system", "You are a helpful AI assistant.")},
        {"role": "user", "content": example["instruction"]},
        {"role": "assistant", "content": example["output"]}
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

    return {"text": text}

def main():
    parser = argparse.ArgumentParser(description="Train model with standard LoRA")
    parser.add_argument("--data", type=str, required=True, help="Path to training JSONL")
    parser.add_argument("--output", type=str, required=True, help="Output directory for LoRA adapter")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-Coder-3B-Instruct")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    args = parser.parse_args()

    print(f"Loading training data from {args.data}...")
    examples = load_training_data(args.data)
    print(f"Loaded {len(examples)} examples")

    print(f"\nLoading base model: {args.base_model}")

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load model in 4-bit for memory efficiency
    from transformers import BitsAndBytesConfig

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
        bnb_4bit_use_double_quant=True
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        quantization_config=bnb_config,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    # Prepare model for training
    model = prepare_model_for_kbit_training(model)

    # Configure LoRA
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Prepare dataset
    print("\nFormatting dataset...")
    dataset = Dataset.from_list(examples)
    dataset = dataset.map(
        lambda x: format_chat_template(x, tokenizer),
        remove_columns=dataset.column_names
    )

    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_seq_length,
            padding=False
        )

    tokenized_dataset = dataset.map(
        tokenize_function,
        batched=True,
        remove_columns=dataset.column_names
    )

    # Training arguments
    training_args = TrainingArguments(
        output_dir=args.output,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.lr,
        fp16=True,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=2,
        warmup_steps=100,
        lr_scheduler_type="cosine",
        optim="paged_adamw_8bit",
        remove_unused_columns=False,
        report_to="none"
    )

    # Data collator
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )

    # Create trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator
    )

    # Train
    print("\n" + "="*50)
    print("Starting training...")
    print("="*50 + "\n")

    trainer.train()

    # Save final model
    print(f"\nSaving LoRA adapter to {args.output}")
    model.save_pretrained(args.output)
    tokenizer.save_pretrained(args.output)

    print("\n✅ Training complete!")
    print(f"LoRA adapter saved to: {args.output}")

if __name__ == "__main__":
    main()
