#!/usr/bin/env python3
"""
LoRA fine-tuning for Veran using Unsloth (Windows/CUDA).

Much faster than standard PEFT training.
"""

import os
# Disable multiprocessing for Windows compatibility
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import torch
from pathlib import Path
from datasets import Dataset
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForSeq2Seq

# Paths
SCRIPT_DIR = Path(__file__).parent
DATA_FILE = SCRIPT_DIR / "veran_combined_v2.jsonl"
OUTPUT_DIR = SCRIPT_DIR / "veran-unsloth-adapters"

# Model config
BASE_MODEL = "unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit"
MAX_SEQ_LENGTH = 2048
LOAD_IN_4BIT = True

# Training config
LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0
EPOCHS = 3
BATCH_SIZE = 2
GRAD_ACCUM = 4
LEARNING_RATE = 2e-4
WARMUP_STEPS = 10

# System prompt
VERAN_SYSTEM = """You are Veran, a 65816 assembly code explanation expert. Given assembly code, explain what it does clearly and concisely.

For each code block:
1. State the PURPOSE (what does it accomplish?)
2. Walk through KEY STEPS (how does it work?)
3. Identify PATTERNS (common idioms if applicable)
4. Note ASSUMPTIONS (register modes, memory state)

Be concise. Focus on understanding, not exhaustive detail."""


def load_data():
    """Load and format training data."""
    examples = []
    with open(DATA_FILE, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                ex = json.loads(line)
                # Format as chat
                text = f"""<|im_start|>system
{VERAN_SYSTEM}<|im_end|>
<|im_start|>user
{ex['instruction']}<|im_end|>
<|im_start|>assistant
{ex['output']}<|im_end|>"""
                examples.append({"text": text})

    return Dataset.from_list(examples)


def main():
    print(f"Loading base model: {BASE_MODEL}")

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=BASE_MODEL,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=LOAD_IN_4BIT,
        dtype=None,  # Auto-detect
    )

    print("Adding LoRA adapters...")
    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=42,
    )

    print("Loading training data...")
    dataset = load_data()
    print(f"Loaded {len(dataset)} examples")

    # Tokenize without multiprocessing
    def formatting_func(examples):
        return examples["text"]

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        args=TrainingArguments(
            output_dir=str(OUTPUT_DIR),
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            num_train_epochs=EPOCHS,
            learning_rate=LEARNING_RATE,
            warmup_steps=WARMUP_STEPS,
            logging_steps=10,
            save_steps=100,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            optim="adamw_8bit",
            seed=42,
            dataloader_num_workers=0,  # Disable multiprocessing
        ),
        formatting_func=formatting_func,
        max_seq_length=MAX_SEQ_LENGTH,
        dataset_num_proc=1,  # Single process tokenization
        packing=False,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving adapters to {OUTPUT_DIR}")
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    print("Done!")


if __name__ == "__main__":
    main()
