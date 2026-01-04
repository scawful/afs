#!/usr/bin/env python3
"""
LoRA fine-tuning for Veran using standard PEFT (Windows/CUDA).
Simpler than unsloth but still fast on GPU.
"""

import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import json
import torch
from pathlib import Path
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from trl import SFTTrainer

# Paths
SCRIPT_DIR = Path(__file__).parent
DATA_FILE = SCRIPT_DIR / "veran_combined_v2.jsonl"
OUTPUT_DIR = SCRIPT_DIR / "veran-peft-adapters"

# Model
BASE_MODEL = "Qwen/Qwen2.5-Coder-7B-Instruct"
MAX_SEQ_LENGTH = 1024

# Training
EPOCHS = 2
BATCH_SIZE = 1
GRAD_ACCUM = 8
LEARNING_RATE = 2e-4

# System prompt
VERAN_SYSTEM = """You are Veran, a 65816 assembly code explanation expert. Given assembly code, explain what it does clearly and concisely.

For each code block:
1. State the PURPOSE (what does it accomplish?)
2. Walk through KEY STEPS (how does it work?)
3. Identify PATTERNS (common idioms if applicable)
4. Note ASSUMPTIONS (register modes, memory state)

Be concise. Focus on understanding, not exhaustive detail."""


def load_data():
    """Load training data."""
    examples = []
    with open(DATA_FILE, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                ex = json.loads(line)
                text = f"""<|im_start|>system
{VERAN_SYSTEM}<|im_end|>
<|im_start|>user
{ex['instruction']}<|im_end|>
<|im_start|>assistant
{ex['output']}<|im_end|>"""
                examples.append({"text": text})
    return Dataset.from_list(examples)


def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("Loading model with 4-bit quantization...")
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    model = prepare_model_for_kbit_training(model)

    print("Adding LoRA adapters...")
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("Loading data...")
    dataset = load_data()
    print(f"Loaded {len(dataset)} examples")

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
            warmup_ratio=0.03,
            logging_steps=5,
            save_steps=50,
            bf16=True,
            optim="paged_adamw_8bit",
            gradient_checkpointing=True,
            max_grad_norm=0.3,
            lr_scheduler_type="cosine",
            seed=42,
            dataloader_num_workers=0,
        ),
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        packing=False,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving to {OUTPUT_DIR}")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    print("Done!")


if __name__ == "__main__":
    main()
