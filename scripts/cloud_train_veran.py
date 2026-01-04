#!/usr/bin/env python3
"""Cloud training script for Veran SNES hardware model."""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
    BitsAndBytesConfig
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# Configuration
MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"
DATA_PATH = "/workspace/training/data/training.jsonl"
OUTPUT_DIR = "/workspace/training/output"

SYSTEM_PROMPT = """You are Veran, a 65816 assembly code explanation expert specializing in SNES/Super Famicom hardware."""

print("=" * 60)
print("Veran SNES Hardware Training v2")
print("=" * 60)

# Load data
print("\n[1/5] Loading training data...")
examples = []
with open(DATA_PATH) as f:
    for line in f:
        ex = json.loads(line)
        # Build chat format
        text = f"<|im_start|>system\n{SYSTEM_PROMPT}<|im_end|>\n<|im_start|>user\n{ex['instruction']}<|im_end|>\n<|im_start|>assistant\n{ex['output']}<|im_end|>"
        examples.append({"text": text})

print(f"  Loaded {len(examples)} examples")

# Load tokenizer
print("\n[2/5] Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token

# Load model with 4-bit quantization
print("\n[3/5] Loading model with 4-bit quantization...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
model = prepare_model_for_kbit_training(model)

# LoRA config
print("\n[4/5] Applying LoRA...")
lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# Tokenize
def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=512,
        padding="max_length"
    )

dataset = Dataset.from_list(examples)
tokenized = dataset.map(tokenize, remove_columns=["text"])

# Training
print("\n[5/5] Training...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=5,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    warmup_steps=20,
    logging_steps=10,
    save_steps=100,
    bf16=True,
    optim="paged_adamw_8bit",
    report_to="none",
    remove_unused_columns=False,
)

trainer = Trainer(
    model=model,
    train_dataset=tokenized,
    args=training_args,
    data_collator=DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False),
)

trainer.train()

# Save
print("\n" + "=" * 60)
print("Saving model...")
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("\nTraining complete!")
print(f"Adapters saved to: {OUTPUT_DIR}")
