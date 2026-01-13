#!/usr/bin/env python3
"""Train Hylia v2 - Narrative/Lore expert for Oracle of Secrets"""

import argparse
import json
from unsloth import FastLanguageModel
from trl import SFTTrainer
from transformers import TrainingArguments
from datasets import Dataset

def load_training_data(path):
    examples = []
    with open(path, 'r') as f:
        for line in f:
            data = json.loads(line)
            # Format as ChatML
            text = ""
            for msg in data['messages']:
                if msg['role'] == 'system':
                    text += f"<|im_start|>system\n{msg['content']}<|im_end|>\n"
                elif msg['role'] == 'user':
                    text += f"<|im_start|>user\n{msg['content']}<|im_end|>\n"
                elif msg['role'] == 'assistant':
                    text += f"<|im_start|>assistant\n{msg['content']}<|im_end|>\n"
            examples.append({"text": text})
    return examples

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', required=True, help='Training data JSONL')
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=2)
    args = parser.parse_args()

    print(f"Loading training data from {args.data}...")
    examples = load_training_data(args.data)
    print(f"Loaded {len(examples)} examples")
    dataset = Dataset.from_list(examples)

    print("Loading base model: unsloth/Qwen2.5-7B-Instruct-bnb-4bit")
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name="unsloth/Qwen2.5-7B-Instruct-bnb-4bit",
        max_seq_length=2048,
        load_in_4bit=True,
    )

    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
        bias="none",
        use_gradient_checkpointing="unsloth",
    )

    training_args = TrainingArguments(
        output_dir=args.output,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=4,
        num_train_epochs=args.epochs,
        learning_rate=2e-4,
        warmup_ratio=0.1,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        bf16=True,
        optim="adamw_8bit",
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=2048,
        args=training_args,
    )

    print("Starting training...")
    trainer.train()

    print(f"Saving model to {args.output}/final/")
    model.save_pretrained(f"{args.output}/final")
    tokenizer.save_pretrained(f"{args.output}/final")
    print("Training complete!")

if __name__ == "__main__":
    main()
