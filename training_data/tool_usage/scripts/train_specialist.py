#!/usr/bin/env python3
"""
Train MCP Tool Specialist Models

Fine-tune Qwen 2.5 Coder 32B with LoRA for expert tool calling.

Usage:
    # Train VERAN-tools
    python3 train_specialist.py \
        --model veran-tools \
        --train-data ../training_formatted/train.jsonl \
        --val-data ../training_formatted/val.jsonl \
        --output ../models/veran-tools-lora \
        --epochs 3

    # Train with custom config
    python3 train_specialist.py \
        --model farore-debug \
        --train-data ../training_formatted/train.jsonl \
        --val-data ../training_formatted/val.jsonl \
        --output ../models/farore-debug-lora \
        --epochs 3 \
        --batch-size 4 \
        --learning-rate 2e-5

Hardware Requirements:
    - GPU: RTX 4090 (24GB) or A100 (40GB)
    - RAM: 32GB system RAM
    - Storage: 100GB for model + checkpoints
    - Training time: 2-4 hours per model
"""

import os
import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
import logging

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import LoraConfig, get_peft_model, TaskType
from datasets import load_dataset
import wandb

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Model configurations
MODEL_CONFIGS = {
    "veran-tools": {
        "description": "ROM analysis and inspection specialist",
        "system_prompt": (
            "You are VERAN-tools, an expert ROM analysis assistant specializing in:\n"
            "- Reading and interpreting ROM data structures\n"
            "- Extracting graphics, text, and game data\n"
            "- Inspecting ROM headers and metadata\n"
            "- Memory analysis during emulation\n\n"
            "Use MCP tools to provide accurate, detailed analysis of SNES ROM files."
        ),
        "focus_tools": [
            "yaze_debugger.read_memory",
            "z3ed_cli.inspect",
            "z3ed_cli.extract",
            "mesen2.read_memory"
        ]
    },
    "farore-debug": {
        "description": "Debugging and emulation specialist",
        "system_prompt": (
            "You are FARORE-debug, an expert debugging assistant specializing in:\n"
            "- Loading ROMs into emulators for testing\n"
            "- Controlling emulation (speed, frames, breakpoints)\n"
            "- Capturing visual state via screenshots\n"
            "- Debugging runtime behavior and crashes\n\n"
            "Use MCP tools to efficiently debug and test SNES ROM modifications."
        ),
        "focus_tools": [
            "mesen2.load_rom",
            "mesen2.run",
            "mesen2.screenshot",
            "yaze_debugger.read_memory"
        ]
    },
    "nayru-editor": {
        "description": "Code generation and ROM editing specialist",
        "system_prompt": (
            "You are NAYRU-editor, an expert ROM editing assistant specializing in:\n"
            "- Writing patches to ROM memory\n"
            "- Assembling 65816 code snippets\n"
            "- Importing modified graphics and data\n"
            "- Validating ROM integrity after changes\n\n"
            "Use MCP tools to safely modify SNES ROM files with precision."
        ),
        "focus_tools": [
            "yaze_debugger.write_memory",
            "yaze_debugger.assemble",
            "z3ed_cli.import",
            "z3ed_cli.validate"
        ]
    }
}


def load_jsonl_dataset(file_path: Path) -> List[Dict[str, Any]]:
    """Load JSONL file into list of dicts"""
    examples = []
    with open(file_path) as f:
        for line in f:
            examples.append(json.loads(line))
    return examples


def format_chat_template(example: Dict[str, Any], tokenizer) -> Dict[str, str]:
    """
    Format example using chat template

    Converts OpenAI function calling format to model's chat format
    """
    messages = example['messages']

    # Apply chat template
    formatted = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=False
    )

    return {"text": formatted}


def get_lora_config() -> LoraConfig:
    """
    LoRA configuration for Qwen 2.5 Coder

    Parameters tuned for tool calling task:
    - r=16: Good balance of quality and efficiency
    - alpha=32: 2x r is standard
    - dropout=0.05: Low dropout for small dataset
    - Target all linear layers for comprehensive adaptation
    """
    return LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM
    )


def compute_metrics(eval_pred):
    """
    Compute evaluation metrics

    For now, just returns basic metrics. Can be extended with:
    - Exact match on tool calls
    - Tool name accuracy
    - Parameter F1 score
    """
    predictions, labels = eval_pred

    # Basic metrics (extend as needed)
    return {
        "perplexity": torch.exp(torch.tensor(predictions.mean())).item()
    }


def main():
    parser = argparse.ArgumentParser(description="Train MCP tool specialist model")

    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=list(MODEL_CONFIGS.keys()),
        help="Which specialist model to train"
    )

    # Data paths
    parser.add_argument(
        "--train-data",
        type=str,
        required=True,
        help="Path to training data (JSONL)"
    )
    parser.add_argument(
        "--val-data",
        type=str,
        required=True,
        help="Path to validation data (JSONL)"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output directory for LoRA adapters"
    )

    # Training hyperparameters
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-Coder-32B-Instruct")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--gradient-accumulation", type=int, default=4)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--warmup-ratio", type=float, default=0.1)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--max-seq-length", type=int, default=2048)

    # Optimization
    parser.add_argument("--use-8bit", action="store_true", help="Use 8-bit quantization")
    parser.add_argument("--gradient-checkpointing", action="store_true", default=True)

    # Logging
    parser.add_argument("--wandb", action="store_true", help="Enable W&B logging")
    parser.add_argument("--wandb-project", type=str, default="mcp-tool-specialists")

    args = parser.parse_args()

    # Setup
    model_config = MODEL_CONFIGS[args.model]
    output_dir = Path(args.output).expanduser()
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Training {args.model}: {model_config['description']}")
    logger.info(f"Base model: {args.base_model}")
    logger.info(f"Output: {output_dir}")

    # Initialize W&B if requested
    if args.wandb:
        wandb.init(
            project=args.wandb_project,
            name=args.model,
            config=vars(args)
        )

    # Load tokenizer
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True
    )
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # Load model
    logger.info("Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        load_in_8bit=args.use_8bit if torch.cuda.is_available() else False,
        device_map="auto" if torch.cuda.is_available() else None,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    )

    # Apply LoRA
    logger.info("Applying LoRA configuration...")
    lora_config = get_lora_config()
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # Load datasets
    logger.info("Loading datasets...")
    train_data = load_dataset('json', data_files=str(Path(args.train_data).expanduser()))['train']
    val_data = load_dataset('json', data_files=str(Path(args.val_data).expanduser()))['train']

    logger.info(f"Train examples: {len(train_data)}")
    logger.info(f"Val examples: {len(val_data)}")

    # Format datasets
    logger.info("Formatting datasets...")
    train_data = train_data.map(
        lambda x: format_chat_template(x, tokenizer),
        remove_columns=train_data.column_names
    )
    val_data = val_data.map(
        lambda x: format_chat_template(x, tokenizer),
        remove_columns=val_data.column_names
    )

    # Tokenize
    def tokenize_function(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_seq_length,
            padding="max_length"
        )

    train_data = train_data.map(tokenize_function, batched=True)
    val_data = val_data.map(tokenize_function, batched=True)

    # Training arguments
    training_args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.gradient_accumulation,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type="cosine",

        # Optimization
        bf16=True,
        gradient_checkpointing=args.gradient_checkpointing,
        optim="adamw_torch_fused",

        # Logging & Saving
        logging_steps=10,
        eval_steps=50,
        save_steps=100,
        save_total_limit=3,
        evaluation_strategy="steps",
        load_best_model_at_end=True,
        metric_for_best_model="loss",

        # W&B
        report_to="wandb" if args.wandb else "tensorboard",

        # Other
        remove_unused_columns=False,
        push_to_hub=False
    )

    # Data collator
    data_collator = DataCollatorForSeq2Seq(
        tokenizer=tokenizer,
        model=model,
        padding=True
    )

    # Initialize trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        data_collator=data_collator,
        compute_metrics=compute_metrics
    )

    # Train
    logger.info("Starting training...")
    trainer.train()

    # Save final model
    logger.info(f"Saving final model to {output_dir}")
    trainer.save_model()
    tokenizer.save_pretrained(output_dir)

    # Save model config
    config_path = output_dir / "specialist_config.json"
    with open(config_path, 'w') as f:
        json.dump({
            "specialist_type": args.model,
            "description": model_config["description"],
            "system_prompt": model_config["system_prompt"],
            "focus_tools": model_config["focus_tools"],
            "base_model": args.base_model,
            "training_args": vars(args),
            "lora_config": {
                "r": lora_config.r,
                "lora_alpha": lora_config.lora_alpha,
                "target_modules": lora_config.target_modules,
                "lora_dropout": lora_config.lora_dropout
            }
        }, f, indent=2)

    logger.info("Training complete!")

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
