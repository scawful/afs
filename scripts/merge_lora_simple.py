#!/usr/bin/env python3
"""
Simple LoRA merge script - merges LoRA adapter with base model.
Run this first, then convert to GGUF separately.
"""

import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapter with base model")
    parser.add_argument("--adapter", type=str, required=True, help="Path to LoRA adapter")
    parser.add_argument("--output", type=str, required=True, help="Output directory for merged model")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-Coder-3B-Instruct", help="Base model")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"MERGING LORA ADAPTER")
    print(f"{'='*60}")
    print(f"Base model: {args.base_model}")
    print(f"LoRA adapter: {args.adapter}")
    print(f"Output: {args.output}")

    print("\n[1/5] Loading base model...")
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True
    )

    print("[2/5] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)

    print("[3/5] Loading LoRA adapter...")
    model = PeftModel.from_pretrained(base_model, args.adapter)

    print("[4/5] Merging weights...")
    model = model.merge_and_unload()

    print("[5/5] Saving merged model...")
    os.makedirs(args.output, exist_ok=True)
    model.save_pretrained(args.output, safe_serialization=True)
    tokenizer.save_pretrained(args.output)

    print(f"\n{'='*60}")
    print(f"✅ MERGE COMPLETE!")
    print(f"{'='*60}")
    print(f"Merged model saved to: {args.output}")

    # Get model size
    total_size = sum(os.path.getsize(os.path.join(args.output, f))
                     for f in os.listdir(args.output)
                     if os.path.isfile(os.path.join(args.output, f)))
    size_gb = total_size / (1024**3)
    print(f"Total size: {size_gb:.2f} GB\n")

if __name__ == "__main__":
    main()
