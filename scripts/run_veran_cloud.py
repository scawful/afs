#!/usr/bin/env python3
"""
Run Veran model with cloud-trained SNES hardware adapters.

This script loads the Qwen2.5-Coder-7B-Instruct base model with LoRA adapters
trained on SNES hardware documentation.

Usage:
    python scripts/run_veran_cloud.py "LDA #$80\nSTA $2100"
    python scripts/run_veran_cloud.py --interactive
"""

import argparse
import sys
import torch
from pathlib import Path

# Check for required packages
try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
except ImportError:
    print("Required packages not found. Install with:")
    print("  pip install transformers peft bitsandbytes accelerate")
    sys.exit(1)

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
# Use v2 adapters (register-emphasis training)
ADAPTER_PATH = PROJECT_ROOT / "models" / "veran-cloud-adapters-v2"
MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"

SYSTEM_PROMPT = """You are Veran, a 65816 assembly code explanation expert specializing in SNES/Super Famicom hardware.

For each code block:
1. Identify hardware registers being accessed (PPU, CPU, DMA, etc.)
2. Explain the bit fields and their effects
3. Describe what the code accomplishes

Be concise and technically accurate."""


def load_model(use_4bit=True):
    """Load base model with cloud-trained adapters."""
    print(f"Loading model from {MODEL_NAME}...")
    
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    if use_4bit and torch.cuda.is_available():
        print("Using 4-bit quantization for GPU...")
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
    else:
        print("Loading in float16...")
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
    
    print(f"Loading adapters from {ADAPTER_PATH}...")
    model = PeftModel.from_pretrained(model, str(ADAPTER_PATH))
    model.eval()
    
    return model, tokenizer


def explain_code(model, tokenizer, code: str, max_tokens: int = 300) -> str:
    """Generate explanation for 65816 assembly code."""
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": f"Explain this 65816 code:\n{code}"}
    ]
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt")
    
    if torch.cuda.is_available():
        inputs = {k: v.cuda() for k, v in inputs.items()}
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )
    
    response = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True
    )
    return response


def interactive_mode(model, tokenizer):
    """Run interactive REPL for code explanation."""
    print("\nVeran SNES Code Explainer - Interactive Mode")
    print("Enter 65816 assembly code (multi-line supported, empty line to submit)")
    print("Type 'quit' to exit\n")
    
    while True:
        print(">>> ", end="")
        lines = []
        while True:
            try:
                line = input()
            except EOFError:
                return
            
            if line.lower() == "quit":
                print("Goodbye!")
                return
            
            if line == "" and lines:
                break
            lines.append(line)
        
        if not lines:
            continue
        
        code = "\n".join(lines)
        print("\nExplaining...\n")
        response = explain_code(model, tokenizer, code)
        print(response)
        print()


def main():
    parser = argparse.ArgumentParser(description="Veran SNES Code Explainer")
    parser.add_argument("code", nargs="?", help="Code to explain (use \\n for newlines)")
    parser.add_argument("--interactive", "-i", action="store_true", help="Interactive mode")
    parser.add_argument("--no-4bit", action="store_true", help="Disable 4-bit quantization")
    parser.add_argument("--max-tokens", type=int, default=300, help="Max response tokens")
    
    args = parser.parse_args()
    
    if not ADAPTER_PATH.exists():
        print(f"Error: Adapters not found at {ADAPTER_PATH}")
        print("Run cloud training first or download adapters.")
        sys.exit(1)
    
    model, tokenizer = load_model(use_4bit=not args.no_4bit)
    
    if args.interactive:
        interactive_mode(model, tokenizer)
    elif args.code:
        # Replace literal \n with actual newlines
        code = args.code.replace("\\n", "\n")
        response = explain_code(model, tokenizer, code, args.max_tokens)
        print(response)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
