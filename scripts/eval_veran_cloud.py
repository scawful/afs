#!/usr/bin/env python3
"""Evaluate Veran cloud-trained model on SNES hardware tasks."""

import json
import sys
from pathlib import Path

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel
except ImportError:
    print("Install: pip install transformers peft bitsandbytes accelerate")
    sys.exit(1)

PROJECT_ROOT = Path(__file__).parent.parent
# Use v2 adapters (register-emphasis training)
ADAPTER_PATH = PROJECT_ROOT / "models" / "veran-cloud-adapters-v2"
MODEL_NAME = "Qwen/Qwen2.5-Coder-7B-Instruct"

SYSTEM_PROMPT = """You are Veran, a 65816 assembly code explanation expert specializing in SNES/Super Famicom hardware."""

# Evaluation cases
EVAL_CASES = [
    # Basic
    {"code": "LDA #$00\nSTA $10", "expected": ["store", "zero", "immediate"], "difficulty": "basic"},
    {"code": "LDA $10\nCLC\nADC $11\nSTA $12", "expected": ["add", "carry", "sum"], "difficulty": "basic"},
    {"code": "LDA $10\nASL A\nASL A\nASL A", "expected": ["shift", "multiply", "8"], "difficulty": "basic"},
    
    # Intermediate
    {"code": "LDA $10\nEOR #$FF\nINC A", "expected": ["complement", "negate", "invert"], "difficulty": "intermediate"},
    {"code": "REP #$20\nLDA $00\nSTA $10\nSEP #$20", "expected": ["16-bit", "mode", "REP", "SEP"], "difficulty": "intermediate"},
    
    # SNES Hardware (the focus of cloud training)
    {"code": "LDA #$80\nSTA $2100", "expected": ["screen", "blank", "force", "INIDISP", "display", "PPU"], "difficulty": "snes_hardware"},
    {"code": "LDA #$0F\nSTA $2100", "expected": ["brightness", "max", "display", "enable", "screen"], "difficulty": "snes_hardware"},
    {"code": "LDA #$01\nSTA $420B", "expected": ["DMA", "channel", "enable", "transfer", "MDMA"], "difficulty": "snes_hardware"},
    {"code": "LDA #$01\nSTA $4200", "expected": ["NMI", "interrupt", "enable", "vblank", "NMITIMEN"], "difficulty": "snes_hardware"},
    {"code": "LDA #$80\nSTA $2115", "expected": ["VRAM", "increment", "address", "VMAIN"], "difficulty": "snes_hardware"},
]


def load_model():
    """Load model with adapters."""
    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    
    if torch.cuda.is_available():
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
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME,
            torch_dtype=torch.float16,
            device_map="auto",
            trust_remote_code=True,
        )
    
    model = PeftModel.from_pretrained(model, str(ADAPTER_PATH))
    model.eval()
    return model, tokenizer


def evaluate(model, tokenizer):
    """Run evaluation."""
    results = {"basic": [], "intermediate": [], "snes_hardware": []}
    
    for i, case in enumerate(EVAL_CASES, 1):
        print(f"\n{'='*60}")
        print(f"Test {i}/{len(EVAL_CASES)} [{case['difficulty']}]")
        print(f"Code:\n{case['code']}")
        
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"Explain this 65816 code:\n{case['code']}"}
        ]
        
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=200, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        
        response = tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        print(f"\nResponse:\n{response[:400]}")
        
        # Score
        response_lower = response.lower()
        found = [kw for kw in case["expected"] if kw.lower() in response_lower]
        score = len(found) / len(case["expected"]) * 100
        
        print(f"\nScore: {score:.0f}%")
        print(f"Found: {found}")
        print(f"Missing: {[k for k in case['expected'] if k.lower() not in response_lower]}")
        
        results[case["difficulty"]].append(score)
    
    # Summary
    print(f"\n{'='*60}")
    print("EVALUATION SUMMARY")
    print("="*60)
    
    total_scores = []
    for difficulty, scores in results.items():
        if scores:
            avg = sum(scores) / len(scores)
            total_scores.extend(scores)
            print(f"{difficulty}: {avg:.0f}% ({len(scores)} tests)")
    
    overall = sum(total_scores) / len(total_scores) if total_scores else 0
    print(f"\nOVERALL: {overall:.0f}%")
    
    return results


def main():
    if not ADAPTER_PATH.exists():
        print(f"Error: Adapters not found at {ADAPTER_PATH}")
        sys.exit(1)
    
    model, tokenizer = load_model()
    evaluate(model, tokenizer)


if __name__ == "__main__":
    main()
