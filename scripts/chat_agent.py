#!/usr/bin/env python3
"""
Interactive chat with AFS agents.
Usage: python3 chat_agent.py --agent nayru
"""

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import PeftModel
import argparse
from pathlib import Path
import sys

AGENTS = {
    "nayru": {
        "name": "Nayru",
        "description": "65816 assembly & SNES code expert",
        "system": "You are Nayru, an expert in 65816 assembly language and SNES programming."
    },
    "majora": {
        "name": "Majora",
        "description": "Oracle of Secrets codebase expert",
        "system": "You are Majora, an expert in the Oracle of Secrets C# codebase and Unity development."
    },
    "din": {
        "name": "Din",
        "description": "Code optimization specialist",
        "system": "You are Din, a code optimization expert focused on performance improvements."
    },
    "farore": {
        "name": "Farore",
        "description": "Debugging specialist",
        "system": "You are Farore, a debugging expert who helps identify and fix code issues."
    },
    "veran": {
        "name": "Veran",
        "description": "SNES hardware expert",
        "system": "You are Veran, an expert in SNES hardware including PPU, DMA, and memory mapping."
    }
}

def main():
    parser = argparse.ArgumentParser(description="Chat with AFS agents")
    parser.add_argument("--agent", type=str, required=True,
                       choices=list(AGENTS.keys()),
                       help="Agent to chat with")
    parser.add_argument("--adapters-dir", type=str,
                       default=str(Path.home() / "Downloads/afs_models"),
                       help="Directory containing LoRA adapters")
    args = parser.parse_args()

    agent_info = AGENTS[args.agent]
    print(f"\n{'='*60}")
    print(f"🤖 {agent_info['name']} - {agent_info['description']}")
    print(f"{'='*60}\n")
    print("Loading model... (this may take a minute)\n")

    # Load base model with 4-bit quantization
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.float16,
    )

    base_model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-Coder-3B-Instruct",
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True
    )

    tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-Coder-3B-Instruct",
        trust_remote_code=True
    )

    # Load adapter
    adapter_path = Path(args.adapters_dir) / f"{args.agent}-lora"
    print(f"Loading {agent_info['name']} adapter...")
    model = PeftModel.from_pretrained(base_model, str(adapter_path))
    print(f"✓ {agent_info['name']} ready!\n")

    print("Type your message (or 'quit' to exit):")
    print(f"{'='*60}\n")

    while True:
        try:
            user_input = input(f"You: ").strip()
            if not user_input:
                continue
            if user_input.lower() in ['quit', 'exit', 'q']:
                print(f"\n👋 Goodbye!")
                break

            # Format prompt with system message
            messages = [
                {"role": "system", "content": agent_info['system']},
                {"role": "user", "content": user_input}
            ]

            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = tokenizer(text, return_tensors="pt").to(model.device)

            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=512,
                    temperature=0.7,
                    do_sample=True,
                    top_p=0.95,
                    pad_token_id=tokenizer.eos_token_id,
                    eos_token_id=tokenizer.eos_token_id
                )

            response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True)
            print(f"\n{agent_info['name']}: {response}\n")

        except KeyboardInterrupt:
            print(f"\n\n👋 Goodbye!")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}\n")
            continue

if __name__ == "__main__":
    main()
