#!/usr/bin/env python3
"""
Merge 7b_asm_v4 LoRA adapters into Qwen2.5-Coder-7B to create 'nayru' model.

Run on medical-mechanica (Windows GPU) via:
  ssh starw@100.104.53.21 "python D:/afs_training/scripts/merge_nayru.py"

Or copy this script there first:
  scp merge_nayru.py starw@100.104.53.21:D:/afs_training/scripts/
"""

import argparse
import os
import sys
from pathlib import Path

def merge_model(
    base_model: str = "Qwen/Qwen2.5-Coder-7B-Instruct",
    lora_path: str = None,
    output_path: str = None,
    device: str = "cuda"
):
    """Merge LoRA adapters into base model."""

    # Import here to fail fast if deps missing
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    # Default paths for medical-mechanica
    if lora_path is None:
        # Try Windows path first, then Linux
        windows_path = Path("D:/afs_training/checkpoints/7b_asm_v4/lora_adapters")
        linux_path = Path("/home/halext/models/afs_training/7b_asm_v4/lora_adapters")

        if windows_path.exists():
            lora_path = str(windows_path)
        elif linux_path.exists():
            lora_path = str(linux_path)
        else:
            raise FileNotFoundError(f"LoRA adapters not found at {windows_path} or {linux_path}")

    if output_path is None:
        # Default output location
        windows_out = Path("D:/afs_training/models/nayru")
        linux_out = Path("/home/halext/models/nayru")

        if Path("D:/").exists():
            output_path = str(windows_out)
        else:
            output_path = str(linux_out)

    print(f"=== Merging LoRA to create 'nayru' model ===")
    print(f"Base model: {base_model}")
    print(f"LoRA path:  {lora_path}")
    print(f"Output:     {output_path}")
    print(f"Device:     {device}")
    print()

    # Determine dtype based on device
    if device == "cuda":
        dtype = torch.bfloat16
        device_map = "auto"
    else:
        dtype = torch.float32
        device_map = "cpu"

    print("Loading base model...")
    base = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=dtype,
        device_map=device_map,
        trust_remote_code=True
    )

    print("Loading LoRA adapters...")
    model = PeftModel.from_pretrained(base, lora_path)

    print("Merging LoRA into base model...")
    merged = model.merge_and_unload()

    print(f"Saving merged model to {output_path}...")
    os.makedirs(output_path, exist_ok=True)
    merged.save_pretrained(output_path)

    print("Saving tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    tokenizer.save_pretrained(output_path)

    # Create model card
    card_path = Path(output_path) / "README.md"
    card_path.write_text(f"""# Nayru - ALTTP ASM Expert Model

Fine-tuned Qwen2.5-Coder-7B for Zelda: A Link to the Past ROM hacking.

## Base Model
- {base_model}

## Training
- LoRA fine-tuned on ALTTP assembly, ROM hacking documentation, and Nintendo source code
- Adapters: 7b_asm_v4

## Usage
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("nayru")
tokenizer = AutoTokenizer.from_pretrained("nayru")
```

## Capabilities
- 65816 assembly code generation and explanation
- ALTTP ROM structure and memory maps
- Hook implementation and vanilla routine modification
- Nintendo development patterns from original source
""")

    print()
    print("=== MERGE COMPLETE ===")
    print(f"Model saved to: {output_path}")
    print()
    print("Next steps:")
    print("1. Convert to GGUF: python llama.cpp/convert_hf_to_gguf.py nayru --outtype q4_k_m")
    print("2. Create Ollama model: ollama create nayru -f Modelfile")

    return output_path


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA adapters into nayru model")
    parser.add_argument("--base", default="Qwen/Qwen2.5-Coder-7B-Instruct", help="Base model")
    parser.add_argument("--lora", default=None, help="Path to LoRA adapters")
    parser.add_argument("--output", default=None, help="Output path for merged model")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Device")

    args = parser.parse_args()

    try:
        merge_model(args.base, args.lora, args.output, args.device)
    except Exception as e:
        print(f"ERROR: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
