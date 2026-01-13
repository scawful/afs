#!/usr/bin/env python3
"""
Merge LoRA adapter with base model and convert to GGUF.

Usage:
    python3 merge_and_convert.py --adapter hylia-v1-lora --output hylia-7b-v1
"""

import argparse
import subprocess
import os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser(description="Merge LoRA and convert to GGUF")
    parser.add_argument("--adapter", type=str, required=True, help="Path to LoRA adapter")
    parser.add_argument("--output", type=str, default="merged-model", help="Output name")
    parser.add_argument("--base-model", type=str, default="Qwen/Qwen2.5-7B-Instruct", help="Base model")
    parser.add_argument("--quant", type=str, default="q8_0", help="Quantization type")
    args = parser.parse_args()

    print("=" * 60)
    print("MERGE AND CONVERT TO GGUF")
    print("=" * 60)

    # Install dependencies
    print("\n[1/6] Installing dependencies...")
    subprocess.run(["pip", "install", "peft", "transformers", "accelerate", "bitsandbytes", "sentencepiece"], check=True)

    # Clone llama.cpp if not exists
    if not Path("llama.cpp").exists():
        print("\n[2/6] Cloning llama.cpp...")
        subprocess.run(["git", "clone", "https://github.com/ggerganov/llama.cpp.git"], check=True)
        subprocess.run(["pip", "install", "-r", "llama.cpp/requirements.txt"], check=True)
    else:
        print("\n[2/6] llama.cpp already exists")

    # Merge LoRA with base model
    print("\n[3/6] Merging LoRA adapter with base model...")
    merge_script = f'''
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import os

print("Loading base model...")
base_model = AutoModelForCausalLM.from_pretrained(
    "{args.base_model}",
    torch_dtype=torch.float16,
    device_map="auto",
    trust_remote_code=True
)

print("Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained("{args.base_model}", trust_remote_code=True)

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base_model, "{args.adapter}")

print("Merging weights...")
model = model.merge_and_unload()

print("Saving merged model...")
os.makedirs("{args.output}-hf", exist_ok=True)
model.save_pretrained("{args.output}-hf", safe_serialization=True)
tokenizer.save_pretrained("{args.output}-hf")

print("Merged model saved to {args.output}-hf")
'''

    with open("merge_temp.py", "w") as f:
        f.write(merge_script)

    subprocess.run(["python", "merge_temp.py"], check=True)

    # Convert to GGUF
    print("\n[4/6] Converting to GGUF format...")
    subprocess.run([
        "python", "llama.cpp/convert_hf_to_gguf.py",
        f"{args.output}-hf",
        "--outfile", f"{args.output}.gguf",
        "--outtype", "f16"
    ], check=True)

    # Quantize
    print(f"\n[5/6] Quantizing to {args.quant}...")
    # Build llama.cpp quantize tool if needed
    if not Path("llama.cpp/build/bin/llama-quantize").exists():
        print("Building llama.cpp...")
        subprocess.run(["cmake", "-B", "build", "-DGGML_CUDA=ON"], cwd="llama.cpp", check=True)
        subprocess.run(["cmake", "--build", "build", "--config", "Release", "-j"], cwd="llama.cpp", check=True)

    subprocess.run([
        "llama.cpp/build/bin/llama-quantize",
        f"{args.output}.gguf",
        f"{args.output}-{args.quant}.gguf",
        args.quant
    ], check=True)

    print("\n[6/6] Cleanup...")
    # Keep the quantized version, remove intermediate files
    os.remove(f"{args.output}.gguf")
    os.remove("merge_temp.py")

    final_path = f"{args.output}-{args.quant}.gguf"
    size_gb = os.path.getsize(final_path) / (1024**3)

    print("\n" + "=" * 60)
    print("CONVERSION COMPLETE!")
    print("=" * 60)
    print(f"Output: {final_path}")
    print(f"Size: {size_gb:.2f} GB")
    print("\nCopy to LMStudio models folder to use.")


if __name__ == "__main__":
    main()
