#!/usr/bin/env python3
"""
Merge LoRA adapters with base model and convert to GGUF.

Run on a machine with GPU and enough VRAM (~16GB for 7B model).

Requirements:
    pip install torch transformers peft accelerate
    # For GGUF conversion, also need llama.cpp or llama-cpp-python
"""

import argparse
import subprocess
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


def merge_adapters(
    base_model: str,
    adapter_path: Path,
    output_path: Path,
    device_map: str = "auto",
):
    """Merge LoRA adapters into base model."""
    print(f"Loading base model: {base_model}")

    # Load in float16 for merging
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map=device_map,
        trust_remote_code=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

    print(f"Loading adapters from: {adapter_path}")
    model = PeftModel.from_pretrained(model, str(adapter_path))

    print("Merging adapters...")
    model = model.merge_and_unload()

    print(f"Saving merged model to: {output_path}")
    output_path.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)

    print("Merge complete!")
    return output_path


def convert_to_gguf(
    model_path: Path,
    output_file: Path,
    quantization: str = "q4_k_m",
):
    """Convert merged model to GGUF format using llama.cpp."""
    print(f"Converting to GGUF: {output_file}")

    # First convert to f16 GGUF
    f16_file = output_file.with_suffix(".f16.gguf")

    # Try using llama.cpp convert script
    convert_script = Path.home() / "llama.cpp" / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        # Try common alternative locations
        for alt in [
            Path("/opt/llama.cpp/convert_hf_to_gguf.py"),
            Path("./llama.cpp/convert_hf_to_gguf.py"),
        ]:
            if alt.exists():
                convert_script = alt
                break

    if not convert_script.exists():
        print("llama.cpp not found. Installing...")
        subprocess.run([
            "git", "clone", "https://github.com/ggerganov/llama.cpp.git",
            "--depth", "1"
        ], check=True)
        convert_script = Path("./llama.cpp/convert_hf_to_gguf.py")
        # Install requirements
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-r",
            "./llama.cpp/requirements.txt"
        ], check=True)

    # Convert to GGUF
    print("Converting HF model to GGUF...")
    subprocess.run([
        sys.executable, str(convert_script),
        str(model_path),
        "--outfile", str(f16_file),
        "--outtype", "f16",
    ], check=True)

    # Quantize if requested
    if quantization != "f16":
        print(f"Quantizing to {quantization}...")
        quantize_bin = Path("./llama.cpp/build/bin/llama-quantize")
        if not quantize_bin.exists():
            # Build llama.cpp
            print("Building llama.cpp...")
            subprocess.run(["cmake", "-B", "build", "-DGGML_CUDA=ON"],
                         cwd="./llama.cpp", check=True)
            subprocess.run(["cmake", "--build", "build", "-j"],
                         cwd="./llama.cpp", check=True)

        subprocess.run([
            str(quantize_bin),
            str(f16_file),
            str(output_file),
            quantization.upper(),
        ], check=True)

        # Remove f16 intermediate file
        f16_file.unlink()
    else:
        f16_file.rename(output_file)

    print(f"GGUF file created: {output_file}")
    return output_file


def main():
    parser = argparse.ArgumentParser(description="Merge LoRA and convert to GGUF")
    parser.add_argument("--adapter-path", type=Path, required=True,
                       help="Path to LoRA adapters")
    parser.add_argument("--output-dir", type=Path, default=Path("./merged"),
                       help="Output directory for merged model")
    parser.add_argument("--base-model", type=str,
                       default="Qwen/Qwen2.5-Coder-7B-Instruct",
                       help="Base model name or path")
    parser.add_argument("--quantization", type=str, default="q4_k_m",
                       choices=["f16", "q8_0", "q6_k", "q5_k_m", "q4_k_m", "q4_0"],
                       help="Quantization level")
    parser.add_argument("--model-name", type=str, default="model",
                       help="Name for output GGUF file")
    parser.add_argument("--skip-merge", action="store_true",
                       help="Skip merge step (use existing merged model)")
    parser.add_argument("--skip-convert", action="store_true",
                       help="Skip GGUF conversion")

    args = parser.parse_args()

    merged_path = args.output_dir / "merged"
    gguf_file = args.output_dir / f"{args.model_name}-{args.quantization}.gguf"

    if not args.skip_merge:
        merge_adapters(args.base_model, args.adapter_path, merged_path)

    if not args.skip_convert:
        convert_to_gguf(merged_path, gguf_file, args.quantization)

    print("\nDone!")
    print(f"GGUF file: {gguf_file}")
    print(f"\nTo create Ollama model:")
    print(f"  1. Copy {gguf_file} to your Ollama machine")
    print(f"  2. Update Modelfile FROM path")
    print(f"  3. Run: ollama create {args.model_name} -f Modelfile")


if __name__ == "__main__":
    main()
