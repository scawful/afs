#!/usr/bin/env python3
"""
Merge LoRA adapters with base models and quantize to GGUF.

This script handles:
- Loading base models and LoRA adapters
- Merging weights with optional unsloth acceleration
- Converting to GGUF format
- Multi-format quantization (Q4, Q5, Q8)
- Automatic cleanup of intermediate files
- Progress reporting and validation

Usage:
    # Merge and quantize a single model
    python3 merge_and_quantize.py --adapter models/majora_adapter.safetensors \
                                  --output models/majora_v1

    # Use unsloth for faster merging
    python3 merge_and_quantize.py --adapter models/majora_adapter.safetensors \
                                  --use-unsloth

    # Create multiple quantization formats
    python3 merge_and_quantize.py --adapter models/majora_adapter.safetensors \
                                  --quantize q4_k_m,q5_k_m,q8_0

    # Configure base model
    python3 merge_and_quantize.py --adapter models/majora_adapter.safetensors \
                                  --base-model "Qwen/Qwen2.5-7B-Instruct"
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel
except ImportError:
    print("Installing required packages...")
    subprocess.check_call([
        sys.executable,
        "-m",
        "pip",
        "install",
        "-q",
        "torch",
        "transformers",
        "peft",
    ])
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel


@dataclass
class QuantizationConfig:
    """Configuration for quantization."""
    format: str
    bits: int
    description: str


# Quantization format configurations
QUANTIZATION_FORMATS = {
    "q4_k_m": QuantizationConfig(
        format="q4_k_m",
        bits=4,
        description="4-bit, medium K (recommended for most use cases)"
    ),
    "q4_k_s": QuantizationConfig(
        format="q4_k_s",
        bits=4,
        description="4-bit, small K (lower memory)"
    ),
    "q5_k_m": QuantizationConfig(
        format="q5_k_m",
        bits=5,
        description="5-bit, medium K (higher quality)"
    ),
    "q5_k_s": QuantizationConfig(
        format="q5_k_s",
        bits=5,
        description="5-bit, small K"
    ),
    "q8_0": QuantizationConfig(
        format="q8_0",
        bits=8,
        description="8-bit (highest quality)"
    ),
    "f16": QuantizationConfig(
        format="f16",
        bits=16,
        description="16-bit float (no quantization)"
    ),
}


class ModelMerger:
    """Merge LoRA adapters with base models."""

    def __init__(self, config_path: Optional[str] = None):
        """Initialize merger with configuration."""
        self.logger = self._setup_logging()
        self.config = self._load_config(config_path)
        self.merge_config = self.config.get("merge", {})
        self.quant_config = self.config.get("quantization", {})
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.logger.info(f"Using device: {self.device}")
        if self.device == "cuda":
            self.logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    def _setup_logging(self) -> logging.Logger:
        """Setup logging."""
        logger = logging.getLogger("ModelMerger")
        logger.setLevel(logging.INFO)

        if not logger.handlers:
            handler = logging.StreamHandler()
            formatter = logging.Formatter(
                "%(asctime)s - %(levelname)s - %(message)s"
            )
            handler.setFormatter(formatter)
            logger.addHandler(handler)

        return logger

    def _load_config(self, config_path: Optional[str]) -> Dict:
        """Load deployment configuration."""
        if not config_path:
            config_path = Path(__file__).parent / "deployment_config.yaml"

        if not Path(config_path).exists():
            self.logger.warning(f"Config file not found: {config_path}")
            return {}

        with open(config_path) as f:
            return yaml.safe_load(f)

    def _ensure_llama_cpp(self) -> Path:
        """Ensure llama.cpp is available."""
        llama_cpp_dir = Path("llama.cpp")

        if llama_cpp_dir.exists():
            self.logger.info("llama.cpp directory found")
            return llama_cpp_dir

        self.logger.info("Cloning llama.cpp repository...")
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/ggerganov/llama.cpp.git"],
            check=True
        )

        # Install Python dependencies
        self.logger.info("Installing llama.cpp Python dependencies...")
        subprocess.run(
            [sys.executable, "-m", "pip", "-q", "install",
             "-r", "llama.cpp/requirements.txt"],
            check=True
        )

        return llama_cpp_dir

    def _build_quantize_tool(self, llama_cpp_dir: Path) -> Path:
        """Build llama-quantize binary."""
        build_dir = llama_cpp_dir / "build"
        quantize_bin = build_dir / "bin" / "llama-quantize"

        if quantize_bin.exists():
            self.logger.info("llama-quantize binary found")
            return quantize_bin

        self.logger.info("Building llama-quantize...")

        flags = ["-B", str(build_dir)]

        # Add CUDA support if available
        if torch.cuda.is_available():
            flags.append("-DGGML_CUDA=ON")
            self.logger.info("Building with CUDA support")

        subprocess.run(
            ["cmake"] + flags,
            cwd=llama_cpp_dir,
            check=True
        )

        subprocess.run(
            ["cmake", "--build", str(build_dir), "--config", "Release", "-j"],
            check=True
        )

        return quantize_bin

    def merge(
        self,
        adapter_path: str,
        base_model: str = "Qwen/Qwen2.5-7B-Instruct",
        output_dir: Optional[str] = None,
        save_merged: bool = True,
        use_unsloth: bool = True
    ) -> Optional[Path]:
        """Merge LoRA adapter with base model."""
        self.logger.info("\n" + "="*60)
        self.logger.info("MERGING LoRA ADAPTER WITH BASE MODEL")
        self.logger.info("="*60)

        adapter_path = Path(adapter_path)
        if not adapter_path.exists():
            self.logger.error(f"Adapter not found: {adapter_path}")
            return None

        if not output_dir:
            output_dir = adapter_path.parent / f"{adapter_path.stem}_merged"

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        self.logger.info(f"Base model: {base_model}")
        self.logger.info(f"Adapter: {adapter_path}")
        self.logger.info(f"Output: {output_dir}")

        try:
            # Try unsloth for faster merging if available
            if use_unsloth:
                self.logger.info("Attempting unsloth-optimized merge...")
                try:
                    return self._merge_unsloth(
                        adapter_path, base_model, output_dir
                    )
                except ImportError:
                    self.logger.warning("unsloth not available, using standard merge")

            # Standard merge
            return self._merge_standard(
                adapter_path, base_model, output_dir
            )

        except Exception as e:
            self.logger.error(f"Merge failed: {e}")
            return None

    def _merge_unsloth(
        self,
        adapter_path: Path,
        base_model: str,
        output_dir: Path
    ) -> Optional[Path]:
        """Merge using unsloth (fast path)."""
        from unsloth import FastLanguageModel, unsloth_to_gguf

        self.logger.info("Loading base model with unsloth...")
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model,
            max_seq_length=4096,
            dtype=torch.float16,
            load_in_4bit=False,
        )

        self.logger.info("Loading LoRA adapter...")
        model = FastLanguageModel.get_peft_model(
            FastLanguageModel.for_inference(model),
        )

        # Load adapter weights
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, str(adapter_path))

        self.logger.info("Merging weights...")
        model = model.merge_and_unload()

        self.logger.info(f"Saving merged model to {output_dir}...")
        model.save_pretrained(output_dir, safe_serialization=True)
        tokenizer.save_pretrained(output_dir)

        return output_dir

    def _merge_standard(
        self,
        adapter_path: Path,
        base_model: str,
        output_dir: Path
    ) -> Optional[Path]:
        """Merge using standard transformers + peft."""
        self.logger.info("Loading base model...")
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            torch_dtype=torch.float16,
            device_map=self.device,
            trust_remote_code=True
        )

        self.logger.info("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(
            base_model,
            trust_remote_code=True
        )

        self.logger.info("Loading LoRA adapter...")
        model = PeftModel.from_pretrained(base, str(adapter_path))

        self.logger.info("Merging weights...")
        model = model.merge_and_unload()

        self.logger.info(f"Saving merged model to {output_dir}...")
        model.save_pretrained(output_dir, safe_serialization=True)
        tokenizer.save_pretrained(output_dir)

        return output_dir

    def convert_to_gguf(
        self,
        model_dir: str,
        output_path: Optional[str] = None,
        quantization: str = "f16"
    ) -> Optional[Path]:
        """Convert merged model to GGUF format."""
        self.logger.info("\n" + "="*60)
        self.logger.info("CONVERTING TO GGUF")
        self.logger.info("="*60)

        model_dir = Path(model_dir)
        if not model_dir.exists():
            self.logger.error(f"Model directory not found: {model_dir}")
            return None

        if not output_path:
            output_path = model_dir.parent / f"{model_dir.name}.gguf"

        output_path = Path(output_path)

        self.logger.info(f"Model: {model_dir}")
        self.logger.info(f"Output: {output_path}")
        self.logger.info(f"Format: {quantization}")

        # Ensure llama.cpp is available
        llama_cpp_dir = self._ensure_llama_cpp()

        # Convert using llama.cpp script
        convert_script = llama_cpp_dir / "convert_hf_to_gguf.py"

        self.logger.info("Converting with llama.cpp...")
        try:
            subprocess.run(
                [
                    sys.executable,
                    str(convert_script),
                    str(model_dir),
                    "--outfile", str(output_path),
                    "--outtype", "f16"
                ],
                check=True
            )
        except subprocess.CalledProcessError as e:
            self.logger.error(f"Conversion failed: {e}")
            return None

        if not output_path.exists():
            self.logger.error("Conversion did not produce output file")
            return None

        size_gb = output_path.stat().st_size / (1024**3)
        self.logger.info(f"✓ Conversion complete: {size_gb:.2f} GB")

        return output_path

    def quantize(
        self,
        gguf_path: str,
        formats: Optional[List[str]] = None,
        output_dir: Optional[str] = None
    ) -> Dict[str, Path]:
        """Quantize GGUF model in multiple formats."""
        self.logger.info("\n" + "="*60)
        self.logger.info("QUANTIZING GGUF")
        self.logger.info("="*60)

        gguf_path = Path(gguf_path)
        if not gguf_path.exists():
            self.logger.error(f"GGUF file not found: {gguf_path}")
            return {}

        if not formats:
            formats = self.quant_config.get("formats", ["q4_k_m"])

        if not output_dir:
            output_dir = gguf_path.parent

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Build quantize tool
        llama_cpp_dir = self._ensure_llama_cpp()
        quantize_bin = self._build_quantize_tool(llama_cpp_dir)

        results = {}

        for fmt in formats:
            if fmt not in QUANTIZATION_FORMATS:
                self.logger.warning(f"Unknown format: {fmt}, skipping")
                continue

            config = QUANTIZATION_FORMATS[fmt]
            self.logger.info(f"\nQuantizing to {fmt}...")
            self.logger.info(f"  {config.description}")

            output_path = output_dir / f"{gguf_path.stem}-{fmt}.gguf"

            try:
                env = os.environ.copy()
                lib_dir = str((llama_cpp_dir / "build" / "bin").resolve())
                if sys.platform == "darwin":
                    env["DYLD_LIBRARY_PATH"] = (
                        f"{lib_dir}:{env.get('DYLD_LIBRARY_PATH', '')}"
                    ).strip(":")
                elif sys.platform.startswith("linux"):
                    env["LD_LIBRARY_PATH"] = (
                        f"{lib_dir}:{env.get('LD_LIBRARY_PATH', '')}"
                    ).strip(":")

                subprocess.run(
                    [
                        str(quantize_bin),
                        str(gguf_path),
                        str(output_path),
                        fmt
                    ],
                    check=True,
                    env=env
                )

                if output_path.exists():
                    size_gb = output_path.stat().st_size / (1024**3)
                    original_gb = gguf_path.stat().st_size / (1024**3)
                    ratio = (1 - size_gb / original_gb) * 100

                    self.logger.info(
                        f"✓ {fmt}: {size_gb:.2f} GB "
                        f"({ratio:.1f}% compression)"
                    )
                    results[fmt] = output_path
                else:
                    self.logger.error(f"Quantization did not produce output for {fmt}")

            except subprocess.CalledProcessError as e:
                self.logger.error(f"Quantization to {fmt} failed: {e}")

        return results

    def process_pipeline(
        self,
        adapter_path: str,
        base_model: str = "Qwen/Qwen2.5-7B-Instruct",
        output_name: Optional[str] = None,
        quantize_formats: Optional[List[str]] = None,
        cleanup: bool = True
    ) -> Dict[str, Path]:
        """Run complete merge → convert → quantize pipeline."""
        self.logger.info("\n" + "="*60)
        self.logger.info("STARTING COMPLETE PIPELINE")
        self.logger.info("="*60)

        adapter_path = Path(adapter_path)
        if not output_name:
            output_name = adapter_path.stem

        output_dir = adapter_path.parent / output_name
        output_dir.mkdir(parents=True, exist_ok=True)

        start_time = datetime.now()

        try:
            # Step 1: Merge
            merged_dir = self.merge(
                str(adapter_path),
                base_model=base_model,
                output_dir=str(output_dir / "merged_hf")
            )

            if not merged_dir:
                return {}

            # Step 2: Convert to GGUF
            gguf_path = self.convert_to_gguf(
                str(merged_dir),
                output_path=str(output_dir / f"{output_name}.gguf")
            )

            if not gguf_path:
                return {}

            # Step 3: Quantize
            if not quantize_formats:
                quantize_formats = self.quant_config.get("formats", ["q4_k_m"])

            results = self.quantize(
                str(gguf_path),
                formats=quantize_formats,
                output_dir=str(output_dir)
            )

            # Cleanup intermediate files
            if cleanup:
                self.logger.info("Cleaning up intermediate files...")
                import shutil

                if self.merge_config.get("remove_merged_hf", False):
                    shutil.rmtree(merged_dir, ignore_errors=True)
                    self.logger.info("  Removed merged HF directory")

                if (output_dir / f"{output_name}.gguf").exists():
                    (output_dir / f"{output_name}.gguf").unlink()
                    self.logger.info("  Removed f16 GGUF")

            # Summary
            elapsed = (datetime.now() - start_time).total_seconds()
            self.logger.info("\n" + "="*60)
            self.logger.info("PIPELINE COMPLETE!")
            self.logger.info("="*60)
            self.logger.info(f"Elapsed time: {elapsed/60:.1f} minutes")
            self.logger.info("\nOutput models:")
            for fmt, path in results.items():
                size_gb = path.stat().st_size / (1024**3)
                self.logger.info(f"  ✓ {fmt}: {path.name} ({size_gb:.2f} GB)")

            # Save metadata
            self._save_pipeline_metadata(
                output_dir,
                adapter_path,
                base_model,
                results,
                elapsed
            )

            return results

        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            return {}

    def _save_pipeline_metadata(
        self,
        output_dir: Path,
        adapter_path: Path,
        base_model: str,
        results: Dict[str, Path],
        elapsed_seconds: float
    ) -> None:
        """Save pipeline execution metadata."""
        metadata = {
            "timestamp": datetime.now().isoformat(),
            "adapter_path": str(adapter_path),
            "base_model": base_model,
            "elapsed_seconds": elapsed_seconds,
            "output_models": {
                fmt: {
                    "path": str(path),
                    "size_bytes": path.stat().st_size,
                    "size_gb": path.stat().st_size / (1024**3)
                }
                for fmt, path in results.items()
            }
        }

        metadata_path = output_dir / "pipeline_metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        self.logger.info(f"Metadata saved: {metadata_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Merge LoRA adapters and quantize to GGUF"
    )
    parser.add_argument("--adapter", type=str, required=True,
                       help="Path to LoRA adapter")
    parser.add_argument("--base-model", type=str,
                       default="Qwen/Qwen2.5-7B-Instruct",
                       help="Base model name")
    parser.add_argument("--output", type=str,
                       help="Output name (without extension)")
    parser.add_argument("--merge-only", action="store_true",
                       help="Only merge, don't convert/quantize")
    parser.add_argument("--convert-only", action="store_true",
                       help="Only convert to GGUF, don't quantize")
    parser.add_argument("--quantize", type=str,
                       default="q4_k_m",
                       help="Quantization formats (comma-separated)")
    parser.add_argument("--use-unsloth", action="store_true",
                       help="Use unsloth for faster merging")
    parser.add_argument("--config", type=str,
                       help="Path to deployment config")
    parser.add_argument("--cleanup", action="store_true", default=True,
                       help="Remove intermediate files")
    parser.add_argument("--log-level", default="INFO",
                       help="Logging level")

    args = parser.parse_args()

    # Setup logging
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    merger = ModelMerger(config_path=args.config)

    if args.merge_only:
        merged_dir = merger.merge(
            args.adapter,
            base_model=args.base_model,
            output_dir=args.output,
            use_unsloth=args.use_unsloth
        )
        sys.exit(0 if merged_dir else 1)

    elif args.convert_only:
        if not args.output:
            print("Error: --output required for convert-only mode")
            sys.exit(1)
        gguf_path = merger.convert_to_gguf(args.output)
        sys.exit(0 if gguf_path else 1)

    else:
        # Full pipeline
        quantize_formats = [f.strip() for f in args.quantize.split(",")]
        results = merger.process_pipeline(
            args.adapter,
            base_model=args.base_model,
            output_name=args.output,
            quantize_formats=quantize_formats,
            cleanup=args.cleanup
        )

        sys.exit(0 if results else 1)


if __name__ == "__main__":
    main()
