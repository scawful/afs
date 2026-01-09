#!/usr/bin/env python3
"""
Verify Training Environment Setup

Checks all requirements for training MCP tool specialist models:
- Python version
- GPU availability and specs
- Required packages
- Disk space
- Memory
- Data files

Usage:
    python3 verify_training_setup.py
    python3 verify_training_setup.py --verbose
"""

import sys
import platform
import subprocess
from pathlib import Path
import argparse


class Colors:
    """ANSI color codes for terminal output"""
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    BLUE = '\033[94m'
    END = '\033[0m'
    BOLD = '\033[1m'


def print_header(text):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text:^60}{Colors.END}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.END}\n")


def check_python_version():
    """Check Python version (>= 3.9 required)"""
    print(f"{Colors.BOLD}Python Version:{Colors.END}")
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}.{version.micro}"

    if version.major == 3 and version.minor >= 9:
        print(f"  ✅ {Colors.GREEN}Python {version_str} (OK){Colors.END}")
        return True
    else:
        print(f"  ❌ {Colors.RED}Python {version_str} (Need >= 3.9){Colors.END}")
        return False


def check_gpu():
    """Check for GPU availability and specs"""
    print(f"\n{Colors.BOLD}GPU Check:{Colors.END}")

    try:
        import torch
        if not torch.cuda.is_available():
            print(f"  ⚠️  {Colors.YELLOW}No CUDA GPU detected{Colors.END}")
            print(f"     Training will be VERY slow on CPU")
            print(f"     Recommended: RTX 4090 (24GB) or A100 (40GB)")
            return False

        # Get GPU info
        gpu_count = torch.cuda.device_count()
        print(f"  ✅ {Colors.GREEN}CUDA available ({gpu_count} GPU(s)){Colors.END}")

        for i in range(gpu_count):
            props = torch.cuda.get_device_properties(i)
            vram_gb = props.total_memory / (1024**3)
            print(f"     GPU {i}: {props.name}")
            print(f"            VRAM: {vram_gb:.1f} GB")
            print(f"            Compute: {props.major}.{props.minor}")

            # Check VRAM
            if vram_gb < 20:
                print(f"     ⚠️  {Colors.YELLOW}Low VRAM ({vram_gb:.1f}GB < 20GB recommended){Colors.END}")
                print(f"        Consider using 8-bit quantization (--use-8bit)")

        return True

    except ImportError:
        print(f"  ❌ {Colors.RED}PyTorch not installed{Colors.END}")
        return False
    except Exception as e:
        print(f"  ❌ {Colors.RED}GPU check failed: {e}{Colors.END}")
        return False


def check_packages(verbose=False):
    """Check required Python packages"""
    print(f"\n{Colors.BOLD}Required Packages:{Colors.END}")

    required = [
        ("torch", "PyTorch"),
        ("transformers", "HuggingFace Transformers"),
        ("peft", "LoRA/PEFT"),
        ("accelerate", "Accelerate"),
        ("datasets", "HF Datasets"),
    ]

    optional = [
        ("wandb", "Weights & Biases (optional)"),
        ("bitsandbytes", "8-bit quantization (optional)"),
        ("deepspeed", "Distributed training (optional)"),
    ]

    all_ok = True

    for package, name in required:
        try:
            __import__(package)
            if verbose:
                mod = __import__(package)
                version = getattr(mod, '__version__', 'unknown')
                print(f"  ✅ {Colors.GREEN}{name:<30} {version}{Colors.END}")
            else:
                print(f"  ✅ {Colors.GREEN}{name}{Colors.END}")
        except ImportError:
            print(f"  ❌ {Colors.RED}{name} (NOT INSTALLED){Colors.END}")
            all_ok = False

    if verbose:
        print(f"\n{Colors.BOLD}Optional Packages:{Colors.END}")
        for package, name in optional:
            try:
                __import__(package)
                mod = __import__(package)
                version = getattr(mod, '__version__', 'unknown')
                print(f"  ✅ {Colors.GREEN}{name:<30} {version}{Colors.END}")
            except ImportError:
                print(f"  ⚠️  {Colors.YELLOW}{name} (not installed){Colors.END}")

    return all_ok


def check_disk_space():
    """Check available disk space"""
    print(f"\n{Colors.BOLD}Disk Space:{Colors.END}")

    try:
        import shutil
        stat = shutil.disk_usage(Path.home())
        free_gb = stat.free / (1024**3)

        print(f"  Free space: {free_gb:.1f} GB")

        if free_gb >= 100:
            print(f"  ✅ {Colors.GREEN}Sufficient space (>100GB){Colors.END}")
            return True
        elif free_gb >= 50:
            print(f"  ⚠️  {Colors.YELLOW}Limited space ({free_gb:.1f}GB){Colors.END}")
            print(f"     Recommended: 100GB for model + checkpoints")
            return True
        else:
            print(f"  ❌ {Colors.RED}Insufficient space ({free_gb:.1f}GB < 50GB){Colors.END}")
            return False

    except Exception as e:
        print(f"  ⚠️  {Colors.YELLOW}Could not check disk space: {e}{Colors.END}")
        return True


def check_memory():
    """Check system RAM"""
    print(f"\n{Colors.BOLD}System Memory:{Colors.END}")

    try:
        import psutil
        mem = psutil.virtual_memory()
        total_gb = mem.total / (1024**3)
        available_gb = mem.available / (1024**3)

        print(f"  Total RAM: {total_gb:.1f} GB")
        print(f"  Available: {available_gb:.1f} GB")

        if total_gb >= 32:
            print(f"  ✅ {Colors.GREEN}Sufficient RAM (>= 32GB){Colors.END}")
            return True
        elif total_gb >= 16:
            print(f"  ⚠️  {Colors.YELLOW}Limited RAM ({total_gb:.1f}GB){Colors.END}")
            print(f"     Recommended: 32GB")
            print(f"     Consider reducing batch size")
            return True
        else:
            print(f"  ❌ {Colors.RED}Insufficient RAM ({total_gb:.1f}GB < 16GB){Colors.END}")
            return False

    except ImportError:
        print(f"  ⚠️  {Colors.YELLOW}psutil not installed, skipping check{Colors.END}")
        return True
    except Exception as e:
        print(f"  ⚠️  {Colors.YELLOW}Could not check memory: {e}{Colors.END}")
        return True


def check_data_files():
    """Check training data files exist"""
    print(f"\n{Colors.BOLD}Training Data:{Colors.END}")

    base_path = Path(__file__).parent.parent / "training_formatted"

    required_files = [
        ("train.jsonl", "Training set"),
        ("val.jsonl", "Validation set"),
        ("test.jsonl", "Test set"),
        ("split_stats.json", "Split statistics")
    ]

    all_ok = True

    for filename, description in required_files:
        filepath = base_path / filename
        if filepath.exists():
            size_kb = filepath.stat().st_size / 1024
            if filename.endswith('.jsonl'):
                # Count lines
                lines = sum(1 for _ in open(filepath))
                print(f"  ✅ {Colors.GREEN}{description:<20} {lines:>4} examples ({size_kb:.1f} KB){Colors.END}")
            else:
                print(f"  ✅ {Colors.GREEN}{description:<20} ({size_kb:.1f} KB){Colors.END}")
        else:
            print(f"  ❌ {Colors.RED}{description} (NOT FOUND){Colors.END}")
            all_ok = False

    return all_ok


def estimate_training_time():
    """Estimate training time based on hardware"""
    print(f"\n{Colors.BOLD}Training Time Estimates:{Colors.END}")

    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_properties(0).name

            if "4090" in gpu_name or "A100" in gpu_name:
                print(f"  GPU: {gpu_name}")
                print(f"  Estimated time per model: 2-4 hours")
                print(f"  All 3 specialists: 6-12 hours total")
            elif "3090" in gpu_name or "A6000" in gpu_name:
                print(f"  GPU: {gpu_name}")
                print(f"  Estimated time per model: 3-5 hours")
                print(f"  All 3 specialists: 9-15 hours total")
            else:
                print(f"  GPU: {gpu_name}")
                print(f"  Estimated time: Varies (4-8 hours per model)")
        else:
            print(f"  ⚠️  No GPU detected")
            print(f"  CPU training: 50-100+ hours per model (NOT RECOMMENDED)")

    except:
        print(f"  Could not estimate training time")


def print_next_steps(all_checks_passed):
    """Print next steps based on verification results"""
    print_header("Next Steps")

    if all_checks_passed:
        print(f"{Colors.GREEN}✅ All critical checks passed!{Colors.END}\n")
        print("Ready to start training. Run:")
        print(f"\n  {Colors.BOLD}python3 train_specialist.py \\{Colors.END}")
        print(f"    --model veran-tools \\")
        print(f"    --train-data ../training_formatted/train.jsonl \\")
        print(f"    --val-data ../training_formatted/val.jsonl \\")
        print(f"    --output ../models/veran-tools-lora \\")
        print(f"    --epochs 3")
        print()
    else:
        print(f"{Colors.YELLOW}⚠️  Some checks failed{Colors.END}\n")
        print("To install missing packages:")
        print(f"\n  {Colors.BOLD}pip install -r ../requirements-training.txt{Colors.END}\n")
        print("Then re-run this script to verify.")


def main():
    parser = argparse.ArgumentParser(description="Verify training environment")
    parser.add_argument('--verbose', '-v', action='store_true', help='Show package versions')
    args = parser.parse_args()

    print_header("MCP Tool Specialist Training Environment Check")

    checks = {
        "Python": check_python_version(),
        "GPU": check_gpu(),
        "Packages": check_packages(args.verbose),
        "Disk": check_disk_space(),
        "Memory": check_memory(),
        "Data": check_data_files()
    }

    estimate_training_time()

    # Summary
    print_header("Summary")

    passed = sum(1 for v in checks.values() if v)
    total = len(checks)

    for name, result in checks.items():
        status = f"{Colors.GREEN}✅ PASS{Colors.END}" if result else f"{Colors.RED}❌ FAIL{Colors.END}"
        print(f"  {name:<15} {status}")

    print(f"\n  {Colors.BOLD}Score: {passed}/{total} checks passed{Colors.END}\n")

    all_passed = all(checks.values())
    print_next_steps(all_passed)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
