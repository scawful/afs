"""Configuration classes for model training."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any


class Framework(str, Enum):
    """Supported training frameworks."""

    MLX = "mlx"
    UNSLOTH = "unsloth"
    LLAMA_CPP = "llama_cpp"


class CotInclusionMode(str, Enum):
    """How to include Chain of Thought in training data."""

    NONE = "none"  # Exclude CoT entirely
    SEPARATE = "separate"  # Keep in separate thinking field
    EMBEDDED = "embedded"  # Embed in output text
    SPECIAL_TOKENS = "special_tokens"  # Use <think>...</think> markers


@dataclass
class LoRAConfig:
    """LoRA adapter configuration."""

    r: int = 16  # Rank
    alpha: int = 32  # Scaling factor
    dropout: float = 0.05
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "r": self.r,
            "alpha": self.alpha,
            "dropout": self.dropout,
            "target_modules": self.target_modules,
        }


@dataclass
class DatasetConfig:
    """Dataset configuration for training."""

    input_path: Path
    output_dir: Path
    train_ratio: float = 0.8
    val_ratio: float = 0.1
    test_ratio: float = 0.1
    stratify_by: str | None = "domain"  # Field to stratify by
    shuffle: bool = True
    seed: int = 42
    cot_mode: CotInclusionMode = CotInclusionMode.SEPARATE
    min_length: int = 10  # Minimum output length
    max_length: int = 8192  # Maximum combined length

    def __post_init__(self):
        """Validate ratios sum to 1.0."""
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Ratios must sum to 1.0, got {total}")


@dataclass
class TrainingConfig:
    """Complete training configuration."""

    run_name: str
    base_model: str  # e.g., "Qwen/Qwen2.5-Coder-7B-Instruct"
    output_dir: Path
    framework: Framework = Framework.MLX

    # Dataset
    dataset: DatasetConfig | None = None

    # LoRA
    lora: LoRAConfig = field(default_factory=LoRAConfig)

    # Training hyperparameters
    num_epochs: int = 3
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 2e-4
    warmup_steps: int = 100
    max_seq_length: int = 2048
    weight_decay: float = 0.01

    # Mixed precision
    bf16: bool = True
    fp16: bool = False

    # Checkpointing
    save_steps: int = 500
    eval_steps: int = 100
    logging_steps: int = 10

    # Memory optimization
    gradient_checkpointing: bool = True
    use_flash_attention: bool = True

    # Metadata
    tags: list[str] = field(default_factory=list)
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "run_name": self.run_name,
            "base_model": self.base_model,
            "output_dir": str(self.output_dir),
            "framework": self.framework.value,
            "lora": self.lora.to_dict(),
            "num_epochs": self.num_epochs,
            "batch_size": self.batch_size,
            "gradient_accumulation_steps": self.gradient_accumulation_steps,
            "learning_rate": self.learning_rate,
            "warmup_steps": self.warmup_steps,
            "max_seq_length": self.max_seq_length,
            "weight_decay": self.weight_decay,
            "bf16": self.bf16,
            "fp16": self.fp16,
            "save_steps": self.save_steps,
            "eval_steps": self.eval_steps,
            "logging_steps": self.logging_steps,
            "gradient_checkpointing": self.gradient_checkpointing,
            "use_flash_attention": self.use_flash_attention,
            "tags": self.tags,
            "description": self.description,
        }


# Preset configurations for different models
MODEL_PRESETS: dict[str, dict[str, Any]] = {
    "qwen2.5-coder-7b": {
        "base_model": "Qwen/Qwen2.5-Coder-7B-Instruct",
        "max_seq_length": 4096,
        "lora_r": 16,
        "lora_alpha": 32,
    },
    "qwen2.5-coder-14b": {
        "base_model": "Qwen/Qwen2.5-Coder-14B-Instruct",
        "max_seq_length": 4096,
        "lora_r": 16,
        "lora_alpha": 32,
        "batch_size": 2,  # Smaller batch for larger model
    },
    "deepseek-r1-distill-7b": {
        "base_model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        "max_seq_length": 4096,
        "lora_r": 16,
        "lora_alpha": 32,
    },
    "deepseek-r1-distill-14b": {
        "base_model": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "max_seq_length": 4096,
        "lora_r": 16,
        "lora_alpha": 32,
        "batch_size": 2,
    },
}


def get_model_preset(preset_name: str) -> dict[str, Any]:
    """Get a preset configuration for a model."""
    if preset_name not in MODEL_PRESETS:
        available = ", ".join(MODEL_PRESETS.keys())
        raise ValueError(f"Unknown preset: {preset_name}. Available: {available}")
    return MODEL_PRESETS[preset_name].copy()
