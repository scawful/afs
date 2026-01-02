"""Format converters for training data."""

from .base import BaseConverter
from .mlx import MLXConverter, MLXCompletionConverter
from .hf import (
    AlpacaConverter,
    ChatMLConverter,
    ShareGPTConverter,
    UnslothThinkingConverter,
)
from .llama_cpp import LlamaCppConverter, GGUFTrainConverter

__all__ = [
    # Base
    "BaseConverter",
    # MLX
    "MLXConverter",
    "MLXCompletionConverter",
    # HuggingFace/Unsloth
    "AlpacaConverter",
    "ChatMLConverter",
    "ShareGPTConverter",
    "UnslothThinkingConverter",
    # llama.cpp
    "LlamaCppConverter",
    "GGUFTrainConverter",
]


def get_converter(
    format_name: str,
    include_cot: bool = True,
    cot_mode: str = "separate",
    **kwargs,
) -> BaseConverter:
    """Factory function to get converter by name.

    Args:
        format_name: Converter name (mlx, alpaca, chatml, sharegpt, llama_cpp, etc.)
        include_cot: Whether to include chain of thought
        cot_mode: CoT mode (none, separate, embedded, special_tokens)
        **kwargs: Additional converter-specific arguments

    Returns:
        Configured converter instance
    """
    from ..config import CotInclusionMode

    # Parse CoT mode
    cot_mode_enum = CotInclusionMode(cot_mode)

    converters = {
        "mlx": MLXConverter,
        "mlx_chat": MLXConverter,
        "mlx_completion": MLXCompletionConverter,
        "alpaca": AlpacaConverter,
        "chatml": ChatMLConverter,
        "sharegpt": ShareGPTConverter,
        "unsloth_thinking": UnslothThinkingConverter,
        "llama_cpp": LlamaCppConverter,
        "gguf": GGUFTrainConverter,
    }

    if format_name not in converters:
        available = ", ".join(converters.keys())
        raise ValueError(f"Unknown format: {format_name}. Available: {available}")

    return converters[format_name](
        include_cot=include_cot,
        cot_mode=cot_mode_enum,
        **kwargs,
    )
