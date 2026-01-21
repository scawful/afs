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
from .toolbench import load_toolbench_dataset, export_toolbench_to_jsonl

_CONVERTER_REGISTRY: dict[str, type[BaseConverter]] = {}

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
    # ToolBench
    "load_toolbench_dataset",
    "export_toolbench_to_jsonl",
    # Registry helpers
    "available_converters",
    "register_converter",
    "get_converter",
]


def register_converter(
    name: str,
    converter: type[BaseConverter],
    *,
    overwrite: bool = False,
) -> None:
    """Register a converter by name."""
    key = name.strip().lower()
    if not overwrite and key in _CONVERTER_REGISTRY:
        raise ValueError(f"Converter already registered: {key}")
    _CONVERTER_REGISTRY[key] = converter


def available_converters() -> list[str]:
    """Return sorted converter names."""
    from ...plugins import load_enabled_plugins

    load_enabled_plugins()
    return sorted(_CONVERTER_REGISTRY)


def _register_builtin_converters() -> None:
    if _CONVERTER_REGISTRY:
        return
    register_converter("mlx", MLXConverter)
    register_converter("mlx_chat", MLXConverter)
    register_converter("mlx_completion", MLXCompletionConverter)
    register_converter("alpaca", AlpacaConverter)
    register_converter("chatml", ChatMLConverter)
    register_converter("sharegpt", ShareGPTConverter)
    register_converter("unsloth_thinking", UnslothThinkingConverter)
    register_converter("llama_cpp", LlamaCppConverter)
    register_converter("gguf", GGUFTrainConverter)


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
    from ...plugins import load_enabled_plugins

    load_enabled_plugins()

    # Parse CoT mode
    cot_mode_enum = CotInclusionMode(cot_mode)

    key = format_name.strip().lower()
    converter = _CONVERTER_REGISTRY.get(key)
    if converter is None:
        available = ", ".join(available_converters())
        raise ValueError(f"Unknown format: {format_name}. Available: {available}")

    return converter(
        include_cot=include_cot,
        cot_mode=cot_mode_enum,
        **kwargs,
    )


_register_builtin_converters()
