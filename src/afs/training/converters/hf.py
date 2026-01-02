"""HuggingFace/Unsloth format converter."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base import BaseConverter

if TYPE_CHECKING:
    from ...generators.base import TrainingSample


class AlpacaConverter(BaseConverter):
    """Convert samples to Alpaca format for HuggingFace/Unsloth.

    Alpaca format:
    {
        "instruction": "...",
        "input": "...",
        "output": "..."
    }
    """

    def convert_sample(self, sample: "TrainingSample") -> dict[str, Any]:
        """Convert sample to Alpaca format.

        Args:
            sample: Training sample

        Returns:
            Dict with instruction, input, output
        """
        output = self._format_output_with_cot(sample)

        return {
            "instruction": sample.instruction,
            "input": sample.input or "",
            "output": output,
        }


class ChatMLConverter(BaseConverter):
    """Convert samples to ChatML format for Unsloth/HuggingFace.

    ChatML format (used by many models including Qwen):
    {
        "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ]
    }

    Or as text:
    <|im_start|>system
    {system_prompt}<|im_end|>
    <|im_start|>user
    {user_message}<|im_end|>
    <|im_start|>assistant
    {assistant_response}<|im_end|>
    """

    DEFAULT_SYSTEM = """You are an expert 65816 SNES assembly programmer. You help with writing, optimizing, and debugging assembly code for the Super Nintendo Entertainment System."""

    def __init__(
        self,
        include_cot: bool = True,
        cot_mode: "CotInclusionMode" = None,
        system_prompt: str | None = None,
        as_text: bool = False,
    ):
        """Initialize ChatML converter.

        Args:
            include_cot: Whether to include chain of thought
            cot_mode: How to include CoT
            system_prompt: Custom system prompt
            as_text: If True, return formatted text instead of messages dict
        """
        super().__init__(
            include_cot=include_cot,
            cot_mode=cot_mode,
            system_prompt=system_prompt or self.DEFAULT_SYSTEM,
        )
        self.as_text = as_text

    def convert_sample(self, sample: "TrainingSample") -> dict[str, Any]:
        """Convert sample to ChatML format.

        Args:
            sample: Training sample

        Returns:
            Dict with 'messages' or 'text' key
        """
        user_content = self._build_user_message(sample)
        output = self._format_output_with_cot(sample)

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output},
        ]

        if self.as_text:
            # Format as ChatML text
            lines = []
            for msg in messages:
                lines.append(f"<|im_start|>{msg['role']}")
                lines.append(f"{msg['content']}<|im_end|>")
            return {"text": "\n".join(lines)}

        return {"messages": messages}


class ShareGPTConverter(BaseConverter):
    """Convert samples to ShareGPT format.

    ShareGPT format (common for fine-tuning):
    {
        "conversations": [
            {"from": "system", "value": "..."},
            {"from": "human", "value": "..."},
            {"from": "gpt", "value": "..."}
        ]
    }
    """

    DEFAULT_SYSTEM = """You are an expert 65816 SNES assembly programmer."""

    def __init__(
        self,
        include_cot: bool = True,
        cot_mode: "CotInclusionMode" = None,
        system_prompt: str | None = None,
    ):
        """Initialize ShareGPT converter."""
        super().__init__(
            include_cot=include_cot,
            cot_mode=cot_mode,
            system_prompt=system_prompt or self.DEFAULT_SYSTEM,
        )

    def convert_sample(self, sample: "TrainingSample") -> dict[str, Any]:
        """Convert sample to ShareGPT format.

        Args:
            sample: Training sample

        Returns:
            Dict with 'conversations' key
        """
        user_content = self._build_user_message(sample)
        output = self._format_output_with_cot(sample)

        conversations = [
            {"from": "system", "value": self.system_prompt},
            {"from": "human", "value": user_content},
            {"from": "gpt", "value": output},
        ]

        return {"conversations": conversations}


class UnslothThinkingConverter(BaseConverter):
    """Convert samples to Unsloth thinking format.

    For models that support extended thinking (DeepSeek-R1, QwQ, etc.),
    uses special tokens to mark thinking sections.
    """

    DEFAULT_SYSTEM = """You are an expert 65816 SNES assembly programmer with deep reasoning capabilities. Think through problems step by step before providing solutions."""

    def __init__(
        self,
        include_cot: bool = True,
        system_prompt: str | None = None,
        thinking_start: str = "<think>",
        thinking_end: str = "</think>",
    ):
        """Initialize thinking converter.

        Args:
            include_cot: Whether to include chain of thought
            system_prompt: Custom system prompt
            thinking_start: Token to mark start of thinking
            thinking_end: Token to mark end of thinking
        """
        from ..config import CotInclusionMode

        super().__init__(
            include_cot=include_cot,
            cot_mode=CotInclusionMode.SPECIAL_TOKENS,
            system_prompt=system_prompt or self.DEFAULT_SYSTEM,
        )
        self.thinking_start = thinking_start
        self.thinking_end = thinking_end

    def convert_sample(self, sample: "TrainingSample") -> dict[str, Any]:
        """Convert sample with explicit thinking section.

        Args:
            sample: Training sample

        Returns:
            Dict with 'messages' key, thinking in special tokens
        """
        user_content = self._build_user_message(sample)

        # Build output with thinking
        if self.include_cot and sample.thinking:
            output = f"{self.thinking_start}\n{sample.thinking}\n{self.thinking_end}\n\n{sample.output}"
        else:
            output = sample.output

        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_content},
            {"role": "assistant", "content": output},
        ]

        return {"messages": messages}
