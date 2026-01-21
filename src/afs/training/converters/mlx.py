"""MLX format converter for Apple Silicon training."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base import BaseConverter

if TYPE_CHECKING:
    from ...generators.base import TrainingSample


# Default system prompt for 65816 assembly tasks
DEFAULT_SYSTEM_PROMPT = """You are an expert 65816 SNES assembly programmer. You help with writing, optimizing, and debugging assembly code for the Super Nintendo Entertainment System. You understand the 65816 processor architecture, SNES memory maps, and common ROM hacking techniques."""


class MLXConverter(BaseConverter):
    """Convert samples to MLX-LM chat format.

    MLX-LM expects chat messages in this format:
    {
        "messages": [
            {"role": "system", "content": "..."},
            {"role": "user", "content": "..."},
            {"role": "assistant", "content": "..."}
        ]
    }
    """

    def __init__(
        self,
        include_cot: bool = True,
        cot_mode: CotInclusionMode = None,
        system_prompt: str | None = None,
        include_system: bool = True,
    ):
        """Initialize MLX converter.

        Args:
            include_cot: Whether to include chain of thought
            cot_mode: How to include CoT
            system_prompt: Custom system prompt (default: assembly expert)
            include_system: Whether to include system message
        """
        super().__init__(
            include_cot=include_cot,
            cot_mode=cot_mode,
            system_prompt=system_prompt or DEFAULT_SYSTEM_PROMPT,
        )
        self.include_system = include_system

    def convert_sample(self, sample: TrainingSample) -> dict[str, Any]:
        """Convert sample to MLX chat format.

        Args:
            sample: Training sample

        Returns:
            Dict with 'messages' list
        """
        from ..config import CotInclusionMode

        messages = []

        # System message
        if self.include_system and self.system_prompt:
            messages.append({
                "role": "system",
                "content": self.system_prompt,
            })

        # User message
        user_content = self._build_user_message(sample)
        messages.append({
            "role": "user",
            "content": user_content,
        })

        # Assistant response
        output = self._format_output_with_cot(sample)

        # For SEPARATE mode with MLX, we can use extended thinking format
        if (
            self.include_cot
            and sample.thinking
            and self.cot_mode == CotInclusionMode.SEPARATE
        ):
            # MLX supports thinking in special format for some models
            messages.append({
                "role": "assistant",
                "content": output,
                "thinking": sample.thinking,  # May be ignored by tokenizer
            })
        else:
            messages.append({
                "role": "assistant",
                "content": output,
            })

        return {"messages": messages}


class MLXCompletionConverter(BaseConverter):
    """Convert samples to MLX-LM completion format.

    For models that don't support chat format, use simple prompt-completion:
    {
        "prompt": "...",
        "completion": "..."
    }
    """

    def __init__(
        self,
        include_cot: bool = True,
        cot_mode: CotInclusionMode = None,
        prompt_template: str | None = None,
    ):
        """Initialize completion converter.

        Args:
            include_cot: Whether to include chain of thought
            cot_mode: How to include CoT
            prompt_template: Template for formatting prompt
        """
        super().__init__(include_cot=include_cot, cot_mode=cot_mode)
        self.prompt_template = prompt_template or "### Instruction:\n{instruction}\n\n### Response:\n"

    def convert_sample(self, sample: TrainingSample) -> dict[str, Any]:
        """Convert sample to completion format.

        Args:
            sample: Training sample

        Returns:
            Dict with 'prompt' and 'completion'
        """
        user_content = self._build_user_message(sample)
        prompt = self.prompt_template.format(instruction=user_content)
        completion = self._format_output_with_cot(sample)

        return {
            "prompt": prompt,
            "completion": completion,
        }
