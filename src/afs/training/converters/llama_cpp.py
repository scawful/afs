"""llama.cpp format converter for CPU/GGUF training."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .base import BaseConverter

if TYPE_CHECKING:
    from ...generators.base import TrainingSample


class LlamaCppConverter(BaseConverter):
    """Convert samples to llama.cpp finetune format.

    llama.cpp finetune expects simple prompt-completion pairs:
    {
        "prompt": "...",
        "completion": "..."
    }

    Or for chat-style:
    {
        "text": "<full formatted text with special tokens>"
    }
    """

    DEFAULT_SYSTEM = """You are an expert 65816 SNES assembly programmer."""

    # Common prompt templates for different model types
    TEMPLATES = {
        "llama3": {
            "bos": "<|begin_of_text|>",
            "system_start": "<|start_header_id|>system<|end_header_id|>\n\n",
            "system_end": "<|eot_id|>",
            "user_start": "<|start_header_id|>user<|end_header_id|>\n\n",
            "user_end": "<|eot_id|>",
            "assistant_start": "<|start_header_id|>assistant<|end_header_id|>\n\n",
            "assistant_end": "<|eot_id|>",
        },
        "chatml": {
            "bos": "",
            "system_start": "<|im_start|>system\n",
            "system_end": "<|im_end|>\n",
            "user_start": "<|im_start|>user\n",
            "user_end": "<|im_end|>\n",
            "assistant_start": "<|im_start|>assistant\n",
            "assistant_end": "<|im_end|>",
        },
        "mistral": {
            "bos": "<s>",
            "system_start": "[INST] ",
            "system_end": "",
            "user_start": "",
            "user_end": " [/INST]",
            "assistant_start": "",
            "assistant_end": "</s>",
        },
        "alpaca": {
            "bos": "",
            "system_start": "### System:\n",
            "system_end": "\n\n",
            "user_start": "### Instruction:\n",
            "user_end": "\n\n### Response:\n",
            "assistant_start": "",
            "assistant_end": "",
        },
    }

    def __init__(
        self,
        include_cot: bool = True,
        cot_mode: CotInclusionMode = None,
        system_prompt: str | None = None,
        template: str = "chatml",
        as_prompt_completion: bool = False,
    ):
        """Initialize llama.cpp converter.

        Args:
            include_cot: Whether to include chain of thought
            cot_mode: How to include CoT
            system_prompt: Custom system prompt
            template: Template name (llama3, chatml, mistral, alpaca)
            as_prompt_completion: If True, return prompt/completion instead of text
        """
        super().__init__(
            include_cot=include_cot,
            cot_mode=cot_mode,
            system_prompt=system_prompt or self.DEFAULT_SYSTEM,
        )

        if template not in self.TEMPLATES:
            available = ", ".join(self.TEMPLATES.keys())
            raise ValueError(f"Unknown template: {template}. Available: {available}")

        self.template_name = template
        self.template = self.TEMPLATES[template]
        self.as_prompt_completion = as_prompt_completion

    def convert_sample(self, sample: TrainingSample) -> dict[str, Any]:
        """Convert sample to llama.cpp format.

        Args:
            sample: Training sample

        Returns:
            Dict with 'text' or 'prompt'/'completion' keys
        """
        t = self.template
        user_content = self._build_user_message(sample)
        output = self._format_output_with_cot(sample)

        if self.as_prompt_completion:
            # Build prompt without assistant response
            prompt = (
                f"{t['bos']}"
                f"{t['system_start']}{self.system_prompt}{t['system_end']}"
                f"{t['user_start']}{user_content}{t['user_end']}"
                f"{t['assistant_start']}"
            )
            completion = f"{output}{t['assistant_end']}"
            return {"prompt": prompt, "completion": completion}
        else:
            # Build full text
            text = (
                f"{t['bos']}"
                f"{t['system_start']}{self.system_prompt}{t['system_end']}"
                f"{t['user_start']}{user_content}{t['user_end']}"
                f"{t['assistant_start']}{output}{t['assistant_end']}"
            )
            return {"text": text}

    def get_template_info(self) -> dict[str, str]:
        """Get information about current template.

        Returns:
            Dict with template name and tokens
        """
        return {
            "name": self.template_name,
            **self.template,
        }


class GGUFTrainConverter(BaseConverter):
    """Convert samples for GGUF training with llama.cpp.

    Produces format compatible with llama.cpp's train-text tool:
    - Simple text file with samples separated by special tokens
    - Or JSONL with text field
    """

    def __init__(
        self,
        include_cot: bool = True,
        cot_mode: CotInclusionMode = None,
        separator: str = "\n<|endoftext|>\n",
    ):
        """Initialize GGUF train converter.

        Args:
            include_cot: Whether to include chain of thought
            cot_mode: How to include CoT
            separator: Token/text to separate samples
        """
        super().__init__(include_cot=include_cot, cot_mode=cot_mode)
        self.separator = separator

    def convert_sample(self, sample: TrainingSample) -> dict[str, Any]:
        """Convert sample to simple text format.

        Args:
            sample: Training sample

        Returns:
            Dict with 'text' key
        """
        user_content = self._build_user_message(sample)
        output = self._format_output_with_cot(sample)

        # Simple Q&A format
        text = f"Question: {user_content}\n\nAnswer: {output}"
        return {"text": text}

    def convert_to_text_file(
        self,
        samples: list[TrainingSample],
        output_path: str,
    ) -> int:
        """Convert samples to a single text file.

        Args:
            samples: List of training samples
            output_path: Path for output text file

        Returns:
            Number of samples written
        """
        from pathlib import Path

        converted = self.convert_all(samples)
        texts = [item["text"] for item in converted]

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            f.write(self.separator.join(texts))

        return len(converted)
