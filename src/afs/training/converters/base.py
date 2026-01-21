"""Base converter interface for training data formats."""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ...generators.base import TrainingSample
    from ..config import CotInclusionMode


class BaseConverter(ABC):
    """Abstract base class for format converters.

    Converters transform TrainingSample objects into framework-specific
    formats for training (MLX, HuggingFace, llama.cpp, etc.).
    """

    def __init__(
        self,
        include_cot: bool = True,
        cot_mode: CotInclusionMode = None,
        system_prompt: str | None = None,
    ):
        """Initialize converter.

        Args:
            include_cot: Whether to include chain of thought
            cot_mode: How to include CoT (separate, embedded, special_tokens)
            system_prompt: Optional system prompt to include
        """
        from ..config import CotInclusionMode

        self.include_cot = include_cot
        self.cot_mode = cot_mode or CotInclusionMode.SEPARATE
        self.system_prompt = system_prompt

    @abstractmethod
    def convert_sample(self, sample: TrainingSample) -> dict[str, Any]:
        """Convert a single sample to target format.

        Args:
            sample: Training sample to convert

        Returns:
            Dictionary in target format
        """
        pass

    def convert_all(self, samples: list[TrainingSample]) -> list[dict[str, Any]]:
        """Convert all samples.

        Args:
            samples: List of training samples

        Returns:
            List of converted dictionaries
        """
        return [self.convert_sample(s) for s in samples]

    def convert_file(
        self,
        input_path: Path,
        output_path: Path,
    ) -> int:
        """Convert a JSONL file.

        Args:
            input_path: Path to input JSONL
            output_path: Path for output file

        Returns:
            Number of samples converted
        """
        import json

        from ...generators.base import read_jsonl

        samples = read_jsonl(input_path)
        converted = self.convert_all(samples)

        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            for item in converted:
                f.write(json.dumps(item) + "\n")

        return len(converted)

    def _format_output_with_cot(self, sample: TrainingSample) -> str:
        """Format output with CoT based on inclusion mode.

        Args:
            sample: Training sample

        Returns:
            Formatted output string
        """
        from ..config import CotInclusionMode

        if not self.include_cot or not sample.thinking:
            return sample.output

        if self.cot_mode == CotInclusionMode.NONE:
            return sample.output

        elif self.cot_mode == CotInclusionMode.EMBEDDED:
            return f"""Let me analyze this step by step:

{sample.thinking}

Based on this analysis, here is the solution:

{sample.output}"""

        elif self.cot_mode == CotInclusionMode.SPECIAL_TOKENS:
            return f"<think>\n{sample.thinking}\n</think>\n\n{sample.output}"

        else:  # SEPARATE - return just output, thinking handled separately
            return sample.output

    def _build_user_message(self, sample: TrainingSample) -> str:
        """Build user message from instruction and input.

        Args:
            sample: Training sample

        Returns:
            Combined user message
        """
        if sample.input:
            return f"{sample.instruction}\n\n{sample.input}"
        return sample.instruction
